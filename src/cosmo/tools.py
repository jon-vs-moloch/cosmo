"""Strict JSON-serializable tool surface over ControlPlane and AtomicHarness.

Tools are the only action channel available to the local chat gateway. They
validate arguments against declared JSON-schema-like contracts, dispatch to
durable control-plane / harness methods, and never grant shell or external
provider actions. Promotion always routes through AtomicHarness validation and
review gates.

No tool performs shell, network, or cloud-provider actions. All handlers return
JSON-serializable records only (never live RepositoryInspector objects).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .atomic import AtomicHarness, AtomicHarnessError, AtomicStore
from .authority import (
    AuthorityError,
    CallerProvenance,
    RepoRootCapability,
    authenticate_operator,
    require_operator,
)
from .context import (
    DEFAULT_MAX_MAP_ENTRIES,
    DEFAULT_MAX_READ_BYTES,
    DEFAULT_MAX_SEARCH_MATCHES,
    ContextError,
    RepositoryInspector,
)
from .control import ControlPlane, ControlPlaneError
from .integration import IntegrationError, IntegrationService
from .help import harness_help_payload
from .planning import PlanningError, PlanningService
from .triage import TriageError, TriageService


class ToolError(RuntimeError):
    """A tool definition, argument, or dispatch failure."""


# Hard caller caps for user-facing repository/context tools. Enforced in
# ToolDispatcher *before* any RepositoryInspector construction or filesystem
# read so a model cannot pass unbounded max_* through the chat surface.
# Lower-level RepositoryInspector APIs remain unbounded for trusted callers.
TOOL_MAX_MAP_ENTRIES = DEFAULT_MAX_MAP_ENTRIES  # 2000
TOOL_MAX_TREE_ENTRIES = DEFAULT_MAX_MAP_ENTRIES  # 2000
TOOL_MAX_READ_BYTES = DEFAULT_MAX_READ_BYTES  # 64 KiB
TOOL_MAX_SEARCH_MATCHES = DEFAULT_MAX_SEARCH_MATCHES  # 50


def _enforce_tool_limit(
    name: str,
    value: Any,
    *,
    cap: int,
) -> int:
    """Validate a positive integer limit that does not exceed *cap*.

    Raises structured ToolError for missing/non-integer/non-positive/over-cap
    values. Does not touch the filesystem.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"{name} must be a positive integer (got {type(value).__name__})")
    if value <= 0:
        raise ToolError(f"{name} must be positive")
    if value > cap:
        raise ToolError(f"{name} {value} exceeds tool cap {cap}")
    return value


# JSON types accepted in tool arguments (strict JSON-serializable subset).
_JSON_TYPES = frozenset({"string", "integer", "number", "boolean", "object", "array"})


@dataclass(frozen=True)
class ToolParameter:
    """One named argument in a tool definition."""

    name: str
    type: str
    description: str
    required: bool = True
    items_type: str | None = None  # for array items

    def to_json_schema_property(self) -> dict[str, Any]:
        if self.type not in _JSON_TYPES:
            raise ToolError(f"unsupported parameter type: {self.type!r}")
        prop: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.type == "array":
            prop["items"] = {"type": self.items_type or "string"}
        return prop


@dataclass(frozen=True)
class ToolDefinition:
    """JSON-serializable tool contract exposed to the chat model."""

    name: str
    description: str
    parameters: tuple[ToolParameter, ...] = ()

    def to_json_dict(self, *, compact: bool = False) -> dict[str, Any]:
        properties = {
            param.name: param.to_json_schema_property() for param in self.parameters
        }
        if compact:
            # Keep names/types/requiredness for tool-call validity while moving
            # long field prose to harness_help. This materially reduces the
            # chat system prompt for small-context local models.
            for property_schema in properties.values():
                property_schema.pop("description", None)
        required = [param.name for param in self.parameters if param.required]
        description = self.description
        if compact:
            # Keep a terse routing hint; detailed field semantics are available
            # from the read-only harness_help tool.
            description = " ".join(self.description.split())[:160]
        return {
            "name": self.name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        }


def _is_json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def validate_tool_arguments(
    definition: ToolDefinition, arguments: Any
) -> dict[str, Any]:
    """Validate and normalize tool arguments against a ToolDefinition.

    Returns a plain dict suitable for dispatch. Raises ToolError with a
    structured message on any violation (unknown keys, missing required,
    wrong types, non-JSON values).
    """
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        raise ToolError("tool arguments must be a JSON object")
    if not _is_json_serializable(dict(arguments)):
        raise ToolError("tool arguments must be JSON-serializable")

    known = {param.name: param for param in definition.parameters}
    unknown = sorted(set(arguments) - set(known))
    if unknown:
        raise ToolError(f"unknown tool arguments: {', '.join(unknown)}")

    missing = [
        param.name
        for param in definition.parameters
        if param.required and param.name not in arguments
    ]
    if missing:
        raise ToolError(f"missing required arguments: {', '.join(missing)}")

    validated: dict[str, Any] = {}
    for name, raw in arguments.items():
        param = known[name]
        validated[name] = _coerce_value(param, raw)
    return validated


def _coerce_value(param: ToolParameter, raw: Any) -> Any:
    expected = param.type
    if expected == "string":
        if not isinstance(raw, str):
            raise ToolError(f"argument {param.name!r} must be a string")
        return raw
    if expected == "integer":
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ToolError(f"argument {param.name!r} must be an integer")
        return raw
    if expected == "number":
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ToolError(f"argument {param.name!r} must be a number")
        return float(raw)
    if expected == "boolean":
        if not isinstance(raw, bool):
            raise ToolError(f"argument {param.name!r} must be a boolean")
        return raw
    if expected == "object":
        if not isinstance(raw, Mapping):
            raise ToolError(f"argument {param.name!r} must be an object")
        if not _is_json_serializable(dict(raw)):
            raise ToolError(f"argument {param.name!r} must be JSON-serializable")
        return dict(raw)
    if expected == "array":
        if not isinstance(raw, list):
            raise ToolError(f"argument {param.name!r} must be an array")
        item_type = param.items_type or "string"
        coerced: list[Any] = []
        for index, item in enumerate(raw):
            if item_type == "string" and not isinstance(item, str):
                raise ToolError(
                    f"argument {param.name!r}[{index}] must be a string"
                )
            if item_type == "integer" and (
                isinstance(item, bool) or not isinstance(item, int)
            ):
                raise ToolError(
                    f"argument {param.name!r}[{index}] must be an integer"
                )
            if item_type == "number" and (
                isinstance(item, bool) or not isinstance(item, (int, float))
            ):
                raise ToolError(
                    f"argument {param.name!r}[{index}] must be a number"
                )
            if item_type == "object" and not isinstance(item, Mapping):
                raise ToolError(
                    f"argument {param.name!r}[{index}] must be an object"
                )
            if item_type == "object":
                coerced.append(dict(item))
            else:
                coerced.append(item)
        if not _is_json_serializable(coerced):
            raise ToolError(f"argument {param.name!r} must be JSON-serializable")
        return coerced
    raise ToolError(f"unsupported parameter type: {expected!r}")


def _p(
    name: str,
    type_: str,
    description: str,
    *,
    required: bool = True,
    items_type: str | None = None,
) -> ToolParameter:
    return ToolParameter(
        name=name,
        type=type_,
        description=description,
        required=required,
        items_type=items_type,
    )


# Canonical tool catalog. Names are stable; the chat system prompt embeds them.
TOOL_DEFINITIONS: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        name="harness_help",
        description=(
            "Return the compact Cosmo lifecycle, plan-shape, worker, review, "
            "and authority contracts. Use when the harness protocol or a tool "
            "output format is uncertain. Read-only; no repository access."
        ),
        parameters=(),
    ),
    ToolDefinition(
        name="submit_request",
        description=(
            "Primary request front door. Without objective+artifact_path, "
            "ingests an immutable Request for triage (no Task yet). With "
            "objective+artifact_path (and repo_root), creates a leaf/parent "
            "Task via the control-plane shortcut with operator triage defaults. "
            "Prefer triage tools for multi-artifact work."
        ),
        parameters=(
            _p("request_text", "string", "Original operator request text"),
            _p(
                "objective",
                "string",
                "Task objective (required for leaf/parent shortcut)",
                required=False,
            ),
            _p(
                "artifact_path",
                "string",
                "Repo-relative writable artifact for leaf shortcut",
                required=False,
            ),
            _p(
                "repo_root",
                "string",
                "Product repo root (required for leaf/parent shortcut)",
                required=False,
            ),
            _p(
                "validators",
                "array",
                "Deterministic validator names",
                required=False,
                items_type="string",
            ),
            _p("desired_outcome", "string", "Optional desired outcome", required=False),
            _p(
                "constraints",
                "array",
                "Explicit constraints",
                required=False,
                items_type="string",
            ),
            _p(
                "priority_components",
                "object",
                "Explainable priority components (spec or legacy names)",
                required=False,
            ),
            _p(
                "plan_kind",
                "string",
                "leaf or parent (default leaf when artifact_path set)",
                required=False,
            ),
            _p(
                "parent_node_id",
                "string",
                "Parent plan node id when nesting",
                required=False,
            ),
            _p(
                "dependencies",
                "array",
                "Plan node ids this task depends on",
                required=False,
                items_type="string",
            ),
            _p(
                "source_message_refs",
                "array",
                "Conversation message refs for the immutable Request",
                required=False,
                items_type="string",
            ),
            _p(
                "requested_priority",
                "string",
                "low|normal|high|urgent hint for triage",
                required=False,
            ),
            _p("summary", "string", "Optional request summary", required=False),
        ),
    ),
    ToolDefinition(
        name="ingest_request",
        description=(
            "Create an immutable Request only (no Task). Use before "
            "propose_triage/review_triage/apply_triage for broad outcomes."
        ),
        parameters=(
            _p("request_text", "string", "Original operator request text"),
            _p("desired_outcome", "string", "Optional desired outcome", required=False),
            _p(
                "constraints",
                "array",
                "Explicit constraints",
                required=False,
                items_type="string",
            ),
            _p(
                "source_message_refs",
                "array",
                "Conversation message refs",
                required=False,
                items_type="string",
            ),
            _p(
                "requested_priority",
                "string",
                "low|normal|high|urgent",
                required=False,
            ),
            _p("summary", "string", "Optional summary", required=False),
        ),
    ),
    ToolDefinition(
        name="propose_triage",
        description=(
            "Model-backed or operator-supplied triage proposal for a Request. "
            "Does not create a Task. Relationship taxonomy: NEW, DUPLICATE_OF, "
            "OVERLAPS_WITH, DEPENDS_ON, SUPERSEDES, CONFLICTS_WITH."
        ),
        parameters=(
            _p("request_id", "string", "Immutable Request id"),
            _p(
                "proposal",
                "object",
                "Operator-supplied triage proposal (skip model when set)",
                required=False,
            ),
            _p(
                "model",
                "string",
                "Local model instance for model-backed triage",
                required=False,
            ),
        ),
    ),
    ToolDefinition(
        name="review_triage",
        description=(
            "Independent triage review. Verdicts: PASS, RETRY_TRIAGE, "
            "NEEDS_USER_CLARIFICATION, ESCALATE. Never applies the proposal."
        ),
        parameters=(
            _p("proposal_id", "string", "Triage proposal id"),
            _p(
                "review",
                "object",
                "Operator-supplied review payload (skip model when set)",
                required=False,
            ),
            _p(
                "model",
                "string",
                "Local model for independent review",
                required=False,
            ),
            _p(
                "independent_reviewer",
                "string",
                "Reviewer label",
                required=False,
            ),
        ),
    ),
    ToolDefinition(
        name="apply_triage",
        description=(
            "Apply an approved (PASS) triage proposal: create/link/update Task "
            "and versioned TaskSpec. No product file mutation."
        ),
        parameters=(
            _p("proposal_id", "string", "Approved triage proposal id"),
            _p(
                "repo_root",
                "string",
                "Product repo root (required for NEW tasks)",
                required=False,
            ),
            _p(
                "validators",
                "array",
                "Validators for NEW leaf tasks",
                required=False,
                items_type="string",
            ),
        ),
    ),
    ToolDefinition(
        name="list_triage_proposals",
        description="List durable triage proposal records.",
        parameters=(),
    ),
    ToolDefinition(
        name="get_triage_proposal",
        description="Read one triage proposal by id.",
        parameters=(_p("proposal_id", "string", "Triage proposal id"),),
    ),
    ToolDefinition(
        name="scheduler_tick",
        description=(
            "Deterministic scheduler step: select highest-priority ready leaves "
            "under concurrency limits. Does not run workers."
        ),
        parameters=(
            _p(
                "max_concurrent",
                "integer",
                "Max concurrent dispatches (default 2)",
                required=False,
            ),
            _p(
                "limit",
                "integer",
                "Max tasks to select this tick",
                required=False,
            ),
        ),
    ),
    ToolDefinition(
        name="reassess_priority",
        description=(
            "Append a new explainable priority assessment for a Task and update "
            "the task projection. Does not mutate product files or TaskSpec "
            "content; scheduling uses the new components with aging."
        ),
        parameters=(
            _p("task_id", "string", "Task identifier"),
            _p(
                "priority_components",
                "object",
                "Explainable priority components (spec or legacy names)",
            ),
        ),
    ),
    ToolDefinition(
        name="dispatch_scheduler_batch",
        description=(
            "Run one scheduler_tick then dispatch selected ready leaves via "
            "ControlPlane.dispatch_ready_leaf (lease + worker + validation + "
            "independent review). Product promotion remains a separate step. "
            "Requires dispatcher InferenceClient."
        ),
        parameters=(
            _p("repo_root", "string", "Product repo root"),
            _p(
                "model",
                "string",
                "Loaded local model instance id (defaults to the chat model)",
                required=False,
            ),
            _p("owner", "string", "Lease owner for dispatched leaves"),
            _p(
                "strategy_summary",
                "string",
                "Worker strategy summary applied to each leaf",
                required=False,
            ),
            _p(
                "max_concurrent",
                "integer",
                "Max concurrent dispatches (default 2)",
                required=False,
            ),
            _p(
                "limit",
                "integer",
                "Max tasks to dispatch this batch (default 1)",
                required=False,
            ),
            _p("reasoning", "string", "off or on", required=False),
            _p("temperature", "number", "Sampling temperature", required=False),
            _p(
                "worker_max_output_tokens",
                "integer",
                "Worker output budget",
                required=False,
            ),
            _p(
                "review_max_output_tokens",
                "integer",
                "Reviewer output budget",
                required=False,
            ),
            _p(
                "lease_ttl_seconds",
                "integer",
                "Lease TTL seconds per dispatch",
                required=False,
            ),
            _p(
                "read_only_paths",
                "array",
                "Optional repo-relative paths injected into worker context",
                required=False,
                items_type="string",
            ),
        ),
    ),
    ToolDefinition(
        name="refresh_dependency_readiness",
        description=(
            "Refresh blocked/ready status from plan dependency satisfaction. "
            "Moves blocked->ready when deps complete, or ready->blocked when "
            "deps become unmet. Does not bypass parent/spec/lease gates."
        ),
        parameters=(_p("task_id", "string", "Task identifier"),),
    ),
    ToolDefinition(
        name="inspect_lifecycle",
        description=(
            "Inspect control-plane lifecycle state. With task_id: task + spec + "
            "plan + readiness gates + lease + attempts + ready-queue position. "
            "Without task_id: compact control-plane snapshot (tasks, requests, "
            "ready queue, leases). Read-only; no product mutation."
        ),
        parameters=(
            _p(
                "task_id",
                "string",
                "Optional task id for a single-task lifecycle projection",
                required=False,
            ),
        ),
    ),
    ToolDefinition(
        name="create_task",
        description=(
            "Create a leaf Task via the control plane (canonical request/task "
            "path with priority, plan, and lifecycle fields). Leaf shortcut only; "
            "AtomicHarness.submit remains legacy-only. Prefer submit_request + "
            "triage for broad multi-artifact work."
        ),
        parameters=(
            _p("request_text", "string", "Original operator request text"),
            _p("objective", "string", "Task objective"),
            _p("artifact_path", "string", "Repo-relative writable artifact"),
            _p("repo_root", "string", "Product repo root"),
            _p(
                "validators",
                "array",
                "Deterministic validator names",
                required=False,
                items_type="string",
            ),
            _p("desired_outcome", "string", "Optional desired outcome", required=False),
            _p(
                "constraints",
                "array",
                "Explicit constraints",
                required=False,
                items_type="string",
            ),
            _p(
                "priority_components",
                "object",
                "Explainable priority components",
                required=False,
            ),
        ),
    ),
    ToolDefinition(
        name="list_tasks",
        description="List all durable task records.",
        parameters=(),
    ),
    ToolDefinition(
        name="get_task",
        description="Read one task record by id.",
        parameters=(_p("task_id", "string", "Task identifier"),),
    ),
    ToolDefinition(
        name="inspect_task",
        description="Inspect a task with its current spec and plan node.",
        parameters=(_p("task_id", "string", "Task identifier"),),
    ),
    ToolDefinition(
        name="inspect_spec",
        description="Inspect a TaskSpec record.",
        parameters=(_p("spec_id", "string", "Spec identifier"),),
    ),
    ToolDefinition(
        name="inspect_plan",
        description="Inspect a plan node record.",
        parameters=(_p("node_id", "string", "Plan node identifier"),),
    ),
    ToolDefinition(
        name="inspect_attempt",
        description="Inspect an attempt (merged with attempt state when present).",
        parameters=(_p("attempt_id", "string", "Attempt identifier"),),
    ),
    ToolDefinition(
        name="inspect_review",
        description="Inspect a review, plan review, or triage review by id.",
        parameters=(_p("review_id", "string", "Review identifier"),),
    ),
    ToolDefinition(
        name="inspect_ledger",
        description=(
            "Inspect ledger projection for a task (builds projection if needed) "
            "or a specific ledger event/projection id."
        ),
        parameters=(
            _p("task_id", "string", "Task id to project ledger for", required=False),
            _p(
                "projection_id",
                "string",
                "Existing ledger projection id",
                required=False,
            ),
            _p("event_id", "string", "Raw ledger event id", required=False),
        ),
    ),
    ToolDefinition(
        name="update_task_status",
        description="Apply a legal control-plane task status transition.",
        parameters=(
            _p("task_id", "string", "Task identifier"),
            _p("status", "string", "Target status"),
            _p("reason", "string", "Optional transition reason", required=False),
            _p("actor", "string", "Actor label (default chat)", required=False),
        ),
    ),
    ToolDefinition(
        name="ready_queue",
        description="List ready leaf tasks ordered by priority (deterministic).",
        parameters=(),
    ),
    ToolDefinition(
        name="acquire_lease",
        description="Acquire an exclusive lease on a task.",
        parameters=(
            _p("task_id", "string", "Task identifier"),
            _p("owner", "string", "Lease owner"),
            _p(
                "ttl_seconds",
                "integer",
                "Lease TTL in seconds",
                required=False,
            ),
        ),
    ),
    ToolDefinition(
        name="renew_lease",
        description="Renew an active lease owned by the given owner.",
        parameters=(
            _p("lease_id", "string", "Lease identifier"),
            _p("owner", "string", "Lease owner"),
            _p(
                "ttl_seconds",
                "integer",
                "Optional new TTL seconds",
                required=False,
            ),
        ),
    ),
    ToolDefinition(
        name="release_lease",
        description="Release an active lease owned by the given owner.",
        parameters=(
            _p("lease_id", "string", "Lease identifier"),
            _p("owner", "string", "Lease owner"),
        ),
    ),
    ToolDefinition(
        name="dispatch_ready",
        description=(
            "Stable underlying ready-leaf dispatch: lease a ready leaf and run "
            "AtomicHarness.run_attempt (worker + validation + independent "
            "review). Does not promote product state. Prefer run_attempt for a "
            "first attempt and retry_attempt when a later attempt needs the "
            "four-field retry contract."
        ),
        parameters=(
            _p("task_id", "string", "Task identifier"),
            _p("repo_root", "string", "Product repo root"),
            _p(
                "model",
                "string",
                "Loaded local model instance id (defaults to the chat model)",
                required=False,
            ),
            _p("strategy_summary", "string", "Worker strategy summary"),
            _p("owner", "string", "Lease owner for dispatch"),
            _p("reasoning", "string", "off or on", required=False),
            _p("temperature", "number", "Sampling temperature", required=False),
            _p(
                "worker_max_output_tokens",
                "integer",
                "Worker output budget",
                required=False,
            ),
            _p(
                "review_max_output_tokens",
                "integer",
                "Reviewer output budget",
                required=False,
            ),
            _p(
                "read_only_paths",
                "array",
                "Optional repo-relative paths injected into the worker context",
                required=False,
                items_type="string",
            ),
        ),
    ),
    ToolDefinition(
        name="run_attempt",
        description=(
            "Lifecycle alias for a normal ready-leaf dispatch (first or "
            "non-retry attempt). Same authority as dispatch_ready; does not "
            "promote product state. Use retry_attempt when prior attempts "
            "exist and the four-field retry contract is required. "
            "No shell/network/provider action."
        ),
        parameters=(
            _p("task_id", "string", "Task identifier"),
            _p("repo_root", "string", "Product repo root"),
            _p(
                "model",
                "string",
                "Loaded local model instance id (defaults to the chat model)",
                required=False,
            ),
            _p("strategy_summary", "string", "Worker strategy summary"),
            _p("owner", "string", "Lease owner for dispatch"),
            _p("reasoning", "string", "off or on", required=False),
            _p("temperature", "number", "Sampling temperature", required=False),
            _p(
                "worker_max_output_tokens",
                "integer",
                "Worker output budget",
                required=False,
            ),
            _p(
                "review_max_output_tokens",
                "integer",
                "Reviewer output budget",
                required=False,
            ),
            _p(
                "read_only_paths",
                "array",
                "Optional repo-relative paths injected into the worker context",
                required=False,
                items_type="string",
            ),
        ),
    ),
    ToolDefinition(
        name="retry_attempt",
        description=(
            "Lifecycle alias for a later ready-leaf attempt. Requires the "
            "four-field retry contract (addresses, preserves, differs_from, "
            "expected_improvement) and passes it into ControlPlane."
            "dispatch_ready_leaf so no-blind-retry and "
            "materially-different-strategy gates remain authoritative. Does "
            "not promote product state. No shell/network/provider action."
        ),
        parameters=(
            _p("task_id", "string", "Task identifier"),
            _p("repo_root", "string", "Product repo root"),
            _p(
                "model",
                "string",
                "Loaded local model instance id (defaults to the chat model)",
                required=False,
            ),
            _p("strategy_summary", "string", "Materially different worker strategy"),
            _p("owner", "string", "Lease owner for dispatch"),
            _p(
                "addresses",
                "array",
                "What this retry addresses (required retry contract)",
                items_type="string",
            ),
            _p(
                "preserves",
                "array",
                "What this retry must preserve (required retry contract)",
                items_type="string",
            ),
            _p(
                "differs_from",
                "array",
                "Prior attempt ids or differences this strategy differs from",
                items_type="string",
            ),
            _p(
                "expected_improvement",
                "array",
                "Expected improvements vs prior attempts",
                items_type="string",
            ),
            _p("reasoning", "string", "off or on", required=False),
            _p("temperature", "number", "Sampling temperature", required=False),
            _p(
                "worker_max_output_tokens",
                "integer",
                "Worker output budget",
                required=False,
            ),
            _p(
                "review_max_output_tokens",
                "integer",
                "Reviewer output budget",
                required=False,
            ),
            _p(
                "read_only_paths",
                "array",
                "Optional repo-relative paths injected into the worker context",
                required=False,
                items_type="string",
            ),
        ),
    ),
    ToolDefinition(
        name="promote_attempt",
        description=(
            "Promote a reviewed attempt through AtomicHarness gates "
            "(Validation PASS + Review PASS required). Never fabricates PASS. "
            "No shell/network/provider action."
        ),
        parameters=(
            _p("attempt_id", "string", "Attempt identifier"),
            _p("repo_root", "string", "Product repo root"),
        ),
    ),
    ToolDefinition(
        name="recover_promotion",
        description=(
            "Recover or complete promotion via control-plane "
            "recover_promotion (same harness gates). "
            "No shell/network/provider action."
        ),
        parameters=(
            _p("attempt_id", "string", "Attempt identifier"),
            _p("repo_root", "string", "Product repo root"),
        ),
    ),
    # ------------------------------------------------------------------
    # Repository / context inspection (approved repo root only)
    # ------------------------------------------------------------------
    ToolDefinition(
        name="repository_map",
        description=(
            "Build a bounded repository map under an approved product repo root. "
            "Read-only; excludes AtomicStore and .git. No shell/network/provider "
            "action."
        ),
        parameters=(
            _p("repo_root", "string", "Approved product repository root"),
            _p(
                "max_map_entries",
                "integer",
                "Optional entry cap (bounded by inspector defaults)",
                required=False,
            ),
        ),
    ),
    ToolDefinition(
        name="list_tree",
        description=(
            "List relative file paths under an approved repo root (optional "
            "prefix). Read-only; no shell/network/provider action."
        ),
        parameters=(
            _p("repo_root", "string", "Approved product repository root"),
            _p("prefix", "string", "Optional relative directory prefix", required=False),
            _p(
                "max_entries",
                "integer",
                "Optional listing cap",
                required=False,
            ),
        ),
    ),
    ToolDefinition(
        name="read_artifact",
        description=(
            "Read a bounded UTF-8 excerpt of a repo-relative path under the "
            "approved root. Read-only; no shell/network/provider action."
        ),
        parameters=(
            _p("repo_root", "string", "Approved product repository root"),
            _p("path", "string", "Repo-relative artifact path"),
            _p("max_bytes", "integer", "Optional read byte budget", required=False),
            _p("start_line", "integer", "Optional 1-based start line", required=False),
            _p("end_line", "integer", "Optional 1-based end line", required=False),
        ),
    ),
    ToolDefinition(
        name="search_text",
        description=(
            "Search for a literal string under an approved repo root. "
            "Read-only; no shell/network/provider action."
        ),
        parameters=(
            _p("repo_root", "string", "Approved product repository root"),
            _p("query", "string", "Literal search string"),
            _p("prefix", "string", "Optional relative prefix", required=False),
            _p(
                "max_matches",
                "integer",
                "Optional match cap",
                required=False,
            ),
        ),
    ),
    ToolDefinition(
        name="inspect_python_symbols",
        description=(
            "Inspect top-level Python symbols in a repo-relative module under "
            "the approved root. Read-only; no shell/network/provider action."
        ),
        parameters=(
            _p("repo_root", "string", "Approved product repository root"),
            _p("path", "string", "Repo-relative .py path"),
        ),
    ),
    # ------------------------------------------------------------------
    # Planning lifecycle (propose -> independent review -> apply)
    # ------------------------------------------------------------------
    ToolDefinition(
        name="build_planning_pack",
        description=(
            "Build a deterministic planning pack and context manifest for a "
            "task/request under an approved repo root. Does not call a model "
            "and does not apply a plan. Returns JSON evidence only (no live "
            "inspector objects). No shell/network/provider action."
        ),
        parameters=(
            _p("repo_root", "string", "Approved product repository root"),
            _p("task_id", "string", "Optional task id", required=False),
            _p("request_id", "string", "Optional request id", required=False),
            _p(
                "selected_paths",
                "array",
                "Optional explicit read-only paths for the pack",
                required=False,
                items_type="string",
            ),
            _p(
                "include_repository_map",
                "boolean",
                "Include repository map (default true)",
                required=False,
            ),
        ),
    ),
    ToolDefinition(
        name="propose_plan",
        description=(
            "Call the local InferenceClient planner and store an immutable "
            "plan version (unreviewed, not applied). Uses the dispatcher's "
            "local client and the supplied model; never a cloud client. "
            "Returns plan_version_id and evidence hashes for sequencing "
            "propose -> review -> apply. No shell/network/provider action."
        ),
        parameters=(
            _p("repo_root", "string", "Approved product repository root"),
            _p(
                "model",
                "string",
                "Loaded local model instance id (defaults to the chat model)",
                required=False,
            ),
            _p("task_id", "string", "Optional task id", required=False),
            _p("request_id", "string", "Optional request id", required=False),
            _p(
                "selected_paths",
                "array",
                "Optional explicit read-only paths",
                required=False,
                items_type="string",
            ),
            _p(
                "parent_node_id",
                "string",
                "Optional existing parent node for the proposal",
                required=False,
            ),
            _p("reasoning", "string", "off or on", required=False),
            _p("temperature", "number", "Sampling temperature", required=False),
            _p(
                "max_output_tokens",
                "integer",
                "Planner output budget",
                required=False,
            ),
        ),
    ),
    ToolDefinition(
        name="review_plan",
        description=(
            "Independently review a stored plan version. Supply an operator "
            "review object or use the local InferenceClient with model. Never "
            "applies the plan. No shell/network/provider action."
        ),
        parameters=(
            _p("plan_version_id", "string", "Plan version identifier"),
            _p(
                "review",
                "object",
                "Operator-supplied plan review payload (verdict, evidence, ...)",
                required=False,
            ),
            _p(
                "model",
                "string",
                "Local model for model-backed review when review is omitted",
                required=False,
            ),
            _p("reasoning", "string", "off or on", required=False),
            _p("temperature", "number", "Sampling temperature", required=False),
            _p(
                "max_output_tokens",
                "integer",
                "Reviewer output budget",
                required=False,
            ),
        ),
    ),
    ToolDefinition(
        name="list_plan_versions",
        description=(
            "List durable plan version records. No shell/network/provider action."
        ),
        parameters=(),
    ),
    ToolDefinition(
        name="get_plan_version",
        description=(
            "Read one plan version by id (proposal, review linkage, apply "
            "status). No shell/network/provider action."
        ),
        parameters=(_p("plan_version_id", "string", "Plan version identifier"),),
    ),
    ToolDefinition(
        name="apply_approved_plan",
        description=(
            "Materialize plan nodes only after independent review PASS. Refuses "
            "unreviewed/non-PASS plans. Does not promote product artifacts. "
            "No shell/network/provider action."
        ),
        parameters=(
            _p("plan_version_id", "string", "Approved plan version identifier"),
            _p("repo_root", "string", "Product repo root for node materialization"),
            _p(
                "request_text",
                "string",
                "Optional root request text override",
                required=False,
            ),
        ),
    ),
    # ------------------------------------------------------------------
    # Integration lifecycle (assemble -> review -> explicit promote)
    # ------------------------------------------------------------------
    ToolDefinition(
        name="list_integrations",
        description=(
            "List durable parent integration runs. No shell/network/provider "
            "action."
        ),
        parameters=(),
    ),
    ToolDefinition(
        name="get_integration",
        description=(
            "Read one integration run: checks, selected children, workspace, "
            "baseline hash, review/promotion receipts. Product is unchanged "
            "until promote_integration. No shell/network/provider action."
        ),
        parameters=(_p("integration_id", "string", "Integration run identifier"),),
    ),
    ToolDefinition(
        name="assemble_integration",
        description=(
            "Assemble a parent integration workspace from node-approved direct "
            "leaf children over a full baseline snapshot. Does not mutate "
            "product files. Nested parents require a prior approved "
            "IntegrationRun. No shell/network/provider action."
        ),
        parameters=(
            _p("parent_node_id", "string", "Parent plan node id"),
            _p("repo_root", "string", "Product repo root"),
            _p(
                "selected",
                "object",
                "Optional map of leaf node_id -> artifact_version_id",
                required=False,
            ),
            _p(
                "expected_baseline_tree_sha256",
                "string",
                "Optional stale-baseline guard",
                required=False,
            ),
        ),
    ),
    ToolDefinition(
        name="review_integration",
        description=(
            "Record an independent integration review (local model-backed by "
            "default when a dispatcher client is available; operator verdict "
            "is an explicit fallback). Does not promote. "
            "No shell/network/provider action."
        ),
        parameters=(
            _p("integration_id", "string", "Integration run identifier"),
            _p(
                "verdict",
                "string",
                "Optional operator verdict; omit for local model review",
                required=False,
            ),
            _p("model", "string", "Optional local model instance override", required=False),
            _p("reasoning", "string", "off or on", required=False),
            _p("max_output_tokens", "integer", "Reviewer output budget", required=False),
            _p("notes", "string", "Optional review notes", required=False),
            _p(
                "requirement_results",
                "array",
                "Optional requirement results (objects)",
                required=False,
                items_type="object",
            ),
        ),
    ),
    ToolDefinition(
        name="promote_integration",
        description=(
            "Explicit parent integration promotion after validation+review PASS. "
            "Stale baselines fail closed; product files stay unchanged until "
            "this succeeds. No shell/network/provider action."
        ),
        parameters=(
            _p("integration_id", "string", "Integration run identifier"),
            _p("repo_root", "string", "Product repo root", required=False),
        ),
    ),
)


# The original/full contracts are an operator surface.  The model catalog is
# derived below by removing authority-bearing arguments and tools.  Keeping one
# canonical definition list avoids the parallel-catalog drift that this split
# is intended to prevent.
OPERATOR_TOOL_DEFINITIONS = TOOL_DEFINITIONS

MODEL_EXCLUDED_TOOLS = frozenset(
    {
        # These shortcuts manufacture operator-supplied triage/recovery state.
        "create_task",
        "recover_promotion",
    }
)

MODEL_HIDDEN_PARAMETERS_BY_TOOL: dict[str, frozenset[str]] = {
    # A model may only ingest an immutable Request through this broad front
    # door.  Operator leaf/parent shortcuts stay in the operator catalog.
    "submit_request": frozenset(
        {
            "objective",
            "artifact_path",
            "repo_root",
            "validators",
            "priority_components",
            "plan_kind",
            "parent_node_id",
            "dependencies",
        }
    ),
    "propose_triage": frozenset({"proposal"}),
    "review_triage": frozenset({"review", "independent_reviewer"}),
    "review_plan": frozenset({"review"}),
    "review_integration": frozenset(
        {"verdict", "requirement_results", "notes"}
    ),
    "update_task_status": frozenset({"actor"}),
}
MODEL_DESCRIPTIONS_BY_TOOL = {
    "submit_request": (
        "Create an immutable Request for triage (no Task and no operator "
        "defaults). Follow with propose_triage, review_triage, and apply_triage."
    ),
    "propose_triage": (
        "Create a local model-backed triage proposal for an immutable Request. "
        "Does not create or modify a Task."
    ),
    "review_triage": (
        "Invoke the local independent triage reviewer. The caller cannot "
        "supply a verdict or reviewer identity."
    ),
    "review_plan": (
        "Invoke the local independent plan reviewer. The caller cannot supply "
        "a verdict. Never applies the plan."
    ),
    "review_integration": (
        "Invoke the local independent integration reviewer. The caller cannot "
        "supply a verdict or evidence. Never promotes product state."
    ),
}


def _model_definition(definition: ToolDefinition) -> ToolDefinition:
    """Project the canonical operator contract into a least-authority model one."""

    hidden = set(MODEL_HIDDEN_PARAMETERS_BY_TOOL.get(definition.name, ()))
    # Repository roots are always injected from an out-of-band capability.
    hidden.add("repo_root")
    return ToolDefinition(
        name=definition.name,
        description=MODEL_DESCRIPTIONS_BY_TOOL.get(
            definition.name, definition.description
        ),
        parameters=tuple(
            parameter
            for parameter in definition.parameters
            if parameter.name not in hidden
        ),
    )


MODEL_TOOL_DEFINITIONS = tuple(
    _model_definition(item)
    for item in OPERATOR_TOOL_DEFINITIONS
    if item.name not in MODEL_EXCLUDED_TOOLS
)
TOOL_DEFINITIONS_BY_NAME: dict[str, ToolDefinition] = {
    item.name: item for item in TOOL_DEFINITIONS
}
MODEL_TOOL_DEFINITIONS_BY_NAME: dict[str, ToolDefinition] = {
    item.name: item for item in MODEL_TOOL_DEFINITIONS
}
MODEL_REPO_ROOT_TOOLS = frozenset(
    item.name
    for item in MODEL_TOOL_DEFINITIONS
    if any(
        parameter.name == "repo_root"
        for parameter in TOOL_DEFINITIONS_BY_NAME[item.name].parameters
    )
    and item.name != "submit_request"
)


def tool_definitions_json(
    *, catalog: str = "model", compact: bool = False
) -> list[dict[str, Any]]:
    """Return a model-safe or operator tool catalog as JSON dictionaries."""

    if catalog == "model":
        definitions = MODEL_TOOL_DEFINITIONS
    elif catalog == "operator":
        definitions = OPERATOR_TOOL_DEFINITIONS
    else:
        raise ToolError(f"unsupported tool catalog: {catalog!r}")
    return [item.to_json_dict(compact=compact) for item in definitions]


@dataclass
class ToolResult:
    """Structured result returned to the chat loop (never free-form shell)."""

    ok: bool
    tool: str
    result: Any = None
    error: str | None = None
    caller_provenance: dict[str, str] | None = None
    repo_capability_id: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "tool": self.tool,
        }
        if self.ok:
            payload["result"] = self.result
        else:
            payload["error"] = self.error
        if self.caller_provenance is not None:
            payload["caller_provenance"] = dict(self.caller_provenance)
        if self.repo_capability_id is not None:
            payload["repo_capability_id"] = self.repo_capability_id
        return payload


@dataclass
class ToolDispatcher:
    """Dispatch validated tool calls onto ControlPlane + AtomicHarness."""

    plane: ControlPlane
    # Optional inference client used by dispatch/planning (local LM Studio only).
    client: Any | None = None
    model: str | None = None
    caller: CallerProvenance | None = None
    repo_capability: RepoRootCapability | None = None
    handlers: dict[str, Callable[[dict[str, Any]], Any]] = field(init=False)

    def __post_init__(self) -> None:
        if self.caller is None:
            # Direct Python construction is an operator API.  ChatSession
            # rejects this default and requires an explicit model principal.
            self.caller = authenticate_operator("trusted-python-api")
        self.handlers = {
            "harness_help": self._harness_help,
            "submit_request": self._submit_request,
            "ingest_request": self._ingest_request,
            "propose_triage": self._propose_triage,
            "review_triage": self._review_triage,
            "apply_triage": self._apply_triage,
            "list_triage_proposals": self._list_triage_proposals,
            "get_triage_proposal": self._get_triage_proposal,
            "scheduler_tick": self._scheduler_tick,
            "reassess_priority": self._reassess_priority,
            "dispatch_scheduler_batch": self._dispatch_scheduler_batch,
            "refresh_dependency_readiness": self._refresh_dependency_readiness,
            "inspect_lifecycle": self._inspect_lifecycle,
            "create_task": self._create_task,
            "list_tasks": self._list_tasks,
            "get_task": self._get_task,
            "inspect_task": self._inspect_task,
            "inspect_spec": self._inspect_spec,
            "inspect_plan": self._inspect_plan,
            "inspect_attempt": self._inspect_attempt,
            "inspect_review": self._inspect_review,
            "inspect_ledger": self._inspect_ledger,
            "update_task_status": self._update_task_status,
            "ready_queue": self._ready_queue,
            "acquire_lease": self._acquire_lease,
            "renew_lease": self._renew_lease,
            "release_lease": self._release_lease,
            "dispatch_ready": self._dispatch_ready,
            "run_attempt": self._run_attempt,
            "retry_attempt": self._retry_attempt,
            "promote_attempt": self._promote_attempt,
            "recover_promotion": self._recover_promotion,
            "repository_map": self._repository_map,
            "list_tree": self._list_tree,
            "read_artifact": self._read_artifact,
            "search_text": self._search_text,
            "inspect_python_symbols": self._inspect_python_symbols,
            "build_planning_pack": self._build_planning_pack,
            "propose_plan": self._propose_plan,
            "review_plan": self._review_plan,
            "list_plan_versions": self._list_plan_versions,
            "get_plan_version": self._get_plan_version,
            "apply_approved_plan": self._apply_approved_plan,
            "list_integrations": self._list_integrations,
            "get_integration": self._get_integration,
            "assemble_integration": self._assemble_integration,
            "review_integration": self._review_integration,
            "promote_integration": self._promote_integration,
        }

    @property
    def harness(self) -> AtomicHarness:
        return self.plane.harness

    @property
    def store(self) -> AtomicStore:
        return self.plane.store

    def _planning(self) -> PlanningService:
        return PlanningService(self.plane)

    def _triage(self) -> TriageService:
        return TriageService(self.plane)

    def _integration(self) -> IntegrationService:
        return IntegrationService(self.store)

    def _operator(self) -> CallerProvenance:
        """Return authenticated operator provenance for operator-only handlers."""

        return require_operator(self.caller)

    def _harness_help(self, _args: dict[str, Any]) -> Any:
        """Return the bounded, model-readable protocol guide."""
        return harness_help_payload()

    def _inspector(self, repo_root: str, **kwargs: Any) -> RepositoryInspector:
        root = Path(repo_root).resolve()
        if not root.is_dir():
            raise ToolError(f"repo_root is not a directory: {root}")
        return RepositoryInspector(
            root,
            state_root=self.store.root,
            **kwargs,
        )

    def execute(self, name: str, arguments: Any = None) -> ToolResult:
        """Validate arguments and run one tool. Always returns ToolResult.

        Inference/transport failures and any handler-level runtime failure are
        converted to structured errors so the chat loop never crashes or
        mutates product state on malformed calls or failed dispatch.
        Promotion remains gated exclusively through AtomicHarness / IntegrationService.
        """
        assert self.caller is not None
        definitions = (
            MODEL_TOOL_DEFINITIONS_BY_NAME
            if self.caller.principal_type == "model"
            else TOOL_DEFINITIONS_BY_NAME
        )
        provenance = self.caller.to_record()
        capability_id = (
            self.repo_capability.capability_id if self.repo_capability else None
        )
        if name not in definitions:
            return ToolResult(
                ok=False,
                tool=str(name),
                error=(
                    "unknown tool or unavailable to "
                    f"{self.caller.principal_type} caller: {name!r}"
                ),
                caller_provenance=provenance,
                repo_capability_id=capability_id,
            )
        definition = definitions[name]
        try:
            validated = validate_tool_arguments(definition, arguments)
            if (
                self.caller.principal_type == "model"
                and name in MODEL_REPO_ROOT_TOOLS
            ):
                if self.repo_capability is not None:
                    validated["repo_root"] = str(
                        self.repo_capability.authorize(self.caller, name)
                    )
                else:
                    raise AuthorityError(
                        f"{name} requires an operator-approved repo-root capability"
                    )
            handler = self.handlers[name]
            result = handler(validated)
            if not _is_json_serializable(result):
                raise ToolError("tool handler returned non-JSON-serializable result")
            return ToolResult(
                ok=True,
                tool=name,
                result=result,
                caller_provenance=provenance,
                repo_capability_id=capability_id,
            )
        except ToolError as exc:
            return ToolResult(
                ok=False,
                tool=name,
                error=str(exc),
                caller_provenance=provenance,
                repo_capability_id=capability_id,
            )
        except (
            AuthorityError,
            ControlPlaneError,
            AtomicHarnessError,
            PlanningError,
            IntegrationError,
            ContextError,
            TriageError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            return ToolResult(
                ok=False,
                tool=name,
                error=str(exc),
                caller_provenance=provenance,
                repo_capability_id=capability_id,
            )
        except Exception as exc:  # noqa: BLE001 — structured tool boundary
            return ToolResult(
                ok=False,
                tool=name,
                error=f"{type(exc).__name__}: {exc}",
                caller_provenance=provenance,
                repo_capability_id=capability_id,
            )

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _submit_request(self, args: dict[str, Any]) -> Any:
        # Broad front door: without objective+repo_root, only ingest Request.
        has_leaf_contract = bool(args.get("objective")) and bool(
            args.get("repo_root")
        )
        if not has_leaf_contract:
            return self.plane.ingest_request(
                request_text=args["request_text"],
                desired_outcome=args.get("desired_outcome"),
                explicit_constraints=args.get("constraints") or (),
                source_message_refs=args.get("source_message_refs") or (),
                requested_priority=args.get("requested_priority"),
                summary=args.get("summary"),
            )
        plan_kind = args.get("plan_kind") or (
            "leaf" if args.get("artifact_path") else "parent"
        )
        return self.plane.submit_request(
            request_text=args["request_text"],
            objective=args["objective"],
            artifact_path=args.get("artifact_path") or "",
            repo_root=Path(args["repo_root"]),
            validators=args.get("validators") or (),
            desired_outcome=args.get("desired_outcome"),
            explicit_constraints=args.get("constraints") or (),
            priority_components=args.get("priority_components"),
            plan_kind=plan_kind,
            parent_node_id=args.get("parent_node_id"),
            dependencies=args.get("dependencies") or (),
            operator_caller=self._operator(),
        )

    def _ingest_request(self, args: dict[str, Any]) -> Any:
        return self.plane.ingest_request(
            request_text=args["request_text"],
            desired_outcome=args.get("desired_outcome"),
            explicit_constraints=args.get("constraints") or (),
            source_message_refs=args.get("source_message_refs") or (),
            requested_priority=args.get("requested_priority"),
            summary=args.get("summary"),
        )

    def _propose_triage(self, args: dict[str, Any]) -> Any:
        proposal = args.get("proposal")
        model = args.get("model") or self.model
        if proposal is not None:
            return self._triage().propose_triage(
                args["request_id"],
                proposal=proposal,
                operator_caller=self._operator(),
            )
        if model is None or self.client is None:
            raise ToolError(
                "propose_triage requires proposal object or dispatcher client+model"
            )
        return self._triage().propose_triage(
            args["request_id"], client=self.client, model=model
        )

    def _review_triage(self, args: dict[str, Any]) -> Any:
        review = args.get("review")
        model = args.get("model") or self.model
        if review is not None:
            return self._triage().review_triage(
                args["proposal_id"],
                review=review,
                independent_reviewer=args.get("independent_reviewer"),
                operator_caller=self._operator(),
            )
        if model is None or self.client is None:
            raise ToolError(
                "review_triage requires review object or dispatcher client+model"
            )
        return self._triage().review_triage(
            args["proposal_id"],
            client=self.client,
            model=model,
            independent_reviewer=args.get("independent_reviewer"),
        )

    def _apply_triage(self, args: dict[str, Any]) -> Any:
        repo_root = Path(args["repo_root"]) if args.get("repo_root") else None
        return self._triage().apply_triage(
            args["proposal_id"],
            repo_root=repo_root,
            validators=args.get("validators") or (),
        )

    def _list_triage_proposals(self, _args: dict[str, Any]) -> Any:
        return self._triage().list_proposals()

    def _get_triage_proposal(self, args: dict[str, Any]) -> Any:
        return self._triage().get_proposal(args["proposal_id"])

    def _scheduler_tick(self, args: dict[str, Any]) -> Any:
        kwargs: dict[str, Any] = {}
        if "max_concurrent" in args:
            kwargs["max_concurrent"] = args["max_concurrent"]
        if "limit" in args:
            kwargs["limit"] = args["limit"]
        return self.plane.scheduler_tick(**kwargs)

    def _reassess_priority(self, args: dict[str, Any]) -> Any:
        return self.plane.reassess_priority(
            args["task_id"],
            args["priority_components"],
        )

    def _dispatch_scheduler_batch(self, args: dict[str, Any]) -> Any:
        if self.client is None:
            raise ToolError(
                "dispatch_scheduler_batch requires an InferenceClient on the "
                "dispatcher"
            )
        model = args.get("model") or self.model
        if not model:
            raise ToolError(
                "dispatch_scheduler_batch requires a model argument or a "
                "dispatcher default model"
            )
        kwargs: dict[str, Any] = {
            "repo_root": Path(args["repo_root"]),
            "client": self.client,
            "model": model,
            "owner": args["owner"],
        }
        if "strategy_summary" in args:
            kwargs["strategy_summary"] = args["strategy_summary"]
        if "max_concurrent" in args:
            kwargs["max_concurrent"] = args["max_concurrent"]
        if "limit" in args:
            kwargs["limit"] = args["limit"]
        if "reasoning" in args:
            kwargs["reasoning"] = args["reasoning"]
        if "temperature" in args:
            kwargs["temperature"] = args["temperature"]
        if "worker_max_output_tokens" in args:
            kwargs["worker_max_output_tokens"] = args["worker_max_output_tokens"]
        if "review_max_output_tokens" in args:
            kwargs["review_max_output_tokens"] = args["review_max_output_tokens"]
        if "lease_ttl_seconds" in args:
            kwargs["lease_ttl_seconds"] = args["lease_ttl_seconds"]
        if "read_only_paths" in args:
            kwargs["read_only_paths"] = args["read_only_paths"]
        return self.plane.dispatch_scheduler_batch(**kwargs)

    def _refresh_dependency_readiness(self, args: dict[str, Any]) -> Any:
        return self.plane.refresh_dependency_readiness(args["task_id"])

    def _inspect_lifecycle(self, args: dict[str, Any]) -> Any:
        return self.plane.inspect_lifecycle(args.get("task_id"))

    def _create_task(self, args: dict[str, Any]) -> Any:
        # Canonical control-plane path only — never a second authority via
        # AtomicHarness.submit (legacy API preserved on the harness itself).
        return self.plane.create_task(
            request_text=args["request_text"],
            objective=args["objective"],
            artifact_path=args["artifact_path"],
            repo_root=Path(args["repo_root"]),
            validators=args.get("validators") or (),
            desired_outcome=args.get("desired_outcome"),
            explicit_constraints=args.get("constraints") or (),
            priority_components=args.get("priority_components"),
            operator_caller=self._operator(),
        )

    def _list_tasks(self, _args: dict[str, Any]) -> Any:
        return self.plane.list_tasks()

    def _get_task(self, args: dict[str, Any]) -> Any:
        return self.plane.get_task(args["task_id"])

    def _inspect_task(self, args: dict[str, Any]) -> Any:
        task = self.plane.get_task(args["task_id"])
        spec = None
        plan = None
        if task.get("current_spec_id"):
            try:
                spec = self.plane.get_task_spec(str(task["current_spec_id"]))
            except (ControlPlaneError, AtomicHarnessError, KeyError, OSError):
                spec = None
        if task.get("plan_node_id"):
            try:
                plan = self.plane.get_plan(str(task["plan_node_id"]))
            except (ControlPlaneError, AtomicHarnessError, KeyError, OSError):
                plan = None
        return {"task": task, "spec": spec, "plan": plan}

    def _inspect_spec(self, args: dict[str, Any]) -> Any:
        return self.plane.get_task_spec(args["spec_id"])

    def _inspect_plan(self, args: dict[str, Any]) -> Any:
        return self.plane.get_plan(args["node_id"])

    def _inspect_attempt(self, args: dict[str, Any]) -> Any:
        return self.plane.get_attempt(args["attempt_id"])

    def _inspect_review(self, args: dict[str, Any]) -> Any:
        return self.plane.get_review(args["review_id"])

    def _inspect_ledger(self, args: dict[str, Any]) -> Any:
        if args.get("event_id"):
            return self.plane.get_ledger_record(args["event_id"])
        if args.get("projection_id"):
            return self.plane.get_ledger_projection(args["projection_id"])
        if args.get("task_id"):
            return self.plane.project_ledger(args["task_id"])
        raise ToolError(
            "inspect_ledger requires task_id, projection_id, or event_id"
        )

    def _update_task_status(self, args: dict[str, Any]) -> Any:
        return self.plane.transition_task(
            args["task_id"],
            args["status"],
            reason=args.get("reason"),
            actor=args.get("actor") or self.caller.principal_id,
        )

    def _ready_queue(self, _args: dict[str, Any]) -> Any:
        return self.plane.ready_queue()

    def _acquire_lease(self, args: dict[str, Any]) -> Any:
        kwargs: dict[str, Any] = {
            "owner": args["owner"],
        }
        if "ttl_seconds" in args:
            kwargs["ttl_seconds"] = args["ttl_seconds"]
        return self.plane.acquire_lease(args["task_id"], **kwargs)

    def _renew_lease(self, args: dict[str, Any]) -> Any:
        kwargs: dict[str, Any] = {"owner": args["owner"]}
        if "ttl_seconds" in args:
            kwargs["ttl_seconds"] = args["ttl_seconds"]
        return self.plane.renew_lease(args["lease_id"], **kwargs)

    def _release_lease(self, args: dict[str, Any]) -> Any:
        return self.plane.release_lease(args["lease_id"], owner=args["owner"])

    def _dispatch_kwargs(
        self, args: dict[str, Any], *, tool_name: str
    ) -> dict[str, Any]:
        if self.client is None:
            raise ToolError(
                f"{tool_name} requires an InferenceClient on the dispatcher"
            )
        kwargs: dict[str, Any] = {
            "repo_root": Path(args["repo_root"]),
            "client": self.client,
            "strategy_summary": args["strategy_summary"],
            "owner": args["owner"],
        }
        model = args.get("model") or self.model
        if not model:
            raise ToolError(
                f"{tool_name} requires a model argument or a dispatcher default model"
            )
        kwargs["model"] = model
        if "reasoning" in args:
            kwargs["reasoning"] = args["reasoning"]
        if "temperature" in args:
            kwargs["temperature"] = args["temperature"]
        if "worker_max_output_tokens" in args:
            kwargs["worker_max_output_tokens"] = args["worker_max_output_tokens"]
        if "review_max_output_tokens" in args:
            kwargs["review_max_output_tokens"] = args["review_max_output_tokens"]
        if "read_only_paths" in args:
            kwargs["read_only_paths"] = args["read_only_paths"]
        return kwargs

    def _dispatch_ready(self, args: dict[str, Any]) -> Any:
        kwargs = self._dispatch_kwargs(args, tool_name="dispatch_ready")
        return self.plane.dispatch_ready_leaf(args["task_id"], **kwargs)

    def _run_attempt(self, args: dict[str, Any]) -> Any:
        # Explicit lifecycle name for a normal ready-leaf dispatch; same
        # underlying ControlPlane.dispatch_ready_leaf path as dispatch_ready.
        kwargs = self._dispatch_kwargs(args, tool_name="run_attempt")
        return self.plane.dispatch_ready_leaf(args["task_id"], **kwargs)

    def _retry_attempt(self, args: dict[str, Any]) -> Any:
        # Later attempt: require four-field retry contract and pass it through
        # so harness no-blind-retry / strategy-difference gates stay authoritative.
        kwargs = self._dispatch_kwargs(args, tool_name="retry_attempt")
        kwargs["retry_contract"] = {
            "addresses": args["addresses"],
            "preserves": args["preserves"],
            "differs_from": args["differs_from"],
            "expected_improvement": args["expected_improvement"],
        }
        return self.plane.dispatch_ready_leaf(args["task_id"], **kwargs)

    def _promote_attempt(self, args: dict[str, Any]) -> Any:
        # Always through harness.promote — validation/review gates apply.
        return self.plane.recover_promotion(
            attempt_id=args["attempt_id"],
            repo_root=Path(args["repo_root"]),
        )

    def _recover_promotion(self, args: dict[str, Any]) -> Any:
        return self.plane.recover_promotion(
            attempt_id=args["attempt_id"],
            repo_root=Path(args["repo_root"]),
        )

    # ------------------------------------------------------------------
    # Repository inspection
    # ------------------------------------------------------------------

    def _repository_map(self, args: dict[str, Any]) -> Any:
        # Cap caller-supplied budget before any inspector / filesystem work.
        kwargs: dict[str, Any] = {}
        if "max_map_entries" in args:
            kwargs["max_map_entries"] = _enforce_tool_limit(
                "max_map_entries",
                args["max_map_entries"],
                cap=TOOL_MAX_MAP_ENTRIES,
            )
        inspector = self._inspector(args["repo_root"], **kwargs)
        return inspector.build_map().to_dict()

    def _list_tree(self, args: dict[str, Any]) -> Any:
        max_entries = args.get("max_entries")
        if max_entries is not None:
            max_entries = _enforce_tool_limit(
                "max_entries",
                max_entries,
                cap=TOOL_MAX_TREE_ENTRIES,
            )
        inspector = self._inspector(args["repo_root"])
        return inspector.list_tree(
            prefix=args.get("prefix") or "",
            max_entries=max_entries,
        )

    def _read_artifact(self, args: dict[str, Any]) -> Any:
        max_bytes = args.get("max_bytes")
        if max_bytes is not None:
            max_bytes = _enforce_tool_limit(
                "max_bytes",
                max_bytes,
                cap=TOOL_MAX_READ_BYTES,
            )
        inspector = self._inspector(args["repo_root"])
        return inspector.read_artifact(
            args["path"],
            max_bytes=max_bytes,
            start_line=args.get("start_line"),
            end_line=args.get("end_line"),
        )

    def _search_text(self, args: dict[str, Any]) -> Any:
        max_matches = args.get("max_matches")
        if max_matches is not None:
            max_matches = _enforce_tool_limit(
                "max_matches",
                max_matches,
                cap=TOOL_MAX_SEARCH_MATCHES,
            )
        inspector = self._inspector(args["repo_root"])
        return inspector.search_text(
            args["query"],
            prefix=args.get("prefix") or "",
            max_matches=max_matches,
        )

    def _inspect_python_symbols(self, args: dict[str, Any]) -> Any:
        inspector = self._inspector(args["repo_root"])
        return inspector.inspect_python_symbols(args["path"])

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _build_planning_pack(self, args: dict[str, Any]) -> Any:
        built = self._planning().build_planning_pack(
            repo_root=Path(args["repo_root"]),
            task_id=args.get("task_id"),
            request_id=args.get("request_id"),
            selected_paths=args.get("selected_paths") or (),
            include_repository_map=args.get("include_repository_map", True),
        )
        # Never return the live RepositoryInspector to the chat surface.
        return {
            "pack": built["pack"],
            "manifest": {
                "context_manifest_id": built["manifest"].get("context_manifest_id"),
                "manifest_sha256": built["manifest"].get("manifest_sha256"),
                "role": built["manifest"].get("role"),
                "item_count": len(built["manifest"].get("items") or []),
            },
            "repository_map": built.get("repository_map"),
            "request": built.get("request"),
            "task": built.get("task"),
            "task_spec": built.get("task_spec"),
            "ledger_projection_id": (
                (built.get("ledger_projection") or {}).get("projection_id")
                if built.get("ledger_projection")
                else None
            ),
            "existing_plan_node_ids": [
                str(node.get("node_id"))
                for node in (built.get("plan_nodes") or [])
            ],
        }

    def _propose_plan(self, args: dict[str, Any]) -> Any:
        if self.client is None:
            raise ToolError("propose_plan requires an InferenceClient on the dispatcher")
        model = args.get("model") or self.model
        if not model:
            raise ToolError(
                "propose_plan requires a model argument or a dispatcher default model"
            )
        result = self._planning().propose_plan(
            repo_root=Path(args["repo_root"]),
            client=self.client,
            model=model,
            task_id=args.get("task_id"),
            request_id=args.get("request_id"),
            selected_paths=args.get("selected_paths") or (),
            parent_node_id=args.get("parent_node_id"),
            reasoning=args.get("reasoning") or "off",
            temperature=args.get("temperature", 0.0),
            max_output_tokens=args.get("max_output_tokens") or 4096,
        )
        # Evidence for chat sequencing: propose -> review -> apply.
        return {
            "plan_version_id": result.get("plan_version_id"),
            "status": result.get("status"),
            "approved": result.get("approved"),
            "applied": result.get("applied"),
            "task_id": result.get("task_id"),
            "request_id": result.get("request_id"),
            "context_manifest_id": result.get("context_manifest_id"),
            "context_manifest_sha256": result.get("context_manifest_sha256"),
            "planning_pack_sha256": result.get("planning_pack_sha256"),
            "repository_map_sha256": result.get("repository_map_sha256"),
            "proposal_sha256": result.get("proposal_sha256"),
            "proposal": result.get("proposal"),
            "parent_node_id": result.get("parent_node_id"),
            "model": result.get("model"),
        }

    def _review_plan(self, args: dict[str, Any]) -> Any:
        review_payload = args.get("review")
        if review_payload is None:
            if self.client is None:
                raise ToolError(
                    "review_plan requires operator review or InferenceClient+model"
                )
            model = args.get("model") or self.model
            if not model:
                raise ToolError(
                    "review_plan requires a model argument or a dispatcher default model"
                )
            record = self._planning().review_plan(
                args["plan_version_id"],
                client=self.client,
                model=model,
                reasoning=args.get("reasoning") or "off",
                temperature=args.get("temperature", 0.0),
                max_output_tokens=args.get("max_output_tokens") or 2048,
            )
        else:
            record = self._planning().review_plan(
                args["plan_version_id"],
                review=review_payload,
                operator_caller=self._operator(),
            )
        version = self._planning().get_plan_version(args["plan_version_id"])
        return {
            "review_id": record.get("review_id"),
            "plan_version_id": record.get("plan_version_id"),
            "verdict": record.get("verdict"),
            "evidence": record.get("evidence"),
            "notes": record.get("notes"),
            "approved": version.get("approved"),
            "status": version.get("status"),
            "context_manifest_id": version.get("context_manifest_id"),
            "context_manifest_sha256": version.get("context_manifest_sha256"),
            "repository_map_sha256": version.get("repository_map_sha256"),
            "proposal_sha256": version.get("proposal_sha256"),
        }

    def _list_plan_versions(self, _args: dict[str, Any]) -> Any:
        return self._planning().list_plan_versions()

    def _get_plan_version(self, args: dict[str, Any]) -> Any:
        return self._planning().get_plan_version(args["plan_version_id"])

    def _apply_approved_plan(self, args: dict[str, Any]) -> Any:
        return self._planning().apply_approved_plan(
            args["plan_version_id"],
            repo_root=Path(args["repo_root"]),
            request_text=args.get("request_text"),
        )

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def _list_integrations(self, _args: dict[str, Any]) -> Any:
        return self.plane.list_integrations()

    def _get_integration(self, args: dict[str, Any]) -> Any:
        return self.plane.get_integration(args["integration_id"])

    def _assemble_integration(self, args: dict[str, Any]) -> Any:
        selected = args.get("selected")
        return self.plane.assemble_integration(
            args["parent_node_id"],
            repo_root=Path(args["repo_root"]),
            selected=selected,
            expected_baseline_tree_sha256=args.get("expected_baseline_tree_sha256"),
        )

    def _review_integration(self, args: dict[str, Any]) -> Any:
        model = args.get("model") or self.model
        return self.plane.review_integration(
            args["integration_id"],
            verdict=args.get("verdict"),
            requirement_results=args.get("requirement_results") or [],
            notes=args.get("notes"),
            client=self.client if model else None,
            model=model,
            reasoning=str(args.get("reasoning") or "off"),
            max_output_tokens=int(args.get("max_output_tokens") or 2048),
            operator_caller=(
                self._operator() if args.get("verdict") is not None else None
            ),
        )

    def _promote_integration(self, args: dict[str, Any]) -> Any:
        repo_root = args.get("repo_root")
        return self.plane.promote_integration(
            args["integration_id"],
            repo_root=Path(repo_root) if repo_root else None,
        )
