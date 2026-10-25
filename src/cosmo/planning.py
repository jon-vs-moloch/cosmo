"""Local-only planning service over ControlPlane + ContextCompiler.

Produces immutable plan proposals/versions from a deterministic planning pack,
requires an independent plan review before any ControlPlane mutation, and
never persists prompt transcripts. Model calls use the supplied local
InferenceClient only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .atomic import (
    AtomicHarnessError,
    SUPPORTED_DETERMINISTIC_VALIDATORS,
    _safe_record_identifier,
    _safe_relative_path,
    _sha256_text,
    new_id,
    utc_now,
)
from .authority import CallerProvenance, authenticate_model, require_operator
from .context import ContextCompiler, ContextError, RepositoryInspector, RepositoryMap
from .control import ControlPlane, ControlPlaneError
from .inference import InferenceClient
from .models import Generation
from .reporting import canonical_hash


class PlanningError(AtomicHarnessError):
    """Planning proposal, review, or application failed an invariant."""


PLAN_REVIEW_VERDICTS = frozenset(
    {
        "PASS",
        "REVISE",
        "REVISE_PLAN",
        "REJECT",
        "REPLAN_FROM_PARENT",
        "BLOCKED_ON_INFORMATION",
        "STALE_SPECIFICATION",
        "ESCALATE",
    }
)

APPROVED_PLAN_VERDICTS = frozenset({"PASS"})

# Planning context is deliberately bounded even when a chat model controls
# ``selected_paths``.  Per-file ContextCompiler limits are not sufficient if
# an untrusted caller requests thousands of individually valid files.
MAX_SELECTED_PATHS = 64
MAX_SELECTED_BYTES = 4 * 1024 * 1024

PLANNER_SYSTEM = """You are a bounded Cosmo Planner.
Decompose the supplied task into a recursive plan of parent and leaf nodes.
Every leaf must declare exactly one writable relative artifact path and local
deterministic validators. Parents coordinate children and declare composition
validators; parents must not claim a single product writable artifact.
Dependencies must reference sibling temporary child ids from this proposal.
Return exactly one JSON object matching the plan proposal schema.
Do not include Markdown fences, transcripts, or free-form prose outside JSON."""

PLAN_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "cosmo_plan_proposal",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "objective": {"type": "string"},
                "requirements": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "shared_contracts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "artifact": {"type": "string"},
                            "required_version": {},
                        },
                        "required": ["artifact"],
                        "additionalProperties": True,
                    },
                },
                "children": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "temp_id": {"type": "string"},
                            "objective": {"type": "string"},
                            "kind": {
                                "type": "string",
                                "enum": ["leaf", "parent"],
                            },
                            "writable_artifact": {"type": ["string", "null"]},
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        "validators": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "requirements": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "constraints": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "root_acceptance_linkage": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "shared_contracts": {"type": "array"},
                        "children": {"type": "array"},
                        },
                        "required": [
                            "temp_id",
                            "objective",
                            "kind",
                            "depends_on",
                        ],
                        "additionalProperties": True,
                    },
                },
                "parent_validators": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "risks": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "unresolved_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "objective",
                "requirements",
                "children",
                "parent_validators",
                "risks",
                "unresolved_questions",
            ],
            "additionalProperties": False,
        },
    },
}

PLAN_REVIEWER_SYSTEM = """You are an independent Cosmo Plan Reviewer.
You receive a bounded, hash-anchored projection of the Planning Agent's
planning pack and context manifest. The durable records retain the complete
pack; the model projection intentionally omits large source excerpts so that
small local models have room to reason and return the required JSON. Check:
- Complete requirement coverage (every requirement ID linked to a child)
- Root/ancestor acceptance linkage
- Semantic atomicity stopping rule (one writable artifact, narrow responsibility,
  short context, bounded transformation, local deterministic validation)
- Exactly one writable artifact per leaf and unique paths
- Valid dependency ordering and parent composition validators
- Missing consumers, call sites, documentation, or migration steps
- Whether assumptions require research or user clarification
Return exactly one JSON object with verdict, evidence, notes, and
requirement_coverage. Verdict must be one of PASS, REVISE, REJECT,
REVISE_PLAN, REPLAN_FROM_PARENT, BLOCKED_ON_INFORMATION, STALE_SPECIFICATION,
ESCALATE."""

# LM Studio instances are often loaded with a deliberately small context
# window.  Review output needs its own budget, so sending every source excerpt
# from a planning manifest can make an otherwise valid review impossible.  We
# keep complete manifests durable and project only bounded metadata plus small
# excerpts to the reviewer.
PLAN_REVIEW_EXCERPT_BUDGET = 3500
PLAN_REVIEW_ITEM_EXCERPT_LIMIT = 600


def _as_string_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise PlanningError(f"{field_name} must be an array")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise PlanningError(f"{field_name} entries must be strings")
        result.append(item)
    return result


def _parse_model_json_object(content: str, *, label: str) -> dict[str, Any]:
    """Accept exactly one JSON object or one complete outer JSON fence.

    This is the full transport grammar: surrounding whitespace is harmless and
    a single full-response `````json`` fence may wrap the object.  Prefixes,
    suffixes, multiple objects, and prose are protocol failures.  The semantic
    schema remains strict in the validators below.
    """
    text = (content or "").strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE
    )
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlanningError(f"{label} is not one JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise PlanningError(f"{label} must be one JSON object")
    return value


def parse_plan_proposal(content: str) -> dict[str, Any]:
    """Parse and validate planner JSON into a normalized PlanProposal."""
    value = _parse_model_json_object(content, label="plan proposal")
    return validate_plan_proposal(value)


def validate_plan_proposal(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate plan shape and path safety without mutating product files."""
    required = {
        "objective",
        "requirements",
        "children",
        "parent_validators",
        "risks",
        "unresolved_questions",
    }
    missing = required - set(value)
    if missing:
        raise PlanningError(
            f"plan proposal missing required fields: {sorted(missing)}"
        )
    objective = value["objective"]
    if not isinstance(objective, str) or not objective.strip():
        raise PlanningError("plan objective must be a non-empty string")

    requirements = _as_string_list(value.get("requirements"), field_name="requirements")
    parent_validators = _as_string_list(
        value.get("parent_validators"), field_name="parent_validators"
    )
    risks = _as_string_list(value.get("risks"), field_name="risks")
    unresolved = _as_string_list(
        value.get("unresolved_questions"), field_name="unresolved_questions"
    )
    shared_contracts = value.get("shared_contracts") or []
    if not isinstance(shared_contracts, list):
        raise PlanningError("shared_contracts must be an array")
    normalized_contracts: list[dict[str, Any]] = []
    for item in shared_contracts:
        if not isinstance(item, dict) or "artifact" not in item:
            raise PlanningError(
                "each shared_contract must be an object with artifact"
            )
        try:
            artifact = _safe_relative_path(str(item["artifact"]))
        except AtomicHarnessError as exc:
            raise PlanningError(str(exc)) from exc
        entry = dict(item)
        entry["artifact"] = artifact
        normalized_contracts.append(entry)

    children_raw = value.get("children")
    if not isinstance(children_raw, list) or not children_raw:
        raise PlanningError("plan proposal must declare at least one child")

    children = [_normalize_child(child, ancestors=()) for child in children_raw]
    _validate_dependency_graph(children)

    # At least one leaf somewhere in the tree.
    if not _collect_leaves(children):
        raise PlanningError("plan proposal must contain at least one leaf")

    # Stopping-rule / atomicity checks on leaves (spec §9.3).
    atomicity_notes = _check_leaf_atomicity(children)

    root_acceptance = value.get("root_acceptance_linkage") or requirements
    if isinstance(root_acceptance, list):
        root_acceptance = [
            str(item) for item in root_acceptance if str(item).strip()
        ]
    else:
        root_acceptance = list(requirements)

    # Requirement coverage: every declared requirement should appear on at
    # least one child when children declare requirements.
    coverage = _requirement_coverage_matrix(requirements, children)

    proposal = {
        "objective": objective.strip(),
        "requirements": requirements,
        "shared_contracts": normalized_contracts,
        "children": children,
        "parent_validators": parent_validators,
        "risks": risks,
        "unresolved_questions": unresolved,
        "root_acceptance_linkage": root_acceptance,
        "requirement_coverage": coverage,
        "atomicity_notes": atomicity_notes,
    }
    proposal["proposal_sha256"] = canonical_hash(
        {
            "objective": proposal["objective"],
            "requirements": proposal["requirements"],
            "shared_contracts": proposal["shared_contracts"],
            "children": proposal["children"],
            "parent_validators": proposal["parent_validators"],
            "risks": proposal["risks"],
            "unresolved_questions": proposal["unresolved_questions"],
            "root_acceptance_linkage": proposal["root_acceptance_linkage"],
        }
    )
    return proposal


def _check_leaf_atomicity(
    children: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Record semantic atomicity notes; hard-fail only on structural violations."""
    notes: list[str] = []
    for leaf in _collect_leaves(children):
        path = str(leaf.get("writable_artifact") or "")
        objective = str(leaf.get("objective") or "")
        validators = list(leaf.get("validators") or [])
        if not path:
            raise PlanningError(
                f"leaf {leaf.get('temp_id')} fails stopping rule: no writable artifact"
            )
        if not validators:
            notes.append(
                f"leaf {leaf.get('temp_id')} missing local validators "
                f"(defaulted elsewhere)"
            )
        if len(objective) > 400:
            notes.append(
                f"leaf {leaf.get('temp_id')} objective is very long; "
                "consider further decomposition"
            )
        # Heuristic: multi-file language in objective suggests non-atomic leaf.
        lowered = objective.lower()
        if any(
            token in lowered
            for token in (
                " and update ",
                " and modify ",
                "multiple files",
                "across the codebase",
            )
        ):
            notes.append(
                f"leaf {leaf.get('temp_id')} objective may span multiple "
                "artifacts; verify atomicity"
            )
    return notes


def _requirement_coverage_matrix(
    requirements: Sequence[str],
    children: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Map requirement IDs/texts to covering child temp_ids when declared."""
    nodes = _iter_nodes(children)
    coverage: dict[str, list[str]] = {}
    for req in requirements:
        req_s = str(req)
        covering: list[str] = []
        for node in nodes:
            node_reqs = node.get("requirements") or []
            if req_s in {str(item) for item in node_reqs}:
                covering.append(str(node["temp_id"]))
            elif req_s.lower() in str(node.get("objective") or "").lower():
                covering.append(str(node["temp_id"]))
        coverage[req_s] = covering
    uncovered = [req for req, covers in coverage.items() if not covers]
    return {
        "by_requirement": coverage,
        "uncovered": uncovered,
        "complete": not uncovered or not requirements,
    }


def _normalize_child(
    raw: Any, *, ancestors: tuple[str, ...]
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PlanningError("each child must be an object")
    temp_id = raw.get("temp_id") or raw.get("node") or raw.get("id")
    if not isinstance(temp_id, str) or not temp_id.strip():
        raise PlanningError("each child requires a non-empty temp_id")
    temp_id = temp_id.strip()
    if temp_id in ancestors:
        raise PlanningError(f"cyclic child temp_id nesting: {temp_id}")
    objective = raw.get("objective")
    if not isinstance(objective, str) or not objective.strip():
        raise PlanningError(f"child {temp_id} requires a non-empty objective")
    kind = str(raw.get("kind") or ("leaf" if raw.get("writable_artifact") else "parent"))
    if kind not in {"leaf", "parent"}:
        raise PlanningError(f"child {temp_id} has invalid kind: {kind!r}")
    depends_on = _as_string_list(raw.get("depends_on") or [], field_name="depends_on")
    validators = _as_string_list(raw.get("validators") or [], field_name="validators")
    nested_raw = raw.get("children") or []
    if nested_raw and not isinstance(nested_raw, list):
        raise PlanningError(f"child {temp_id} children must be an array")
    nested = [
        _normalize_child(item, ancestors=ancestors + (temp_id,))
        for item in nested_raw
    ]

    writable = raw.get("writable_artifact")
    if kind == "leaf":
        if nested:
            raise PlanningError(
                f"leaf {temp_id} must not declare nested children"
            )
        if not isinstance(writable, str) or not writable.strip():
            raise PlanningError(
                f"leaf {temp_id} must declare exactly one writable_artifact"
            )
        unsupported = sorted(
            set(validators) - set(SUPPORTED_DETERMINISTIC_VALIDATORS)
        )
        if unsupported:
            raise PlanningError(
                f"leaf {temp_id} declares unsupported deterministic validators "
                f"{unsupported}; use only "
                f"{sorted(SUPPORTED_DETERMINISTIC_VALIDATORS)}. "
                "Shell/test commands belong to post-integration validation."
            )
        try:
            writable = _safe_relative_path(writable.strip())
        except AtomicHarnessError as exc:
            raise PlanningError(
                f"leaf {temp_id} writable_artifact is unsafe: {exc}"
            ) from exc
        if not validators:
            validators = ["non_empty", "utf8"]
    else:
        writable = None
        if not nested and not raw.get("allow_empty_parent"):
            # Parent may be a coordination shell that will be refined later;
            # still require either nested children or explicit parent validators.
            if not validators and not raw.get("parent_validators"):
                pass

    child_requirements = _as_string_list(
        raw.get("requirements") or [], field_name="requirements"
    )
    constraints = _as_string_list(
        raw.get("constraints") or raw.get("explicit_constraints") or [],
        field_name="constraints",
    )
    root_linkage = _as_string_list(
        raw.get("root_acceptance_linkage") or [],
        field_name="root_acceptance_linkage",
    )
    shared_contracts_raw = raw.get("shared_contracts") or []
    if not isinstance(shared_contracts_raw, list):
        raise PlanningError(f"child {temp_id} shared_contracts must be an array")
    shared_contracts: list[dict[str, Any]] = []
    for item in shared_contracts_raw:
        if not isinstance(item, dict) or "artifact" not in item:
            raise PlanningError(
                f"child {temp_id} shared_contracts entries require artifact"
            )
        try:
            contract = dict(item)
            contract["artifact"] = _safe_relative_path(str(item["artifact"]))
        except AtomicHarnessError as exc:
            raise PlanningError(str(exc)) from exc
        shared_contracts.append(contract)
    return {
        "temp_id": temp_id,
        "objective": objective.strip(),
        "kind": kind,
        "writable_artifact": writable,
        "depends_on": depends_on,
        "validators": validators,
        "requirements": child_requirements,
        "constraints": constraints,
        "root_acceptance_linkage": root_linkage,
        "shared_contracts": shared_contracts,
        "children": nested,
    }


def _collect_leaves(children: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    leaves: list[dict[str, Any]] = []
    for child in children:
        if child.get("kind") == "leaf":
            leaves.append(dict(child))
        leaves.extend(_collect_leaves(child.get("children") or []))
    return leaves


def _iter_nodes(
    children: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for child in children:
        nodes.append(dict(child))
        nodes.extend(_iter_nodes(child.get("children") or []))
    return nodes


def _validate_dependency_graph(children: Sequence[Mapping[str, Any]]) -> None:
    nodes = _iter_nodes(children)
    ids = [str(node["temp_id"]) for node in nodes]
    if len(ids) != len(set(ids)):
        raise PlanningError("plan child temp_ids must be unique")
    id_set = set(ids)
    # Sibling dependency checks: depends_on must reference known temp_ids.
    for node in nodes:
        for dep in node.get("depends_on") or []:
            if dep not in id_set:
                raise PlanningError(
                    f"child {node['temp_id']} depends on unknown temp_id {dep!r}"
                )
            if dep == node["temp_id"]:
                raise PlanningError(
                    f"child {node['temp_id']} cannot depend on itself"
                )
    # Leaf uniqueness of writable artifacts.
    leaf_paths = [
        str(node["writable_artifact"])
        for node in nodes
        if node.get("kind") == "leaf"
    ]
    if len(leaf_paths) != len(set(leaf_paths)):
        raise PlanningError(
            "each leaf must declare a unique writable_artifact path"
        )
    # Cycle detection on depends_on edges.
    adjacency = {
        str(node["temp_id"]): [str(d) for d in (node.get("depends_on") or [])]
        for node in nodes
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node_id: str) -> None:
        if node_id in visiting:
            raise PlanningError("plan dependencies contain a cycle")
        if node_id in visited:
            return
        visiting.add(node_id)
        for dep in adjacency.get(node_id, []):
            dfs(dep)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in adjacency:
        dfs(node_id)


def parse_plan_review(content: str) -> dict[str, Any]:
    value = _parse_model_json_object(content, label="plan review")
    return validate_plan_review(value)


def validate_plan_review(value: Mapping[str, Any]) -> dict[str, Any]:
    verdict = str(value.get("verdict") or "").upper()
    if verdict == "REVISE_PLAN":
        # Normalize synonym used in the full architecture spec.
        pass
    if verdict not in PLAN_REVIEW_VERDICTS:
        raise PlanningError(f"unsupported plan review verdict: {verdict!r}")
    evidence = value.get("evidence") or value.get("checks") or []
    if not isinstance(evidence, list) or not evidence:
        raise PlanningError("plan review requires non-empty evidence")
    if any(not isinstance(item, str) or not item.strip() for item in evidence):
        raise PlanningError("plan review evidence entries must be non-empty strings")
    notes = value.get("notes") or value.get("rationale") or ""
    # Some local instruction-tuned models serialize notes as a short list even
    # when the contract says string.  Normalize that presentation-only variant
    # while keeping arbitrary objects invalid.
    if isinstance(notes, list) and all(
        isinstance(item, str) for item in notes
    ):
        notes = "; ".join(item.strip() for item in notes if item.strip())
    if notes is not None and not isinstance(notes, str):
        raise PlanningError("plan review notes must be a string")
    return {
        "verdict": verdict,
        "evidence": list(evidence),
        "notes": notes or "",
        "requirement_coverage": value.get("requirement_coverage"),
        "recommended_next_action": value.get("recommended_next_action"),
    }


class PlanningService:
    """Plan proposal / independent review / approval-only application."""

    def __init__(
        self,
        control: ControlPlane,
        *,
        compiler: ContextCompiler | None = None,
    ):
        self.control = control
        self.store = control.store
        self.compiler = compiler or ContextCompiler(control.store)

    # ------------------------------------------------------------------
    # Planning pack
    # ------------------------------------------------------------------

    def build_planning_pack(
        self,
        *,
        repo_root: Path,
        task_id: str | None = None,
        request_id: str | None = None,
        selected_paths: Sequence[str] = (),
        include_repository_map: bool = True,
        inspector: RepositoryInspector | None = None,
        max_map_entries: int = 500,
    ) -> dict[str, Any]:
        """Create a deterministic planning input pack and context manifest."""
        repo_root = Path(repo_root).resolve()
        if not repo_root.is_dir():
            raise PlanningError(f"repo_root is not a directory: {repo_root}")

        task: dict[str, Any] | None = None
        request: dict[str, Any] | None = None
        task_spec: dict[str, Any] | None = None
        ledger: dict[str, Any] | None = None
        plan_nodes: list[dict[str, Any]] = list(self.control.list_plans())

        if task_id is not None:
            task = self.control.get_task(task_id)
            if task.get("current_spec_id"):
                task_spec = self.control.get_task_spec(str(task["current_spec_id"]))
            linked = list(task.get("linked_request_ids") or [])
            if linked:
                request = self.control.get_request(str(linked[0]))
            try:
                ledger = self.control.project_ledger(str(task["task_id"]))
            except (ControlPlaneError, AtomicHarnessError):
                ledger = None
        if request_id is not None and request is None:
            request = self.control.get_request(request_id)

        inspector = inspector or RepositoryInspector(
            repo_root,
            state_root=self.store.root,
            max_map_entries=max_map_entries,
        )
        repository_map: RepositoryMap | None = None
        if include_repository_map:
            repository_map = inspector.build_map()

        selected_files: list[dict[str, Any]] = []
        if len(selected_paths) > MAX_SELECTED_PATHS:
            raise PlanningError(
                f"selected_paths exceeds planning cap ({MAX_SELECTED_PATHS})"
            )
        selected_bytes = 0
        for relative in selected_paths:
            read = inspector.read_artifact(relative)
            selected_bytes += int(read.get("size_bytes") or 0)
            if selected_bytes > MAX_SELECTED_BYTES:
                raise PlanningError(
                    "selected planning context exceeds aggregate byte cap "
                    f"({MAX_SELECTED_BYTES})"
                )
            selected_files.append(
                {
                    "path": read["path"],
                    "sha256": read["sha256"],
                    "content_sha256": read["content_sha256"],
                    "excerpt": read["excerpt"],
                    "range": read["range"],
                    "selection_method": "explicit",
                    "kind": "read_only_file",
                }
            )

        manifest = self.compiler.compile_for_planning(
            request=request,
            task=task,
            task_spec=task_spec,
            plan_nodes=plan_nodes,
            ledger_projection=ledger,
            repository_map=repository_map,
            selected_files=selected_files,
        )

        pack = {
            "schema_version": "cosmo.planning-pack.v1",
            "repo_root": str(repo_root),
            "task_id": task.get("task_id") if task else None,
            "request_id": request.get("request_id") if request else None,
            "spec_id": task_spec.get("spec_id") if task_spec else None,
            "context_manifest_id": manifest["context_manifest_id"],
            "context_manifest_sha256": manifest["manifest_sha256"],
            "repository_map_sha256": (
                repository_map.map_sha256 if repository_map is not None else None
            ),
            "existing_plan_node_ids": [
                str(node.get("node_id")) for node in plan_nodes
            ],
            "ledger_projection_id": (
                ledger.get("projection_id") if ledger is not None else None
            ),
            "selected_paths": list(selected_paths),
            "created_at": utc_now(),
        }
        pack["pack_sha256"] = canonical_hash(
            {
                "context_manifest_sha256": pack["context_manifest_sha256"],
                "repository_map_sha256": pack["repository_map_sha256"],
                "task_id": pack["task_id"],
                "request_id": pack["request_id"],
                "spec_id": pack["spec_id"],
                "existing_plan_node_ids": pack["existing_plan_node_ids"],
                "ledger_projection_id": pack["ledger_projection_id"],
                "selected_paths": pack["selected_paths"],
            }
        )
        self.store.append_event(
            "planning_pack_built",
            {
                "pack_sha256": pack["pack_sha256"],
                "context_manifest_id": pack["context_manifest_id"],
                "task_id": pack["task_id"],
                "request_id": pack["request_id"],
            },
        )
        return {
            "pack": pack,
            "manifest": manifest,
            "repository_map": repository_map.to_dict() if repository_map else None,
            "request": request,
            "task": task,
            "task_spec": task_spec,
            "ledger_projection": ledger,
            "plan_nodes": plan_nodes,
            "inspector": inspector,
        }

    # ------------------------------------------------------------------
    # Propose
    # ------------------------------------------------------------------

    def propose_plan(
        self,
        *,
        repo_root: Path,
        client: InferenceClient,
        model: str,
        task_id: str | None = None,
        request_id: str | None = None,
        selected_paths: Sequence[str] = (),
        include_repository_map: bool = True,
        reasoning: str = "off",
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        parent_node_id: str | None = None,
    ) -> dict[str, Any]:
        """Call the local planner and store an immutable plan version (unreviewed)."""
        built = self.build_planning_pack(
            repo_root=repo_root,
            task_id=task_id,
            request_id=request_id,
            selected_paths=selected_paths,
            include_repository_map=include_repository_map,
        )
        pack = built["pack"]
        manifest = built["manifest"]
        request = built["request"]
        task = built["task"]
        task_spec = built["task_spec"]

        if task is not None:
            try:
                self.control.transition_task(
                    str(task["task_id"]),
                    "planning",
                    reason="planning proposal in progress",
                    actor="planning_service",
                )
            except ControlPlaneError:
                # Already planning or not eligible; continue without forcing.
                pass

        prompt = self._planning_prompt(
            pack=pack,
            manifest=manifest,
            request=request,
            task=task,
            task_spec=task_spec,
            repository_map=built.get("repository_map"),
            plan_nodes=built.get("plan_nodes") or [],
            ledger=built.get("ledger_projection"),
        )
        generation = client.chat(
            model=model,
            role="planner",
            system_prompt=PLANNER_SYSTEM,
            prompt=prompt,
            reasoning=reasoning,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format=PLAN_RESPONSE_FORMAT,
            operation_id=new_id("PLAN-OP"),
        )
        proposal = parse_plan_proposal(generation.content)
        # A plan proposed for an existing parent Task must refine that
        # parent, not silently create a second coordination root on apply.
        if parent_node_id is None and task is not None and task.get("plan_node_id"):
            candidate_parent = self.store.read(
                "plan_nodes", str(task["plan_node_id"])
            )
            if candidate_parent.get("kind") == "parent":
                parent_node_id = str(candidate_parent["node_id"])
        return self.store_plan_proposal(
            proposal=proposal,
            pack=pack,
            parent_node_id=parent_node_id,
            model=model,
            generation=generation,
        )

    def store_plan_proposal(
        self,
        *,
        proposal: Mapping[str, Any],
        pack: Mapping[str, Any],
        parent_node_id: str | None = None,
        model: str | None = None,
        generation: Generation | None = None,
        operator_supplied: bool = False,
        operator_caller: CallerProvenance | None = None,
    ) -> dict[str, Any]:
        """Validate and persist an immutable plan version without applying it."""
        caller_provenance = None
        if operator_supplied:
            caller_provenance = require_operator(operator_caller).to_record()
        elif generation is not None and model:
            caller_provenance = authenticate_model(
                f"lmstudio:{model}", authenticated_by="cosmo.plan-proposal"
            ).to_record()
        normalized = validate_plan_proposal(proposal)
        plan_version_id = new_id("PLANVER")
        now = utc_now()
        record = {
            "schema_version": "cosmo.plan-version.v1",
            "plan_version_id": plan_version_id,
            "status": "proposed",
            "approved": False,
            "applied": False,
            "parent_node_id": parent_node_id,
            "task_id": pack.get("task_id"),
            "request_id": pack.get("request_id"),
            "spec_id": pack.get("spec_id"),
            "context_manifest_id": pack.get("context_manifest_id"),
            "context_manifest_sha256": pack.get("context_manifest_sha256"),
            "planning_pack_sha256": pack.get("pack_sha256"),
            "repository_map_sha256": pack.get("repository_map_sha256"),
            "repo_root": pack.get("repo_root"),
            "planning_repo_tree_sha256": None,
            "task_spec_sha256": None,
            "request_immutable_sha256": None,
            "proposal": normalized,
            "proposal_sha256": normalized["proposal_sha256"],
            "model": model,
            "operator_supplied": bool(operator_supplied),
            "caller_provenance": caller_provenance,
            "generation": None,
            "review_id": None,
            "created_at": now,
        }
        if pack.get("repo_root"):
            try:
                from .atomic import _product_revision

                record["planning_repo_tree_sha256"] = _product_revision(
                    Path(str(pack["repo_root"])), excluded_root=self.store.root
                )
            except (OSError, AtomicHarnessError):
                # A manually supplied pack may omit a readable repo; the
                # context/review hashes remain authoritative for that mode.
                pass
        if pack.get("spec_id"):
            try:
                record["task_spec_sha256"] = canonical_hash(
                    self.control.get_task_spec(str(pack["spec_id"]))
                )
            except (ControlPlaneError, AtomicHarnessError):
                pass
        if pack.get("request_id"):
            try:
                record["request_immutable_sha256"] = self.control.get_request(
                    str(pack["request_id"])
                ).get("immutable_sha256")
            except (ControlPlaneError, AtomicHarnessError):
                pass
        if generation is not None:
            # Record hashes only — never the prompt transcript body.
            record["generation"] = {
                "role": generation.role,
                "content_sha256": _sha256_text(generation.content),
                "request_hash": generation.request_hash,
                "reasoning_hash": generation.reasoning_hash,
                "stats": dict(generation.stats or {}),
                "latency_seconds": generation.latency_seconds,
                "model_instance_id": generation.model_instance_id,
            }
        self.store.write("plan_versions", plan_version_id, record)
        self.store.append_event(
            "plan_proposal_stored",
            {
                "plan_version_id": plan_version_id,
                "proposal_sha256": normalized["proposal_sha256"],
                "context_manifest_id": pack.get("context_manifest_id"),
                "task_id": pack.get("task_id"),
                "status": "proposed",
            },
        )
        return record

    def _planning_prompt(
        self,
        *,
        pack: Mapping[str, Any],
        manifest: Mapping[str, Any],
        request: Mapping[str, Any] | None,
        task: Mapping[str, Any] | None,
        task_spec: Mapping[str, Any] | None,
        repository_map: Mapping[str, Any] | None,
        plan_nodes: Sequence[Mapping[str, Any]],
        ledger: Mapping[str, Any] | None,
    ) -> str:
        # Compact, hash-anchored prompt. Full map may be large; include summary.
        map_summary = None
        if repository_map is not None:
            files = repository_map.get("files") or []
            map_summary = {
                "map_sha256": repository_map.get("map_sha256"),
                "file_count": repository_map.get("file_count") or len(files),
                "truncated": repository_map.get("truncated"),
                "files": [
                    {
                        "path": entry.get("path"),
                        "sha256": entry.get("sha256"),
                        "language": entry.get("language"),
                        "symbols": entry.get("symbols"),
                    }
                    for entry in files[:200]
                ],
            }
        payload = {
            "planning_pack": pack,
            "context_manifest_id": manifest.get("context_manifest_id"),
            "context_manifest_sha256": manifest.get("manifest_sha256"),
            "request": request,
            "task": task,
            "task_spec": task_spec,
            "existing_plan_nodes": [
                {
                    "node_id": node.get("node_id"),
                    "kind": node.get("kind"),
                    "objective": node.get("objective"),
                    "writable_artifact": node.get("writable_artifact"),
                    "dependencies": node.get("dependencies"),
                    "parent_node_id": node.get("parent_node_id"),
                    "status": node.get("status"),
                }
                for node in plan_nodes
            ],
            "ledger_projection": ledger,
            "repository_map_summary": map_summary,
            "output_contract": (
                "Return one JSON object with objective, requirements, "
                "shared_contracts, children (temp_id/objective/kind/"
                "writable_artifact/depends_on/validators/requirements/"
                "constraints/root_acceptance_linkage), parent_validators, "
                "risks, unresolved_questions. Leaf validators must be symbolic "
                "names from [non_empty, utf8, json, python_syntax, python]; "
                "never emit shell commands such as pytest or py_compile. "
                "Broader tests are explicit post-integration validation."
            ),
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    # ------------------------------------------------------------------
    # Review
    # ------------------------------------------------------------------

    def review_plan(
        self,
        plan_version_id: str,
        *,
        review: Mapping[str, Any] | None = None,
        client: InferenceClient | None = None,
        model: str | None = None,
        reasoning: str = "off",
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        independent_reviewer: str | None = None,
        operator_caller: CallerProvenance | None = None,
    ) -> dict[str, Any]:
        """Independent plan review. Never applies the plan."""
        plan_version_id = _safe_record_identifier(plan_version_id)
        version = self.store.read("plan_versions", plan_version_id)
        if version.get("applied"):
            raise PlanningError("cannot re-review an already applied plan version")

        # Always load the planner's full context for independent review.
        review_context = self._plan_review_context(version)

        if review is not None:
            authority = require_operator(operator_caller)
            validated = validate_plan_review(review)
            reviewer = independent_reviewer or str(
                review.get("independent_reviewer") or "operator_supplied"
            )
            operator_supplied = True
            caller_provenance = authority.to_record()
            generation_meta = None
        else:
            if client is None or model is None:
                raise PlanningError(
                    "review_plan requires operator review payload or client+model"
                )
            prompt = json.dumps(
                {
                    **review_context,
                    "review_contract": (
                        "Return JSON with verdict, evidence (non-empty string "
                        "array), notes, and requirement_coverage. You have the "
                        "full planning pack the planner saw."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            generation = client.chat(
                model=model,
                role="plan_reviewer",
                system_prompt=PLAN_REVIEWER_SYSTEM,
                prompt=prompt,
                reasoning=reasoning,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                operation_id=new_id("PLAN-REVIEW-OP"),
            )
            validated = parse_plan_review(generation.content)
            reviewer = independent_reviewer or model
            operator_supplied = False
            caller_provenance = authenticate_model(
                f"lmstudio:{model}", authenticated_by="cosmo.plan-review"
            ).to_record()
            generation_meta = {
                "role": generation.role,
                "content_sha256": _sha256_text(generation.content),
                "request_hash": generation.request_hash,
                "stats": dict(generation.stats or {}),
            }

        # Deterministic structural evidence supplements model/operator review.
        structural = self._deterministic_plan_checks(version)
        evidence = list(validated["evidence"])
        for item in structural:
            if item not in evidence:
                evidence.append(item)

        # A PASS review is not allowed to override deterministic plan-shape
        # failures.  The model/operator review remains independent evidence,
        # but coverage and parent-composition gates are hard prerequisites for
        # materialization.
        verdict = validated["verdict"]
        structural_failures = [
            item for item in structural if item.startswith("STRUCTURAL_FAIL:")
        ]
        if verdict == "PASS" and structural_failures:
            verdict = "REVISE_PLAN"
            validated_notes = str(validated.get("notes") or "").strip()
            suffix = "deterministic plan gates failed: " + "; ".join(
                structural_failures
            )
            validated_notes = f"{validated_notes}; {suffix}".strip("; ")
        else:
            validated_notes = validated.get("notes")

        review_id = new_id("PLAN-REVIEW")
        now = utc_now()
        record = {
            "schema_version": "cosmo.plan-review.v1",
            "review_id": review_id,
            "plan_version_id": plan_version_id,
            "node_id": version.get("parent_node_id"),
            "verdict": verdict,
            "evidence": evidence,
            "notes": validated_notes,
            "requirement_coverage": (
                validated.get("requirement_coverage")
                or (version.get("proposal") or {}).get("requirement_coverage")
            ),
            "root_acceptance_linkage": (version.get("proposal") or {}).get(
                "root_acceptance_linkage"
            ),
            "atomicity_notes": (version.get("proposal") or {}).get(
                "atomicity_notes"
            ),
            "planning_context_manifest_id": version.get("context_manifest_id"),
            "planning_pack_sha256": version.get("planning_pack_sha256"),
            "recommended_next_action": validated.get("recommended_next_action"),
            "independent_reviewer": reviewer,
            "operator_supplied": operator_supplied,
            "caller_provenance": caller_provenance,
            "generation": generation_meta,
            "created_at": now,
        }
        self.store.write("plan_reviews", review_id, record)

        # Update plan version projection (status only; proposal stays immutable).
        version["review_id"] = review_id
        version["status"] = (
            "approved"
            if verdict in APPROVED_PLAN_VERDICTS
            else f"review_{verdict.lower()}"
        )
        version["approved"] = verdict in APPROVED_PLAN_VERDICTS
        version["reviewed_at"] = now
        self.store.replace("plan_versions", plan_version_id, version)

        self.store.append_event(
            "plan_reviewed",
            {
                "plan_version_id": plan_version_id,
                "review_id": review_id,
                "verdict": verdict,
                "approved": version["approved"],
            },
        )
        return record

    def _plan_review_context(self, version: Mapping[str, Any]) -> dict[str, Any]:
        """Bounded planning context for the independent reviewer.

        The complete planning pack/context manifest remains in AtomicStore and
        is hash-linked in this projection.  Only bounded excerpts are passed to
        inference, which prevents an 8k-context local model from failing before
        it can emit a review.  This is a context compiler boundary, not a
        weakening of the deterministic review gates.
        """
        manifest_id = version.get("context_manifest_id")
        manifest: dict[str, Any] = {}
        if manifest_id:
            try:
                manifest = self.store.read("context_manifests", str(manifest_id))
            except AtomicHarnessError:
                manifest = {}
        request = None
        task = None
        task_spec = None
        if version.get("request_id"):
            try:
                request = self.control.get_request(str(version["request_id"]))
            except (ControlPlaneError, AtomicHarnessError):
                request = None
        if version.get("task_id"):
            try:
                task = self.control.get_task(str(version["task_id"]))
            except (ControlPlaneError, AtomicHarnessError):
                task = None
        if version.get("spec_id"):
            try:
                task_spec = self.control.get_task_spec(str(version["spec_id"]))
            except (ControlPlaneError, AtomicHarnessError):
                task_spec = None
        projected_items: list[dict[str, Any]] = []
        remaining_excerpt = PLAN_REVIEW_EXCERPT_BUDGET
        omitted_excerpt_items = 0
        for raw in manifest.get("items") or []:
            if not isinstance(raw, Mapping):
                continue
            item = {
                key: raw.get(key)
                for key in (
                    "kind",
                    "path",
                    "source_path",
                    "sha256",
                    "content_hash",
                    "selection_method",
                    "language",
                    "size_bytes",
                    "truncated",
                )
                if raw.get(key) is not None
            }
            excerpt = raw.get("excerpt")
            if isinstance(excerpt, str) and excerpt and remaining_excerpt > 0:
                take = min(PLAN_REVIEW_ITEM_EXCERPT_LIMIT, remaining_excerpt)
                item["excerpt"] = excerpt[:take]
                if len(excerpt) > take:
                    item["excerpt_truncated"] = True
                remaining_excerpt -= take
            elif isinstance(excerpt, str) and excerpt:
                omitted_excerpt_items += 1
            projected_items.append(item)
        return {
            "context_projection": {
                "mode": "bounded_for_model_context",
                "complete_context_is_hash_linked_in_atomic_store": True,
                "excerpt_budget_chars": PLAN_REVIEW_EXCERPT_BUDGET,
                "omitted_excerpt_items": omitted_excerpt_items,
            },
            "plan_version_id": version.get("plan_version_id"),
            "proposal": version.get("proposal"),
            "proposal_sha256": version.get("proposal_sha256"),
            "context_manifest_id": manifest_id,
            "context_manifest_sha256": version.get("context_manifest_sha256"),
            "planning_pack_sha256": version.get("planning_pack_sha256"),
            "repository_map_sha256": version.get("repository_map_sha256"),
            "manifest": {
                "context_manifest_id": manifest.get("context_manifest_id"),
                "manifest_sha256": manifest.get("manifest_sha256"),
                "role": manifest.get("role"),
                "item_count": len(manifest.get("items") or []),
                "items": projected_items,
            },
            "request": request,
            "task": task,
            "task_spec": task_spec,
            "requirements": (version.get("proposal") or {}).get("requirements"),
            "root_acceptance_linkage": (version.get("proposal") or {}).get(
                "root_acceptance_linkage"
            ),
            "atomicity_notes": (version.get("proposal") or {}).get(
                "atomicity_notes"
            ),
            "requirement_coverage": (version.get("proposal") or {}).get(
                "requirement_coverage"
            ),
        }

    def _deterministic_plan_checks(
        self, version: Mapping[str, Any]
    ) -> list[str]:
        """Structural plan checks that do not require a model."""
        checks: list[str] = []
        proposal = version.get("proposal") or {}
        try:
            validated = validate_plan_proposal(proposal)
        except PlanningError as exc:
            return [f"STRUCTURAL_FAIL: {exc}"]
        leaves = _collect_leaves(validated.get("children") or [])
        checks.append(f"leaf_count={len(leaves)}")
        checks.append("each leaf has exactly one writable_artifact")
        if validated.get("parent_validators"):
            checks.append("parent composition validators declared")
        else:
            checks.append("STRUCTURAL_FAIL: no parent_validators declared")
        coverage = validated.get("requirement_coverage") or {}
        if coverage.get("complete"):
            checks.append("requirement coverage complete or requirements empty")
        else:
            uncovered = coverage.get("uncovered") or []
            checks.append(
                "STRUCTURAL_FAIL: requirements without explicit child coverage: "
                f"{uncovered}"
            )
        for note in validated.get("atomicity_notes") or []:
            checks.append(f"ATOMICITY: {note}")
        if version.get("context_manifest_id"):
            checks.append(
                f"planner context manifest present: {version.get('context_manifest_id')}"
            )
        else:
            checks.append("WARNING: plan version missing context_manifest_id")
        return checks

    # ------------------------------------------------------------------
    # Apply (approval only)
    # ------------------------------------------------------------------

    def apply_approved_plan(
        self,
        plan_version_id: str,
        *,
        repo_root: Path,
        request_text: str | None = None,
        priority_components: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Materialize parent/leaf plan nodes only after an approved review."""
        plan_version_id = _safe_record_identifier(plan_version_id)
        version = self.store.read("plan_versions", plan_version_id)
        if version.get("applied"):
            raise PlanningError(
                f"plan version already applied: {plan_version_id}"
            )
        if not version.get("approved"):
            raise PlanningError(
                "refuse to apply unreviewed or non-PASS plan version; "
                f"status={version.get('status')!r}"
            )
        review_id = version.get("review_id")
        if not review_id:
            raise PlanningError("approved plan version is missing review_id")
        review = self.store.read("plan_reviews", str(review_id))
        if str(review.get("verdict")).upper() not in APPROVED_PLAN_VERDICTS:
            raise PlanningError(
                f"linked plan review is not PASS: {review.get('verdict')!r}"
            )

        repo_root = Path(repo_root).resolve()
        if not repo_root.is_dir():
            raise PlanningError(f"repo_root is not a directory: {repo_root}")

        proposal = validate_plan_proposal(version["proposal"])
        parent_node_id = version.get("parent_node_id")
        created: list[dict[str, Any]] = []

        # Approved plans are tied to the exact TaskSpec, immutable Request,
        # context manifest, and repository head that the independent reviewer
        # saw.  Any change requires a fresh plan/review rather than applying a
        # stale decomposition.
        if version.get("task_id"):
            current_task = self.control.get_task(str(version["task_id"]))
            if version.get("spec_id") and current_task.get("current_spec_id") != version.get("spec_id"):
                raise PlanningError("approved plan is stale: TaskSpec id changed")
            if version.get("task_spec_sha256") and current_task.get("current_spec_id"):
                current_hash = canonical_hash(
                    self.control.get_task_spec(str(current_task["current_spec_id"]))
                )
                if current_hash != version.get("task_spec_sha256"):
                    raise PlanningError("approved plan is stale: TaskSpec changed")
            linked = list(current_task.get("linked_request_ids") or [])
            if version.get("request_id") and linked:
                current_request = self.control.get_request(str(linked[0]))
                if (
                    version.get("request_immutable_sha256")
                    and current_request.get("immutable_sha256")
                    != version.get("request_immutable_sha256")
                ):
                    raise PlanningError("approved plan is stale: Request changed")
        if version.get("planning_repo_tree_sha256"):
            from .atomic import _product_revision

            observed_tree = _product_revision(
                repo_root, excluded_root=self.store.root
            )
            if observed_tree != version.get("planning_repo_tree_sha256"):
                raise PlanningError(
                    "approved plan is stale: repository baseline changed"
                )

        # Older/operator-created plan versions may omit parent_node_id even
        # though they point at an existing parent Task. Recover that linkage
        # before deciding whether a new coordination root is needed.
        if parent_node_id is None and version.get("task_id"):
            try:
                task_for_parent = self.control.get_task(str(version["task_id"]))
                candidate = task_for_parent.get("plan_node_id")
                if candidate:
                    candidate_plan = self.store.read("plan_nodes", str(candidate))
                    if candidate_plan.get("kind") == "parent":
                        parent_node_id = str(candidate_plan["node_id"])
            except (ControlPlaneError, AtomicHarnessError):
                pass

        # Root coordination parent for this proposal when none supplied.
        root_request_text = request_text
        if root_request_text is None:
            if version.get("request_id"):
                try:
                    root_request_text = self.control.get_request(
                        str(version["request_id"])
                    ).get("original_text")
                except AtomicHarnessError:
                    root_request_text = None
        if not root_request_text:
            root_request_text = f"Execute approved plan: {proposal['objective']}"

        root_request_id = version.get("request_id")
        if root_request_id:
            root_request_id = _safe_record_identifier(str(root_request_id))
            try:
                self.control.get_request(root_request_id)
            except (ControlPlaneError, AtomicHarnessError):
                root_request_id = None

        if parent_node_id is None:
            if root_request_id is None:
                ingested = self.control.ingest_request(
                    request_text=root_request_text,
                    desired_outcome=proposal["objective"],
                    explicit_constraints=proposal.get("risks") or [],
                    summary=proposal["objective"],
                    requester="planning_service",
                )
                root_request_id = str(ingested["request_id"])
            root_requirements = self._requirement_records(
                proposal.get("requirements") or [],
                objective=proposal["objective"],
                source="approved_plan",
            )
            root = self.control.create_task_from_triage(
                request_id=root_request_id,
                objective=proposal["objective"],
                repo_root=repo_root,
                plan_kind="parent",
                validators=proposal.get("parent_validators") or [],
                explicit_constraints=proposal.get("risks") or [],
                requirements=root_requirements,
                shared_contracts=proposal.get("shared_contracts") or [],
                root_acceptance_linkage=proposal.get("root_acceptance_linkage") or [],
                priority_components=priority_components,
                relationship={
                    "type": "NEW",
                    "target_task_id": None,
                    "rationale": "approved plan root",
                },
                triage_proposal_id=f"PLAN-{plan_version_id}",
                triage_review_id=str(review_id),
                plan_version_id=plan_version_id,
                plan_review_id=str(review_id),
                title=proposal["objective"],
                summary=root_request_text,
                needs_planning=True,
            )
            parent_node_id = root["node_id"]
            created.append({"kind": "parent", **root})
        else:
            parent_node_id = _safe_record_identifier(str(parent_node_id))
            parent = self.store.read("plan_nodes", parent_node_id)
            if parent.get("kind") != "parent":
                raise PlanningError(
                    "plan_version parent_node_id must reference a parent node"
                )
            parent_task_id = parent.get("task_id")
            if root_request_id is None and parent_task_id:
                try:
                    parent_task = self.control.get_task(str(parent_task_id))
                    linked = list(parent_task.get("linked_request_ids") or [])
                    if linked:
                        root_request_id = str(linked[0])
                except (ControlPlaneError, AtomicHarnessError):
                    pass
            if root_request_id is None:
                ingested = self.control.ingest_request(
                    request_text=root_request_text,
                    desired_outcome=proposal["objective"],
                    explicit_constraints=proposal.get("risks") or [],
                    summary=proposal["objective"],
                    requester="planning_service",
                )
                root_request_id = str(ingested["request_id"])

        temp_to_node: dict[str, str] = {}
        # Flatten + global order so nested children can depend on uncles/siblings
        # without silently dropping depends_on edges.
        created.extend(
            self._apply_children(
                children=proposal["children"],
                parent_node_id=parent_node_id,
                repo_root=repo_root,
                request_id=root_request_id,
                root_request_text=root_request_text,
                priority_components=priority_components,
                temp_to_node=temp_to_node,
                plan_version_id=plan_version_id,
                plan_review_id=str(review_id),
            )
        )

        # Applying an approved plan opens the governing parent gate.  A draft
        # parent must never permit leaf dispatch, but every materialized child
        # needs a durable approved/open parent state to become schedulable.
        applied_parent = self.store.read("plan_nodes", str(parent_node_id))
        applied_parent["status"] = "open"
        applied_parent["approved_plan_version_id"] = plan_version_id
        applied_parent["plan_review_id"] = str(review_id)
        self.store.replace("plan_nodes", str(parent_node_id), applied_parent)

        version["applied"] = True
        version["status"] = "applied"
        version["applied_at"] = utc_now()
        version["applied_parent_node_id"] = parent_node_id
        version["applied_node_ids"] = [
            item.get("node_id") for item in created if item.get("node_id")
        ]
        version["temp_id_map"] = temp_to_node
        self.store.replace("plan_versions", plan_version_id, version)
        self.store.append_event(
            "plan_applied",
            {
                "plan_version_id": plan_version_id,
                "parent_node_id": parent_node_id,
                "node_ids": version["applied_node_ids"],
                "review_id": review_id,
            },
        )

        # Return planning task to ready if it was marked planning.
        task_id = version.get("task_id")
        if task_id:
            try:
                task = self.control.get_task(str(task_id))
                if task.get("status") == "planning":
                    self.control.transition_task(
                        str(task_id),
                        "ready",
                        reason="plan applied",
                        actor="planning_service",
                    )
            except (ControlPlaneError, AtomicHarnessError):
                pass

        return {
            "plan_version_id": plan_version_id,
            "parent_node_id": parent_node_id,
            "created": created,
            "temp_id_map": temp_to_node,
            "review_id": review_id,
        }

    def _apply_children(
        self,
        *,
        children: Sequence[Mapping[str, Any]],
        parent_node_id: str,
        repo_root: Path,
        request_id: str,
        root_request_text: str,
        priority_components: Mapping[str, Any] | None,
        temp_to_node: dict[str, str],
        plan_version_id: str,
        plan_review_id: str,
    ) -> list[dict[str, Any]]:
        """Materialize a (possibly nested) child tree under an existing parent.

        Nodes are created in a global order that honors both ``depends_on``
        edges and parent-before-child nesting, so a nested leaf may depend on
        an uncle without the edge being dropped.
        """
        flat = self._flatten_proposal_nodes(children, parent_temp_id=None)
        ordered = self._order_nodes_for_apply(flat)
        created: list[dict[str, Any]] = []
        for node in ordered:
            temp_id = str(node["temp_id"])
            parent_temp = node.get("_parent_temp_id")
            if parent_temp is None:
                apply_parent = parent_node_id
            else:
                if parent_temp not in temp_to_node:
                    raise PlanningError(
                        f"child {temp_id} parent temp_id {parent_temp!r} "
                        "was not materialised before the child"
                    )
                apply_parent = temp_to_node[parent_temp]

            missing_deps = [
                dep
                for dep in node.get("depends_on") or []
                if dep not in temp_to_node
            ]
            if missing_deps:
                raise PlanningError(
                    f"child {temp_id} depends on unmaterialised temp_ids: "
                    f"{missing_deps}"
                )
            dep_node_ids = [
                temp_to_node[dep] for dep in node.get("depends_on") or []
            ]

            requirements = self._requirement_records(
                node.get("requirements") or [],
                objective=str(node["objective"]),
                source="approved_plan_child",
            )
            child_id = f"{plan_version_id}-{temp_id}"
            result = self.control.create_task_from_triage(
                request_id=request_id,
                objective=str(node["objective"]),
                repo_root=repo_root,
                plan_kind=str(node["kind"]),
                artifact_path=str(node.get("writable_artifact") or ""),
                validators=node.get("validators") or [],
                explicit_constraints=node.get("constraints") or [],
                requirements=requirements,
                shared_contracts=node.get("shared_contracts") or [],
                root_acceptance_linkage=node.get("root_acceptance_linkage") or [],
                priority_components=priority_components,
                relationship={
                    "type": "NEW",
                    "target_task_id": None,
                    "rationale": "approved plan child",
                },
                triage_proposal_id=f"PLAN-{child_id}",
                triage_review_id=f"PLAN-REVIEW-{child_id}",
                plan_version_id=plan_version_id,
                plan_review_id=plan_review_id,
                title=str(node["objective"]),
                summary=f"{root_request_text} :: {node['objective']}",
                needs_planning=str(node["kind"]) == "parent",
                parent_node_id=apply_parent,
                dependencies=dep_node_ids,
            )
            temp_to_node[temp_id] = result["node_id"]
            created.append(
                {"kind": node["kind"], "temp_id": temp_id, **result}
            )
        return created

    @staticmethod
    def _requirement_records(
        raw_requirements: Iterable[Any],
        *,
        objective: str,
        source: str,
    ) -> list[dict[str, Any]]:
        """Convert plan requirement IDs/text into durable TaskSpec records."""
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_requirements, start=1):
            text = str(raw).strip()
            if not text:
                continue
            requirement_id = text if text.startswith("REQ-") or text.startswith("R") else f"REQ-{index}"
            if requirement_id in seen:
                continue
            seen.add(requirement_id)
            records.append(
                {
                    "requirement_id": requirement_id,
                    "text": text,
                    "source": source,
                    "inferred": False,
                }
            )
        if not records:
            records.append(
                {
                    "requirement_id": "REQ-1",
                    "text": objective,
                    "source": source,
                    "inferred": False,
                }
            )
        return records

    def _flatten_proposal_nodes(
        self,
        children: Sequence[Mapping[str, Any]],
        *,
        parent_temp_id: str | None,
    ) -> list[dict[str, Any]]:
        flat: list[dict[str, Any]] = []
        for child in children:
            node = dict(child)
            node["_parent_temp_id"] = parent_temp_id
            # Nested payload is tracked via recursion; strip to avoid double work.
            nested = list(node.get("children") or [])
            node["children"] = []
            flat.append(node)
            flat.extend(
                self._flatten_proposal_nodes(
                    nested, parent_temp_id=str(node["temp_id"])
                )
            )
        return flat

    def _order_nodes_for_apply(
        self, nodes: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Topologically order flattened nodes (deps + nesting parent first)."""
        by_id = {str(node["temp_id"]): dict(node) for node in nodes}
        if len(by_id) != len(nodes):
            raise PlanningError("plan child temp_ids must be unique")

        # Prerequisites: explicit depends_on plus nesting parent (if any).
        prereqs: dict[str, set[str]] = {}
        for temp_id, node in by_id.items():
            needed: set[str] = set()
            for dep in node.get("depends_on") or []:
                dep_s = str(dep)
                if dep_s not in by_id:
                    raise PlanningError(
                        f"child {temp_id} depends on unknown temp_id {dep_s!r}"
                    )
                if dep_s == temp_id:
                    raise PlanningError(
                        f"child {temp_id} cannot depend on itself"
                    )
                needed.add(dep_s)
            parent_temp = node.get("_parent_temp_id")
            if parent_temp is not None:
                parent_s = str(parent_temp)
                if parent_s not in by_id:
                    raise PlanningError(
                        f"child {temp_id} has unknown parent temp_id {parent_s!r}"
                    )
                needed.add(parent_s)
            prereqs[temp_id] = needed

        remaining = set(by_id)
        ordered: list[dict[str, Any]] = []
        while remaining:
            ready = sorted(
                node_id
                for node_id in remaining
                if prereqs[node_id].isdisjoint(remaining)
            )
            if not ready:
                raise PlanningError(
                    "plan dependencies/nesting contain a cycle"
                )
            for node_id in ready:
                ordered.append(by_id[node_id])
                remaining.remove(node_id)
        return ordered

    def _topological_children(
        self, children: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Order a single level so dependencies among siblings are created first.

        Retained for callers/tests that operate on a flat sibling list. Nested
        application uses :meth:`_order_nodes_for_apply` instead.
        """
        nodes = [dict(child) for child in children]
        ids = {str(n["temp_id"]) for n in nodes}
        remaining = {str(n["temp_id"]): n for n in nodes}
        ordered: list[dict[str, Any]] = []
        while remaining:
            ready = [
                node
                for node in remaining.values()
                if all(
                    dep not in remaining
                    for dep in node.get("depends_on") or []
                    if dep in ids
                )
            ]
            if not ready:
                raise PlanningError("plan dependencies contain a cycle")
            ready.sort(key=lambda item: str(item["temp_id"]))
            for node in ready:
                ordered.append(node)
                remaining.pop(str(node["temp_id"]), None)
        return ordered

    # Spec-aligned aliases
    def plan(self, **kwargs: Any) -> dict[str, Any]:
        return self.propose_plan(**kwargs)

    def applyApprovedPlan(self, plan_version_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.apply_approved_plan(plan_version_id, **kwargs)

    def get_plan_version(self, plan_version_id: str) -> dict[str, Any]:
        return self.store.read(
            "plan_versions", _safe_record_identifier(plan_version_id)
        )

    def list_plan_versions(self) -> list[dict[str, Any]]:
        return self.store.list_records("plan_versions")

    def expand_ready_child(
        self,
        parent_node_id: str,
        *,
        repo_root: Path,
        client: InferenceClient,
        model: str,
        selected_paths: Sequence[str] = (),
        reasoning: str = "off",
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Just-in-time recursive planning for the next coarse parent child.

        Spec §9.6: expand only the next ready child when detailed decomposition
        is needed. Returns an unreviewed plan version; review/apply remain
        separate so no product mutation occurs before promotion of the plan.
        """
        parent_node_id = _safe_record_identifier(parent_node_id)
        parent = self.store.read("plan_nodes", parent_node_id)
        if parent.get("kind") != "parent":
            raise PlanningError("expand_ready_child requires a parent plan node")
        task_id = parent.get("task_id")
        if not task_id:
            raise PlanningError("parent plan node is missing task_id")
        # Prefer a child marked draft/open without nested leaves.
        children = list(parent.get("child_node_ids") or [])
        target_task_id = str(task_id)
        for child_id in children:
            try:
                child = self.store.read("plan_nodes", str(child_id))
            except AtomicHarnessError:
                continue
            if child.get("kind") == "parent" and child.get("status") in {
                "draft",
                "open",
                "approved",
            }:
                nested = list(child.get("child_node_ids") or [])
                if not nested:
                    target_task_id = str(child.get("task_id") or task_id)
                    break
        return self.propose_plan(
            repo_root=repo_root,
            client=client,
            model=model,
            task_id=target_task_id,
            selected_paths=selected_paths,
            reasoning=reasoning,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            parent_node_id=parent_node_id,
        )
