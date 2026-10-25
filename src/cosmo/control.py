"""Control-plane facade over the Cosmo atomic-artifact store.

This layer schedules and records work on top of :class:`AtomicStore` and
:class:`AtomicHarness`. It does not invent a second persistence system and
does not call external providers.

    immutable Request
      -> triage proposal + independent triage review
         (model-backed via TriageService, or explicit operator fallback)
      -> Task + versioned TaskSpec + relationships + priority assessment
      -> parent/leaf plan graph with dependency-aware readiness
      -> exclusive leases and a deterministic ready queue / scheduler
      -> dispatch into AtomicHarness (Worker/Validation/Review/Promotion)
      -> append-only convergence assessments and a compact ledger projection

Canonical product path: ControlPlane + PlanningService + TriageService +
AtomicStore. AtomicHarness.submit remains legacy-only.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .atomic import (
    AtomicHarness,
    AtomicHarnessError,
    AtomicStore,
    _artifact_target,
    _exclusive_lock,
    _product_revision,
    _read_text,
    _safe_record_identifier,
    _safe_relative_path,
    _sha256_text,
    new_id,
    utc_now,
)
from .authority import CallerProvenance, authenticate_operator, require_operator
from .inference import InferenceClient
from .reporting import canonical_hash


class ControlPlaneError(AtomicHarnessError):
    """A control-plane invariant or protocol failed."""


# Spec §19 task statuses plus harness-friendly ready/running/stale/planning.
# "executing" is accepted as an alias of "running"; "integrating" is parent join.
TASK_STATUSES = frozenset(
    {
        "ready",
        "planning",
        "queued",
        "running",
        "executing",  # alias recorded as running on transition
        "integrating",
        "blocked",
        "stale",
        "completed",
        "canceled",
    }
)

# Legal directed transitions. Terminal statuses have empty outbound sets.
LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    "ready": frozenset(
        {
            "planning",
            "queued",
            "running",
            "executing",
            "integrating",
            "blocked",
            "stale",
            "canceled",
        }
    ),
    "planning": frozenset(
        {"ready", "queued", "blocked", "stale", "canceled", "running"}
    ),
    "queued": frozenset(
        {"ready", "running", "executing", "blocked", "stale", "canceled"}
    ),
    "running": frozenset(
        {
            "ready",
            "blocked",
            "stale",
            "completed",
            "canceled",
            "integrating",
            "planning",
        }
    ),
    "executing": frozenset(
        {
            "ready",
            "blocked",
            "stale",
            "completed",
            "canceled",
            "integrating",
            "planning",
        }
    ),
    "integrating": frozenset(
        {"ready", "running", "blocked", "stale", "completed", "canceled"}
    ),
    "blocked": frozenset(
        {"ready", "planning", "queued", "stale", "canceled"}
    ),
    "stale": frozenset({"ready", "planning", "canceled"}),
    "completed": frozenset(),
    "canceled": frozenset(),
}

# Spec §8.3 relationship taxonomy.
RELATIONSHIP_TYPES = frozenset(
    {
        "NEW",
        "DUPLICATE_OF",
        "OVERLAPS_WITH",
        "DEPENDS_ON",
        "SUPERSEDES",
        "CONFLICTS_WITH",
    }
)

# Spec §8.4 triage review verdicts.
TRIAGE_REVIEW_VERDICTS = frozenset(
    {
        "PASS",
        "RETRY_TRIAGE",
        "NEEDS_USER_CLARIFICATION",
        "ESCALATE",
    }
)

# Explainable priority components (spec §8.3 + legacy aliases).
# Higher positive weights raise priority; negative weights are costs/penalties.
# Weights are fixed so ready-queue ordering is deterministic across processes.
PRIORITY_WEIGHTS: dict[str, float] = {
    # Spec components (0-1 normalized; 0-100 inputs accepted).
    "user_importance": 3.0,
    "urgency": 4.0,
    "deadline_pressure": 3.0,
    "dependency_unblocking": 2.5,
    "risk_reduction": 2.0,
    "age": 1.0,
    "estimated_effort": -1.5,
    # Legacy aliases retained so existing operators/tests keep working.
    "impact": 3.0,
    "freshness": 1.0,
    "operator_boost": 2.0,
    "blocked_penalty": -5.0,
    "dependency_penalty": -3.0,
}

# Map legacy names onto canonical spec names for projections.
PRIORITY_CANONICAL_ALIASES: dict[str, str] = {
    "impact": "user_importance",
    "freshness": "age",
}

# Legacy names are accepted at the boundary, but must never contribute a
# second time when their canonical spec component is also present.  Keeping
# the alias records in the breakdown preserves old inspection clients while
# making the score itself unambiguous.
_PRIORITY_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "user_importance": ("user_importance", "impact"),
    "age": ("age", "freshness"),
}

# Aging: each full day since created_at adds this much to the age component
# (clamped at 1.0) when computing the effective ready-queue score.
PRIORITY_AGE_PER_DAY = 0.05
DEFAULT_LEASE_SECONDS = 300
DEFAULT_MAX_CONCURRENT_DISPATCHES = 2


def _parse_utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _clamp_unit(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _normalize_priority_value(name: str, number: float) -> float:
    """Normalize 0-100 scale inputs to 0-1; leave penalties as magnitudes."""
    if name.endswith("_penalty") or name == "estimated_effort":
        # Allow 0-100 effort/penalty magnitudes by scaling down when > 1.
        if number < 0.0:
            raise ControlPlaneError(
                f"priority component {name!r} must be non-negative"
            )
        if number > 1.0:
            return _clamp_unit(number / 100.0)
        return float(number)
    if number > 1.0:
        # Spec documents 0-100; normalize to unit interval.
        return _clamp_unit(number / 100.0)
    return _clamp_unit(number)


def score_priority_components(components: Mapping[str, Any]) -> dict[str, Any]:
    """Score explainable priority components into a single total.

    Unknown component names are rejected so operators cannot smuggle opaque
    scores past the explainable contract. Accepts both spec names
    (user_importance, urgency, ...) and legacy aliases (impact, freshness, ...).
    Values may be 0-1 or 0-100.
    """
    if not isinstance(components, Mapping) or not components:
        raise ControlPlaneError("priority components must be a non-empty mapping")
    normalized: dict[str, float] = {}
    supplied: set[str] = set()
    for key, raw in components.items():
        name = str(key)
        if name not in PRIORITY_WEIGHTS:
            raise ControlPlaneError(f"unknown priority component: {name!r}")
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ControlPlaneError(
                f"priority component {name!r} must be numeric"
            ) from exc
        normalized[name] = _normalize_priority_value(name, number)
        supplied.add(name)
    # Fill omitted components with zero so the breakdown is always complete.
    for name in PRIORITY_WEIGHTS:
        normalized.setdefault(name, 0.0)
    # Canonical projection: an explicitly supplied spec name wins over its
    # legacy alias, including an intentional explicit zero.  If only the
    # alias was supplied, it is promoted to the canonical component.
    canonical: dict[str, float] = {}
    for canonical_name, names in _PRIORITY_ALIAS_GROUPS.items():
        if canonical_name in supplied:
            canonical[canonical_name] = normalized[canonical_name]
        else:
            canonical[canonical_name] = max(
                (normalized[name] for name in names if name in supplied),
                default=0.0,
            )
    for name in (
        "urgency",
        "deadline_pressure",
        "dependency_unblocking",
        "risk_reduction",
        "estimated_effort",
    ):
        canonical[name] = normalized[name]

    # Score canonical components exactly once.  Alias rows remain visible but
    # carry zero weight and an ``alias_of`` pointer, so old consumers can still
    # inspect them without accidentally double-counting them.
    score_values = dict(normalized)
    for canonical_name, names in _PRIORITY_ALIAS_GROUPS.items():
        score_values[canonical_name] = canonical[canonical_name]
    alias_names = {
        alias
        for canonical_name, names in _PRIORITY_ALIAS_GROUPS.items()
        for alias in names
        if alias != canonical_name
    }
    breakdown: dict[str, dict[str, Any]] = {}
    for name in sorted(PRIORITY_WEIGHTS):
        if name in alias_names:
            canonical_name = PRIORITY_CANONICAL_ALIASES[name]
            breakdown[name] = {
                "value": score_values[name],
                "weight": 0.0,
                "contribution": 0.0,
                "alias_of": canonical_name,
            }
            continue
        value = score_values[name]
        breakdown[name] = {
            "value": value,
            "weight": PRIORITY_WEIGHTS[name],
            "contribution": round(value * PRIORITY_WEIGHTS[name], 6),
        }
    total = round(sum(item["contribution"] for item in breakdown.values()), 6)
    return {
        "components": score_values,
        "canonical_components": canonical,
        "breakdown": breakdown,
        "score": total,
        "weights": dict(PRIORITY_WEIGHTS),
        "canonical_aliases": dict(PRIORITY_CANONICAL_ALIASES),
    }


def compute_age_component(
    created_at: str | None,
    *,
    now: str | None = None,
    base_age: float = 0.0,
) -> float:
    """Derive an age component from wall-clock aging plus a stored base."""
    base = _clamp_unit(float(base_age or 0.0))
    if not created_at:
        return base
    try:
        created = _parse_utc(str(created_at))
        stamp = _parse_utc(now or utc_now())
    except (TypeError, ValueError):
        return base
    days = max(0.0, (stamp - created).total_seconds() / 86400.0)
    return _clamp_unit(base + days * PRIORITY_AGE_PER_DAY)


def effective_priority_score(
    components: Mapping[str, Any],
    *,
    created_at: str | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Score priority with scheduler aging applied to the age component."""
    merged = dict(components)
    base_age = float(merged.get("age") or merged.get("freshness") or 0.0)
    if base_age > 1.0:
        base_age = base_age / 100.0
    aged = compute_age_component(created_at, now=now, base_age=base_age)
    merged["age"] = aged
    # Keep freshness in sync for legacy breakdown readers.  The scorer treats
    # it as an alias of ``age`` and therefore does not double-count it.
    if "freshness" in PRIORITY_WEIGHTS:
        merged["freshness"] = max(float(merged.get("freshness") or 0.0), aged)
    scored = score_priority_components(merged)
    scored["aged"] = True
    scored["age_component"] = aged
    return scored


class ControlPlane:
    """JSON-serializable control-plane facade over AtomicStore + AtomicHarness."""

    def __init__(self, store: AtomicStore):
        self.store = store
        self.harness = AtomicHarness(store)

    # ------------------------------------------------------------------
    # Request / task creation
    # ------------------------------------------------------------------

    def submit_request(
        self,
        *,
        request_text: str,
        objective: str,
        artifact_path: str,
        repo_root: Path,
        validators: Iterable[str] = (),
        desired_outcome: str | None = None,
        explicit_constraints: Iterable[str] = (),
        triage_proposal: Mapping[str, Any] | None = None,
        triage_review: Mapping[str, Any] | None = None,
        relationship: Mapping[str, Any] | None = None,
        priority_components: Mapping[str, Any] | None = None,
        plan_kind: str = "leaf",
        parent_node_id: str | None = None,
        dependencies: Iterable[str] = (),
        operator_caller: CallerProvenance | None = None,
    ) -> dict[str, Any]:
        """Create an immutable Request and a ready Task with explicit triage.

        Model triage/planning hooks are operator-supplied: callers must pass
        explicit triage proposal/review payloads (or accept the documented
        operator-supplied leaf defaults). This method never invokes a model.
        """
        if not request_text.strip():
            raise ControlPlaneError("request_text must not be empty")
        # This method is the trusted Python/operator shortcut. Model-facing
        # catalogs cannot reach it with a leaf contract. Direct callers get an
        # explicit local-API principal so every operator-supplied PASS remains
        # attributable; front doors should pass their authenticated principal.
        authority = require_operator(
            operator_caller
            or authenticate_operator(
                "trusted-control-api", authenticated_by="cosmo.control-api"
            )
        )
        caller_provenance = authority.to_record()
        if not objective.strip():
            raise ControlPlaneError("objective must not be empty")
        if plan_kind not in {"leaf", "parent"}:
            raise ControlPlaneError(
                f"plan_kind must be 'leaf' or 'parent', got {plan_kind!r}"
            )
        repo_root = Path(repo_root).resolve()
        if not repo_root.is_dir():
            raise ControlPlaneError(f"repo_root is not a directory: {repo_root}")

        safe_path: str | None = None
        if plan_kind == "leaf":
            safe_path = _safe_relative_path(artifact_path)
        elif artifact_path:
            # Parents may name a coordination path but must not claim a single
            # writable product artifact for workers.
            safe_path = None

        constraints = [str(item) for item in explicit_constraints]
        validator_list = list(dict.fromkeys(validators)) or (
            ["non_empty", "utf8"] if plan_kind == "leaf" else []
        )
        dependency_ids = [
            _safe_record_identifier(str(item)) for item in dependencies
        ]
        for dep_id in dependency_ids:
            self.store.read("plan_nodes", dep_id)

        if parent_node_id is not None:
            parent_node_id = _safe_record_identifier(parent_node_id)
            parent = self.store.read("plan_nodes", parent_node_id)
            if parent.get("kind") != "parent":
                raise ControlPlaneError(
                    f"parent_node_id must reference a parent plan node: "
                    f"{parent_node_id}"
                )

        request_id = new_id("REQ")
        task_id = new_id("TASK")
        spec_id = new_id("SPEC")
        node_id = new_id("NODE")
        relationship_id = new_id("REL")
        priority_id = new_id("PRIO")
        now = utc_now()

        request = {
            "schema_version": "cosmo.request.v1",
            "request_id": request_id,
            "original_text": request_text,
            "desired_outcome": desired_outcome,
            "explicit_constraints": constraints,
            "created_at": now,
            "immutable_sha256": canonical_hash(
                {
                    "original_text": request_text,
                    "desired_outcome": desired_outcome,
                    "explicit_constraints": constraints,
                }
            ),
        }

        # Operator-supplied triage is recorded explicitly for the leaf/parent
        # shortcut. Prefer TriageService for model-backed autonomous triage.
        proposal_payload = dict(triage_proposal or {})
        proposal_id = str(proposal_payload.pop("proposal_id", new_id("TRIAGE")))
        _safe_record_identifier(proposal_id)
        relationship_body = dict(
            relationship
            or proposal_payload.get("relationship")
            or {"type": "NEW", "target_task_id": None, "rationale": ""}
        )
        rel_type = str(relationship_body.get("type") or "NEW").upper()
        if rel_type not in RELATIONSHIP_TYPES:
            raise ControlPlaneError(
                f"unsupported relationship type: {rel_type!r}"
            )
        relationship_body["type"] = rel_type
        if "target_task_id" not in relationship_body:
            relationship_body["target_task_id"] = None
        triage_proposal_record = {
            "schema_version": "cosmo.triage-proposal.v1",
            "proposal_id": proposal_id,
            "request_id": request_id,
            "relationship": relationship_body,
            "proposed_task_id": task_id,
            "status": "applied",
            "applied": True,
            "mode": str(
                proposal_payload.get("mode") or "operator_supplied_leaf_contract"
            ),
            "notes": proposal_payload.get("notes"),
            "created_at": now,
            "operator_supplied": True,
            "caller_provenance": caller_provenance,
        }

        review_payload = dict(triage_review or {})
        review_id = str(review_payload.pop("review_id", new_id("TRIAGE-REVIEW")))
        _safe_record_identifier(review_id)
        triage_verdict = str(review_payload.get("verdict") or "PASS").upper()
        if triage_verdict == "REJECT":
            triage_verdict = "RETRY_TRIAGE"
        elif triage_verdict == "REVISE":
            triage_verdict = "RETRY_TRIAGE"
        if triage_verdict not in TRIAGE_REVIEW_VERDICTS:
            raise ControlPlaneError(
                f"unsupported triage review verdict: {triage_verdict!r}"
            )
        triage_review_record = {
            "schema_version": "cosmo.triage-review.v1",
            "review_id": review_id,
            "proposal_id": proposal_id,
            "verdict": triage_verdict,
            "evidence": list(
                review_payload.get("evidence")
                or review_payload.get("checks")
                or ["operator_supplied leaf/parent contract"]
            ),
            "independent_reviewer": str(
                review_payload.get("independent_reviewer")
                or "operator_leaf_contract"
            ),
            "rationale": review_payload.get("rationale"),
            "notes": review_payload.get("notes") or review_payload.get("rationale"),
            "created_at": now,
            "operator_supplied": True,
            "caller_provenance": caller_provenance,
        }
        if triage_verdict != "PASS":
            raise ControlPlaneError(
                "submit_request requires an independent triage review PASS; "
                f"got {triage_verdict!r}"
            )

        if plan_kind == "leaf" and safe_path is None:
            raise ControlPlaneError("leaf plans require a writable artifact_path")

        # Baseline capture for leaves only; parents coordinate without a
        # single product artifact.
        baseline_artifact_sha256 = None
        baseline_tree_sha256 = None
        if plan_kind == "leaf":
            baseline_content = _read_text(_artifact_target(repo_root, safe_path))
            baseline_artifact_sha256 = (
                _sha256_text(baseline_content)
                if baseline_content is not None
                else None
            )
            baseline_tree_sha256 = _product_revision(
                repo_root, excluded_root=self.store.root
            )

        spec = {
            "schema_version": "cosmo.task-spec.v1",
            "spec_id": spec_id,
            "task_id": task_id,
            "version": 1,
            "objective": objective,
            "requirements": [
                {"requirement_id": "objective", "text": objective}
            ],
            "acceptance_criteria": [
                (
                    f"the declared artifact exists at {safe_path}"
                    if safe_path
                    else "all dependent leaf plans complete"
                ),
                "all declared deterministic validators pass",
                "an independent review returns PASS",
            ],
            "constraints": constraints,
            "artifact_path": safe_path,
            "artifact_type": (
                "source_file"
                if safe_path and safe_path.endswith(".py")
                else ("coordination" if plan_kind == "parent" else "artifact")
            ),
            "validators": validator_list,
            "repo_root": str(repo_root),
            "baseline_artifact_sha256": baseline_artifact_sha256,
            "baseline_tree_sha256": baseline_tree_sha256,
            "created_at": now,
        }

        components = dict(
            priority_components
            or {
                "user_importance": 0.5,
                "urgency": 0.5,
                "deadline_pressure": 0.0,
                "dependency_unblocking": 0.0,
                "risk_reduction": 0.0,
                "age": 0.0,
                "estimated_effort": 0.3,
            }
        )
        if dependency_ids:
            if float(components.get("dependency_penalty") or 0) == 0 and float(
                components.get("dependency_unblocking") or 0
            ) == 0:
                components["dependency_penalty"] = 1.0
        scored = score_priority_components(components)
        initial_status = "ready"
        if dependency_ids:
            # Dependency-aware readiness: not ready until all deps complete.
            if not self._dependencies_satisfied(dependency_ids):
                initial_status = "blocked"

        task = {
            "schema_version": "cosmo.task.v1",
            "task_id": task_id,
            "status": initial_status,
            "linked_request_ids": [request_id],
            "current_spec_id": spec_id,
            "current_spec_version": 1,
            "plan_node_id": node_id,
            "priority_assessment_id": priority_id,
            "priority_score": scored["score"],
            "priority_components": scored["components"],
            "created_at": now,
            "updated_at": now,
        }

        plan = {
            "schema_version": "cosmo.plan-node.v1",
            "node_id": node_id,
            "task_id": task_id,
            "kind": plan_kind,
            "status": "approved" if plan_kind == "leaf" else "open",
            "objective": objective,
            "writable_artifact": safe_path if plan_kind == "leaf" else None,
            "validators": validator_list,
            "dependencies": dependency_ids,
            "parent_node_id": parent_node_id,
            "requirements": ["objective"],
            "root_acceptance_linkage": ["objective"],
            "child_node_ids": [],
            "created_at": now,
        }
        if plan_kind == "leaf" and safe_path is None:
            raise ControlPlaneError("leaf plan must declare exactly one writable artifact")

        plan_review = {
            "schema_version": "cosmo.plan-review.v1",
            "review_id": new_id("PLAN-REVIEW"),
            "node_id": node_id,
            "verdict": "PASS",
            "checks": (
                [
                    "exactly one writable artifact",
                    "local deterministic validators declared",
                ]
                if plan_kind == "leaf"
                else [
                    "parent coordinates children without a writable artifact",
                    "dependencies are explicit plan node identifiers",
                ]
            ),
            "operator_supplied": True,
            "caller_provenance": caller_provenance,
            "created_at": now,
        }

        relationship_record = {
            "schema_version": "cosmo.relationship.v1",
            "relationship_id": relationship_id,
            "request_id": request_id,
            "task_id": task_id,
            "type": str(relationship_body.get("type") or "NEW"),
            "target_task_id": relationship_body.get("target_task_id"),
            "rationale": relationship_body.get("rationale"),
            "created_at": now,
        }

        priority_record = {
            "schema_version": "cosmo.priority-assessment.v1",
            "assessment_id": priority_id,
            "task_id": task_id,
            "request_id": request_id,
            "score": scored["score"],
            "components": scored["components"],
            "breakdown": scored["breakdown"],
            "weights": scored["weights"],
            "created_at": now,
        }

        self.store.write("requests", request_id, request)
        self.store.write("tasks", task_id, task)
        self.store.write("task_specs", spec_id, spec)
        self.store.write("plan_nodes", node_id, plan)
        self.store.write("triage_proposals", proposal_id, triage_proposal_record)
        self.store.write("triage_reviews", review_id, triage_review_record)
        self.store.write("plan_reviews", plan_review["review_id"], plan_review)
        self.store.write("relationships", relationship_id, relationship_record)
        self.store.write("priority_assessments", priority_id, priority_record)

        if parent_node_id is not None:
            parent = self.store.read("plan_nodes", parent_node_id)
            children = list(parent.get("child_node_ids") or [])
            if node_id not in children:
                children.append(node_id)
            parent["child_node_ids"] = children
            self.store.replace("plan_nodes", parent_node_id, parent)

        self.store.append_event(
            "control_request_submitted",
            {
                "request_id": request_id,
                "task_id": task_id,
                "spec_id": spec_id,
                "node_id": node_id,
                "status": initial_status,
                "priority_score": scored["score"],
                "plan_kind": plan_kind,
            },
        )

        return {
            "request_id": request_id,
            "task_id": task_id,
            "spec_id": spec_id,
            "node_id": node_id,
            "relationship_id": relationship_id,
            "priority_assessment_id": priority_id,
            "triage_proposal_id": proposal_id,
            "triage_review_id": review_id,
            "plan_review_id": plan_review["review_id"],
            "status": initial_status,
            "priority_score": scored["score"],
            "priority_breakdown": scored["breakdown"],
            "artifact_path": safe_path,
            "plan_kind": plan_kind,
            "state_root": str(self.store.root),
        }

    def create_task(
        self,
        *,
        request_text: str,
        objective: str,
        artifact_path: str,
        repo_root: Path,
        validators: Iterable[str] = (),
        desired_outcome: str | None = None,
        explicit_constraints: Iterable[str] = (),
        priority_components: Mapping[str, Any] | None = None,
        operator_caller: CallerProvenance | None = None,
    ) -> dict[str, Any]:
        """Create a leaf task via the canonical control-plane request path.

        User-facing tools must use this (or :meth:`submit_request`) so the task
        receives plan/priority/lifecycle fields and participates in
        ``ready_queue``, leases, dispatch, and promotion. Prefer this over
        :meth:`AtomicHarness.submit`, which remains available for explicit
        legacy harness-only call sites.
        """
        return self.submit_request(
            request_text=request_text,
            objective=objective,
            artifact_path=artifact_path,
            repo_root=repo_root,
            validators=validators,
            desired_outcome=desired_outcome,
            explicit_constraints=explicit_constraints,
            priority_components=priority_components,
            plan_kind="leaf",
            operator_caller=operator_caller,
        )

    # ------------------------------------------------------------------
    # Request intake (no Task yet) and triage-backed task linkage
    # ------------------------------------------------------------------

    def ingest_request(
        self,
        *,
        request_text: str,
        desired_outcome: str | None = None,
        explicit_constraints: Iterable[str] = (),
        source_message_refs: Iterable[str] = (),
        requested_priority: str | None = None,
        summary: str | None = None,
        requester: str | None = None,
    ) -> dict[str, Any]:
        """Create an immutable Request without creating a Task.

        Broad chat outcomes enter here. TriageService proposes relationships
        and only after independent review PASS does a Task get created/linked.
        """
        if not request_text or not str(request_text).strip():
            raise ControlPlaneError("request_text must not be empty")
        constraints = [str(item) for item in explicit_constraints]
        refs = [str(item) for item in source_message_refs]
        priority = None
        if requested_priority is not None:
            priority = str(requested_priority).lower()
            if priority not in {"low", "normal", "high", "urgent"}:
                raise ControlPlaneError(
                    "requested_priority must be low|normal|high|urgent"
                )
        request_id = new_id("REQ")
        now = utc_now()
        body = {
            "original_text": str(request_text),
            "desired_outcome": desired_outcome,
            "explicit_constraints": constraints,
            "source_message_refs": refs,
            "requested_priority": priority,
            "summary": summary,
            "requester": requester,
        }
        request = {
            "schema_version": "cosmo.request.v1",
            "request_id": request_id,
            "original_text": body["original_text"],
            "desired_outcome": desired_outcome,
            "explicit_constraints": constraints,
            "source_message_refs": refs,
            "requested_priority": priority,
            "summary": summary,
            "requester": requester,
            "created_at": now,
            "immutable_sha256": canonical_hash(body),
        }
        self.store.write("requests", request_id, request)
        self.update_request_state(
            request_id,
            triage_status="pending",
            linked_task_id=None,
            linked_task_ids=[],
        )
        self.store.append_event(
            "request_ingested",
            {
                "request_id": request_id,
                "requested_priority": priority,
                "source_message_ref_count": len(refs),
            },
        )
        return {
            "request_id": request_id,
            "triage_status": "pending",
            "immutable_sha256": request["immutable_sha256"],
            "created_at": now,
            "state_root": str(self.store.root),
        }

    def create_task_from_triage(
        self,
        *,
        request_id: str,
        objective: str,
        repo_root: Path,
        plan_kind: str = "parent",
        artifact_path: str = "",
        validators: Iterable[str] = (),
        explicit_constraints: Iterable[str] = (),
        requirements: Iterable[Mapping[str, Any]] = (),
        shared_contracts: Iterable[Mapping[str, Any]] = (),
        root_acceptance_linkage: Iterable[str] = (),
        priority_components: Mapping[str, Any] | None = None,
        relationship: Mapping[str, Any] | None = None,
        triage_proposal_id: str | None = None,
        triage_review_id: str | None = None,
        plan_version_id: str | None = None,
        plan_review_id: str | None = None,
        title: str | None = None,
        summary: str | None = None,
        needs_planning: bool = False,
        parent_node_id: str | None = None,
        dependencies: Iterable[str] = (),
    ) -> dict[str, Any]:
        """Create Task/Spec/plan node linked to an existing immutable Request.

        Used by TriageService.apply for NEW relationships. Does not re-create
        the Request or invent a second triage authority.
        """
        request_id = _safe_record_identifier(request_id)
        self.store.read("requests", request_id)
        if not objective or not str(objective).strip():
            raise ControlPlaneError("objective must not be empty")
        if plan_kind not in {"leaf", "parent"}:
            raise ControlPlaneError(
                f"plan_kind must be 'leaf' or 'parent', got {plan_kind!r}"
            )
        repo_root = Path(repo_root).resolve()
        if not repo_root.is_dir():
            raise ControlPlaneError(f"repo_root is not a directory: {repo_root}")

        safe_path: str | None = None
        if plan_kind == "leaf":
            if not artifact_path:
                raise ControlPlaneError("leaf plans require artifact_path")
            safe_path = _safe_relative_path(artifact_path)

        dependency_ids = [
            _safe_record_identifier(str(item)) for item in dependencies
        ]
        for dep_id in dependency_ids:
            self.store.read("plan_nodes", dep_id)
        if parent_node_id is not None:
            parent_node_id = _safe_record_identifier(parent_node_id)
            parent = self.store.read("plan_nodes", parent_node_id)
            if parent.get("kind") != "parent":
                raise ControlPlaneError(
                    f"parent_node_id must reference a parent plan node: "
                    f"{parent_node_id}"
                )

        constraints = [str(item) for item in explicit_constraints]
        contracts: list[dict[str, Any]] = []
        for raw_contract in shared_contracts:
            if not isinstance(raw_contract, Mapping) or "artifact" not in raw_contract:
                raise ControlPlaneError(
                    "shared_contracts entries must be mappings with artifact"
                )
            contract = dict(raw_contract)
            contract["artifact"] = _safe_relative_path(str(contract["artifact"]))
            contracts.append(contract)
        validator_list = list(dict.fromkeys(validators)) or (
            ["non_empty", "utf8"] if plan_kind == "leaf" else []
        )
        req_list = [dict(item) for item in requirements]
        if not req_list:
            req_list = [
                {
                    "requirement_id": "REQ-1",
                    "text": objective,
                    "source": "request",
                    "inferred": False,
                }
            ]
        acceptance_linkage = [str(item) for item in root_acceptance_linkage if str(item)]
        if not acceptance_linkage:
            acceptance_linkage = [
                str(item.get("requirement_id"))
                for item in req_list
                if item.get("requirement_id")
            ]

        task_id = new_id("TASK")
        spec_id = new_id("SPEC")
        node_id = new_id("NODE")
        relationship_id = new_id("REL")
        priority_id = new_id("PRIO")
        now = utc_now()

        baseline_artifact_sha256 = None
        baseline_tree_sha256 = None
        if plan_kind == "leaf" and safe_path is not None:
            baseline_content = _read_text(_artifact_target(repo_root, safe_path))
            baseline_artifact_sha256 = (
                _sha256_text(baseline_content)
                if baseline_content is not None
                else None
            )
            baseline_tree_sha256 = _product_revision(
                repo_root, excluded_root=self.store.root
            )

        components = dict(
            priority_components
            or {
                "user_importance": 0.5,
                "urgency": 0.5,
                "age": 0.0,
                "estimated_effort": 0.3,
            }
        )
        scored = score_priority_components(components)
        initial_status = "planning" if needs_planning or plan_kind == "parent" else "ready"
        if dependency_ids and not self._dependencies_satisfied(dependency_ids):
            initial_status = "blocked"

        relationship_body = dict(
            relationship
            or {"type": "NEW", "target_task_id": None, "rationale": "triage NEW"}
        )
        rel_type = str(relationship_body.get("type") or "NEW").upper()
        if rel_type not in RELATIONSHIP_TYPES:
            raise ControlPlaneError(f"unsupported relationship type: {rel_type!r}")

        spec = {
            "schema_version": "cosmo.task-spec.v1",
            "spec_id": spec_id,
            "task_id": task_id,
            "version": 1,
            "title": title or objective,
            "summary": summary or objective,
            "objective": objective,
            "requirements": req_list,
            "acceptance_criteria": [
                (
                    f"the declared artifact exists at {safe_path}"
                    if safe_path
                    else "root acceptance: all active requirements covered by approved children"
                ),
                "all declared deterministic validators pass",
                "an independent review returns PASS",
            ],
            "constraints": constraints,
            "shared_contracts": contracts,
            "artifact_path": safe_path,
            "artifact_type": (
                "source_file"
                if safe_path and str(safe_path).endswith(".py")
                else ("coordination" if plan_kind == "parent" else "artifact")
            ),
            # Parent validators are composition gates and must remain on the
            # plan node for IntegrationService to execute them.
            "validators": validator_list,
            "repo_root": str(repo_root),
            "baseline_artifact_sha256": baseline_artifact_sha256,
            "baseline_tree_sha256": baseline_tree_sha256,
            "needs_planning": bool(needs_planning or plan_kind == "parent"),
            "root_acceptance_linkage": acceptance_linkage,
            "created_at": now,
            "triage_proposal_id": triage_proposal_id,
            "triage_review_id": triage_review_id,
            "plan_version_id": plan_version_id,
            "plan_review_id": plan_review_id,
        }
        task = {
            "schema_version": "cosmo.task.v1",
            "task_id": task_id,
            "status": initial_status,
            "linked_request_ids": [request_id],
            "current_spec_id": spec_id,
            "current_spec_version": 1,
            "plan_node_id": node_id,
            "priority_assessment_id": priority_id,
            "priority_score": scored["score"],
            "priority_components": scored["components"],
            "title": title or objective,
            "needs_planning": bool(needs_planning or plan_kind == "parent"),
            "created_at": now,
            "updated_at": now,
        }
        plan = {
            "schema_version": "cosmo.plan-node.v1",
            "node_id": node_id,
            "task_id": task_id,
            "kind": plan_kind,
            "status": (
                "draft"
                if needs_planning or plan_kind == "parent"
                else "approved"
            ),
            "objective": objective,
            "writable_artifact": safe_path if plan_kind == "leaf" else None,
            "validators": validator_list,
            "dependencies": dependency_ids,
            "parent_node_id": parent_node_id,
            "child_node_ids": [],
            "requirements": [item.get("requirement_id") for item in req_list],
            "root_acceptance_linkage": acceptance_linkage,
            "shared_contracts": contracts,
            "plan_version_id": plan_version_id,
            "plan_review_id": plan_review_id,
            "created_at": now,
        }
        relationship_record = {
            "schema_version": "cosmo.relationship.v1",
            "relationship_id": relationship_id,
            "request_id": request_id,
            "task_id": task_id,
            "type": rel_type,
            "target_task_id": relationship_body.get("target_task_id"),
            "rationale": relationship_body.get("rationale"),
            "triage_proposal_id": triage_proposal_id,
            "created_at": now,
        }
        priority_record = {
            "schema_version": "cosmo.priority-assessment.v1",
            "assessment_id": priority_id,
            "task_id": task_id,
            "request_id": request_id,
            "score": scored["score"],
            "components": scored["components"],
            "breakdown": scored["breakdown"],
            "weights": scored["weights"],
            "created_at": now,
        }

        self.store.write("tasks", task_id, task)
        self.store.write("task_specs", spec_id, spec)
        self.store.write("plan_nodes", node_id, plan)
        self.store.write("relationships", relationship_id, relationship_record)
        self.store.write("priority_assessments", priority_id, priority_record)

        # Link request projection without rewriting the immutable Request.
        current_request = self.get_request(request_id)
        linked = list(current_request.get("linked_task_ids") or [])
        if task_id not in linked:
            linked.append(task_id)
        self.update_request_state(
            request_id,
            linked_task_id=task_id,
            linked_task_ids=linked,
        )

        if parent_node_id is not None:
            parent = self.store.read("plan_nodes", parent_node_id)
            children = list(parent.get("child_node_ids") or [])
            if node_id not in children:
                children.append(node_id)
            parent["child_node_ids"] = children
            self.store.replace("plan_nodes", parent_node_id, parent)

        self.store.append_event(
            "task_created_from_triage",
            {
                "request_id": request_id,
                "task_id": task_id,
                "spec_id": spec_id,
                "node_id": node_id,
                "plan_kind": plan_kind,
                "status": initial_status,
                "triage_proposal_id": triage_proposal_id,
                "needs_planning": bool(needs_planning or plan_kind == "parent"),
            },
        )
        return {
            "request_id": request_id,
            "task_id": task_id,
            "spec_id": spec_id,
            "node_id": node_id,
            "relationship_id": relationship_id,
            "priority_assessment_id": priority_id,
            "triage_proposal_id": triage_proposal_id,
            "triage_review_id": triage_review_id,
            "status": initial_status,
            "priority_score": scored["score"],
            "priority_breakdown": scored["breakdown"],
            "artifact_path": safe_path,
            "plan_kind": plan_kind,
            "needs_planning": bool(needs_planning or plan_kind == "parent"),
            "state_root": str(self.store.root),
        }

    def link_request_to_task(
        self,
        *,
        request_id: str,
        task_id: str,
        relationship: Mapping[str, Any],
        triage_proposal_id: str | None = None,
        triage_review_id: str | None = None,
        priority_components: Mapping[str, Any] | None = None,
        added_requirements: Iterable[str] = (),
        inferred_requirements: Iterable[str] = (),
        modified_requirements: Iterable[str] = (),
        constraints: Iterable[str] = (),
        create_spec_version: bool = True,
        mark_stale: bool = False,
        summary: str = "",
        title: str = "",
    ) -> dict[str, Any]:
        """Link an immutable Request to an existing Task; optionally version Spec."""
        request_id = _safe_record_identifier(request_id)
        task_id = _safe_record_identifier(task_id)
        self.store.read("requests", request_id)
        task = self.store.read("tasks", task_id)
        rel_type = str(relationship.get("type") or "").upper()
        if rel_type not in RELATIONSHIP_TYPES or rel_type == "NEW":
            raise ControlPlaneError(
                f"link_request_to_task requires a non-NEW relationship, got {rel_type!r}"
            )
        if str(relationship.get("target_task_id") or "") not in {"", task_id}:
            # Target must be this task.
            if relationship.get("target_task_id") != task_id:
                raise ControlPlaneError(
                    "relationship.target_task_id must match task_id"
                )

        relationship_id = new_id("REL")
        now = utc_now()
        relationship_record = {
            "schema_version": "cosmo.relationship.v1",
            "relationship_id": relationship_id,
            "request_id": request_id,
            "task_id": task_id,
            "type": rel_type,
            "target_task_id": task_id,
            "rationale": relationship.get("rationale"),
            "triage_proposal_id": triage_proposal_id,
            "created_at": now,
        }
        self.store.write("relationships", relationship_id, relationship_record)

        linked = list(task.get("linked_request_ids") or [])
        if request_id not in linked:
            linked.append(request_id)
        task["linked_request_ids"] = linked
        task["updated_at"] = now

        new_spec_id = task.get("current_spec_id")
        new_version = int(task.get("current_spec_version") or 1)
        if create_spec_version:
            old_spec = self.store.read("task_specs", str(task["current_spec_id"]))
            new_version = int(old_spec.get("version") or 1) + 1
            new_spec_id = new_id("SPEC")
            reqs = list(old_spec.get("requirements") or [])
            next_idx = len(reqs) + 1
            for text in added_requirements:
                reqs.append(
                    {
                        "requirement_id": f"REQ-{next_idx}",
                        "text": str(text),
                        "source": "request",
                        "inferred": False,
                    }
                )
                next_idx += 1
            for text in inferred_requirements:
                reqs.append(
                    {
                        "requirement_id": f"REQ-{next_idx}",
                        "text": str(text),
                        "source": "triage_inferred",
                        "inferred": True,
                    }
                )
                next_idx += 1
            # modified_requirements recorded as notes; full semantic merge is
            # operator/planning work, not silent rewrite of prior IDs.
            merged_constraints = list(old_spec.get("constraints") or [])
            for item in constraints:
                if item not in merged_constraints:
                    merged_constraints.append(str(item))
            spec = {
                **old_spec,
                "spec_id": new_spec_id,
                "version": new_version,
                "requirements": reqs,
                "constraints": merged_constraints,
                "title": title or old_spec.get("title"),
                "summary": summary or old_spec.get("summary"),
                "supersedes_spec_id": old_spec.get("spec_id"),
                "modified_requirement_notes": list(modified_requirements),
                "linked_request_ids": linked,
                "triage_proposal_id": triage_proposal_id,
                "triage_review_id": triage_review_id,
                "created_at": now,
            }
            # Drop immutable identity fields that must change.
            self.store.write("task_specs", new_spec_id, spec)
            task["current_spec_id"] = new_spec_id
            task["current_spec_version"] = new_version
            # Spec change invalidates in-flight work when requested.
            if mark_stale and task.get("status") in {
                "ready",
                "queued",
                "running",
                "executing",
                "planning",
                "integrating",
            }:
                task["status"] = "stale"
                task["status_reason"] = (
                    f"task specification v{new_version} from {rel_type} triage"
                )

        if priority_components is not None:
            scored = score_priority_components(priority_components)
            priority_id = new_id("PRIO")
            priority_record = {
                "schema_version": "cosmo.priority-assessment.v1",
                "assessment_id": priority_id,
                "task_id": task_id,
                "request_id": request_id,
                "score": scored["score"],
                "components": scored["components"],
                "breakdown": scored["breakdown"],
                "weights": scored["weights"],
                "created_at": now,
            }
            self.store.write("priority_assessments", priority_id, priority_record)
            task["priority_assessment_id"] = priority_id
            task["priority_score"] = scored["score"]
            task["priority_components"] = scored["components"]

        self.store.replace("tasks", task_id, task)
        if mark_stale:
            self._mark_related_work_stale(
                task_id,
                reason=(
                    f"TaskSpec superseded by {rel_type} triage request "
                    f"{request_id}"
                ),
            )
        current_request = self.get_request(request_id)
        req_tasks = list(current_request.get("linked_task_ids") or [])
        if task_id not in req_tasks:
            req_tasks.append(task_id)
        self.update_request_state(
            request_id,
            linked_task_id=task_id,
            linked_task_ids=req_tasks,
        )

        self.store.append_event(
            "request_linked_to_task",
            {
                "request_id": request_id,
                "task_id": task_id,
                "relationship_id": relationship_id,
                "relationship_type": rel_type,
                "spec_id": new_spec_id,
                "spec_version": new_version,
                "mark_stale": mark_stale,
                "triage_proposal_id": triage_proposal_id,
            },
        )
        return {
            "request_id": request_id,
            "task_id": task_id,
            "spec_id": new_spec_id,
            "spec_version": new_version,
            "relationship_id": relationship_id,
            "status": task.get("status"),
            "priority_score": task.get("priority_score"),
            "triage_proposal_id": triage_proposal_id,
            "triage_review_id": triage_review_id,
        }

    def _mark_related_work_stale(self, task_id: str, *, reason: str) -> None:
        """Invalidate derived plan/attempt projections after a spec change."""
        for node in self.store.list_records("plan_nodes"):
            if str(node.get("task_id") or "") != task_id:
                continue
            if node.get("status") in {"completed", "canceled"}:
                continue
            node["status"] = "stale"
            node["stale_reason"] = reason
            node["updated_at"] = utc_now()
            self.store.replace("plan_nodes", str(node["node_id"]), node)

        terminal = {
            "promoted",
            "failed",
            "review_failed",
            "worker_failed",
            "worker_protocol_failed",
            "invalid_write_scope",
            "no_op",
        }
        for base in self.store.list_records("attempts"):
            if str(base.get("task_id") or "") != task_id:
                continue
            attempt_id = str(base["attempt_id"])
            path = self.store.path("attempt_states", attempt_id)
            state = self.store.read("attempt_states", attempt_id) if path.is_file() else dict(base)
            if state.get("status") not in terminal:
                state["status"] = "stale"
            state["stale"] = True
            state["stale_reason"] = reason
            state["updated_at"] = utc_now()
            if path.is_file():
                self.store.replace("attempt_states", attempt_id, state)

    def record_chat_tool_event(
        self,
        *,
        tool: str,
        ok: bool,
        operation_id: str | None = None,
        error: str | None = None,
        task_id: str | None = None,
        attempt_id: str | None = None,
        caller_provenance: Mapping[str, Any] | None = None,
        repo_capability_id: str | None = None,
    ) -> dict[str, Any]:
        """Append a durable chat tool-loop event without prompt/transcript bodies.

        Transcript content stays ephemeral on the chat session. This records
        only operation metadata for operator observability.
        """
        event_record = {
            "schema_version": "cosmo.chat-tool-event.v2",
            "event_kind": "chat_tool_call",
            "operation_id": operation_id,
            "tool": str(tool),
            "ok": bool(ok),
            "error": error,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "caller_provenance": dict(caller_provenance or {}) or None,
            "repo_capability_id": repo_capability_id,
            "created_at": utc_now(),
        }
        # Stable id for the optional chat_events projection record.
        chat_event_id = new_id("CHAT")
        event_record["chat_event_id"] = chat_event_id
        try:
            self.store.write("chat_events", chat_event_id, event_record)
        except AtomicHarnessError:
            # Immutable collision is extremely unlikely; still journal the event.
            pass
        self.store.append_event(
            "chat_tool_call",
            {
                "chat_event_id": chat_event_id,
                "operation_id": operation_id,
                "tool": str(tool),
                "ok": bool(ok),
                "error": error,
                "task_id": task_id,
                "attempt_id": attempt_id,
                "caller_provenance": dict(caller_provenance or {}) or None,
                "repo_capability_id": repo_capability_id,
            },
        )
        return event_record

    # ------------------------------------------------------------------
    # Record accessors
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> dict[str, Any]:
        return self.store.read("tasks", _safe_record_identifier(task_id))

    def list_tasks(self) -> list[dict[str, Any]]:
        return self.store.list_records("tasks")

    def get_request(self, request_id: str) -> dict[str, Any]:
        request_id = _safe_record_identifier(request_id)
        request = self.store.read("requests", request_id)
        state_path = self.store.path("request_states", request_id)
        if state_path.is_file():
            return {**request, **self.store.read("request_states", request_id)}
        return request

    def list_requests(self) -> list[dict[str, Any]]:
        return [self.get_request(str(item["request_id"])) for item in self.store.list_records("requests")]

    def update_request_state(self, request_id: str, **updates: Any) -> dict[str, Any]:
        """Update the derived request lifecycle projection, never the Request.

        The original Request record is content-addressed and immutable. Triage
        status, proposal/review ids, and linked-task projections belong in a
        separate mutable state record so audit readers can distinguish the
        user's original words from later workflow state.
        """
        request_id = _safe_record_identifier(request_id)
        # Verify the immutable record exists before creating a projection.
        self.store.read("requests", request_id)
        state_path = self.store.path("request_states", request_id)
        if state_path.is_file():
            state = self.store.read("request_states", request_id)
        else:
            state = {
                "schema_version": "cosmo.request-state.v1",
                "request_id": request_id,
                "created_at": utc_now(),
            }
        state.update(updates)
        state["request_id"] = request_id
        state["updated_at"] = utc_now()
        if state_path.is_file():
            self.store.replace("request_states", request_id, state)
        else:
            self.store.write("request_states", request_id, state)
        return {**self.store.read("requests", request_id), **state}

    def get_task_spec(self, spec_id: str) -> dict[str, Any]:
        return self.store.read("task_specs", _safe_record_identifier(spec_id))

    def list_task_specs(self) -> list[dict[str, Any]]:
        return self.store.list_records("task_specs")

    def get_plan(self, node_id: str) -> dict[str, Any]:
        return self.store.read("plan_nodes", _safe_record_identifier(node_id))

    def list_plans(self) -> list[dict[str, Any]]:
        return self.store.list_records("plan_nodes")

    def get_attempt(self, attempt_id: str) -> dict[str, Any]:
        attempt_id = _safe_record_identifier(attempt_id)
        base = self.store.read("attempts", attempt_id)
        state_path = self.store.path("attempt_states", attempt_id)
        if not state_path.is_file():
            return base
        state = self.store.read("attempt_states", attempt_id)
        return {**base, **state}

    def list_attempts(self) -> list[dict[str, Any]]:
        return self.store.list_records("attempts")

    def get_review(self, review_id: str) -> dict[str, Any]:
        review_id = _safe_record_identifier(review_id)
        # Independent attempt reviews and plan/triage reviews share accessors.
        for kind in ("reviews", "plan_reviews", "triage_reviews"):
            path = self.store.path(kind, review_id)
            if path.is_file():
                return self.store.read(kind, review_id)
        raise ControlPlaneError(f"record not found: review/{review_id}")

    def list_reviews(self) -> list[dict[str, Any]]:
        return (
            self.store.list_records("reviews")
            + self.store.list_records("plan_reviews")
            + self.store.list_records("triage_reviews")
        )

    def get_ledger_record(self, event_id: str) -> dict[str, Any]:
        return self.store.read("ledger", _safe_record_identifier(event_id))

    def list_ledger_records(self) -> list[dict[str, Any]]:
        return self.store.list_records("ledger")

    def get_priority_assessment(self, assessment_id: str) -> dict[str, Any]:
        return self.store.read(
            "priority_assessments", _safe_record_identifier(assessment_id)
        )

    def get_lease(self, lease_id: str) -> dict[str, Any]:
        return self.store.read("leases", _safe_record_identifier(lease_id))

    def list_leases(self) -> list[dict[str, Any]]:
        return self.store.list_records("leases")

    def get_convergence_assessment(self, assessment_id: str) -> dict[str, Any]:
        return self.store.read(
            "convergence_assessments", _safe_record_identifier(assessment_id)
        )

    def get_ledger_projection(self, projection_id: str) -> dict[str, Any]:
        return self.store.read(
            "ledger_projections", _safe_record_identifier(projection_id)
        )

    def inspect_lifecycle(
        self, task_id: str | None = None, *, now: str | None = None
    ) -> dict[str, Any]:
        """Project task or control-plane lifecycle state without mutating product.

        With ``task_id``, returns a single-task lifecycle view: task, spec, plan,
        readiness gates, active lease, attempts, priority assessment, and
        position on the aged ready queue. Without ``task_id``, returns the
        compact control-plane snapshot from :meth:`to_json_dict`.
        """
        if task_id is None:
            return self.to_json_dict()
        task_id = _safe_record_identifier(task_id)
        stamp = now or utc_now()
        task = self.get_task(task_id)
        spec = None
        plan = None
        if task.get("current_spec_id"):
            try:
                spec = self.get_task_spec(str(task["current_spec_id"]))
            except (ControlPlaneError, AtomicHarnessError):
                spec = None
        if task.get("plan_node_id"):
            try:
                plan = self.get_plan(str(task["plan_node_id"]))
            except (ControlPlaneError, AtomicHarnessError):
                plan = None
        priority = None
        if task.get("priority_assessment_id"):
            try:
                priority = self.get_priority_assessment(
                    str(task["priority_assessment_id"])
                )
            except (ControlPlaneError, AtomicHarnessError):
                priority = None
        attempts: list[dict[str, Any]] = []
        for item in self.list_attempts():
            if item.get("task_id") != task_id:
                continue
            try:
                attempts.append(self.get_attempt(str(item["attempt_id"])))
            except AtomicHarnessError:
                attempts.append(item)
        attempts.sort(key=lambda item: str(item.get("created_at") or ""))
        active_lease = self._active_lease_for_task(task_id, now=stamp)
        dispatchable = self._is_dispatchable(task, now=stamp)
        queue = self.ready_queue(now=stamp)
        queue_ids = [str(item["task_id"]) for item in queue]
        try:
            queue_position = queue_ids.index(task_id)
        except ValueError:
            queue_position = None
        linked_requests: list[dict[str, Any]] = []
        for req_id in task.get("linked_request_ids") or []:
            try:
                linked_requests.append(self.get_request(str(req_id)))
            except (ControlPlaneError, AtomicHarnessError):
                continue
        return {
            "schema_version": "cosmo.lifecycle-inspection.v1",
            "inspected_at": stamp,
            "task": task,
            "spec": spec,
            "plan": plan,
            "priority_assessment": priority,
            "linked_requests": linked_requests,
            "active_lease": active_lease,
            "dispatchable": dispatchable,
            "dependencies_satisfied": (
                self._dependencies_satisfied(
                    [str(item) for item in (plan.get("dependencies") or [])]
                )
                if plan is not None
                else None
            ),
            "parent_permits_dispatch": (
                self._parent_permits_dispatch(plan) if plan is not None else None
            ),
            "spec_is_current": self._spec_is_current(task),
            "ready_queue_position": queue_position,
            "ready_queue_size": len(queue),
            "attempts": attempts,
            "attempt_count": len(attempts),
        }

    # ------------------------------------------------------------------
    # Task transitions
    # ------------------------------------------------------------------

    def transition_task(
        self,
        task_id: str,
        new_status: str,
        *,
        reason: str | None = None,
        actor: str = "operator",
    ) -> dict[str, Any]:
        """Apply a legal task status transition with an append-only event."""
        task_id = _safe_record_identifier(task_id)
        requested_status = str(new_status)
        # Spec alias: executing -> running (stored as running for queue logic).
        if requested_status == "executing":
            new_status = "running"
        else:
            new_status = requested_status
        if new_status not in TASK_STATUSES - {"executing"}:
            raise ControlPlaneError(f"unknown task status: {requested_status!r}")
        task = self.store.read("tasks", task_id)
        current = str(task.get("status"))
        if current == "executing":
            current = "running"
        if current not in TASK_STATUSES:
            raise ControlPlaneError(f"task has unknown status: {current!r}")
        allowed = LEGAL_TRANSITIONS.get(current, frozenset())
        # Accept either the stored name or the executing alias on the edge.
        if new_status not in allowed and requested_status not in allowed:
            raise ControlPlaneError(
                f"illegal task transition {current!r} -> {requested_status!r}"
            )
        if new_status == "ready":
            plan = self.store.read("plan_nodes", str(task["plan_node_id"]))
            deps = [str(item) for item in (plan.get("dependencies") or [])]
            if deps and not self._dependencies_satisfied(deps):
                raise ControlPlaneError(
                    "cannot mark task ready while plan dependencies are unmet"
                )
            if not self._parent_permits_dispatch(plan):
                raise ControlPlaneError(
                    "cannot mark task ready while parent plan is not approved"
                )
            if not self._spec_is_current(task):
                raise ControlPlaneError(
                    "cannot mark task ready against a non-current specification"
                )
        previous = current
        task["status"] = new_status
        task["updated_at"] = utc_now()
        if reason:
            task["status_reason"] = reason
        self.store.replace("tasks", task_id, task)
        self.store.append_event(
            "task_status_transition",
            {
                "task_id": task_id,
                "from_status": previous,
                "to_status": new_status,
                "reason": reason,
                "actor": actor,
            },
        )
        return task

    # ------------------------------------------------------------------
    # Priority and ready queue
    # ------------------------------------------------------------------

    def reassess_priority(
        self,
        task_id: str,
        priority_components: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Append a new priority assessment and update the task projection."""
        task_id = _safe_record_identifier(task_id)
        task = self.store.read("tasks", task_id)
        scored = score_priority_components(priority_components)
        assessment_id = new_id("PRIO")
        record = {
            "schema_version": "cosmo.priority-assessment.v1",
            "assessment_id": assessment_id,
            "task_id": task_id,
            "request_id": (task.get("linked_request_ids") or [None])[0],
            "score": scored["score"],
            "components": scored["components"],
            "breakdown": scored["breakdown"],
            "weights": scored["weights"],
            "created_at": utc_now(),
        }
        self.store.write("priority_assessments", assessment_id, record)
        task["priority_assessment_id"] = assessment_id
        task["priority_score"] = scored["score"]
        task["priority_components"] = scored["components"]
        task["updated_at"] = utc_now()
        self.store.replace("tasks", task_id, task)
        self.store.append_event(
            "priority_reassessed",
            {
                "task_id": task_id,
                "assessment_id": assessment_id,
                "score": scored["score"],
            },
        )
        return record

    def ready_queue(
        self, *, now: str | None = None, include_parents: bool = False
    ) -> list[dict[str, Any]]:
        """Return ready work ordered by effective priority (with aging).

        Ordering is fully deterministic: higher effective score first, then
        older ``created_at``, then lexicographic ``task_id`` as a final
        tie-break. A task is available only when:

        - status is ready (not blocked/canceled/completed/stale)
        - plan dependencies are accepted/completed
        - no active foreign lease
        - parent plan permits dispatch (approved/open parent)
        - task specification is current
        - for leaves: plan is leaf with a writable artifact
        """
        self.expire_stale_leases(now=now)
        stamp = now or utc_now()
        ready: list[dict[str, Any]] = []
        for task in self.list_tasks():
            if task.get("status") != "ready":
                continue
            if not self._is_dispatchable(task, now=stamp, include_parents=include_parents):
                continue
            enriched = dict(task)
            components = task.get("priority_components")
            if not components and task.get("priority_assessment_id"):
                try:
                    assessment = self.get_priority_assessment(
                        str(task["priority_assessment_id"])
                    )
                    components = assessment.get("components") or {}
                except (ControlPlaneError, AtomicHarnessError):
                    components = {}
            if components:
                scored = effective_priority_score(
                    components,
                    created_at=str(task.get("created_at") or ""),
                    now=stamp,
                )
                enriched["priority_score"] = scored["score"]
                enriched["priority_score_aged"] = scored["score"]
                enriched["age_component"] = scored.get("age_component")
                enriched["priority_breakdown"] = scored.get("breakdown")
            ready.append(enriched)
        ready.sort(
            key=lambda item: (
                -float(item.get("priority_score") or 0.0),
                str(item.get("created_at") or ""),
                str(item.get("task_id") or ""),
            )
        )
        return ready

    def _is_dispatchable(
        self,
        task: Mapping[str, Any],
        *,
        now: str | None = None,
        include_parents: bool = False,
    ) -> bool:
        """Readiness gates: deps, lease, parent permit, current spec, kind."""
        try:
            plan = self.store.read("plan_nodes", str(task["plan_node_id"]))
        except AtomicHarnessError:
            return False
        kind = plan.get("kind")
        if kind == "leaf":
            if not plan.get("writable_artifact"):
                return False
        elif kind == "parent":
            if not include_parents:
                return False
        else:
            return False
        if plan.get("status") in {
            "stale",
            "blocked",
            "canceled",
            "node_approved",
            "complete",
            "completed",
        }:
            return False
        deps = [str(item) for item in (plan.get("dependencies") or [])]
        if deps and not self._dependencies_satisfied(deps):
            return False
        if not self._parent_permits_dispatch(plan):
            return False
        if not self._spec_is_current(task):
            return False
        active = self._active_lease_for_task(str(task["task_id"]), now=now)
        if active is not None:
            return False
        return True

    def _parent_permits_dispatch(self, plan: Mapping[str, Any]) -> bool:
        parent_id = plan.get("parent_node_id")
        if not parent_id:
            return True
        try:
            parent = self.store.read("plan_nodes", str(parent_id))
        except AtomicHarnessError:
            return False
        # A draft/planning parent is not an approved composition boundary.
        # Plan application moves the governing node to ``open``; integration
        # promotion moves it to ``approved``.  Only those post-review states
        # may permit child dispatch.
        if parent.get("status") in {
            None,
            "draft",
            "planning",
            "triage_pending",
            "triage_review",
            "canceled",
            "stale",
            "blocked",
        }:
            return False
        if parent.get("dispatch_blocked"):
            return False
        return True

    def _spec_is_current(self, task: Mapping[str, Any]) -> bool:
        spec_id = task.get("current_spec_id")
        if not spec_id:
            return False
        try:
            spec = self.store.read("task_specs", str(spec_id))
        except AtomicHarnessError:
            return False
        if spec.get("superseded_by"):
            return False
        if int(spec.get("version") or 0) != int(task.get("current_spec_version") or 0):
            return False
        return True

    # ------------------------------------------------------------------
    # Exclusive leases
    # ------------------------------------------------------------------

    def acquire_lease(
        self,
        task_id: str,
        *,
        owner: str,
        ttl_seconds: int = DEFAULT_LEASE_SECONDS,
        now: str | None = None,
    ) -> dict[str, Any]:
        """Acquire an exclusive lease on a task for ``owner``."""
        task_id = _safe_record_identifier(task_id)
        if not owner or not str(owner).strip():
            raise ControlPlaneError("lease owner must be a non-empty string")
        if ttl_seconds <= 0:
            raise ControlPlaneError("lease ttl_seconds must be positive")
        renew_lease_id: str | None = None
        lease: dict[str, Any] | None = None
        # Lease acquisition is a filesystem compare-and-create operation.  A
        # per-state-root lock closes the read/read/write race between worker
        # processes that would otherwise each observe no active lease.
        with _exclusive_lock(self.store.root / "locks" / "leases.lock"):
            self.expire_stale_leases(now=now)
            task = self.store.read("tasks", task_id)
            if task.get("status") not in {"ready", "queued", "running"}:
                raise ControlPlaneError(
                    f"cannot lease task in status {task.get('status')!r}"
                )
            existing = self._active_lease_for_task(task_id, now=now)
            if existing is not None:
                if existing.get("owner") == owner:
                    renew_lease_id = str(existing["lease_id"])
                else:
                    raise ControlPlaneError(
                        f"task {task_id} already leased by {existing.get('owner')!r}"
                    )
            else:
                stamp = now or utc_now()
                expires = (
                    _parse_utc(stamp) + timedelta(seconds=int(ttl_seconds))
                ).isoformat()
                lease_id = new_id("LEASE")
                lease = {
                    "schema_version": "cosmo.lease.v1",
                    "lease_id": lease_id,
                    "task_id": task_id,
                    "node_id": task.get("plan_node_id"),
                    "owner": str(owner),
                    "status": "active",
                    "acquired_at": stamp,
                    "renewed_at": stamp,
                    "expires_at": expires,
                    "ttl_seconds": int(ttl_seconds),
                    "released_at": None,
                }
                self.store.write("leases", lease_id, lease)
        if renew_lease_id is not None:
            return self.renew_lease(
                renew_lease_id,
                owner=owner,
                ttl_seconds=ttl_seconds,
                now=now,
            )
        if lease is None:
            raise ControlPlaneError("lease acquisition produced no lease")
        self.store.append_event(
            "lease_acquired",
            {
                "lease_id": lease["lease_id"],
                "task_id": task_id,
                "owner": owner,
                "expires_at": lease["expires_at"],
            },
        )
        return lease

    def renew_lease(
        self,
        lease_id: str,
        *,
        owner: str,
        ttl_seconds: int | None = None,
        now: str | None = None,
    ) -> dict[str, Any]:
        """Renew an active lease owned by ``owner``."""
        lease_id = _safe_record_identifier(lease_id)
        self.expire_stale_leases(now=now)
        lease = self.store.read("leases", lease_id)
        if lease.get("status") != "active":
            raise ControlPlaneError(f"lease is not active: {lease_id}")
        if lease.get("owner") != owner:
            raise ControlPlaneError(
                f"lease {lease_id} is owned by {lease.get('owner')!r}"
            )
        stamp = now or utc_now()
        if _parse_utc(str(lease["expires_at"])) <= _parse_utc(stamp):
            lease["status"] = "expired"
            self.store.replace("leases", lease_id, lease)
            raise ControlPlaneError(f"lease has expired: {lease_id}")
        ttl = int(ttl_seconds if ttl_seconds is not None else lease["ttl_seconds"])
        if ttl <= 0:
            raise ControlPlaneError("lease ttl_seconds must be positive")
        expires = (_parse_utc(stamp) + timedelta(seconds=ttl)).isoformat()
        lease["renewed_at"] = stamp
        lease["expires_at"] = expires
        lease["ttl_seconds"] = ttl
        self.store.replace("leases", lease_id, lease)
        self.store.append_event(
            "lease_renewed",
            {
                "lease_id": lease_id,
                "task_id": lease.get("task_id"),
                "owner": owner,
                "expires_at": expires,
            },
        )
        return lease

    def release_lease(
        self,
        lease_id: str,
        *,
        owner: str,
        now: str | None = None,
    ) -> dict[str, Any]:
        """Release an active lease owned by ``owner``."""
        lease_id = _safe_record_identifier(lease_id)
        lease = self.store.read("leases", lease_id)
        if lease.get("status") != "active":
            raise ControlPlaneError(f"lease is not active: {lease_id}")
        if lease.get("owner") != owner:
            raise ControlPlaneError(
                f"lease {lease_id} is owned by {lease.get('owner')!r}"
            )
        stamp = now or utc_now()
        lease["status"] = "released"
        lease["released_at"] = stamp
        self.store.replace("leases", lease_id, lease)
        self.store.append_event(
            "lease_released",
            {
                "lease_id": lease_id,
                "task_id": lease.get("task_id"),
                "owner": owner,
            },
        )
        return lease

    def expire_stale_leases(self, *, now: str | None = None) -> list[dict[str, Any]]:
        """Mark active leases past their expiry as expired (append-only event)."""
        stamp = now or utc_now()
        current = _parse_utc(stamp)
        expired: list[dict[str, Any]] = []
        for lease in self.list_leases():
            if lease.get("status") != "active":
                continue
            if _parse_utc(str(lease["expires_at"])) > current:
                continue
            lease_id = str(lease["lease_id"])
            lease["status"] = "expired"
            lease["expired_at"] = stamp
            self.store.replace("leases", lease_id, lease)
            self.store.append_event(
                "lease_expired",
                {
                    "lease_id": lease_id,
                    "task_id": lease.get("task_id"),
                    "owner": lease.get("owner"),
                },
            )
            expired.append(lease)
        return expired

    def _active_lease_for_task(
        self, task_id: str, *, now: str | None = None
    ) -> dict[str, Any] | None:
        stamp = _parse_utc(now or utc_now())
        active: dict[str, Any] | None = None
        for lease in self.list_leases():
            if lease.get("task_id") != task_id:
                continue
            if lease.get("status") != "active":
                continue
            if _parse_utc(str(lease["expires_at"])) <= stamp:
                continue
            active = lease
        return active

    # ------------------------------------------------------------------
    # Plan graph helpers
    # ------------------------------------------------------------------

    def _dependencies_satisfied(self, dependency_ids: Iterable[str]) -> bool:
        for dep_id in dependency_ids:
            plan = self.store.read("plan_nodes", str(dep_id))
            # Node-approved leaves count as satisfied for sibling dispatch even
            # before product promotion when the plan marks accept_node_approved.
            if plan.get("status") in {"node_approved", "complete", "completed"}:
                if plan.get("accept_node_approved") or plan.get("status") in {
                    "complete",
                    "completed",
                }:
                    continue
            task = self.store.read("tasks", str(plan["task_id"]))
            if task.get("status") != "completed":
                # Also accept node_approved attempt state via plan flag.
                if plan.get("status") == "node_approved" and plan.get(
                    "accept_node_approved"
                ):
                    continue
                return False
        return True

    def active_dispatch_count(self, *, now: str | None = None) -> int:
        """Count tasks currently leased or in running/executing/integrating."""
        self.expire_stale_leases(now=now)
        count = 0
        for task in self.list_tasks():
            if task.get("status") in {"running", "executing", "integrating"}:
                count += 1
                continue
            if self._active_lease_for_task(str(task["task_id"]), now=now):
                count += 1
        return count

    def scheduler_tick(
        self,
        *,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT_DISPATCHES,
        now: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Deterministic scheduler step: pick highest-priority ready leaves.

        Does not run workers. Returns the ordered dispatch candidates that fit
        under the concurrency limit so an operator or dispatch loop can lease
        and execute them while preserving safety gates.
        """
        if max_concurrent <= 0:
            raise ControlPlaneError("max_concurrent must be positive")
        stamp = now or utc_now()
        active = self.active_dispatch_count(now=stamp)
        capacity = max(0, int(max_concurrent) - active)
        queue = self.ready_queue(now=stamp)
        if limit is not None:
            capacity = min(capacity, max(0, int(limit)))
        selected = queue[:capacity]
        decision = {
            "schema_version": "cosmo.scheduler-tick.v1",
            "created_at": stamp,
            "max_concurrent": int(max_concurrent),
            "active_dispatches": active,
            "capacity": capacity,
            "ready_count": len(queue),
            "selected_task_ids": [item["task_id"] for item in selected],
            "selected": [
                {
                    "task_id": item["task_id"],
                    "priority_score": item.get("priority_score"),
                    "age_component": item.get("age_component"),
                    "created_at": item.get("created_at"),
                    "plan_node_id": item.get("plan_node_id"),
                }
                for item in selected
            ],
        }
        self.store.append_event(
            "scheduler_tick",
            {
                "active_dispatches": active,
                "capacity": capacity,
                "selected_task_ids": decision["selected_task_ids"],
                "ready_count": len(queue),
            },
        )
        return decision

    def dispatch_scheduler_batch(
        self,
        *,
        repo_root: Path,
        client: InferenceClient,
        model: str,
        owner: str,
        strategy_summary: str = "scheduler-dispatched leaf attempt",
        max_concurrent: int = DEFAULT_MAX_CONCURRENT_DISPATCHES,
        limit: int | None = 1,
        reasoning: str = "off",
        temperature: float = 0.0,
        worker_max_output_tokens: int = 2048,
        review_max_output_tokens: int = 2048,
        lease_ttl_seconds: int = DEFAULT_LEASE_SECONDS,
        read_only_paths: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Run one scheduler tick and dispatch selected ready leaves.

        Each leaf still goes through AtomicHarness validation + independent
        review. Product promotion remains a separate explicit step.
        """
        tick = self.scheduler_tick(max_concurrent=max_concurrent, limit=limit)
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for task_id in tick["selected_task_ids"]:
            try:
                result = self.dispatch_ready_leaf(
                    str(task_id),
                    repo_root=Path(repo_root),
                    client=client,
                    model=model,
                    strategy_summary=strategy_summary,
                    owner=owner,
                    reasoning=reasoning,
                    temperature=temperature,
                    worker_max_output_tokens=worker_max_output_tokens,
                    review_max_output_tokens=review_max_output_tokens,
                    lease_ttl_seconds=lease_ttl_seconds,
                    read_only_paths=read_only_paths,
                )
                results.append(result)
            except (ControlPlaneError, AtomicHarnessError) as exc:
                errors.append({"task_id": task_id, "error": str(exc)})
        return {
            "scheduler_tick": tick,
            "dispatched": results,
            "errors": errors,
        }

    def refresh_dependency_readiness(self, task_id: str) -> dict[str, Any]:
        """Move a blocked task to ready when all plan dependencies complete."""
        task_id = _safe_record_identifier(task_id)
        task = self.store.read("tasks", task_id)
        if task.get("status") not in {"blocked", "ready"}:
            return task
        plan = self.store.read("plan_nodes", str(task["plan_node_id"]))
        deps = [str(item) for item in (plan.get("dependencies") or [])]
        satisfied = self._dependencies_satisfied(deps) if deps else True
        if task.get("status") == "blocked" and satisfied:
            return self.transition_task(
                task_id,
                "ready",
                reason="plan dependencies satisfied",
                actor="control_plane",
            )
        if task.get("status") == "ready" and deps and not satisfied:
            return self.transition_task(
                task_id,
                "blocked",
                reason="plan dependencies unmet",
                actor="control_plane",
            )
        return task

    # ------------------------------------------------------------------
    # Convergence assessments and ledger projection
    # ------------------------------------------------------------------

    def record_convergence_assessment(
        self,
        task_id: str,
        *,
        summary: str,
        converged: bool,
        evidence: Mapping[str, Any] | None = None,
        operator_supplied: bool = True,
    ) -> dict[str, Any]:
        """Append an immutable convergence assessment for a task."""
        task_id = _safe_record_identifier(task_id)
        self.store.read("tasks", task_id)  # ensure exists
        if not summary.strip():
            raise ControlPlaneError("convergence summary must not be empty")
        assessment_id = new_id("CONV")
        record = {
            "schema_version": "cosmo.convergence-assessment.v1",
            "assessment_id": assessment_id,
            "task_id": task_id,
            "summary": summary,
            "converged": bool(converged),
            "evidence": dict(evidence or {}),
            "operator_supplied": bool(operator_supplied),
            "created_at": utc_now(),
        }
        self.store.write("convergence_assessments", assessment_id, record)
        self.store.append_event(
            "convergence_assessed",
            {
                "assessment_id": assessment_id,
                "task_id": task_id,
                "converged": bool(converged),
            },
        )
        return record

    def project_ledger(self, task_id: str) -> dict[str, Any]:
        """Build a compact ledger projection from attempts/validation/reviews.

        Preserves observations, hypotheses, retained successes, invalidated
        assumptions, and rejected strategies without rewriting source records.
        """
        task_id = _safe_record_identifier(task_id)
        task = self.store.read("tasks", task_id)
        attempts = [
            item
            for item in self.list_attempts()
            if item.get("task_id") == task_id
        ]
        # Prefer derived attempt_states when present.
        merged_attempts: list[dict[str, Any]] = []
        for item in attempts:
            attempt_id = str(item["attempt_id"])
            try:
                merged_attempts.append(self.get_attempt(attempt_id))
            except AtomicHarnessError:
                merged_attempts.append(item)

        observations: list[Any] = []
        hypotheses: list[Any] = []
        retained_successes: list[Any] = []
        invalidated_assumptions: list[Any] = []
        rejected_strategies: list[dict[str, Any]] = []
        validation_summaries: list[dict[str, Any]] = []
        review_summaries: list[dict[str, Any]] = []

        for attempt in merged_attempts:
            attempt_id = str(attempt["attempt_id"])
            strategy = str(attempt.get("strategy_summary") or "").strip()
            status = str(attempt.get("status") or "")
            validation_id = attempt.get("validation_id")
            review_id = attempt.get("review_id")
            validation = None
            review = None
            if validation_id:
                try:
                    validation = self.store.read(
                        "validations", str(validation_id)
                    )
                except AtomicHarnessError:
                    validation = None
            if review_id:
                try:
                    review = self.store.read("reviews", str(review_id))
                except AtomicHarnessError:
                    review = None

            if validation is not None:
                validation_summaries.append(
                    {
                        "attempt_id": attempt_id,
                        "validation_id": validation.get("validation_id"),
                        "passed": validation.get("passed"),
                        "candidate_sha256": validation.get("candidate_sha256"),
                    }
                )
                if not validation.get("passed"):
                    observations.append(
                        {
                            "source": "validation",
                            "attempt_id": attempt_id,
                            "text": "deterministic validation failed",
                            "checks": validation.get("checks"),
                        }
                    )
                    if strategy:
                        rejected_strategies.append(
                            {
                                "attempt_id": attempt_id,
                                "strategy_summary": strategy,
                                "reason": "validation_failed",
                            }
                        )

            if review is not None:
                body = review.get("review") if isinstance(
                    review.get("review"), dict
                ) else review
                # Attempt reviews store structured fields either nested or flat.
                def _field(name: str) -> list[Any]:
                    value = body.get(name) if isinstance(body, dict) else None
                    if value is None and name in review:
                        value = review.get(name)
                    return list(value) if isinstance(value, list) else []

                obs = _field("observations")
                hyp = _field("hypotheses")
                ret = _field("retained_successes")
                inv = _field("invalidated_assumptions")
                observations.extend(
                    {"source": "review", "attempt_id": attempt_id, "item": item}
                    for item in obs
                )
                hypotheses.extend(
                    {"source": "review", "attempt_id": attempt_id, "item": item}
                    for item in hyp
                )
                retained_successes.extend(
                    {"source": "review", "attempt_id": attempt_id, "item": item}
                    for item in ret
                )
                invalidated_assumptions.extend(
                    {"source": "review", "attempt_id": attempt_id, "item": item}
                    for item in inv
                )
                verdict = None
                if isinstance(body, dict):
                    verdict = body.get("verdict")
                verdict = verdict or review.get("verdict")
                review_summaries.append(
                    {
                        "attempt_id": attempt_id,
                        "review_id": review.get("review_id") or review_id,
                        "verdict": verdict,
                    }
                )
                if verdict and verdict != "PASS" and strategy:
                    rejected_strategies.append(
                        {
                            "attempt_id": attempt_id,
                            "strategy_summary": strategy,
                            "reason": f"review_{verdict}",
                        }
                    )
            elif status in {"failed", "blocked", "rejected"} and strategy:
                rejected_strategies.append(
                    {
                        "attempt_id": attempt_id,
                        "strategy_summary": strategy,
                        "reason": status,
                    }
                )

        # Deduplicate rejected strategies by attempt_id while preserving order.
        seen_attempts: set[str] = set()
        unique_rejected: list[dict[str, Any]] = []
        for item in rejected_strategies:
            key = str(item.get("attempt_id"))
            if key in seen_attempts:
                continue
            seen_attempts.add(key)
            unique_rejected.append(item)

        projection_id = new_id("LEDGER")
        projection = {
            "schema_version": "cosmo.ledger-projection.v1",
            "projection_id": projection_id,
            "task_id": task_id,
            "spec_id": task.get("current_spec_id"),
            "observations": observations,
            "hypotheses": hypotheses,
            "retained_successes": retained_successes,
            "invalidated_assumptions": invalidated_assumptions,
            "rejected_strategies": unique_rejected,
            "validation_summaries": validation_summaries,
            "review_summaries": review_summaries,
            "attempt_ids": [str(item["attempt_id"]) for item in merged_attempts],
            "created_at": utc_now(),
        }
        self.store.write("ledger_projections", projection_id, projection)
        self.store.append_event(
            "ledger_projected",
            {
                "projection_id": projection_id,
                "task_id": task_id,
                "attempt_count": len(merged_attempts),
            },
        )
        return projection

    # ------------------------------------------------------------------
    # Dispatch into AtomicHarness (no validation/review bypass)
    # ------------------------------------------------------------------

    def dispatch_ready_leaf(
        self,
        task_id: str,
        *,
        repo_root: Path,
        client: InferenceClient,
        model: str,
        strategy_summary: str,
        owner: str,
        reasoning: str = "off",
        temperature: float = 0.0,
        worker_max_output_tokens: int = 2048,
        review_max_output_tokens: int = 2048,
        retry_contract: dict[str, Any] | None = None,
        lease_ttl_seconds: int = DEFAULT_LEASE_SECONDS,
        read_only_paths: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Lease a ready leaf and run it through AtomicHarness.run_attempt.

        Validation and independent Review remain inside AtomicHarness; this
        facade only enforces control-plane readiness, leasing, and status
        transitions around that primitive.
        """
        task_id = _safe_record_identifier(task_id)
        task = self.store.read("tasks", task_id)
        plan = self.store.read("plan_nodes", str(task["plan_node_id"]))
        if plan.get("kind") != "leaf":
            raise ControlPlaneError("dispatch_ready_leaf requires a leaf plan")
        if not plan.get("writable_artifact"):
            raise ControlPlaneError("leaf plan must declare one writable artifact")
        deps = [str(item) for item in (plan.get("dependencies") or [])]
        if deps and not self._dependencies_satisfied(deps):
            raise ControlPlaneError(
                "cannot dispatch leaf while plan dependencies are unmet"
            )
        if task.get("status") not in {"ready", "queued"}:
            raise ControlPlaneError(
                f"task must be ready or queued to dispatch, got {task.get('status')!r}"
            )

        lease = self.acquire_lease(
            task_id, owner=owner, ttl_seconds=lease_ttl_seconds
        )
        if task.get("status") == "ready":
            self.transition_task(
                task_id, "queued", reason="leased for dispatch", actor=owner
            )
        self.transition_task(
            task_id, "running", reason="dispatch_ready_leaf", actor=owner
        )
        try:
            result = self.harness.run_attempt(
                task_id=task_id,
                repo_root=Path(repo_root),
                client=client,
                model=model,
                reasoning=reasoning,
                temperature=temperature,
                worker_max_output_tokens=worker_max_output_tokens,
                review_max_output_tokens=review_max_output_tokens,
                strategy_summary=strategy_summary,
                retry_contract=retry_contract,
                read_only_paths=read_only_paths,
            )
        except Exception:
            # Return task to ready so a later operator can retry; do not hide
            # the original failure.
            try:
                self.transition_task(
                    task_id,
                    "ready",
                    reason="dispatch failed; returned to ready",
                    actor=owner,
                )
            except ControlPlaneError:
                pass
            try:
                self.release_lease(str(lease["lease_id"]), owner=owner)
            except ControlPlaneError:
                pass
            raise

        # Successful harness completion leaves the attempt reviewed; the task
        # stays non-terminal until explicit promotion. Return to ready for
        # retry or promotion recovery unless the attempt is promotable and the
        # operator will promote next.
        if result.get("promotable"):
            # Leave running so promotion is the next legal control action via
            # recover_promotion / promote helpers; operator may also return to
            # ready deliberately.
            pass
        else:
            try:
                self.transition_task(
                    task_id,
                    "ready",
                    reason="attempt completed but not promotable",
                    actor=owner,
                )
            except ControlPlaneError:
                pass
        try:
            self.release_lease(str(lease["lease_id"]), owner=owner)
        except ControlPlaneError:
            pass

        self.store.append_event(
            "leaf_dispatched",
            {
                "task_id": task_id,
                "attempt_id": result.get("attempt_id"),
                "lease_id": lease["lease_id"],
                "promotable": result.get("promotable"),
                "owner": owner,
            },
        )
        return {
            **result,
            "lease_id": lease["lease_id"],
            "task_id": task_id,
        }

    def recover_promotion(
        self, *, attempt_id: str, repo_root: Path
    ) -> dict[str, Any]:
        """Expose AtomicHarness promotion with recovery, without bypassing gates.

        Promotion still requires Validation PASS and Review PASS inside the
        harness. This method only wraps that entry point for control-plane
        callers and records a control-plane event around the outcome.
        """
        attempt_id = _safe_record_identifier(attempt_id)
        result = self.harness.promote(
            attempt_id=attempt_id, repo_root=Path(repo_root)
        )
        attempt = self.get_attempt(attempt_id)
        task_id = str(attempt.get("task_id"))
        # Harness marks the task completed on successful promotion; mirror a
        # control-plane event for observability.
        self.store.append_event(
            "control_promotion_completed",
            {
                "attempt_id": attempt_id,
                "task_id": task_id,
                "promotion_id": result.get("promotion_id"),
                "recovered": bool(result.get("recovered")),
            },
        )
        return result

    # ------------------------------------------------------------------
    # Parent integration dispatch (no second promotion path)
    # ------------------------------------------------------------------

    def assemble_integration(
        self,
        parent_node_id: str,
        *,
        repo_root: Path,
        selected: Mapping[str, str] | None = None,
        require_all_leaves: bool = True,
        expected_baseline_tree_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Assemble a parent integration run via IntegrationService.

        Does not mutate product files. Child leaves must already be
        node-approved (validation PASS + review PASS).
        """
        from .integration import IntegrationService

        parent_node_id = _safe_record_identifier(parent_node_id)
        parent = self.store.read("plan_nodes", parent_node_id)
        if parent.get("kind") != "parent":
            raise ControlPlaneError("assemble_integration requires a parent plan node")
        service = IntegrationService(self.store)
        result = service.assemble(
            parent_node_id,
            repo_root=Path(repo_root),
            selected=selected,
            require_all_leaves=require_all_leaves,
            expected_baseline_tree_sha256=expected_baseline_tree_sha256,
        )
        self.store.append_event(
            "control_integration_assembled",
            {
                "integration_id": result.get("integration_id"),
                "parent_node_id": parent_node_id,
                "validation_passed": result.get("validation_passed"),
            },
        )
        return result

    def get_integration(self, integration_id: str) -> dict[str, Any]:
        return self.store.read(
            "integration_runs", _safe_record_identifier(integration_id)
        )

    def list_integrations(self) -> list[dict[str, Any]]:
        return self.store.list_records("integration_runs")

    def review_integration(
        self,
        integration_id: str,
        *,
        verdict: str | None = None,
        requirement_results: Iterable[Mapping[str, Any]] | None = None,
        notes: str | None = None,
        client: InferenceClient | None = None,
        model: str | None = None,
        reasoning: str = "off",
        max_output_tokens: int = 2048,
        operator_caller: CallerProvenance | None = None,
    ) -> dict[str, Any]:
        """Record an independent integration review (does not promote).

        A local client+model uses the canonical model-backed reviewer.  An
        explicit verdict remains available for a clearly recorded operator
        fallback; neither path can promote product state.
        """
        from .integration import IntegrationService

        service = IntegrationService(self.store)
        if client is not None and model:
            return service.review_with_client(
                _safe_record_identifier(integration_id),
                client=client,
                model=model,
                reasoning=reasoning,
                max_output_tokens=max_output_tokens,
            )
        if not verdict:
            raise ControlPlaneError(
                "review_integration requires verdict or local client+model"
            )
        result = service.record_review(
            _safe_record_identifier(integration_id),
            verdict=verdict,
            requirement_results=list(requirement_results or []),
            notes=notes,
            operator_caller=operator_caller,
        )
        return result

    def promote_integration(
        self, integration_id: str, *, repo_root: Path | None = None
    ) -> dict[str, Any]:
        """Explicit parent integration promotion after validation+review PASS.

        Uses IntegrationService.promote under the existing promotion lock;
        product files stay unchanged until this call.
        """
        from .integration import IntegrationService

        service = IntegrationService(self.store)
        result = service.promote(
            _safe_record_identifier(integration_id),
            repo_root=Path(repo_root) if repo_root is not None else None,
        )
        self.store.append_event(
            "control_integration_promoted",
            {
                "integration_id": integration_id,
                "promotion_id": result.get("promotion_id"),
            },
        )
        return result

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of control-plane projections."""
        return {
            "schema_version": "cosmo.control-plane.v1",
            "state_root": str(self.store.root),
            "tasks": self.list_tasks(),
            "requests": self.list_requests(),
            "plans": self.list_plans(),
            "ready_queue": [
                {
                    "task_id": item["task_id"],
                    "priority_score": item.get("priority_score"),
                    "status": item.get("status"),
                    "created_at": item.get("created_at"),
                }
                for item in self.ready_queue()
            ],
            "active_leases": [
                item
                for item in self.list_leases()
                if item.get("status") == "active"
            ],
            "task_statuses": sorted(TASK_STATUSES),
            "legal_transitions": {
                key: sorted(value) for key, value in LEGAL_TRANSITIONS.items()
            },
            "priority_weights": dict(PRIORITY_WEIGHTS),
            "relationship_types": sorted(RELATIONSHIP_TYPES),
            "triage_review_verdicts": sorted(TRIAGE_REVIEW_VERDICTS),
            "max_concurrent_default": DEFAULT_MAX_CONCURRENT_DISPATCHES,
        }
