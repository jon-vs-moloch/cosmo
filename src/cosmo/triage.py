"""Model-backed request triage with independent review.

Flow (spec §8):

    immutable Request
      -> Triage Agent proposal
      -> independent Triage Review
      -> link / create / update Task + TaskSpec + relationships + priority

This service never mutates product files. Model calls use the supplied local
``InferenceClient`` only. Operator-supplied proposal/review payloads remain a
first-class fallback and are recorded as such — they are not presented as
autonomous model triage.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .atomic import (
    AtomicHarnessError,
    _safe_record_identifier,
    _sha256_text,
    new_id,
    utc_now,
)
from .authority import CallerProvenance, authenticate_model, require_operator
from .control import (
    ControlPlane,
    ControlPlaneError,
    RELATIONSHIP_TYPES,
    TRIAGE_REVIEW_VERDICTS,
    score_priority_components,
)
from .inference import InferenceClient
from .models import Generation
from .reporting import canonical_hash


class TriageError(AtomicHarnessError):
    """Triage proposal, review, or application failed an invariant."""


# Spec §8.4 verdicts (plus legacy REJECT/REVISE aliases normalized on parse).
APPROVED_TRIAGE_VERDICTS = frozenset({"PASS"})

TRIAGE_PROPOSAL_SYSTEM = """You are a bounded Cosmo Triage Agent.
Classify one immutable user Request against a compact index of active Tasks.
Infer exactly one primary relationship (NEW, DUPLICATE_OF, OVERLAPS_WITH,
DEPENDS_ON, SUPERSEDES, CONFLICTS_WITH), preserve every requirement from the
Request, mark inferred requirements explicitly, and emit explainable priority
components. Return exactly one JSON object. No Markdown fences or prose."""

TRIAGE_REVIEWER_SYSTEM = """You are an independent Cosmo Triage Reviewer.
Evaluate a triage proposal against the original Request and task index. Check
relationship support, lost requirements, inferred-vs-stated marking, priority
sanity, and whether new scope invalidates work underway. Return exactly one
JSON object with verdict PASS | RETRY_TRIAGE | NEEDS_USER_CLARIFICATION |
ESCALATE, evidence, and notes. No Markdown fences or prose."""

TRIAGE_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "cosmo_triage_proposal",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "relationship": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": sorted(RELATIONSHIP_TYPES),
                        },
                        "target_task_id": {"type": ["string", "null"]},
                        "rationale": {"type": "string"},
                    },
                    "required": ["type", "target_task_id", "rationale"],
                    "additionalProperties": False,
                },
                "proposed_task_change": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "added_requirements": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "requirements": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "modified_requirements": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "inferred_requirements": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "constraints": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "plan_kind": {
                            "type": "string",
                            "enum": ["leaf", "parent", "needs_planning"],
                        },
                        "artifact_path": {"type": ["string", "null"]},
                        "objective": {"type": "string"},
                    },
                    "required": [
                        "title",
                        "summary",
                        "added_requirements",
                        "inferred_requirements",
                        "constraints",
                    ],
                    "additionalProperties": True,
                },
                "priority_components": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
                "confidence": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                },
                "open_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "relationship",
                "proposed_task_change",
                "priority_components",
                "confidence",
                "open_questions",
            ],
            "additionalProperties": False,
        },
    },
}


def _as_string_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TriageError(f"{field_name} must be an array")
    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                result.append(text)
        elif isinstance(item, Mapping):
            text = str(item.get("text") or item.get("requirement") or "").strip()
            if text:
                result.append(text)
        else:
            raise TriageError(f"{field_name} entries must be strings or objects")
    return result


def _parse_model_json_object(content: str, *, label: str) -> dict[str, Any]:
    """Accept exactly one JSON object or one complete outer JSON fence.

    Surrounding whitespace is harmless. Prefixes, suffixes, multiple objects,
    and prose are protocol failures before semantic validation.
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
        raise TriageError(f"{label} is not one JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise TriageError(f"{label} must be one JSON object")
    return value


def validate_relationship(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize a relationship object to the spec taxonomy."""
    body = dict(value or {"type": "NEW", "target_task_id": None, "rationale": ""})
    rel_type = str(body.get("type") or "NEW").upper()
    if rel_type not in RELATIONSHIP_TYPES:
        raise TriageError(
            f"unsupported relationship type: {rel_type!r}; "
            f"expected one of {sorted(RELATIONSHIP_TYPES)}"
        )
    target = body.get("target_task_id")
    if target is not None and str(target).strip():
        target = _safe_record_identifier(str(target).strip())
    else:
        target = None
    if rel_type == "NEW" and target is not None:
        raise TriageError("NEW relationships must not target an existing task")
    if rel_type != "NEW" and target is None:
        raise TriageError(
            f"relationship type {rel_type} requires target_task_id"
        )
    return {
        "type": rel_type,
        "target_task_id": target,
        "rationale": str(body.get("rationale") or body.get("notes") or ""),
    }


def validate_priority_components(value: Any) -> dict[str, float]:
    """Validate priority components via the control-plane scorer."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TriageError("priority_components must be an object")
    # score_priority_components rejects unknowns and normalizes scales.
    scored = score_priority_components(value if value else {
        "user_importance": 0.5,
        "urgency": 0.5,
    })
    return dict(scored["components"])


def validate_triage_proposal(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a triage proposal artifact."""
    if not isinstance(value, Mapping):
        raise TriageError("triage proposal must be an object")
    relationship = validate_relationship(
        value.get("relationship") if isinstance(value.get("relationship"), Mapping) else None
    )
    change_raw = value.get("proposed_task_change") or {}
    if not isinstance(change_raw, Mapping):
        raise TriageError("proposed_task_change must be an object")
    title = str(change_raw.get("title") or "").strip()
    summary = str(change_raw.get("summary") or "").strip()
    if not title:
        # Fall back to summary / objective so sparse operator payloads work.
        title = str(
            change_raw.get("objective")
            or value.get("title")
            or "Untitled triage task"
        ).strip()
    if not summary:
        summary = str(
            change_raw.get("objective")
            or value.get("summary")
            or title
        ).strip()
    plan_kind = str(
        change_raw.get("plan_kind") or value.get("plan_kind") or "needs_planning"
    ).lower()
    if plan_kind not in {"leaf", "parent", "needs_planning"}:
        raise TriageError(
            f"plan_kind must be leaf, parent, or needs_planning; got {plan_kind!r}"
        )
    artifact_path = change_raw.get("artifact_path")
    if artifact_path is not None and str(artifact_path).strip():
        artifact_path = str(artifact_path).strip()
    else:
        artifact_path = None
    if plan_kind == "leaf" and not artifact_path:
        # Leaf shortcut requires a path; otherwise force planning.
        plan_kind = "needs_planning"
    objective = str(
        change_raw.get("objective") or title or summary
    ).strip()
    proposed = {
        "title": title,
        "summary": summary,
        "objective": objective,
        "added_requirements": _as_string_list(
            change_raw.get("added_requirements")
            if change_raw.get("added_requirements") is not None
            else change_raw.get("requirements"),
            field_name="added_requirements",
        ),
        "modified_requirements": _as_string_list(
            change_raw.get("modified_requirements"),
            field_name="modified_requirements",
        ),
        "inferred_requirements": _as_string_list(
            change_raw.get("inferred_requirements"),
            field_name="inferred_requirements",
        ),
        "constraints": _as_string_list(
            change_raw.get("constraints"), field_name="constraints"
        ),
        "plan_kind": plan_kind,
        "artifact_path": artifact_path,
    }
    try:
        priority = validate_priority_components(value.get("priority_components"))
    except ControlPlaneError as exc:
        raise TriageError(str(exc)) from exc
    confidence = str(value.get("confidence") or "medium").lower()
    if confidence not in {"low", "medium", "high"}:
        raise TriageError(f"confidence must be low|medium|high, got {confidence!r}")
    open_questions = _as_string_list(
        value.get("open_questions"), field_name="open_questions"
    )
    proposal = {
        "relationship": relationship,
        "proposed_task_change": proposed,
        "priority_components": priority,
        "confidence": confidence,
        "open_questions": open_questions,
    }
    proposal["proposal_sha256"] = canonical_hash(proposal)
    return proposal


def parse_triage_proposal(content: str) -> dict[str, Any]:
    value = _parse_model_json_object(content, label="triage proposal")
    return validate_triage_proposal(value)


def validate_triage_review(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate independent triage review verdict and evidence."""
    if not isinstance(value, Mapping):
        raise TriageError("triage review must be an object")
    verdict = str(value.get("verdict") or "").upper()
    # Legacy aliases used in earlier scaffolding.
    if verdict == "REJECT":
        verdict = "RETRY_TRIAGE"
    elif verdict == "REVISE":
        verdict = "RETRY_TRIAGE"
    if verdict not in TRIAGE_REVIEW_VERDICTS:
        raise TriageError(
            f"unsupported triage review verdict: {verdict!r}; "
            f"expected one of {sorted(TRIAGE_REVIEW_VERDICTS)}"
        )
    evidence = value.get("evidence") or value.get("checks") or []
    if not isinstance(evidence, list):
        raise TriageError("triage review evidence must be an array")
    if verdict == "PASS" and not evidence:
        # Operator-supplied PASS still needs at least one check note.
        evidence = ["operator or model asserted triage PASS"]
    cleaned_evidence: list[str] = []
    for item in evidence:
        if not isinstance(item, str) or not item.strip():
            raise TriageError("triage review evidence entries must be non-empty strings")
        cleaned_evidence.append(item.strip())
    notes = value.get("notes") or value.get("rationale") or ""
    if notes is not None and not isinstance(notes, str):
        raise TriageError("triage review notes must be a string")
    lost = _as_string_list(
        value.get("lost_requirements"), field_name="lost_requirements"
    )
    if verdict == "PASS" and lost:
        raise TriageError(
            "triage review cannot PASS while listing lost_requirements"
        )
    return {
        "verdict": verdict,
        "evidence": cleaned_evidence,
        "notes": notes or "",
        "lost_requirements": lost,
        "recommended_next_action": value.get("recommended_next_action"),
        "requirement_coverage": value.get("requirement_coverage"),
    }


def parse_triage_review(content: str) -> dict[str, Any]:
    value = _parse_model_json_object(content, label="triage review")
    return validate_triage_review(value)


class TriageService:
    """Request intake, triage proposal/review, and task linkage application."""

    def __init__(self, control: ControlPlane):
        self.control = control
        self.store = control.store

    # ------------------------------------------------------------------
    # Intake
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

        This is the chat front-door entry for broad user outcomes. Downstream
        triage proposal/review/apply creates or links Tasks.
        """
        return self.control.ingest_request(
            request_text=request_text,
            desired_outcome=desired_outcome,
            explicit_constraints=explicit_constraints,
            source_message_refs=source_message_refs,
            requested_priority=requested_priority,
            summary=summary,
            requester=requester,
        )

    # ------------------------------------------------------------------
    # Task index (compact, for triage context)
    # ------------------------------------------------------------------

    def build_task_index(
        self, *, limit: int = 40, include_completed: bool = False
    ) -> list[dict[str, Any]]:
        """Compact index of active/recent tasks for triage context."""
        tasks = self.control.list_tasks()
        rows: list[dict[str, Any]] = []
        for task in tasks:
            status = str(task.get("status") or "")
            if not include_completed and status in {"completed", "canceled"}:
                continue
            spec = None
            if task.get("current_spec_id"):
                try:
                    spec = self.control.get_task_spec(str(task["current_spec_id"]))
                except (ControlPlaneError, AtomicHarnessError):
                    spec = None
            plan = None
            if task.get("plan_node_id"):
                try:
                    plan = self.control.get_plan(str(task["plan_node_id"]))
                except (ControlPlaneError, AtomicHarnessError):
                    plan = None
            linked = []
            for req_id in task.get("linked_request_ids") or []:
                try:
                    req = self.control.get_request(str(req_id))
                    linked.append(
                        {
                            "request_id": req.get("request_id"),
                            "summary": (req.get("summary") or req.get("original_text") or "")[
                                :200
                            ],
                        }
                    )
                except (ControlPlaneError, AtomicHarnessError):
                    continue
            rows.append(
                {
                    "task_id": task.get("task_id"),
                    "status": status,
                    "priority_score": task.get("priority_score"),
                    "created_at": task.get("created_at"),
                    "linked_requests": linked,
                    "objective": (spec or {}).get("objective"),
                    "requirements": [
                        item.get("text") if isinstance(item, Mapping) else item
                        for item in ((spec or {}).get("requirements") or [])
                    ][:12],
                    "plan_kind": (plan or {}).get("kind"),
                    "writable_artifact": (plan or {}).get("writable_artifact"),
                }
            )
        rows.sort(
            key=lambda item: (
                str(item.get("created_at") or ""),
                str(item.get("task_id") or ""),
            ),
            reverse=True,
        )
        return rows[: max(1, int(limit))]

    # ------------------------------------------------------------------
    # Propose
    # ------------------------------------------------------------------

    def propose_triage(
        self,
        request_id: str,
        *,
        proposal: Mapping[str, Any] | None = None,
        client: InferenceClient | None = None,
        model: str | None = None,
        reasoning: str = "off",
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        task_index_limit: int = 40,
        operator_caller: CallerProvenance | None = None,
    ) -> dict[str, Any]:
        """Create a durable triage proposal for an immutable Request.

        Either pass an operator-supplied ``proposal`` or a local
        ``client`` + ``model`` for model-backed triage. Never both-empty.
        """
        request_id = _safe_record_identifier(request_id)
        request = self.control.get_request(request_id)
        task_index = self.build_task_index(limit=task_index_limit)
        now = utc_now()

        if proposal is not None:
            authority = require_operator(operator_caller)
            normalized = validate_triage_proposal(proposal)
            operator_supplied = True
            caller_provenance = authority.to_record()
            generation_meta = None
            model_used = None
        else:
            if client is None or model is None:
                raise TriageError(
                    "propose_triage requires operator proposal or client+model"
                )
            prompt = self._proposal_prompt(request=request, task_index=task_index)
            generation = client.chat(
                model=model,
                role="triage_agent",
                system_prompt=TRIAGE_PROPOSAL_SYSTEM,
                prompt=prompt,
                reasoning=reasoning,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_format=TRIAGE_RESPONSE_FORMAT,
                operation_id=new_id("TRIAGE-OP"),
            )
            normalized = parse_triage_proposal(generation.content)
            operator_supplied = False
            caller_provenance = authenticate_model(
                f"lmstudio:{model}", authenticated_by="cosmo.triage-proposal"
            ).to_record()
            model_used = model
            generation_meta = {
                "role": generation.role,
                "content_sha256": _sha256_text(generation.content),
                "request_hash": generation.request_hash,
                "reasoning_hash": generation.reasoning_hash,
                "stats": dict(generation.stats or {}),
                "latency_seconds": generation.latency_seconds,
                "model_instance_id": generation.model_instance_id,
            }

        # Validate target task exists when relationship needs one.
        target = normalized["relationship"].get("target_task_id")
        if target is not None:
            try:
                self.control.get_task(str(target))
            except (ControlPlaneError, AtomicHarnessError) as exc:
                raise TriageError(
                    f"relationship target_task_id not found: {target}"
                ) from exc

        proposal_id = new_id("TRIAGE")
        record = {
            "schema_version": "cosmo.triage-proposal.v1",
            "proposal_id": proposal_id,
            "request_id": request_id,
            "status": "proposed",
            "relationship": normalized["relationship"],
            "proposed_task_change": normalized["proposed_task_change"],
            "priority_components": normalized["priority_components"],
            "confidence": normalized["confidence"],
            "open_questions": normalized["open_questions"],
            "proposal_sha256": normalized["proposal_sha256"],
            "task_index_snapshot": task_index,
            "task_index_count": len(task_index),
            "operator_supplied": operator_supplied,
            "caller_provenance": caller_provenance,
            "model": model_used,
            "generation": generation_meta,
            "review_id": None,
            "applied": False,
            "created_at": now,
        }
        self.store.write("triage_proposals", proposal_id, record)
        self.store.append_event(
            "triage_proposal_stored",
            {
                "proposal_id": proposal_id,
                "request_id": request_id,
                "relationship_type": normalized["relationship"]["type"],
                "operator_supplied": operator_supplied,
            },
        )
        # Mark request as under triage in the derived projection; the
        # immutable Request record itself is never rewritten.
        self.control.update_request_state(
            request_id,
            triage_status="proposed",
            latest_triage_proposal_id=proposal_id,
        )
        return record

    def _proposal_prompt(
        self,
        *,
        request: Mapping[str, Any],
        task_index: Sequence[Mapping[str, Any]],
    ) -> str:
        payload = {
            "request": {
                "request_id": request.get("request_id"),
                "original_text": request.get("original_text"),
                "desired_outcome": request.get("desired_outcome"),
                "explicit_constraints": request.get("explicit_constraints"),
                "summary": request.get("summary"),
                "requested_priority": request.get("requested_priority"),
                "source_message_refs": request.get("source_message_refs"),
            },
            "active_task_index": list(task_index),
            "relationship_taxonomy": sorted(RELATIONSHIP_TYPES),
            "priority_policy": {
                "components": [
                    "user_importance",
                    "urgency",
                    "deadline_pressure",
                    "dependency_unblocking",
                    "risk_reduction",
                    "age",
                    "estimated_effort",
                ],
                "scale": "0.0-1.0 preferred (0-100 also accepted)",
                "notes": (
                    "estimated_effort is a cost (higher lowers priority). "
                    "age may start near 0; the scheduler applies aging."
                ),
            },
            "output_contract": (
                "Return JSON with relationship, proposed_task_change "
                "(title/summary/added_requirements/modified_requirements/"
                "inferred_requirements/constraints/plan_kind/"
                "artifact_path/objective), priority_components, confidence, "
                "open_questions. Preserve every request requirement; mark "
                "inferred requirements separately."
            ),
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    # ------------------------------------------------------------------
    # Review
    # ------------------------------------------------------------------

    def review_triage(
        self,
        proposal_id: str,
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
        """Independent triage review. Never applies the proposal."""
        proposal_id = _safe_record_identifier(proposal_id)
        proposal = self.store.read("triage_proposals", proposal_id)
        if proposal.get("applied"):
            raise TriageError("cannot re-review an already applied triage proposal")

        request = self.control.get_request(str(proposal["request_id"]))

        if review is not None:
            authority = require_operator(operator_caller)
            validated = validate_triage_review(review)
            reviewer = independent_reviewer or str(
                review.get("independent_reviewer") or "operator_supplied"
            )
            operator_supplied = True
            caller_provenance = authority.to_record()
            generation_meta = None
        else:
            if client is None or model is None:
                raise TriageError(
                    "review_triage requires operator review payload or client+model"
                )
            prompt = self._review_prompt(request=request, proposal=proposal)
            generation = client.chat(
                model=model,
                role="triage_reviewer",
                system_prompt=TRIAGE_REVIEWER_SYSTEM,
                prompt=prompt,
                reasoning=reasoning,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                operation_id=new_id("TRIAGE-REVIEW-OP"),
            )
            validated = parse_triage_review(generation.content)
            reviewer = independent_reviewer or model
            operator_supplied = False
            caller_provenance = authenticate_model(
                f"lmstudio:{model}", authenticated_by="cosmo.triage-review"
            ).to_record()
            generation_meta = {
                "role": generation.role,
                "content_sha256": _sha256_text(generation.content),
                "request_hash": generation.request_hash,
                "stats": dict(generation.stats or {}),
            }

        # Deterministic requirement-preservation checks supplement the model.
        deterministic_checks = self._deterministic_review_checks(
            request=request, proposal=proposal
        )
        evidence = list(validated["evidence"])
        for check in deterministic_checks:
            if check not in evidence:
                evidence.append(check)
        if any(item.startswith("LOST_REQUIREMENT:") for item in deterministic_checks):
            if validated["verdict"] == "PASS":
                # Fail closed: do not allow PASS when request text is dropped.
                validated = {
                    **validated,
                    "verdict": "RETRY_TRIAGE",
                    "notes": (
                        (validated.get("notes") or "")
                        + " [harness] request requirements appear lost"
                    ).strip(),
                }

        review_id = new_id("TRIAGE-REVIEW")
        now = utc_now()
        record = {
            "schema_version": "cosmo.triage-review.v1",
            "review_id": review_id,
            "proposal_id": proposal_id,
            "request_id": proposal.get("request_id"),
            "verdict": validated["verdict"],
            "evidence": evidence,
            "notes": validated.get("notes") or "",
            "lost_requirements": validated.get("lost_requirements") or [],
            "requirement_coverage": validated.get("requirement_coverage"),
            "recommended_next_action": validated.get("recommended_next_action"),
            "independent_reviewer": reviewer,
            "operator_supplied": operator_supplied,
            "caller_provenance": caller_provenance,
            "generation": generation_meta,
            "created_at": now,
        }
        self.store.write("triage_reviews", review_id, record)

        proposal["review_id"] = review_id
        proposal["status"] = (
            "approved"
            if validated["verdict"] in APPROVED_TRIAGE_VERDICTS
            else f"review_{validated['verdict'].lower()}"
        )
        proposal["approved"] = validated["verdict"] in APPROVED_TRIAGE_VERDICTS
        proposal["reviewed_at"] = now
        self.store.replace("triage_proposals", proposal_id, proposal)

        self.control.update_request_state(
            str(proposal["request_id"]),
            triage_status=proposal["status"],
            latest_triage_review_id=review_id,
        )

        self.store.append_event(
            "triage_reviewed",
            {
                "proposal_id": proposal_id,
                "review_id": review_id,
                "verdict": validated["verdict"],
                "approved": proposal["approved"],
            },
        )
        return record

    def _review_prompt(
        self,
        *,
        request: Mapping[str, Any],
        proposal: Mapping[str, Any],
    ) -> str:
        payload = {
            "request": {
                "request_id": request.get("request_id"),
                "original_text": request.get("original_text"),
                "desired_outcome": request.get("desired_outcome"),
                "explicit_constraints": request.get("explicit_constraints"),
            },
            "proposal": {
                "proposal_id": proposal.get("proposal_id"),
                "proposal_sha256": proposal.get("proposal_sha256"),
                "relationship": proposal.get("relationship"),
                "proposed_task_change": proposal.get("proposed_task_change"),
                "priority_components": proposal.get("priority_components"),
                "confidence": proposal.get("confidence"),
                "open_questions": proposal.get("open_questions"),
                "operator_supplied": proposal.get("operator_supplied"),
            },
            "task_index_snapshot": proposal.get("task_index_snapshot") or [],
            "review_contract": (
                "Return JSON with verdict (PASS|RETRY_TRIAGE|"
                "NEEDS_USER_CLARIFICATION|ESCALATE), evidence (non-empty string "
                "array), notes, optional lost_requirements, optional "
                "requirement_coverage."
            ),
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def _deterministic_review_checks(
        self,
        *,
        request: Mapping[str, Any],
        proposal: Mapping[str, Any],
    ) -> list[str]:
        checks: list[str] = []
        change = proposal.get("proposed_task_change") or {}
        original = str(request.get("original_text") or "").strip().lower()
        summary = str(change.get("summary") or "").strip().lower()
        title = str(change.get("title") or "").strip().lower()
        objective = str(change.get("objective") or "").strip().lower()
        combined = " ".join([title, summary, objective])
        # Soft lexical preservation: very short requests must appear somewhere.
        if original and len(original) <= 80:
            # Use a few content tokens longer than 3 chars.
            tokens = [t for t in original.replace(".", " ").split() if len(t) > 3]
            missing = [t for t in tokens[:6] if t not in combined]
            if tokens and len(missing) == len(tokens[:6]):
                checks.append(
                    "LOST_REQUIREMENT: proposal text shares no content tokens "
                    "with the original request"
                )
            else:
                checks.append("request content tokens appear in proposal text")
        else:
            # Long requests cannot be validated by a short lexical overlap
            # heuristic. Require the proposal to enumerate the request's
            # stated/inferred requirements so the independent reviewer has a
            # concrete coverage surface rather than a blind PASS path.
            enumerated = (
                list(change.get("added_requirements") or [])
                + list(change.get("modified_requirements") or [])
                + list(change.get("inferred_requirements") or [])
            )
            if not enumerated:
                checks.append(
                    "LOST_REQUIREMENT: long request has no explicit requirement enumeration"
                )
            else:
                checks.append(
                    f"long request requirements explicitly enumerated ({len(enumerated)})"
                )
        constraints = list(request.get("explicit_constraints") or [])
        proposed_constraints = {
            str(item).strip().lower()
            for item in (change.get("constraints") or [])
        }
        for constraint in constraints:
            if str(constraint).strip().lower() not in proposed_constraints:
                checks.append(
                    f"LOST_REQUIREMENT: explicit constraint not preserved: {constraint}"
                )
            else:
                checks.append(f"constraint preserved: {constraint}")
        rel = proposal.get("relationship") or {}
        if rel.get("type") in RELATIONSHIP_TYPES:
            checks.append(f"relationship type in taxonomy: {rel.get('type')}")
        else:
            checks.append("LOST_REQUIREMENT: relationship type outside taxonomy")
        return checks

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply_triage(
        self,
        proposal_id: str,
        *,
        repo_root: Path | None = None,
        validators: Iterable[str] = (),
        force_plan_kind: str | None = None,
    ) -> dict[str, Any]:
        """Apply an approved triage proposal: link/create/update Task + Spec.

        Product files are never modified. On NEW, creates a Task (parent shell
        when multi-artifact / needs_planning, or leaf when a single artifact is
        declared). On DUPLICATE_OF / OVERLAPS_WITH / etc., links the Request to
        the target Task and versions the TaskSpec when requirements change.
        """
        proposal_id = _safe_record_identifier(proposal_id)
        proposal = self.store.read("triage_proposals", proposal_id)
        if proposal.get("applied"):
            raise TriageError(f"triage proposal already applied: {proposal_id}")
        if not proposal.get("approved"):
            raise TriageError(
                "refuse to apply unreviewed or non-PASS triage proposal; "
                f"status={proposal.get('status')!r}"
            )
        review_id = proposal.get("review_id")
        if not review_id:
            raise TriageError("approved triage proposal is missing review_id")
        review = self.store.read("triage_reviews", str(review_id))
        if str(review.get("verdict")).upper() not in APPROVED_TRIAGE_VERDICTS:
            raise TriageError(
                f"linked triage review is not PASS: {review.get('verdict')!r}"
            )

        request_id = str(proposal["request_id"])
        request = self.control.get_request(request_id)
        relationship = validate_relationship(proposal.get("relationship"))
        change = dict(proposal.get("proposed_task_change") or {})
        plan_kind = force_plan_kind or str(change.get("plan_kind") or "needs_planning")
        if plan_kind == "needs_planning":
            plan_kind = "parent"
        priority_components = proposal.get("priority_components") or {
            "user_importance": 0.5,
            "urgency": 0.5,
        }

        result: dict[str, Any]
        rel_type = relationship["type"]
        if rel_type == "NEW":
            if repo_root is None:
                raise TriageError(
                    "apply_triage for NEW relationships requires repo_root"
                )
            result = self._apply_new(
                request=request,
                proposal=proposal,
                change=change,
                plan_kind=plan_kind,
                repo_root=Path(repo_root),
                validators=validators,
                priority_components=priority_components,
                relationship=relationship,
            )
        elif rel_type == "DUPLICATE_OF":
            result = self._apply_link_existing(
                request=request,
                proposal=proposal,
                relationship=relationship,
                change=change,
                priority_components=priority_components,
                create_spec_version=False,
                mark_stale=False,
            )
        elif rel_type in {"OVERLAPS_WITH", "DEPENDS_ON", "SUPERSEDES", "CONFLICTS_WITH"}:
            result = self._apply_link_existing(
                request=request,
                proposal=proposal,
                relationship=relationship,
                change=change,
                priority_components=priority_components,
                create_spec_version=True,
                mark_stale=rel_type in {"SUPERSEDES", "CONFLICTS_WITH", "OVERLAPS_WITH"},
            )
        else:
            raise TriageError(f"unhandled relationship type: {rel_type}")

        now = utc_now()
        proposal["applied"] = True
        proposal["status"] = "applied"
        proposal["applied_at"] = now
        proposal["applied_task_id"] = result.get("task_id")
        proposal["applied_spec_id"] = result.get("spec_id")
        proposal["applied_relationship_id"] = result.get("relationship_id")
        self.store.replace("triage_proposals", proposal_id, proposal)

        self.control.update_request_state(
            request_id,
            triage_status="applied",
            linked_task_id=result.get("task_id"),
        )

        self.store.append_event(
            "triage_applied",
            {
                "proposal_id": proposal_id,
                "review_id": review_id,
                "request_id": request_id,
                "task_id": result.get("task_id"),
                "relationship_type": rel_type,
            },
        )
        return {
            "proposal_id": proposal_id,
            "review_id": review_id,
            "request_id": request_id,
            "relationship": relationship,
            **result,
        }

    def _apply_new(
        self,
        *,
        request: Mapping[str, Any],
        proposal: Mapping[str, Any],
        change: Mapping[str, Any],
        plan_kind: str,
        repo_root: Path,
        validators: Iterable[str],
        priority_components: Mapping[str, Any],
        relationship: Mapping[str, Any],
    ) -> dict[str, Any]:
        objective = str(
            change.get("objective")
            or change.get("title")
            or change.get("summary")
            or request.get("desired_outcome")
            or request.get("original_text")
            or "Triaged request"
        ).strip()
        requirements = list(change.get("added_requirements") or [])
        inferred = list(change.get("inferred_requirements") or [])
        constraints = list(
            change.get("constraints") or request.get("explicit_constraints") or []
        )
        artifact_path = change.get("artifact_path") or ""
        if plan_kind == "leaf" and not artifact_path:
            plan_kind = "parent"

        # Build structured requirements for the TaskSpec.  The immutable user
        # request is always retained as a first-class requirement; otherwise a
        # terse model summary could silently erase important wording before a
        # Worker or Reviewer ever sees it.  Triage additions/modifications and
        # inferred requirements are retained alongside it with provenance.
        structured_reqs: list[dict[str, Any]] = []
        seen_requirements: set[str] = set()

        def add_requirement(text: Any, *, source: str, inferred_flag: bool) -> None:
            value = str(text or "").strip()
            key = value.casefold()
            if not value or key in seen_requirements:
                return
            seen_requirements.add(key)
            structured_reqs.append(
                {
                    "requirement_id": f"REQ-{len(structured_reqs) + 1}",
                    "text": value,
                    "source": source,
                    "inferred": inferred_flag,
                }
            )

        add_requirement(
            request.get("original_text"), source="request", inferred_flag=False
        )
        if request.get("desired_outcome"):
            add_requirement(
                request.get("desired_outcome"),
                source="request_desired_outcome",
                inferred_flag=False,
            )
        for text in requirements:
            add_requirement(text, source="triage_proposed", inferred_flag=False)
        for text in change.get("modified_requirements") or []:
            add_requirement(text, source="triage_modified", inferred_flag=False)
        for text in inferred:
            add_requirement(text, source="triage_inferred", inferred_flag=True)
        if not structured_reqs:
            add_requirement(objective, source="request", inferred_flag=False)

        # Reuse control-plane task creation but with full triage records
        # already present — submit_request_from_triage avoids double triage.
        return self.control.create_task_from_triage(
            request_id=str(request["request_id"]),
            objective=objective,
            repo_root=repo_root,
            plan_kind=plan_kind if plan_kind in {"leaf", "parent"} else "parent",
            artifact_path=str(artifact_path) if artifact_path else "",
            validators=validators,
            explicit_constraints=constraints,
            requirements=structured_reqs,
            priority_components=priority_components,
            relationship=relationship,
            triage_proposal_id=str(proposal["proposal_id"]),
            triage_review_id=str(proposal.get("review_id")),
            title=str(change.get("title") or objective),
            summary=str(change.get("summary") or objective),
            needs_planning=str(change.get("plan_kind") or "") == "needs_planning"
            or plan_kind == "parent",
        )

    def _apply_link_existing(
        self,
        *,
        request: Mapping[str, Any],
        proposal: Mapping[str, Any],
        relationship: Mapping[str, Any],
        change: Mapping[str, Any],
        priority_components: Mapping[str, Any],
        create_spec_version: bool,
        mark_stale: bool,
    ) -> dict[str, Any]:
        target_task_id = str(relationship["target_task_id"])
        return self.control.link_request_to_task(
            request_id=str(request["request_id"]),
            task_id=target_task_id,
            relationship=relationship,
            triage_proposal_id=str(proposal["proposal_id"]),
            triage_review_id=str(proposal.get("review_id")),
            priority_components=priority_components,
            added_requirements=list(change.get("added_requirements") or []),
            inferred_requirements=list(change.get("inferred_requirements") or []),
            modified_requirements=list(change.get("modified_requirements") or []),
            constraints=list(change.get("constraints") or []),
            create_spec_version=create_spec_version,
            mark_stale=mark_stale,
            summary=str(change.get("summary") or ""),
            title=str(change.get("title") or ""),
        )

    # ------------------------------------------------------------------
    # End-to-end convenience (still records proposal + independent review)
    # ------------------------------------------------------------------

    def triage_request(
        self,
        *,
        request_text: str,
        repo_root: Path | None = None,
        client: InferenceClient | None = None,
        model: str | None = None,
        proposal: Mapping[str, Any] | None = None,
        review: Mapping[str, Any] | None = None,
        desired_outcome: str | None = None,
        explicit_constraints: Iterable[str] = (),
        source_message_refs: Iterable[str] = (),
        requested_priority: str | None = None,
        validators: Iterable[str] = (),
        auto_apply: bool = True,
        operator_caller: CallerProvenance | None = None,
    ) -> dict[str, Any]:
        """Ingest → propose → review → optionally apply.

        For deterministic tests, pass ``proposal`` and ``review`` without a
        client. For model-backed runs, pass local ``client`` + ``model``.
        Operator review remains an explicit fallback when ``review`` is given.
        """
        receipt = self.ingest_request(
            request_text=request_text,
            desired_outcome=desired_outcome,
            explicit_constraints=explicit_constraints,
            source_message_refs=source_message_refs,
            requested_priority=requested_priority,
        )
        proposed = self.propose_triage(
            receipt["request_id"],
            proposal=proposal,
            client=client,
            model=model,
            operator_caller=operator_caller if proposal is not None else None,
        )
        reviewed = self.review_triage(
            proposed["proposal_id"],
            review=review,
            client=client if review is None else None,
            model=model if review is None else None,
            operator_caller=operator_caller if review is not None else None,
        )
        outcome: dict[str, Any] = {
            "request_id": receipt["request_id"],
            "proposal_id": proposed["proposal_id"],
            "review_id": reviewed["review_id"],
            "verdict": reviewed["verdict"],
            "relationship": proposed.get("relationship"),
            "applied": False,
        }
        if reviewed["verdict"] not in APPROVED_TRIAGE_VERDICTS:
            outcome["status"] = reviewed["verdict"]
            return outcome
        if not auto_apply:
            outcome["status"] = "approved_pending_apply"
            return outcome
        applied = self.apply_triage(
            proposed["proposal_id"],
            repo_root=repo_root,
            validators=validators,
        )
        outcome.update(applied)
        outcome["applied"] = True
        outcome["status"] = "applied"
        return outcome

    def get_proposal(self, proposal_id: str) -> dict[str, Any]:
        return self.store.read(
            "triage_proposals", _safe_record_identifier(proposal_id)
        )

    def list_proposals(self) -> list[dict[str, Any]]:
        return self.store.list_records("triage_proposals")

    def get_review(self, review_id: str) -> dict[str, Any]:
        return self.store.read(
            "triage_reviews", _safe_record_identifier(review_id)
        )
