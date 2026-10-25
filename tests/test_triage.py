"""Deterministic tests for request triage proposal/review/apply."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosmo.atomic import AtomicStore
from cosmo.authority import authenticate_operator
from cosmo.control import ControlPlane, ControlPlaneError
from cosmo.models import Generation
from cosmo.tools import ToolDispatcher
from cosmo.triage import (
    TriageError,
    TriageService,
    parse_triage_proposal,
    parse_triage_review,
    validate_relationship,
)


_OPERATOR = authenticate_operator("pytest-operator")


def _plane(tmp_path: Path) -> tuple[ControlPlane, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "target.py").write_text("value = 1\n", encoding="utf-8")
    (repo / "other.py").write_text("x = 0\n", encoding="utf-8")
    state = tmp_path / "state"
    return ControlPlane(AtomicStore(state)), repo, state


def _new_proposal(**overrides):
    body = {
        "relationship": {
            "type": "NEW",
            "target_task_id": None,
            "rationale": "no related work",
        },
        "proposed_task_change": {
            "title": "Set target value to two",
            "summary": "Make the target value two as requested",
            "objective": "Set value to 2 in target.py",
            "added_requirements": ["Set value to 2"],
            "modified_requirements": [],
            "inferred_requirements": ["Keep file syntactically valid"],
            "constraints": [],
            "plan_kind": "leaf",
            "artifact_path": "target.py",
        },
        "priority_components": {
            "user_importance": 0.6,
            "urgency": 0.5,
            "deadline_pressure": 0.0,
            "dependency_unblocking": 0.0,
            "risk_reduction": 0.1,
            "age": 0.0,
            "estimated_effort": 0.2,
        },
        "confidence": "high",
        "open_questions": [],
    }
    body.update(overrides)
    return body


def _pass_review(**overrides):
    body = {
        "verdict": "PASS",
        "evidence": [
            "relationship NEW is supported",
            "requirements preserved",
            "priority components sane",
        ],
        "notes": "ok",
        "independent_reviewer": "operator_test",
    }
    body.update(overrides)
    return body


class FakeTriageClient:
    def __init__(self, proposal=None, review=None):
        self.proposal = proposal or json.dumps(_new_proposal())
        self.review = review or json.dumps(_pass_review())
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        content = (
            self.proposal if kwargs["role"] == "triage_agent" else self.review
        )
        return Generation(
            role=kwargs["role"],
            content=content,
            reasoning_hash=None,
            stats={
                "input_tokens": 10,
                "total_output_tokens": 8,
                "reasoning_output_tokens": 0,
            },
            latency_seconds=0.01,
            request_hash=f"req-{len(self.calls)}",
            model_instance_id=kwargs["model"],
        )


def test_validate_relationship_taxonomy():
    assert validate_relationship({"type": "NEW"})["type"] == "NEW"
    with pytest.raises(TriageError, match="target_task_id"):
        validate_relationship({"type": "DUPLICATE_OF", "target_task_id": None})
    with pytest.raises(TriageError, match="unsupported relationship"):
        validate_relationship({"type": "MAYBE"})


def test_parse_triage_review_verdicts_and_aliases():
    assert parse_triage_review(
        json.dumps({"verdict": "PASS", "evidence": ["ok"]})
    )["verdict"] == "PASS"
    assert parse_triage_review(
        json.dumps({"verdict": "REJECT", "evidence": ["no"]})
    )["verdict"] == "RETRY_TRIAGE"
    assert parse_triage_review(
        json.dumps({"verdict": "NEEDS_USER_CLARIFICATION", "evidence": ["q"]})
    )["verdict"] == "NEEDS_USER_CLARIFICATION"
    with pytest.raises(TriageError, match="unsupported triage review"):
        parse_triage_review(json.dumps({"verdict": "MEH", "evidence": ["x"]}))


def test_ingest_request_does_not_create_task(tmp_path):
    plane, _repo, _state = _plane(tmp_path)
    receipt = plane.ingest_request(
        request_text="Please improve the logging subsystem end to end.",
        desired_outcome="Better logging",
        explicit_constraints=["no cloud calls"],
        requested_priority="high",
        source_message_refs=["msg-1"],
    )
    assert receipt["request_id"].startswith("REQ-")
    assert receipt["triage_status"] == "pending"
    assert plane.list_tasks() == []
    request = plane.get_request(receipt["request_id"])
    assert request["original_text"].startswith("Please improve")
    assert request["immutable_sha256"]
    assert request["requested_priority"] == "high"


def test_operator_triage_new_leaf_end_to_end(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    triage = TriageService(plane)
    outcome = triage.triage_request(
        request_text="Make the target value two.",
        repo_root=repo,
        proposal=_new_proposal(),
        review=_pass_review(),
        operator_caller=_OPERATOR,
        validators=["python_syntax"],
    )
    assert outcome["status"] == "applied"
    assert outcome["applied"] is True
    assert outcome["relationship"]["type"] == "NEW"
    task = plane.get_task(outcome["task_id"])
    assert task["status"] == "ready"
    assert task["priority_score"] is not None
    spec = plane.get_task_spec(outcome["spec_id"])
    assert any(r.get("inferred") for r in spec["requirements"])
    assert any(not r.get("inferred") for r in spec["requirements"])
    plan = plane.get_plan(outcome["node_id"])
    assert plan["kind"] == "leaf"
    assert plan["writable_artifact"] == "target.py"
    # Ready queue sees the leaf.
    assert any(t["task_id"] == outcome["task_id"] for t in plane.ready_queue())


def test_request_record_remains_immutable_as_triage_state_changes(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    triage = TriageService(plane)
    receipt = triage.ingest_request(
        request_text="Make the target value two.",
    )
    request_id = receipt["request_id"]
    immutable_before = plane.store.read("requests", request_id)

    proposal = triage.propose_triage(
        request_id, proposal=_new_proposal(), operator_caller=_OPERATOR
    )
    triage.review_triage(
        proposal["proposal_id"],
        review=_pass_review(),
        operator_caller=_OPERATOR,
    )
    triage.apply_triage(
        proposal["proposal_id"],
        repo_root=repo,
        validators=["python_syntax"],
    )

    assert plane.store.read("requests", request_id) == immutable_before
    state_record = plane.store.read("request_states", request_id)
    assert state_record["triage_status"] == "applied"
    assert state_record["linked_task_id"]


def test_triage_needs_planning_creates_parent_not_leaf(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    triage = TriageService(plane)
    proposal = _new_proposal()
    proposal["proposed_task_change"]["plan_kind"] = "needs_planning"
    proposal["proposed_task_change"]["artifact_path"] = None
    proposal["proposed_task_change"]["title"] = "Improve logging subsystem"
    proposal["proposed_task_change"]["summary"] = (
        "Improve logging across multiple files"
    )
    proposal["proposed_task_change"]["objective"] = (
        "Improve logging across multiple modules"
    )
    proposal["proposed_task_change"]["added_requirements"] = [
        "Structured logs",
        "No cloud providers",
    ]
    outcome = triage.triage_request(
        request_text="Improve logging across the project.",
        repo_root=repo,
        proposal=proposal,
        review=_pass_review(),
        operator_caller=_OPERATOR,
    )
    assert outcome["applied"] is True
    plan = plane.get_plan(outcome["node_id"])
    assert plan["kind"] == "parent"
    assert plan["writable_artifact"] is None
    task = plane.get_task(outcome["task_id"])
    assert task["status"] == "planning"
    assert task.get("needs_planning") is True
    # Parent not on leaf ready queue.
    assert outcome["task_id"] not in [
        t["task_id"] for t in plane.ready_queue()
    ]


def test_triage_duplicate_links_existing_task(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    existing = plane.submit_request(
        request_text="Make the target value two.",
        objective="Set value to 2.",
        artifact_path="target.py",
        repo_root=repo,
        validators=["python_syntax"],
    )
    triage = TriageService(plane)
    proposal = _new_proposal(
        relationship={
            "type": "DUPLICATE_OF",
            "target_task_id": existing["task_id"],
            "rationale": "same leaf outcome",
        }
    )
    outcome = triage.triage_request(
        request_text="Please set target.py value to 2 again.",
        repo_root=repo,
        proposal=proposal,
        review=_pass_review(),
        operator_caller=_OPERATOR,
    )
    assert outcome["applied"] is True
    assert outcome["task_id"] == existing["task_id"]
    task = plane.get_task(existing["task_id"])
    assert outcome["request_id"] in task["linked_request_ids"]


def test_triage_overlaps_versions_spec_and_may_mark_stale(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    existing = plane.submit_request(
        request_text="Make the target value two.",
        objective="Set value to 2.",
        artifact_path="target.py",
        repo_root=repo,
        validators=["python_syntax"],
    )
    triage = TriageService(plane)
    proposal = _new_proposal(
        relationship={
            "type": "OVERLAPS_WITH",
            "target_task_id": existing["task_id"],
            "rationale": "extends acceptance",
        }
    )
    proposal["proposed_task_change"]["added_requirements"] = [
        "Also keep comments intact"
    ]
    outcome = triage.triage_request(
        request_text="Set value to 2 and keep comments.",
        repo_root=repo,
        proposal=proposal,
        review=_pass_review(),
        operator_caller=_OPERATOR,
    )
    assert outcome["task_id"] == existing["task_id"]
    task = plane.get_task(existing["task_id"])
    assert task["current_spec_version"] == 2
    assert task["status"] == "stale"
    spec = plane.get_task_spec(task["current_spec_id"])
    assert any("comments" in str(r.get("text")) for r in spec["requirements"])


def test_triage_review_retry_does_not_apply(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    triage = TriageService(plane)
    outcome = triage.triage_request(
        request_text="Ambiguous multi-part work.",
        repo_root=repo,
        proposal=_new_proposal(),
        review={
            "verdict": "RETRY_TRIAGE",
            "evidence": ["relationship unsupported"],
            "notes": "try again",
        },
        operator_caller=_OPERATOR,
    )
    assert outcome["applied"] is False
    assert outcome["verdict"] == "RETRY_TRIAGE"
    assert plane.list_tasks() == []


def test_model_backed_triage_with_fake_client(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    client = FakeTriageClient()
    triage = TriageService(plane)
    receipt = triage.ingest_request(request_text="Make the target value two.")
    proposed = triage.propose_triage(
        receipt["request_id"], client=client, model="local-model"
    )
    assert proposed["operator_supplied"] is False
    assert proposed["generation"]["content_sha256"]
    assert any(c["role"] == "triage_agent" for c in client.calls)
    # Independent review uses a second call.
    reviewed = triage.review_triage(
        proposed["proposal_id"], client=client, model="local-model"
    )
    assert reviewed["verdict"] == "PASS"
    assert reviewed["operator_supplied"] is False
    assert len(client.calls) == 2
    applied = triage.apply_triage(
        proposed["proposal_id"],
        repo_root=repo,
        validators=["python_syntax"],
    )
    assert applied["task_id"]
    # Reviewer prompt must include the original request (full context).
    review_prompt = client.calls[1]["prompt"]
    assert "Make the target value two" in review_prompt
    assert proposed["proposal_id"] in review_prompt


def test_chat_submit_request_broad_ingests_only(tmp_path):
    plane, _repo, _state = _plane(tmp_path)
    dispatcher = ToolDispatcher(plane)
    result = dispatcher.execute(
        "submit_request",
        {
            "request_text": "Refactor the whole parser subsystem carefully.",
            "desired_outcome": "Safer parser",
            "requested_priority": "high",
        },
    )
    assert result.ok is True
    assert result.result["request_id"].startswith("REQ-")
    assert result.result["triage_status"] == "pending"
    assert plane.list_tasks() == []


def test_chat_triage_tools_operator_path(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    dispatcher = ToolDispatcher(plane)
    ingested = dispatcher.execute(
        "ingest_request",
        {"request_text": "Make the target value two."},
    )
    assert ingested.ok is True
    proposed = dispatcher.execute(
        "propose_triage",
        {
            "request_id": ingested.result["request_id"],
            "proposal": _new_proposal(),
        },
    )
    assert proposed.ok is True
    reviewed = dispatcher.execute(
        "review_triage",
        {
            "proposal_id": proposed.result["proposal_id"],
            "review": _pass_review(),
        },
    )
    assert reviewed.ok is True
    applied = dispatcher.execute(
        "apply_triage",
        {
            "proposal_id": proposed.result["proposal_id"],
            "repo_root": str(repo),
            "validators": ["python_syntax"],
        },
    )
    assert applied.ok is True
    listed = dispatcher.execute("list_triage_proposals", {})
    assert listed.ok is True
    assert len(listed.result) >= 1


def test_scheduler_tick_respects_concurrency(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    a = plane.submit_request(
        request_text="A",
        objective="Set value to 2.",
        artifact_path="target.py",
        repo_root=repo,
        priority_components={
            "urgency": 1.0,
            "impact": 1.0,
            "freshness": 0.0,
            "operator_boost": 0.0,
            "blocked_penalty": 0.0,
            "dependency_penalty": 0.0,
        },
    )
    b = plane.submit_request(
        request_text="B",
        objective="Set other value.",
        artifact_path="other.py",
        repo_root=repo,
        priority_components={
            "urgency": 0.2,
            "impact": 0.2,
            "freshness": 0.0,
            "operator_boost": 0.0,
            "blocked_penalty": 0.0,
            "dependency_penalty": 0.0,
        },
    )
    tick = plane.scheduler_tick(max_concurrent=1, limit=5)
    assert tick["capacity"] == 1
    assert tick["selected_task_ids"] == [a["task_id"]]
    # Lease a task so capacity drops.
    lease = plane.acquire_lease(a["task_id"], owner="worker-1", ttl_seconds=60)
    tick2 = plane.scheduler_tick(max_concurrent=1, limit=5)
    assert tick2["capacity"] == 0
    assert tick2["selected_task_ids"] == []
    plane.release_lease(lease["lease_id"], owner="worker-1")
    # After release, lower-priority b can be selected if a is running.
    plane.transition_task(a["task_id"], "queued", reason="t")
    plane.transition_task(a["task_id"], "running", reason="t")
    tick3 = plane.scheduler_tick(max_concurrent=2, limit=5)
    assert b["task_id"] in tick3["selected_task_ids"]
    assert a["task_id"] not in tick3["selected_task_ids"]


def test_priority_spec_components_and_aging():
    from cosmo.control import (
        compute_age_component,
        effective_priority_score,
        score_priority_components,
    )

    scored = score_priority_components(
        {
            "user_importance": 50,  # 0-100 scale
            "urgency": 0.5,
            "estimated_effort": 0.2,
        }
    )
    assert scored["components"]["user_importance"] == 0.5
    assert "deadline_pressure" in scored["components"]
    aged = compute_age_component(
        "2026-07-01T00:00:00+00:00",
        now="2026-08-01T00:00:00+00:00",
        base_age=0.0,
    )
    assert aged > 0.0
    effective = effective_priority_score(
        {"user_importance": 0.5, "urgency": 0.5, "age": 0.0},
        created_at="2026-07-01T00:00:00+00:00",
        now="2026-08-01T00:00:00+00:00",
    )
    assert effective["age_component"] > 0.0
    assert effective["score"] > scored["score"] or effective["aged"] is True


def test_parse_triage_proposal_requires_relationship():
    raw = parse_triage_proposal(json.dumps(_new_proposal()))
    assert raw["relationship"]["type"] == "NEW"
    assert raw["proposal_sha256"]
    for malformed_wrapper in (
        "Here is the triage: " + json.dumps(_new_proposal()),
        json.dumps(_new_proposal()) + "\nThanks.",
        json.dumps(_new_proposal()) + "\n{}",
    ):
        with pytest.raises(TriageError, match="one JSON object"):
            parse_triage_proposal(malformed_wrapper)
