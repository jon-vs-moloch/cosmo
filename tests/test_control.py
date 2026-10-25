"""Focused tests for the Cosmo control-plane facade."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cosmo.atomic import AtomicHarnessError, AtomicStore
from cosmo.control import (
    ControlPlane,
    ControlPlaneError,
    LEGAL_TRANSITIONS,
    score_priority_components,
)
from cosmo.models import Generation


def _review(verdict="PASS"):
    return json.dumps(
        {
            "verdict": verdict,
            "requirement_results": [
                {
                    "requirement_id": "objective",
                    "result": "PASS" if verdict == "PASS" else "FAIL",
                    "evidence": ["validation:declared-checks"],
                }
            ],
            "observations": ["candidate differs from baseline assignment"],
            "retained_successes": ["candidate is syntactically valid"],
            "hypotheses": ["assignment was the only blocker"],
            "invalidated_assumptions": ["value was already two"],
            "recommended_state_changes": [],
            "recommended_next_action": "promote" if verdict == "PASS" else "retry",
        }
    )


class FakeAtomicClient:
    def __init__(self, candidate="value = 2\n", review=None):
        self.candidate = candidate
        self.review = review or _review()
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        content = self.candidate if kwargs["role"] == "atomic_worker" else self.review
        return Generation(
            role=kwargs["role"],
            content=content,
            reasoning_hash=None,
            stats={
                "input_tokens": 12,
                "total_output_tokens": 5,
                "reasoning_output_tokens": 0,
            },
            latency_seconds=0.01,
            request_hash=f"request-{len(self.calls)}",
            model_instance_id=kwargs["model"],
        )


def _plane(tmp_path: Path) -> tuple[ControlPlane, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "target.py").write_text("value = 1\n", encoding="utf-8")
    state = tmp_path / "state"
    return ControlPlane(AtomicStore(state)), repo, state


def _submit_leaf(plane: ControlPlane, repo: Path, **kwargs):
    payload = {
        "request_text": "Make the target value two.",
        "objective": "Set value to 2.",
        "artifact_path": "target.py",
        "repo_root": repo,
        "validators": ["python_syntax"],
        "priority_components": {
            "urgency": 0.5,
            "impact": 0.5,
            "freshness": 1.0,
            "operator_boost": 0.0,
            "blocked_penalty": 0.0,
            "dependency_penalty": 0.0,
        },
    }
    payload.update(kwargs)
    return plane.submit_request(**payload)


# ---------------------------------------------------------------------------
# Request / task creation
# ---------------------------------------------------------------------------


def test_submit_request_creates_immutable_request_task_spec_and_triage(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    result = _submit_leaf(
        plane,
        repo,
        triage_proposal={
            "mode": "operator_supplied_leaf_contract",
            "notes": "explicit operator triage",
        },
        triage_review={
            "verdict": "PASS",
            "independent_reviewer": "operator_test",
            "rationale": "leaf contract is sufficient",
        },
        relationship={"type": "NEW", "target_task_id": None},
    )
    assert result["status"] == "ready"
    request = plane.get_request(result["request_id"])
    assert request["original_text"] == "Make the target value two."
    assert request["immutable_sha256"]
    task = plane.get_task(result["task_id"])
    assert task["status"] == "ready"
    assert task["current_spec_version"] == 1
    assert task["priority_score"] is not None
    spec = plane.get_task_spec(result["spec_id"])
    assert spec["version"] == 1
    assert spec["artifact_path"] == "target.py"
    plan = plane.get_plan(result["node_id"])
    assert plan["kind"] == "leaf"
    assert plan["writable_artifact"] == "target.py"
    assert len(plan["writable_artifact"].split("/")) >= 1
    triage = plane.store.read("triage_proposals", result["triage_proposal_id"])
    assert triage["operator_supplied"] is True
    review = plane.get_review(result["triage_review_id"])
    assert review["verdict"] == "PASS"
    rel = plane.store.read("relationships", result["relationship_id"])
    assert rel["type"] == "NEW"
    prio = plane.get_priority_assessment(result["priority_assessment_id"])
    assert "breakdown" in prio
    assert set(prio["breakdown"]) >= {
        "urgency",
        "impact",
        "freshness",
        "operator_boost",
        "blocked_penalty",
        "dependency_penalty",
    }


def test_submit_request_rejects_failed_triage_review(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    with pytest.raises(ControlPlaneError, match="triage review PASS"):
        _submit_leaf(
            plane,
            repo,
            triage_review={"verdict": "REJECT", "independent_reviewer": "op"},
        )


def test_list_and_get_accessors(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    result = _submit_leaf(plane, repo)
    assert any(item["task_id"] == result["task_id"] for item in plane.list_tasks())
    assert any(
        item["request_id"] == result["request_id"] for item in plane.list_requests()
    )
    assert any(item["spec_id"] == result["spec_id"] for item in plane.list_task_specs())
    assert any(item["node_id"] == result["node_id"] for item in plane.list_plans())
    assert plane.list_ledger_records()
    snapshot = plane.to_json_dict()
    assert snapshot["schema_version"] == "cosmo.control-plane.v1"
    assert json.loads(json.dumps(snapshot))  # fully JSON-serializable


def test_create_task_facade_matches_control_plane_leaf_path(tmp_path):
    """create_task is a leaf facade over submit_request, not harness.submit."""
    plane, repo, _state = _plane(tmp_path)
    result = plane.create_task(
        request_text="Facade leaf.",
        objective="Set value to 2.",
        artifact_path="target.py",
        repo_root=repo,
        validators=["python_syntax"],
    )
    task = plane.get_task(result["task_id"])
    assert task["status"] == "ready"
    assert task["priority_score"] is not None
    assert task["priority_assessment_id"]
    assert result["plan_kind"] == "leaf"
    assert any(item["task_id"] == result["task_id"] for item in plane.ready_queue())
    lease = plane.acquire_lease(result["task_id"], owner="facade-owner", ttl_seconds=30)
    plane.release_lease(lease["lease_id"], owner="facade-owner")


def test_record_chat_tool_event_is_append_only(tmp_path):
    plane, _repo, state = _plane(tmp_path)
    recorded = plane.record_chat_tool_event(
        tool="list_tasks",
        ok=True,
        operation_id="chat/1/tool/list_tasks",
        task_id=None,
        attempt_id=None,
    )
    assert recorded["tool"] == "list_tasks"
    assert recorded["ok"] is True
    assert recorded["operation_id"] == "chat/1/tool/list_tasks"
    assert "prompt" not in recorded
    events = plane.store.list_records("chat_events")
    assert any(item.get("chat_event_id") == recorded["chat_event_id"] for item in events)
    lines = (state / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert any('"chat_tool_call"' in line for line in lines)


# ---------------------------------------------------------------------------
# Legal transitions
# ---------------------------------------------------------------------------


def test_legal_transitions_and_append_only_events(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    result = _submit_leaf(plane, repo)
    task_id = result["task_id"]
    plane.transition_task(task_id, "planning", reason="operator planning")
    plane.transition_task(task_id, "ready", reason="plan approved")
    plane.transition_task(task_id, "queued", reason="scheduled")
    plane.transition_task(task_id, "running", reason="worker started")
    plane.transition_task(task_id, "blocked", reason="waiting on input")
    plane.transition_task(task_id, "ready", reason="unblocked")
    plane.transition_task(task_id, "stale", reason="baseline moved")
    plane.transition_task(task_id, "canceled", reason="abandoned")
    task = plane.get_task(task_id)
    assert task["status"] == "canceled"
    events = [
        item
        for item in plane.list_ledger_records()
        if item.get("type") == "task_status_transition"
        and item.get("task_id") == task_id
    ]
    assert len(events) >= 8
    assert events[-1]["to_status"] == "canceled"


def test_illegal_transition_rejected(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    result = _submit_leaf(plane, repo)
    with pytest.raises(ControlPlaneError, match="illegal task transition"):
        plane.transition_task(result["task_id"], "completed")
    # Terminal statuses have no outbound edges.
    plane.transition_task(result["task_id"], "canceled")
    with pytest.raises(ControlPlaneError, match="illegal task transition"):
        plane.transition_task(result["task_id"], "ready")
    assert "completed" not in LEGAL_TRANSITIONS["canceled"]


# ---------------------------------------------------------------------------
# Priority scoring and ready queue ordering
# ---------------------------------------------------------------------------


def test_priority_scoring_is_explainable():
    scored = score_priority_components(
        {
            "urgency": 1.0,
            "impact": 0.5,
            "freshness": 0.0,
            "operator_boost": 0.25,
            "blocked_penalty": 0.0,
            "dependency_penalty": 0.0,
        }
    )
    assert scored["score"] == round(4.0 + 1.5 + 0.0 + 0.5, 6)
    assert scored["breakdown"]["urgency"]["contribution"] == 4.0
    with pytest.raises(ControlPlaneError, match="unknown priority component"):
        score_priority_components({"mystery": 1.0})


def test_ready_queue_orders_by_score_then_created_then_id(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    low = _submit_leaf(
        plane,
        repo,
        priority_components={
            "urgency": 0.1,
            "impact": 0.1,
            "freshness": 0.0,
            "operator_boost": 0.0,
            "blocked_penalty": 0.0,
            "dependency_penalty": 0.0,
        },
    )
    high = _submit_leaf(
        plane,
        repo,
        request_text="High priority request.",
        objective="Set value to 2 with high urgency.",
        priority_components={
            "urgency": 1.0,
            "impact": 1.0,
            "freshness": 1.0,
            "operator_boost": 1.0,
            "blocked_penalty": 0.0,
            "dependency_penalty": 0.0,
        },
    )
    mid = _submit_leaf(
        plane,
        repo,
        request_text="Mid priority request.",
        objective="Set value to 2 with medium urgency.",
        priority_components={
            "urgency": 0.6,
            "impact": 0.6,
            "freshness": 0.5,
            "operator_boost": 0.0,
            "blocked_penalty": 0.0,
            "dependency_penalty": 0.0,
        },
    )
    queue = plane.ready_queue()
    ids = [item["task_id"] for item in queue]
    assert ids.index(high["task_id"]) < ids.index(mid["task_id"]) < ids.index(
        low["task_id"]
    )
    # Non-ready tasks are excluded.
    plane.transition_task(mid["task_id"], "canceled")
    queue2 = plane.ready_queue()
    assert mid["task_id"] not in [item["task_id"] for item in queue2]


# ---------------------------------------------------------------------------
# Exclusive leases
# ---------------------------------------------------------------------------


def test_lease_acquire_renew_release_and_exclusivity(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    result = _submit_leaf(plane, repo)
    task_id = result["task_id"]
    now = datetime.now(timezone.utc).isoformat()
    lease = plane.acquire_lease(task_id, owner="worker-a", ttl_seconds=60, now=now)
    assert lease["status"] == "active"
    assert lease["owner"] == "worker-a"
    assert lease["expires_at"]
    with pytest.raises(ControlPlaneError, match="already leased"):
        plane.acquire_lease(task_id, owner="worker-b", ttl_seconds=60, now=now)
    # Same owner renews instead of conflicting.
    renewed = plane.acquire_lease(
        task_id, owner="worker-a", ttl_seconds=120, now=now
    )
    assert renewed["lease_id"] == lease["lease_id"]
    renewed2 = plane.renew_lease(
        lease["lease_id"], owner="worker-a", ttl_seconds=90, now=now
    )
    assert renewed2["ttl_seconds"] == 90
    with pytest.raises(ControlPlaneError, match="owned by"):
        plane.renew_lease(lease["lease_id"], owner="worker-b")
    released = plane.release_lease(lease["lease_id"], owner="worker-a", now=now)
    assert released["status"] == "released"
    # After release another owner may acquire.
    other = plane.acquire_lease(task_id, owner="worker-b", ttl_seconds=30, now=now)
    assert other["owner"] == "worker-b"


def test_stale_lease_expiry(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    result = _submit_leaf(plane, repo)
    past = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    lease = plane.acquire_lease(
        result["task_id"], owner="worker-a", ttl_seconds=1, now=past
    )
    # Force expiry timestamp into the past.
    lease["expires_at"] = past
    plane.store.replace("leases", lease["lease_id"], lease)
    expired = plane.expire_stale_leases(
        now=datetime.now(timezone.utc).isoformat()
    )
    assert any(item["lease_id"] == lease["lease_id"] for item in expired)
    assert plane.get_lease(lease["lease_id"])["status"] == "expired"
    # Task re-enters the ready queue once the stale lease is cleared.
    assert any(
        item["task_id"] == result["task_id"] for item in plane.ready_queue()
    )


# ---------------------------------------------------------------------------
# Plan graph and dependency-aware readiness
# ---------------------------------------------------------------------------


def test_parent_leaf_plan_graph_and_dependencies(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    parent = plane.submit_request(
        request_text="Coordinate two leaves.",
        objective="Parent coordination objective.",
        artifact_path="",
        repo_root=repo,
        plan_kind="parent",
        priority_components={
            "urgency": 0.2,
            "impact": 0.2,
            "freshness": 1.0,
            "operator_boost": 0.0,
            "blocked_penalty": 0.0,
            "dependency_penalty": 0.0,
        },
    )
    assert parent["plan_kind"] == "parent"
    parent_plan = plane.get_plan(parent["node_id"])
    assert parent_plan["kind"] == "parent"
    assert parent_plan["writable_artifact"] is None

    dep = _submit_leaf(
        plane,
        repo,
        request_text="Dependency leaf.",
        objective="Finish dependency first.",
        parent_node_id=parent["node_id"],
        priority_components={
            "urgency": 0.4,
            "impact": 0.4,
            "freshness": 1.0,
            "operator_boost": 0.0,
            "blocked_penalty": 0.0,
            "dependency_penalty": 0.0,
        },
    )
    blocked = _submit_leaf(
        plane,
        repo,
        request_text="Depends on other leaf.",
        objective="Wait for dependency.",
        parent_node_id=parent["node_id"],
        dependencies=[dep["node_id"]],
        priority_components={
            "urgency": 0.9,
            "impact": 0.9,
            "freshness": 1.0,
            "operator_boost": 0.0,
            "blocked_penalty": 0.0,
            "dependency_penalty": 1.0,
        },
    )
    assert blocked["status"] == "blocked"
    assert plane.get_task(blocked["task_id"])["status"] == "blocked"
    # High priority but blocked must not appear in the ready queue.
    ready_ids = [item["task_id"] for item in plane.ready_queue()]
    assert blocked["task_id"] not in ready_ids
    assert dep["task_id"] in ready_ids

    parent_plan = plane.get_plan(parent["node_id"])
    assert dep["node_id"] in parent_plan["child_node_ids"]
    assert blocked["node_id"] in parent_plan["child_node_ids"]

    # Completing the dependency unblocks the dependent task.
    plane.transition_task(dep["task_id"], "queued")
    plane.transition_task(dep["task_id"], "running")
    plane.transition_task(dep["task_id"], "completed")
    refreshed = plane.refresh_dependency_readiness(blocked["task_id"])
    assert refreshed["status"] == "ready"
    assert blocked["task_id"] in [
        item["task_id"] for item in plane.ready_queue()
    ]

    # Leaf plans must declare exactly one writable artifact.
    leaf = plane.get_plan(dep["node_id"])
    assert leaf["writable_artifact"] == "target.py"
    assert leaf["kind"] == "leaf"
    plan_reviews = [
        item
        for item in plane.list_reviews()
        if item.get("node_id") in {dep["node_id"], blocked["node_id"], parent["node_id"]}
    ]
    assert plan_reviews


def test_cannot_mark_ready_with_unmet_dependencies(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    dep = _submit_leaf(plane, repo, request_text="Dep.", objective="Dep objective.")
    blocked = _submit_leaf(
        plane,
        repo,
        request_text="Blocked.",
        objective="Blocked objective.",
        dependencies=[dep["node_id"]],
    )
    assert plane.get_task(blocked["task_id"])["status"] == "blocked"
    with pytest.raises(ControlPlaneError, match="dependencies are unmet"):
        plane.transition_task(blocked["task_id"], "ready")


# ---------------------------------------------------------------------------
# Ledger projection and convergence assessments
# ---------------------------------------------------------------------------


def test_ledger_projection_and_convergence_assessment(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    result = _submit_leaf(plane, repo)
    # Seed attempt/validation/review records via AtomicHarness so projection
    # reads real durable kinds without inventing a second store.
    client = FakeAtomicClient(candidate="value = 2\n", review=_review("REVISE"))
    plane.transition_task(result["task_id"], "queued")
    plane.transition_task(result["task_id"], "running")
    attempt_result = plane.harness.run_attempt(
        task_id=result["task_id"],
        repo_root=repo,
        client=client,
        model="local-model",
        reasoning="off",
        temperature=0,
        worker_max_output_tokens=100,
        review_max_output_tokens=100,
        strategy_summary="replace the assignment with value 2",
    )
    assert attempt_result["promotable"] is False
    projection = plane.project_ledger(result["task_id"])
    assert projection["task_id"] == result["task_id"]
    assert projection["observations"]
    assert projection["hypotheses"]
    assert projection["retained_successes"]
    assert projection["invalidated_assumptions"]
    assert any(
        item["strategy_summary"] == "replace the assignment with value 2"
        for item in projection["rejected_strategies"]
    )
    loaded = plane.get_ledger_projection(projection["projection_id"])
    assert loaded["projection_id"] == projection["projection_id"]

    assessment = plane.record_convergence_assessment(
        result["task_id"],
        summary="Not yet converged; review requested revise.",
        converged=False,
        evidence={"attempt_id": attempt_result["attempt_id"]},
    )
    assert assessment["converged"] is False
    assert (
        plane.get_convergence_assessment(assessment["assessment_id"])["summary"]
        == assessment["summary"]
    )
    # Assessments are append-only: a second write is a new record.
    second = plane.record_convergence_assessment(
        result["task_id"],
        summary="Still open.",
        converged=False,
    )
    assert second["assessment_id"] != assessment["assessment_id"]


# ---------------------------------------------------------------------------
# Dispatch and promotion recovery safety
# ---------------------------------------------------------------------------


def test_dispatch_ready_leaf_uses_harness_without_product_mutation(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    result = _submit_leaf(plane, repo)
    client = FakeAtomicClient()
    dispatched = plane.dispatch_ready_leaf(
        result["task_id"],
        repo_root=repo,
        client=client,
        model="local-model",
        strategy_summary="replace the assignment with the requested value",
        owner="dispatcher-1",
        worker_max_output_tokens=100,
        review_max_output_tokens=100,
    )
    assert dispatched["promotable"] is True
    assert dispatched["lease_id"]
    assert (repo / "target.py").read_text(encoding="utf-8") == "value = 1\n"
    assert [call["role"] for call in client.calls] == [
        "atomic_worker",
        "atomic_reviewer",
    ]
    # Lease released after dispatch.
    assert plane.get_lease(dispatched["lease_id"])["status"] == "released"


def test_recover_promotion_requires_validation_and_review(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    result = _submit_leaf(plane, repo)
    client = FakeAtomicClient()
    dispatched = plane.dispatch_ready_leaf(
        result["task_id"],
        repo_root=repo,
        client=client,
        model="local-model",
        strategy_summary="replace the assignment",
        owner="dispatcher-1",
        worker_max_output_tokens=100,
        review_max_output_tokens=100,
    )
    # Non-promotable attempts cannot be promoted through the control facade.
    failing = plane.harness.run_attempt(
        task_id=result["task_id"],
        repo_root=repo,
        client=FakeAtomicClient(
            candidate="value =\n",
            review=_review("REVISE"),
        ),
        model="local-model",
        reasoning="off",
        temperature=0,
        worker_max_output_tokens=100,
        review_max_output_tokens=100,
        strategy_summary="broken syntax retry",
        retry_contract={
            "addresses": ["syntax"],
            "preserves": ["file path"],
            "differs_from": [dispatched["attempt_id"]],
            "expected_improvement": "valid syntax",
        },
    )
    assert failing["promotable"] is False
    with pytest.raises(AtomicHarnessError, match="validation PASS and review PASS"):
        plane.recover_promotion(attempt_id=failing["attempt_id"], repo_root=repo)

    promotion = plane.recover_promotion(
        attempt_id=dispatched["attempt_id"], repo_root=repo
    )
    assert promotion["promoted_sha256"]
    assert (repo / "target.py").read_text(encoding="utf-8") == "value = 2\n"
    assert plane.get_task(result["task_id"])["status"] == "completed"


def test_dispatch_rejects_non_ready_and_parent_plans(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    parent = plane.submit_request(
        request_text="Parent only.",
        objective="Coordinate.",
        artifact_path="",
        repo_root=repo,
        plan_kind="parent",
    )
    with pytest.raises(ControlPlaneError, match="leaf plan"):
        plane.dispatch_ready_leaf(
            parent["task_id"],
            repo_root=repo,
            client=FakeAtomicClient(),
            model="local-model",
            strategy_summary="n/a",
            owner="dispatcher-1",
        )
    leaf = _submit_leaf(plane, repo)
    plane.transition_task(leaf["task_id"], "canceled")
    with pytest.raises(ControlPlaneError, match="ready or queued"):
        plane.dispatch_ready_leaf(
            leaf["task_id"],
            repo_root=repo,
            client=FakeAtomicClient(),
            model="local-model",
            strategy_summary="n/a",
            owner="dispatcher-1",
        )


def test_unsafe_record_identifiers_rejected(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    result = _submit_leaf(plane, repo)
    with pytest.raises(AtomicHarnessError, match="invalid atomic record identifier"):
        plane.get_task("../escape")
    with pytest.raises(AtomicHarnessError, match="invalid atomic record identifier"):
        plane.transition_task(result["task_id"] + "/../x", "queued")
