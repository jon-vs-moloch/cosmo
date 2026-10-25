"""Focused tests for expanded Cosmo tool catalog and dispatch handlers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosmo.atomic import AtomicHarness, AtomicStore
from cosmo.control import ControlPlane
from cosmo.models import Generation
from cosmo.tools import (
    TOOL_DEFINITIONS_BY_NAME,
    TOOL_MAX_MAP_ENTRIES,
    TOOL_MAX_READ_BYTES,
    TOOL_MAX_SEARCH_MATCHES,
    TOOL_MAX_TREE_ENTRIES,
    ToolDispatcher,
    tool_definitions_json,
    validate_tool_arguments,
)


def _review(verdict="PASS"):
    return json.dumps(
        {
            "verdict": verdict,
            "requirement_results": [
                {
                    "requirement_id": "objective",
                    "result": "PASS",
                    "evidence": ["validation:declared-checks"],
                }
            ],
            "observations": [],
            "retained_successes": ["ok"],
            "hypotheses": [],
            "invalidated_assumptions": [],
            "recommended_state_changes": [],
            "recommended_next_action": "promote",
        }
    )


def _plan_proposal():
    return json.dumps(
        {
            "objective": "Compose two modules",
            "requirements": ["R1"],
            "parent_validators": ["python_syntax"],
            "children": [
                {
                    "temp_id": "c1",
                    "kind": "leaf",
                    "objective": "Write a.py",
                    "writable_artifact": "a.py",
                    "depends_on": [],
                    "validators": ["python_syntax"],
                    "requirements": ["R1"],
                },
                {
                    "temp_id": "c2",
                    "kind": "leaf",
                    "objective": "Write b.py",
                    "writable_artifact": "b.py",
                    "depends_on": ["c1"],
                    "validators": ["python_syntax"],
                },
            ],
            "risks": [],
            "unresolved_questions": [],
        }
    )


def _plan_review(verdict="PASS"):
    return json.dumps(
        {
            "verdict": verdict,
            "evidence": ["covers requirements", "one artifact per leaf"],
            "notes": "ok",
            "requirement_coverage": ["R1"],
            "recommended_next_action": "apply",
        }
    )


class ScriptedClient:
    def __init__(self):
        self.calls = []
        self.atomic_candidate = "A = 1\n"
        self.plan_mode = "propose"

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        role = kwargs.get("role")
        if role == "atomic_worker":
            content = self.atomic_candidate
        elif role == "atomic_reviewer":
            content = _review()
        elif role == "planner":
            content = _plan_proposal()
        elif role == "plan_reviewer":
            content = _plan_review()
        else:
            content = "ok"
        return Generation(
            role=str(role or "chat"),
            content=content,
            reasoning_hash=None,
            stats={
                "input_tokens": 8,
                "total_output_tokens": 4,
                "reasoning_output_tokens": 0,
            },
            latency_seconds=0.01,
            request_hash=f"request-{len(self.calls)}",
            model_instance_id=kwargs.get("model"),
        )


def _repo(tmp_path: Path) -> tuple[Path, Path, ControlPlane, ToolDispatcher, ScriptedClient]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("A = 0\n", encoding="utf-8")
    (repo / "b.py").write_text("B = 0\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    state = tmp_path / "state"
    plane = ControlPlane(AtomicStore(state))
    client = ScriptedClient()
    dispatcher = ToolDispatcher(plane, client=client)
    return repo, state, plane, dispatcher, client


def test_tool_catalog_includes_new_surfaces_and_no_shell_claims():
    names = set(TOOL_DEFINITIONS_BY_NAME)
    for required in (
        "repository_map",
        "list_tree",
        "read_artifact",
        "search_text",
        "inspect_python_symbols",
        "build_planning_pack",
        "propose_plan",
        "review_plan",
        "list_plan_versions",
        "get_plan_version",
        "apply_approved_plan",
        "list_integrations",
        "get_integration",
        "assemble_integration",
        "review_integration",
        "promote_integration",
        "reassess_priority",
        "dispatch_scheduler_batch",
        "refresh_dependency_readiness",
        "inspect_lifecycle",
        "scheduler_tick",
        "ready_queue",
    ):
        assert required in names
    catalog = tool_definitions_json()
    blob = json.dumps(catalog).lower()
    assert "no shell" in blob or "shell" in blob
    # Optional read_only_paths on lifecycle tools.
    for name in ("dispatch_ready", "run_attempt", "retry_attempt"):
        props = TOOL_DEFINITIONS_BY_NAME[name].to_json_dict()["parameters"][
            "properties"
        ]
        assert "read_only_paths" in props


def test_dispatch_tools_default_to_dispatcher_model(tmp_path):
    repo, _state, plane, _dispatcher, client = _repo(tmp_path)
    dispatcher = ToolDispatcher(plane, client=client, model="local-model")
    submitted = plane.submit_request(
        request_text="Set A to one.",
        objective="Set A to one.",
        artifact_path="a.py",
        repo_root=repo,
        validators=["python_syntax"],
    )

    result = dispatcher.execute(
        "run_attempt",
        {
            "task_id": submitted["task_id"],
            "repo_root": str(repo),
            "strategy_summary": "set A conservatively",
            "owner": "chat",
        },
    )

    assert result.ok, result.error
    assert result.result["promotable"] is True
    assert all(call.get("model") == "local-model" for call in client.calls)


def test_repository_inspection_tools_bounded_and_json(tmp_path):
    repo, _state, _plane, dispatcher, _client = _repo(tmp_path)
    mapped = dispatcher.execute(
        "repository_map", {"repo_root": str(repo)}
    )
    assert mapped.ok
    assert "map_sha256" in mapped.result or "files" in mapped.result
    tree = dispatcher.execute(
        "list_tree", {"repo_root": str(repo), "prefix": "src"}
    )
    assert tree.ok
    assert any(e["path"] == "src/mod.py" for e in tree.result["entries"])
    read = dispatcher.execute(
        "read_artifact", {"repo_root": str(repo), "path": "src/mod.py"}
    )
    assert read.ok and "VALUE" in read.result["excerpt"]
    search = dispatcher.execute(
        "search_text", {"repo_root": str(repo), "query": "VALUE"}
    )
    assert search.ok
    symbols = dispatcher.execute(
        "inspect_python_symbols",
        {"repo_root": str(repo), "path": "src/mod.py"},
    )
    assert symbols.ok
    # Unsafe path fails closed.
    bad = dispatcher.execute(
        "read_artifact",
        {"repo_root": str(repo), "path": "../outside.py"},
    )
    assert bad.ok is False
    # Non-directory root.
    missing = dispatcher.execute(
        "list_tree", {"repo_root": str(tmp_path / "nope")}
    )
    assert missing.ok is False


def test_dispatcher_rejects_over_cap_repository_limits_before_inspection(
    tmp_path, monkeypatch
):
    """Huge max_* from the chat surface must fail closed before FS work."""
    repo, _state, _plane, dispatcher, _client = _repo(tmp_path)

    calls: list[str] = []

    class _NoCallInspector:
        def __init__(self, *args, **kwargs):
            calls.append("construct")
            raise AssertionError("RepositoryInspector must not be constructed")

    monkeypatch.setattr("cosmo.tools.RepositoryInspector", _NoCallInspector)

    over_map = dispatcher.execute(
        "repository_map",
        {"repo_root": str(repo), "max_map_entries": TOOL_MAX_MAP_ENTRIES + 1},
    )
    assert over_map.ok is False
    assert "max_map_entries" in over_map.error
    assert "cap" in over_map.error.lower() or str(TOOL_MAX_MAP_ENTRIES) in over_map.error

    over_tree = dispatcher.execute(
        "list_tree",
        {"repo_root": str(repo), "max_entries": TOOL_MAX_TREE_ENTRIES + 1},
    )
    assert over_tree.ok is False
    assert "max_entries" in over_tree.error

    over_read = dispatcher.execute(
        "read_artifact",
        {
            "repo_root": str(repo),
            "path": "src/mod.py",
            "max_bytes": TOOL_MAX_READ_BYTES + 1,
        },
    )
    assert over_read.ok is False
    assert "max_bytes" in over_read.error

    over_search = dispatcher.execute(
        "search_text",
        {
            "repo_root": str(repo),
            "query": "VALUE",
            "max_matches": TOOL_MAX_SEARCH_MATCHES + 1,
        },
    )
    assert over_search.ok is False
    assert "max_matches" in over_search.error

    zero = dispatcher.execute(
        "list_tree",
        {"repo_root": str(repo), "max_entries": 0},
    )
    assert zero.ok is False
    assert "positive" in zero.error.lower()

    # No inspector construction on any over-cap / non-positive rejection.
    assert calls == []


def test_planning_tools_use_local_client_and_never_return_inspector(tmp_path):
    repo, state, plane, dispatcher, client = _repo(tmp_path)
    submitted = plane.submit_request(
        request_text="Compose a and b",
        objective="Compose modules",
        artifact_path="",
        repo_root=repo,
        plan_kind="parent",
        priority_components={
            "urgency": 0.5,
            "impact": 0.5,
            "freshness": 1.0,
            "operator_boost": 0.0,
            "blocked_penalty": 0.0,
            "dependency_penalty": 0.0,
        },
    )
    pack = dispatcher.execute(
        "build_planning_pack",
        {
            "repo_root": str(repo),
            "task_id": submitted["task_id"],
            "selected_paths": ["a.py"],
        },
    )
    assert pack.ok
    assert "inspector" not in pack.result
    assert pack.result["pack"]["pack_sha256"]
    assert "context_manifest_id" in pack.result["manifest"]

    proposed = dispatcher.execute(
        "propose_plan",
        {
            "repo_root": str(repo),
            "model": "local-model",
            "task_id": submitted["task_id"],
        },
    )
    assert proposed.ok
    assert proposed.result["plan_version_id"]
    assert any(c["role"] == "planner" for c in client.calls)
    plan_version_id = proposed.result["plan_version_id"]

    reviewed = dispatcher.execute(
        "review_plan",
        {
            "plan_version_id": plan_version_id,
            "review": {
                "verdict": "PASS",
                "evidence": ["coverage ok", "leaf atomicity ok"],
                "notes": "operator review",
            },
        },
    )
    assert reviewed.ok
    assert reviewed.result["verdict"] == "PASS"
    assert reviewed.result["approved"] is True

    listed = dispatcher.execute("list_plan_versions", {})
    assert listed.ok and any(
        item["plan_version_id"] == plan_version_id for item in listed.result
    )
    got = dispatcher.execute(
        "get_plan_version", {"plan_version_id": plan_version_id}
    )
    assert got.ok and got.result["approved"] is True

    applied = dispatcher.execute(
        "apply_approved_plan",
        {"plan_version_id": plan_version_id, "repo_root": str(repo)},
    )
    assert applied.ok
    assert applied.result["parent_node_id"]
    assert applied.result["created"]

    # Refuse apply without approval (fresh proposal).
    client.calls.clear()
    second = dispatcher.execute(
        "propose_plan",
        {"repo_root": str(repo), "model": "local-model"},
    )
    assert second.ok
    refused = dispatcher.execute(
        "apply_approved_plan",
        {
            "plan_version_id": second.result["plan_version_id"],
            "repo_root": str(repo),
        },
    )
    assert refused.ok is False
    assert "unreviewed" in refused.error.lower() or "refuse" in refused.error.lower()


def test_integration_tools_expose_receipts_and_gate_promotion(tmp_path):
    repo, state, plane, dispatcher, client = _repo(tmp_path)
    parent = plane.submit_request(
        request_text="Compose",
        objective="Compose",
        artifact_path="",
        repo_root=repo,
        plan_kind="parent",
        validators=["python_syntax"],
        priority_components={
            "urgency": 0.5,
            "impact": 0.5,
            "freshness": 1.0,
            "operator_boost": 0.0,
            "blocked_penalty": 0.0,
            "dependency_penalty": 0.0,
        },
    )
    for path, content in (("a.py", "A = 1\n"), ("b.py", "B = 2\n")):
        leaf = plane.submit_request(
            request_text=f"set {path}",
            objective=f"set {path}",
            artifact_path=path,
            repo_root=repo,
            validators=["python_syntax"],
            plan_kind="leaf",
            parent_node_id=parent["node_id"],
            priority_components={
                "urgency": 0.5,
                "impact": 0.5,
                "freshness": 1.0,
                "operator_boost": 0.0,
                "blocked_penalty": 0.0,
                "dependency_penalty": 0.0,
            },
        )
        client.atomic_candidate = content
        AtomicHarness(plane.store).run_attempt(
            task_id=leaf["task_id"],
            repo_root=repo,
            client=client,
            model="local-model",
            reasoning="off",
            temperature=0,
            worker_max_output_tokens=100,
            review_max_output_tokens=100,
            strategy_summary=f"write {path}",
        )

    assembled = dispatcher.execute(
        "assemble_integration",
        {"parent_node_id": parent["node_id"], "repo_root": str(repo)},
    )
    assert assembled.ok
    run = assembled.result
    assert run["validation_passed"] is True
    assert run["baseline_tree_sha256"]
    assert run["workspace"]
    assert run["selected_children"]
    assert run["checks"]
    assert (repo / "a.py").read_text(encoding="utf-8") == "A = 0\n"

    listed = dispatcher.execute("list_integrations", {})
    assert listed.ok and listed.result
    got = dispatcher.execute(
        "get_integration", {"integration_id": run["integration_id"]}
    )
    assert got.ok and got.result["integration_id"] == run["integration_id"]

    refused = dispatcher.execute(
        "promote_integration",
        {"integration_id": run["integration_id"], "repo_root": str(repo)},
    )
    assert refused.ok is False

    reviewed = dispatcher.execute(
        "review_integration",
        {
            "integration_id": run["integration_id"],
            "verdict": "PASS",
            "notes": "composition ok",
        },
    )
    assert reviewed.ok
    assert reviewed.result["integration"]["promotable"] is True

    promoted = dispatcher.execute(
        "promote_integration",
        {"integration_id": run["integration_id"], "repo_root": str(repo)},
    )
    assert promoted.ok
    assert (repo / "a.py").read_text(encoding="utf-8") == "A = 1\n"
    assert promoted.result["promotion_id"]


def test_unknown_and_malformed_tools_fail_closed(tmp_path):
    _repo_path, _state, _plane, dispatcher, _client = _repo(tmp_path)
    unknown = dispatcher.execute("not_a_real_tool", {})
    assert unknown.ok is False and "unknown tool" in unknown.error
    malformed = dispatcher.execute("list_tasks", "not-an-object")
    assert malformed.ok is False
    missing = dispatcher.execute("get_task", {})
    assert missing.ok is False


def test_read_only_paths_passed_to_dispatch(tmp_path):
    repo, _state, plane, dispatcher, client = _repo(tmp_path)
    (repo / "contract.txt").write_text("must set A to 1\n", encoding="utf-8")
    leaf = plane.submit_request(
        request_text="set a",
        objective="set a",
        artifact_path="a.py",
        repo_root=repo,
        validators=["python_syntax"],
        plan_kind="leaf",
        priority_components={
            "urgency": 0.5,
            "impact": 0.5,
            "freshness": 1.0,
            "operator_boost": 0.0,
            "blocked_penalty": 0.0,
            "dependency_penalty": 0.0,
        },
    )
    client.atomic_candidate = "A = 1\n"
    result = dispatcher.execute(
        "run_attempt",
        {
            "task_id": leaf["task_id"],
            "repo_root": str(repo),
            "model": "local-model",
            "strategy_summary": "use contract",
            "owner": "tester",
            "worker_max_output_tokens": 100,
            "review_max_output_tokens": 100,
            "read_only_paths": ["contract.txt"],
        },
    )
    assert result.ok
    attempt = plane.get_attempt(result.result["attempt_id"])
    assert attempt["read_only_paths"] == ["contract.txt"]


def test_validate_tool_arguments_rejects_unknown_keys():
    definition = TOOL_DEFINITIONS_BY_NAME["get_task"]
    with pytest.raises(Exception):
        validate_tool_arguments(definition, {"task_id": "T1", "extra": 1})


def test_priority_scheduler_lifecycle_tools_are_reachable(tmp_path):
    """Priority reassessment, dep refresh, batch dispatch, lifecycle inspect."""
    repo, _state, plane, dispatcher, client = _repo(tmp_path)

    parent = plane.submit_request(
        request_text="Parent for deps",
        objective="Parent objective",
        artifact_path="",
        repo_root=repo,
        plan_kind="parent",
        priority_components={
            "urgency": 0.5,
            "impact": 0.5,
            "freshness": 1.0,
            "operator_boost": 0.0,
            "blocked_penalty": 0.0,
            "dependency_penalty": 0.0,
        },
    )
    dep = plane.submit_request(
        request_text="Dep leaf",
        objective="Dep leaf",
        artifact_path="a.py",
        repo_root=repo,
        validators=["python_syntax"],
        plan_kind="leaf",
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
    blocked = plane.submit_request(
        request_text="Blocked leaf",
        objective="Blocked leaf",
        artifact_path="b.py",
        repo_root=repo,
        validators=["python_syntax"],
        plan_kind="leaf",
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
    assert plane.get_task(blocked["task_id"])["status"] == "blocked"

    # reassess_priority: new assessment, no product write, score changes.
    prior_score = plane.get_task(dep["task_id"])["priority_score"]
    reassessed = dispatcher.execute(
        "reassess_priority",
        {
            "task_id": dep["task_id"],
            "priority_components": {
                "user_importance": 0.95,
                "urgency": 0.95,
                "deadline_pressure": 0.5,
                "dependency_unblocking": 0.0,
                "risk_reduction": 0.0,
                "age": 0.0,
                "estimated_effort": 0.1,
            },
        },
    )
    assert reassessed.ok, reassessed.error
    assert reassessed.result["assessment_id"]
    assert plane.get_task(dep["task_id"])["priority_assessment_id"] == (
        reassessed.result["assessment_id"]
    )
    assert plane.get_task(dep["task_id"])["priority_score"] != prior_score
    assert (repo / "a.py").read_text(encoding="utf-8") == "A = 0\n"

    # inspect_lifecycle: single task projection includes readiness gates.
    life = dispatcher.execute(
        "inspect_lifecycle", {"task_id": blocked["task_id"]}
    )
    assert life.ok, life.error
    assert life.result["task"]["task_id"] == blocked["task_id"]
    assert life.result["dispatchable"] is False
    assert life.result["dependencies_satisfied"] is False
    assert life.result["spec_is_current"] is True
    assert life.result["ready_queue_position"] is None

    # Plane snapshot without task_id.
    snap = dispatcher.execute("inspect_lifecycle", {})
    assert snap.ok
    assert "tasks" in snap.result
    assert "ready_queue" in snap.result

    # refresh_dependency_readiness: blocked stays blocked until dep completes.
    still = dispatcher.execute(
        "refresh_dependency_readiness", {"task_id": blocked["task_id"]}
    )
    assert still.ok
    assert still.result["status"] == "blocked"

    plane.transition_task(dep["task_id"], "queued")
    plane.transition_task(dep["task_id"], "running")
    plane.transition_task(dep["task_id"], "completed")

    refreshed = dispatcher.execute(
        "refresh_dependency_readiness", {"task_id": blocked["task_id"]}
    )
    assert refreshed.ok, refreshed.error
    assert refreshed.result["status"] == "ready"
    life_ready = dispatcher.execute(
        "inspect_lifecycle", {"task_id": blocked["task_id"]}
    )
    assert life_ready.ok
    assert life_ready.result["dependencies_satisfied"] is True
    assert life_ready.result["dispatchable"] is True

    # dispatch_scheduler_batch: tick + dispatch through validation+review.
    client.atomic_candidate = "B = 1\n"
    batch = dispatcher.execute(
        "dispatch_scheduler_batch",
        {
            "repo_root": str(repo),
            "model": "local-model",
            "owner": "batch-owner",
            "strategy_summary": "set B",
            "max_concurrent": 2,
            "limit": 1,
            "worker_max_output_tokens": 100,
            "review_max_output_tokens": 100,
        },
    )
    assert batch.ok, batch.error
    assert batch.result["scheduler_tick"]["selected_task_ids"]
    assert batch.result["dispatched"]
    assert (repo / "b.py").read_text(encoding="utf-8") == "B = 0\n"  # no promote

    # Batch without client fails closed.
    no_client = ToolDispatcher(plane, client=None)
    refused = no_client.execute(
        "dispatch_scheduler_batch",
        {
            "repo_root": str(repo),
            "model": "local-model",
            "owner": "x",
        },
    )
    assert refused.ok is False
    assert "InferenceClient" in refused.error
