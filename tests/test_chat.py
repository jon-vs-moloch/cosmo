"""Fake-client tests for the Cosmo chat/tool gateway."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosmo.atomic import AtomicStore
from cosmo.chat import (
    ChatSession,
    build_system_prompt,
    parse_model_turn,
    run_chat_turn,
)
from cosmo.control import ControlPlane
from cosmo.models import Generation
from cosmo.tools import (
    TOOL_DEFINITIONS_BY_NAME,
    ToolDispatcher,
    ToolError,
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


class ScriptedClient:
    """Inference client that returns scripted contents and records calls."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = list(responses or [])
        self.calls: list[dict] = []
        self.atomic_candidate = "value = 2\n"
        self.atomic_review = _review()

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        role = kwargs.get("role")
        # Harness worker/reviewer paths used by dispatch_ready.
        if role == "atomic_worker":
            content = self.atomic_candidate
        elif role == "atomic_reviewer":
            content = self.atomic_review
        elif self.responses:
            content = self.responses.pop(0)
        else:
            content = "No further scripted response."
        return Generation(
            role=str(role or "chat"),
            content=content,
            reasoning_hash=None,
            stats={
                "input_tokens": 10,
                "total_output_tokens": 5,
                "reasoning_output_tokens": 0,
            },
            latency_seconds=0.01,
            request_hash=f"request-{len(self.calls)}",
            model_instance_id=kwargs.get("model"),
        )


def _plane(tmp_path: Path) -> tuple[ControlPlane, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "target.py").write_text("value = 1\n", encoding="utf-8")
    state = tmp_path / "state"
    return ControlPlane(AtomicStore(state)), repo, state


def _dispatcher(tmp_path: Path, client: ScriptedClient | None = None):
    plane, repo, state = _plane(tmp_path)
    dispatcher = ToolDispatcher(plane, client=client)
    return dispatcher, plane, repo, state


# ---------------------------------------------------------------------------
# Tool definitions / validation
# ---------------------------------------------------------------------------


def test_tool_definitions_are_json_serializable():
    catalog = tool_definitions_json()
    assert catalog
    encoded = json.dumps(catalog)
    assert json.loads(encoded) == catalog
    names = {item["name"] for item in catalog}
    for required in (
        "harness_help",
        "submit_request",
        "list_tasks",
        "get_task",
        "inspect_task",
        "inspect_spec",
        "inspect_plan",
        "inspect_attempt",
        "inspect_review",
        "inspect_ledger",
        "inspect_lifecycle",
        "update_task_status",
        "ready_queue",
        "scheduler_tick",
        "reassess_priority",
        "dispatch_scheduler_batch",
        "refresh_dependency_readiness",
        "acquire_lease",
        "renew_lease",
        "release_lease",
        "dispatch_ready",
        "run_attempt",
        "retry_attempt",
        "promote_attempt",
    ):
        assert required in names
    assert "create_task" not in names
    assert "recover_promotion" not in names
    # No shell / external action tools.
    for forbidden in ("shell", "bash", "run_command", "http", "curl", "exec"):
        assert forbidden not in names


def test_compact_model_catalog_omits_field_prose_but_keeps_schema():
    full = tool_definitions_json(catalog="model")
    compact = tool_definitions_json(catalog="model", compact=True)
    assert {item["name"] for item in compact} == {item["name"] for item in full}
    for definition in compact:
        for property_schema in definition["parameters"]["properties"].values():
            assert "description" not in property_schema
        assert definition["parameters"]["type"] == "object"
        assert definition["parameters"]["additionalProperties"] is False


def test_harness_help_tool_and_slash_shortcut(tmp_path):
    client = ScriptedClient()
    dispatcher, _plane, _repo, _state = _dispatcher(tmp_path, client=client)
    help_result = dispatcher.execute("harness_help", {})
    assert help_result.ok is True
    assert help_result.result["plan_shape"]["important"]
    assert "python_syntax" in help_result.result["contracts"]["leaf"]

    session = ChatSession(client=client, dispatcher=dispatcher, model="local-model")
    result = session.handle_user_message("/help")
    assert result.stop_reason == "help"
    assert "Plan shape" in result.final_text
    assert client.calls == []


def test_run_attempt_is_ready_leaf_dispatch_alias(tmp_path):
    client = ScriptedClient()
    dispatcher, plane, repo, _state = _dispatcher(tmp_path, client=client)
    created = dispatcher.execute(
        "submit_request",
        {
            "request_text": "Make value two.",
            "objective": "Set value to 2.",
            "artifact_path": "target.py",
            "repo_root": str(repo),
            "validators": ["python_syntax"],
        },
    )
    task_id = created.result["task_id"]
    assert (repo / "target.py").read_text(encoding="utf-8") == "value = 1\n"

    ran = dispatcher.execute(
        "run_attempt",
        {
            "task_id": task_id,
            "repo_root": str(repo),
            "model": "local-model",
            "strategy_summary": "replace assignment with value 2",
            "owner": "chat-run",
            "worker_max_output_tokens": 100,
            "review_max_output_tokens": 100,
        },
    )
    assert ran.ok is True
    assert ran.result["promotable"] is True
    assert ran.result["attempt_id"]
    # Same harness pipeline as dispatch_ready: no product mutation.
    assert (repo / "target.py").read_text(encoding="utf-8") == "value = 1\n"
    roles = [call["role"] for call in client.calls]
    assert "atomic_worker" in roles
    assert "atomic_reviewer" in roles


def test_retry_attempt_requires_four_field_contract(tmp_path):
    # Non-promotable first attempt returns the leaf to ready for retries.
    client = ScriptedClient()
    client.atomic_review = _review("REVISE")
    dispatcher, plane, repo, _state = _dispatcher(tmp_path, client=client)
    created = dispatcher.execute(
        "submit_request",
        {
            "request_text": "Make value two.",
            "objective": "Set value to 2.",
            "artifact_path": "target.py",
            "repo_root": str(repo),
            "validators": ["python_syntax"],
        },
    )
    task_id = created.result["task_id"]
    first = dispatcher.execute(
        "run_attempt",
        {
            "task_id": task_id,
            "repo_root": str(repo),
            "model": "local-model",
            "strategy_summary": "first strategy",
            "owner": "chat-retry",
            "worker_max_output_tokens": 100,
            "review_max_output_tokens": 100,
        },
    )
    assert first.ok is True
    assert first.result["promotable"] is False
    first_attempt = first.result["attempt_id"]
    assert plane.get_task(task_id)["status"] == "ready"

    # Schema: missing retry fields are structured tool errors.
    missing = dispatcher.execute(
        "retry_attempt",
        {
            "task_id": task_id,
            "repo_root": str(repo),
            "model": "local-model",
            "strategy_summary": "second strategy",
            "owner": "chat-retry",
        },
    )
    assert missing.ok is False
    assert "missing required" in (missing.error or "").lower()

    # Blind retry / incomplete contract rejected by harness gates.
    blind = dispatcher.execute(
        "retry_attempt",
        {
            "task_id": task_id,
            "repo_root": str(repo),
            "model": "local-model",
            "strategy_summary": "second strategy",
            "owner": "chat-retry",
            "addresses": [],
            "preserves": ["path"],
            "differs_from": [first_attempt],
            "expected_improvement": ["better"],
            "worker_max_output_tokens": 100,
            "review_max_output_tokens": 100,
        },
    )
    assert blind.ok is False
    assert "retries require" in (blind.error or "").lower() or "addresses" in (
        blind.error or ""
    ).lower()

    # Identical strategy rejected even with full contract.
    same_strategy = dispatcher.execute(
        "retry_attempt",
        {
            "task_id": task_id,
            "repo_root": str(repo),
            "model": "local-model",
            "strategy_summary": "first strategy",
            "owner": "chat-retry",
            "addresses": ["failed objective"],
            "preserves": ["file path"],
            "differs_from": [first_attempt],
            "expected_improvement": ["satisfy objective"],
            "worker_max_output_tokens": 100,
            "review_max_output_tokens": 100,
        },
    )
    assert same_strategy.ok is False
    assert "materially different" in (same_strategy.error or "").lower()

    # Valid contract + different strategy goes through dispatch_ready_leaf.
    client.atomic_candidate = "value = 2\n"
    client.atomic_review = _review("PASS")
    retried = dispatcher.execute(
        "retry_attempt",
        {
            "task_id": task_id,
            "repo_root": str(repo),
            "model": "local-model",
            "strategy_summary": "materially different assignment rewrite",
            "owner": "chat-retry",
            "addresses": ["failed objective"],
            "preserves": ["file path"],
            "differs_from": [first_attempt],
            "expected_improvement": ["satisfy objective"],
            "worker_max_output_tokens": 100,
            "review_max_output_tokens": 100,
        },
    )
    assert retried.ok is True
    assert retried.result["attempt_id"] != first_attempt
    assert (repo / "target.py").read_text(encoding="utf-8") == "value = 1\n"


def test_validate_tool_arguments_rejects_unknown_and_malformed():
    definition = TOOL_DEFINITIONS_BY_NAME["get_task"]
    with pytest.raises(ToolError, match="missing required"):
        validate_tool_arguments(definition, {})
    with pytest.raises(ToolError, match="unknown tool arguments"):
        validate_tool_arguments(definition, {"task_id": "T1", "extra": 1})
    with pytest.raises(ToolError, match="must be a string"):
        validate_tool_arguments(definition, {"task_id": 12})
    ok = validate_tool_arguments(definition, {"task_id": "TASK-1"})
    assert ok == {"task_id": "TASK-1"}


def test_malformed_args_become_structured_errors(tmp_path):
    dispatcher, _plane, _repo, _state = _dispatcher(tmp_path)
    result = dispatcher.execute("get_task", {"task_id": 99})
    assert result.ok is False
    assert "string" in (result.error or "").lower()
    payload = result.to_json_dict()
    assert payload["ok"] is False
    assert payload["tool"] == "get_task"
    assert "error" in payload


def test_unknown_tools_are_rejected(tmp_path):
    dispatcher, _plane, _repo, _state = _dispatcher(tmp_path)
    result = dispatcher.execute("rm_rf", {"path": "/"})
    assert result.ok is False
    assert "unknown tool" in (result.error or "")
    result2 = dispatcher.execute("shell", {"cmd": "echo hi"})
    assert result2.ok is False


# ---------------------------------------------------------------------------
# Tool execution against ControlPlane
# ---------------------------------------------------------------------------


def test_submit_request_and_list_get_inspect(tmp_path):
    dispatcher, plane, repo, _state = _dispatcher(tmp_path)
    created = dispatcher.execute(
        "submit_request",
        {
            "request_text": "Make value two.",
            "objective": "Set value to 2.",
            "artifact_path": "target.py",
            "repo_root": str(repo),
            "validators": ["python_syntax"],
        },
    )
    assert created.ok is True
    task_id = created.result["task_id"]
    listed = dispatcher.execute("list_tasks", {})
    assert listed.ok is True
    assert any(item["task_id"] == task_id for item in listed.result)
    got = dispatcher.execute("get_task", {"task_id": task_id})
    assert got.ok and got.result["status"] == "ready"
    inspected = dispatcher.execute("inspect_task", {"task_id": task_id})
    assert inspected.ok
    assert inspected.result["spec"]["objective"] == "Set value to 2."
    assert inspected.result["plan"]["writable_artifact"] == "target.py"
    queue = dispatcher.execute("ready_queue", {})
    assert queue.ok
    assert task_id in [item["task_id"] for item in queue.result]


def test_create_task_uses_control_plane_path(tmp_path):
    """create_task must use ControlPlane authority, not bare AtomicHarness.submit."""
    dispatcher, plane, repo, _state = _dispatcher(tmp_path)
    result = dispatcher.execute(
        "create_task",
        {
            "request_text": "Leaf via control plane.",
            "objective": "Set value to 2.",
            "artifact_path": "target.py",
            "repo_root": str(repo),
            "validators": ["python_syntax"],
            "priority_components": {
                "urgency": 0.9,
                "impact": 0.5,
                "freshness": 1.0,
                "operator_boost": 0.0,
                "blocked_penalty": 0.0,
                "dependency_penalty": 0.0,
            },
        },
    )
    assert result.ok is True
    task_id = result.result["task_id"]
    task = plane.get_task(task_id)
    assert task["status"] == "ready"
    # Control-plane lifecycle/priority fields required for queue/leases/dispatch.
    assert task.get("priority_score") is not None
    assert task.get("priority_assessment_id")
    assert task.get("plan_node_id")
    assert result.result.get("priority_assessment_id")
    assert result.result.get("plan_kind") == "leaf"
    # Participates in ready_queue, inspect, and lease lifecycle.
    queue = dispatcher.execute("ready_queue", {})
    assert queue.ok and task_id in [item["task_id"] for item in queue.result]
    inspected = dispatcher.execute("inspect_task", {"task_id": task_id})
    assert inspected.ok
    assert inspected.result["spec"]["objective"] == "Set value to 2."
    lease = dispatcher.execute(
        "acquire_lease",
        {"task_id": task_id, "owner": "create-task-owner", "ttl_seconds": 30},
    )
    assert lease.ok is True
    dispatcher.execute(
        "release_lease",
        {"lease_id": lease.result["lease_id"], "owner": "create-task-owner"},
    )


def test_execute_turns_runtime_failures_into_structured_errors(tmp_path):
    dispatcher, plane, repo, _state = _dispatcher(tmp_path)

    def boom(_args):
        raise RuntimeError("simulated transport failure")

    dispatcher.handlers["list_tasks"] = boom
    result = dispatcher.execute("list_tasks", {})
    assert result.ok is False
    assert "RuntimeError" in (result.error or "")
    assert "simulated transport failure" in (result.error or "")
    # Product state untouched.
    assert (repo / "target.py").read_text(encoding="utf-8") == "value = 1\n"
    assert plane.list_tasks() == []


def test_lease_lifecycle_and_status_update(tmp_path):
    dispatcher, plane, repo, _state = _dispatcher(tmp_path)
    created = dispatcher.execute(
        "submit_request",
        {
            "request_text": "Lease me.",
            "objective": "Set value to 2.",
            "artifact_path": "target.py",
            "repo_root": str(repo),
        },
    )
    task_id = created.result["task_id"]
    lease = dispatcher.execute(
        "acquire_lease",
        {"task_id": task_id, "owner": "chat-owner", "ttl_seconds": 60},
    )
    assert lease.ok is True
    lease_id = lease.result["lease_id"]
    renewed = dispatcher.execute(
        "renew_lease",
        {"lease_id": lease_id, "owner": "chat-owner", "ttl_seconds": 90},
    )
    assert renewed.ok is True
    released = dispatcher.execute(
        "release_lease",
        {"lease_id": lease_id, "owner": "chat-owner"},
    )
    assert released.ok is True
    status = dispatcher.execute(
        "update_task_status",
        {"task_id": task_id, "status": "queued", "reason": "operator"},
    )
    assert status.ok and status.result["status"] == "queued"


def test_dispatch_and_promote_use_harness_gates(tmp_path):
    client = ScriptedClient()
    dispatcher, plane, repo, _state = _dispatcher(tmp_path, client=client)
    created = dispatcher.execute(
        "submit_request",
        {
            "request_text": "Make value two.",
            "objective": "Set value to 2.",
            "artifact_path": "target.py",
            "repo_root": str(repo),
            "validators": ["python_syntax"],
        },
    )
    task_id = created.result["task_id"]
    # Product state untouched before dispatch.
    assert (repo / "target.py").read_text(encoding="utf-8") == "value = 1\n"

    dispatched = dispatcher.execute(
        "dispatch_ready",
        {
            "task_id": task_id,
            "repo_root": str(repo),
            "model": "local-model",
            "strategy_summary": "replace assignment with value 2",
            "owner": "chat-dispatch",
            "worker_max_output_tokens": 100,
            "review_max_output_tokens": 100,
        },
    )
    assert dispatched.ok is True
    assert dispatched.result["promotable"] is True
    attempt_id = dispatched.result["attempt_id"]
    # Dispatch must not mutate product state.
    assert (repo / "target.py").read_text(encoding="utf-8") == "value = 1\n"

    inspected = dispatcher.execute(
        "inspect_attempt", {"attempt_id": attempt_id}
    )
    assert inspected.ok
    assert inspected.result["attempt_id"] == attempt_id

    promoted = dispatcher.execute(
        "promote_attempt",
        {"attempt_id": attempt_id, "repo_root": str(repo)},
    )
    assert promoted.ok is True
    assert (repo / "target.py").read_text(encoding="utf-8") == "value = 2\n"

    # Roles used by harness only — chat model cannot inject fake PASS reviews.
    roles = [call["role"] for call in client.calls]
    assert "atomic_worker" in roles
    assert "atomic_reviewer" in roles


def test_promote_rejects_non_promotable_attempt(tmp_path):
    client = ScriptedClient()
    client.atomic_review = _review("REVISE")
    dispatcher, plane, repo, _state = _dispatcher(tmp_path, client=client)
    created = dispatcher.execute(
        "submit_request",
        {
            "request_text": "Make value two.",
            "objective": "Set value to 2.",
            "artifact_path": "target.py",
            "repo_root": str(repo),
            "validators": ["python_syntax"],
        },
    )
    dispatched = dispatcher.execute(
        "dispatch_ready",
        {
            "task_id": created.result["task_id"],
            "repo_root": str(repo),
            "model": "local-model",
            "strategy_summary": "try something",
            "owner": "chat-dispatch",
            "worker_max_output_tokens": 100,
            "review_max_output_tokens": 100,
        },
    )
    assert dispatched.ok is True
    assert dispatched.result["promotable"] is False
    promote = dispatcher.execute(
        "promote_attempt",
        {
            "attempt_id": dispatched.result["attempt_id"],
            "repo_root": str(repo),
        },
    )
    assert promote.ok is False
    assert (repo / "target.py").read_text(encoding="utf-8") == "value = 1\n"


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------


def test_parse_model_turn_variants():
    assert parse_model_turn("Hello user")["kind"] == "final"
    tool = parse_model_turn(
        json.dumps({"tool_calls": [{"name": "list_tasks", "arguments": {}}]})
    )
    assert tool["kind"] == "tool_calls"
    assert tool["calls"][0]["name"] == "list_tasks"
    fenced = parse_model_turn(
        'Sure.\n```json\n{"tool_calls":[{"name":"ready_queue","arguments":{}}]}\n```'
    )
    assert fenced["kind"] == "tool_calls"
    bad = parse_model_turn('{"tool_calls":[]}')
    assert bad["kind"] == "error"


def test_chat_loop_terminates_on_final_text(tmp_path):
    client = ScriptedClient(["Here is the answer without tools."])
    dispatcher, _plane, _repo, _state = _dispatcher(tmp_path, client=client)
    result = run_chat_turn(
        client=client,
        dispatcher=dispatcher,
        model="local-model",
        message="What can you do?",
    )
    assert result.stop_reason == "final"
    assert result.final_text == "Here is the answer without tools."
    assert result.tool_turns == 0
    assert len(client.calls) == 1
    assert client.calls[0]["role"] == "chat"
    assert "What can you do?" in client.calls[0]["prompt"]
    assert "Tool catalog" in client.calls[0]["system_prompt"]


def test_chat_loop_executes_tools_then_finalizes(tmp_path):
    dispatcher, plane, repo, _state = _dispatcher(tmp_path)
    # Seed a task so list_tasks has content.
    plane.submit_request(
        request_text="Seed.",
        objective="Seed objective.",
        artifact_path="target.py",
        repo_root=repo,
        validators=["python_syntax"],
    )
    client = ScriptedClient(
        [
            json.dumps(
                {"tool_calls": [{"name": "list_tasks", "arguments": {}}]}
            ),
            "There is at least one task in the store.",
        ]
    )
    dispatcher.client = client
    result = run_chat_turn(
        client=client,
        dispatcher=dispatcher,
        model="local-model",
        message="List my tasks.",
    )
    assert result.stop_reason == "final"
    assert result.tool_turns == 1
    assert "task" in result.final_text.lower() or "There is" in result.final_text
    tool_turns = [t for t in result.transcript if t.role == "tool"]
    assert tool_turns
    assert '"ok": true' in tool_turns[0].content or '"ok": true'.replace(
        " ", ""
    ) in tool_turns[0].content.replace(" ", "")
    # Second model prompt includes tool results.
    assert len(client.calls) == 2
    assert "Tool results" in client.calls[1]["prompt"]


def test_chat_loop_malformed_tool_args_continue(tmp_path):
    client = ScriptedClient(
        [
            json.dumps(
                {
                    "tool_calls": [
                        {"name": "get_task", "arguments": {"task_id": 123}}
                    ]
                }
            ),
            "I received a structured error and stopped.",
        ]
    )
    dispatcher, _plane, _repo, _state = _dispatcher(tmp_path, client=client)
    result = run_chat_turn(
        client=client,
        dispatcher=dispatcher,
        model="local-model",
        message="Get a task.",
    )
    assert result.stop_reason == "final"
    assert result.tool_turns == 1
    tool_body = result.transcript[-2].content if result.transcript[-2].role == "tool" else ""
    # Prefer scanning any tool turn.
    tool_text = "\n".join(t.content for t in result.transcript if t.role == "tool")
    assert "error" in tool_text.lower()
    assert "string" in tool_text.lower() or "must be" in tool_text.lower()


def test_chat_loop_unknown_tool_rejected(tmp_path):
    client = ScriptedClient(
        [
            json.dumps(
                {
                    "tool_calls": [
                        {"name": "shell", "arguments": {"cmd": "rm -rf /"}}
                    ]
                }
            ),
            "I cannot run shell commands.",
        ]
    )
    dispatcher, _plane, _repo, _state = _dispatcher(tmp_path, client=client)
    result = run_chat_turn(
        client=client,
        dispatcher=dispatcher,
        model="local-model",
        message="Delete everything.",
    )
    assert result.stop_reason == "final"
    tool_text = "\n".join(t.content for t in result.transcript if t.role == "tool")
    assert "unknown tool" in tool_text


def test_chat_loop_respects_max_tool_turns(tmp_path):
    infinite_tools = json.dumps(
        {"tool_calls": [{"name": "list_tasks", "arguments": {}}]}
    )
    client = ScriptedClient([infinite_tools, infinite_tools, infinite_tools])
    dispatcher, _plane, _repo, _state = _dispatcher(tmp_path, client=client)
    result = run_chat_turn(
        client=client,
        dispatcher=dispatcher,
        model="local-model",
        message="Keep calling tools.",
        max_tool_turns=2,
    )
    assert result.stop_reason == "max_tool_turns"
    assert result.tool_turns == 2
    assert "max tool turns" in result.final_text.lower()


def test_chat_transcript_ephemeral_requests_durable(tmp_path):
    client = ScriptedClient(
        [
            json.dumps(
                {
                    "tool_calls": [
                        {
                            "name": "submit_request",
                            "arguments": {
                                "request_text": "Make value two.",
                            },
                        }
                    ]
                }
            ),
        "Recorded the request for triage.",
        ]
    )
    dispatcher, plane, repo, state = _dispatcher(tmp_path, client=client)
    session = ChatSession(
        client=client,
        dispatcher=dispatcher,
        model="local-model",
        max_tool_turns=4,
    )
    result = session.handle_user_message("Please submit a request.")
    assert result.stop_reason == "final"
    # Model intake is durable but cannot manufacture an operator-supplied Task.
    assert len(plane.list_requests()) == 1
    assert plane.list_tasks() == []
    # Transcript is only on the session object (ephemeral).
    assert session.transcript
    # No durable chat transcript payloads under state root (store kind dirs may exist).
    transcript_files = [
        path
        for path in state.rglob("*")
        if path.is_file() and "transcript" in path.name.lower()
    ]
    assert transcript_files == []


def test_product_state_safety_no_promotion_without_gates(tmp_path):
    """Chat tools cannot fabricate PASS evidence to mutate product state."""
    client = ScriptedClient(
        [
            # Model tries to promote a non-existent attempt.
            json.dumps(
                {
                    "tool_calls": [
                        {
                            "name": "promote_attempt",
                            "arguments": {
                                "attempt_id": "ATTEMPT-fake",
                                "repo_root": "IGNORED",
                            },
                        }
                    ]
                }
            ),
            "Promotion failed as expected.",
        ]
    )
    dispatcher, plane, repo, _state = _dispatcher(tmp_path, client=client)
    # Fix repo path in the scripted call.
    client.responses[0] = json.dumps(
        {
            "tool_calls": [
                {
                    "name": "promote_attempt",
                    "arguments": {
                        "attempt_id": "ATTEMPT-fake",
                        "repo_root": str(repo),
                    },
                }
            ]
        }
    )
    result = run_chat_turn(
        client=client,
        dispatcher=dispatcher,
        model="local-model",
        message="Promote fake attempt.",
    )
    assert result.stop_reason == "final"
    tool_text = "\n".join(t.content for t in result.transcript if t.role == "tool")
    assert '"ok": false' in tool_text.replace(" ", "") or "error" in tool_text.lower()
    assert (repo / "target.py").read_text(encoding="utf-8") == "value = 1\n"
    assert plane.list_tasks() == []


def test_system_prompt_includes_tools_not_hidden_eval():
    prompt = build_system_prompt()
    assert "submit_request" in prompt
    assert "promote_attempt" in prompt
    assert "shell" not in prompt.lower() or "no shell" in prompt.lower()
    assert "hidden" not in prompt.lower() or "No hidden" in prompt


def test_chat_tool_loop_journals_denied_operator_tool(tmp_path):
    """Tool-loop operations are durable append-only events (no transcript body)."""
    client = ScriptedClient(
        [
            # filled below with real repo path
            "placeholder",
            "The operator-only shortcut was denied.",
        ]
    )
    dispatcher, plane, repo, state = _dispatcher(tmp_path, client=client)
    client.responses[0] = json.dumps(
        {
            "tool_calls": [
                {
                    "name": "create_task",
                    "arguments": {
                        "request_text": "Journal me.",
                        "objective": "Set value to 2.",
                        "artifact_path": "target.py",
                        "repo_root": str(repo),
                        "validators": ["python_syntax"],
                    },
                }
            ]
        }
    )
    result = run_chat_turn(
        client=client,
        dispatcher=dispatcher,
        model="local-model",
        message="Create a leaf task.",
    )
    assert result.stop_reason == "final"
    assert plane.list_tasks() == []

    chat_events = plane.store.list_records("chat_events")
    assert chat_events, "expected durable chat_events records"
    matching = [
        item
        for item in chat_events
        if item.get("tool") == "create_task" and item.get("ok") is False
    ]
    assert matching
    event = matching[0]
    assert event.get("operation_id")
    assert event.get("task_id") is None
    assert "prompt" not in event
    assert "transcript" not in event
    assert "content" not in event

    events_path = state / "events.jsonl"
    assert events_path.is_file()
    lines = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tool_events = [item for item in lines if item.get("type") == "chat_tool_call"]
    assert tool_events
    assert any(
        item.get("tool") == "create_task"
        and item.get("ok") is False
        and item.get("task_id") is None
        for item in tool_events
    )
