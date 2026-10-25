"""Worker sandbox tools, context manifests, and atomic harness routing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosmo.atomic import AtomicHarness, AtomicStore
from cosmo.models import Generation
from cosmo.sandbox import SandboxPolicy, SandboxViolation, build_worker_policy
from cosmo.worker import WorkerError, WorkerExecutor, WorkerToolset


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
            "retained_successes": ["candidate is syntactically valid"],
            "hypotheses": [],
            "invalidated_assumptions": [],
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


def _repo_fixture(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "mod.py").write_text(
        "VALUE = 1\n\ndef run():\n    return VALUE\n",
        encoding="utf-8",
    )
    (repo / "src" / "util.py").write_text(
        "from mod import run\nHELPER = 3\n",
        encoding="utf-8",
    )
    (repo / "target.py").write_text("value = 1\n", encoding="utf-8")
    (repo / "contract.txt").write_text("must set value to 2\n", encoding="utf-8")
    state = tmp_path / "state"
    return repo, state


def _toolset(repo: Path, state: Path, writable: str = "target.py") -> WorkerToolset:
    store = AtomicStore(state)
    workspace = state / "workspaces" / "ATTEMPT-test"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / writable).write_text("value = 1\n", encoding="utf-8")
    policy = build_worker_policy(
        repo_root=repo,
        state_root=state,
        attempt_workspace=workspace,
        writable_artifact=writable,
    )
    return WorkerToolset(
        store=store, policy=policy, attempt_id="ATTEMPT-test"
    )


def test_sandbox_policy_is_deterministic_and_denies_os_isolation(tmp_path):
    repo, state = _repo_fixture(tmp_path)
    workspace = state / "workspaces" / "a"
    workspace.mkdir(parents=True)
    policy = build_worker_policy(
        repo_root=repo,
        state_root=state,
        attempt_workspace=workspace,
        writable_artifact="target.py",
    )
    assert policy.os_process_isolation is False
    assert "not guaranteed" in policy.os_process_isolation_note.lower()
    first = policy.policy_sha256
    second = build_worker_policy(
        repo_root=repo,
        state_root=state,
        attempt_workspace=workspace,
        writable_artifact="target.py",
    ).policy_sha256
    assert first == second
    evidence = policy.evidence_record()
    assert evidence["policy_sha256"] == first
    with pytest.raises(SandboxViolation):
        SandboxPolicy(
            repo_root=str(repo),
            writable_artifact="target.py",
            os_process_isolation=True,
        )


def test_read_search_symbol_tools(tmp_path):
    repo, state = _repo_fixture(tmp_path)
    tools = _toolset(repo, state)
    tree = tools.list_tree(prefix="src")
    paths = {entry["path"] for entry in tree["entries"]}
    assert "src/mod.py" in paths
    read = tools.read_artifact("src/mod.py")
    assert "VALUE" in read["excerpt"]
    search = tools.search_text("HELPER")
    assert any(m["path"] == "src/util.py" for m in search["matches"])
    symbols = tools.inspect_python_symbols("src/mod.py")
    assert "run" in symbols["symbols"]
    assert "VALUE" in symbols["symbols"]


def test_write_candidate_scope_and_rejections(tmp_path):
    repo, state = _repo_fixture(tmp_path)
    tools = _toolset(repo, state, writable="target.py")
    ok = tools.write_candidate("target.py", "value = 2\n")
    assert ok["sha256"]
    workspace_file = Path(tools.policy.attempt_workspace) / "target.py"
    assert workspace_file.read_text(encoding="utf-8") == "value = 2\n"
    # Product tree untouched.
    assert (repo / "target.py").read_text(encoding="utf-8") == "value = 1\n"

    with pytest.raises(WorkerError, match="not the declared"):
        tools.write_candidate("src/mod.py", "nope\n")
    with pytest.raises(WorkerError):
        tools.write_candidate("../outside.py", "nope\n")
    with pytest.raises(WorkerError):
        tools.read_artifact("../outside.py")
    with pytest.raises(WorkerError):
        tools.read_artifact("/etc/passwd")

    # State-root contents are refused for reads.
    AtomicStore(state)  # ensure layout
    (state / "events.jsonl").write_text("{}\n", encoding="utf-8")
    # When state is outside repo, relative path into state is not under repo.
    with pytest.raises(WorkerError):
        tools.read_artifact("events.jsonl")

    # Symlink escape refused.
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    link = repo / "escape"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported")
    with pytest.raises(WorkerError):
        tools.read_artifact("escape")


def test_tool_events_are_hash_only_durable(tmp_path):
    repo, state = _repo_fixture(tmp_path)
    tools = _toolset(repo, state)
    secret = "super secret body that must not be stored raw"
    tools.write_candidate("target.py", secret)
    tools.read_artifact("src/mod.py")
    tools.search_text("VALUE")

    events = [
        json.loads(line)
        for line in (state / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    tool_events = [e for e in events if e.get("type") == "worker_tool_call"]
    assert len(tool_events) >= 3
    blob = json.dumps(tool_events)
    assert secret not in blob
    assert "super secret" not in blob

    records = AtomicStore(state).list_records("worker_tool_events")
    assert records
    for record in records:
        assert "arguments_hash" in record
        assert "result_hash" in record or record.get("status") != "ok"
        dumped = json.dumps(record)
        assert secret not in dumped
        # Content field is hashed, not stored raw.
        if record["tool_name"] == "write_candidate":
            assert "sha256" in json.dumps(record.get("arguments_digest") or {})


def test_canonical_worker_context_and_read_only_paths(tmp_path):
    repo, state = _repo_fixture(tmp_path)
    store = AtomicStore(state)
    harness = AtomicHarness(store)
    submitted = harness.submit(
        request_text="Set value to 2 using the contract.",
        objective="Set value to 2.",
        artifact_path="target.py",
        repo_root=repo,
        validators=["python_syntax"],
    )
    client = FakeAtomicClient()
    result = harness.run_attempt(
        task_id=submitted["task_id"],
        repo_root=repo,
        client=client,
        model="local-model",
        reasoning="off",
        temperature=0,
        worker_max_output_tokens=100,
        review_max_output_tokens=100,
        strategy_summary="use the contract text as guidance",
        read_only_paths=["contract.txt", "src/mod.py"],
    )
    assert result["promotable"] is True
    attempt = harness._read_attempt(result["attempt_id"])
    assert attempt["read_only_paths"] == ["contract.txt", "src/mod.py"]
    manifest = store.read("context_manifests", attempt["context_manifest_id"])
    assert manifest["role"] == "worker"
    assert manifest["context_manifest_id"] == attempt["context_manifest_id"]
    assert manifest.get("manifest_sha256")
    kinds = {item.get("kind") for item in manifest["items"]}
    assert "task_spec" in kinds
    assert "writable_artifact" in kinds
    assert "dependency" in kinds
    worker_prompt = client.calls[0]["prompt"]
    assert "CONTEXT MANIFEST" in worker_prompt
    assert "contract.txt" in worker_prompt or "must set value" in worker_prompt
    assert "manifest_sha256" in worker_prompt
    # Product still baseline until promote.
    assert (repo / "target.py").read_text(encoding="utf-8") == "value = 1\n"


def test_worker_executor_routes_through_atomic_harness(tmp_path):
    repo, state = _repo_fixture(tmp_path)
    store = AtomicStore(state)
    harness = AtomicHarness(store)
    submitted = harness.submit(
        request_text="Make value two.",
        objective="Set value to 2.",
        artifact_path="target.py",
        repo_root=repo,
        validators=["python_syntax"],
    )
    executor = WorkerExecutor(store)
    client = FakeAtomicClient()
    result = executor.execute(
        task_id=submitted["task_id"],
        repo_root=repo,
        client=client,
        model="local-model",
        strategy_summary="replace assignment",
        worker_max_output_tokens=100,
        review_max_output_tokens=100,
    )
    assert result["promotable"] is True
    assert result["sandbox_os_process_isolation"] is False
    assert result["sandbox_policy_sha256"]
    assert [c["role"] for c in client.calls] == ["atomic_worker", "atomic_reviewer"]
    # Still requires explicit promote.
    assert (repo / "target.py").read_text(encoding="utf-8") == "value = 1\n"
    harness.promote(attempt_id=result["attempt_id"], repo_root=repo)
    assert (repo / "target.py").read_text(encoding="utf-8") == "value = 2\n"


def test_worker_rejects_workspace_outside_state_root(tmp_path):
    repo, state = _repo_fixture(tmp_path)
    store = AtomicStore(state)
    outside = tmp_path / "outside_ws"
    outside.mkdir()
    policy = build_worker_policy(
        repo_root=repo,
        state_root=state,
        attempt_workspace=outside,
        writable_artifact="target.py",
    )
    with pytest.raises(WorkerError, match="under the AtomicStore state root"):
        WorkerToolset(store=store, policy=policy, attempt_id="ATTEMPT-out")


def test_worker_rejects_over_policy_limits_before_read(tmp_path):
    repo, state = _repo_fixture(tmp_path)
    tools = _toolset(repo, state)
    # Over policy max_read_bytes must fail before loading content.
    with pytest.raises(WorkerError, match="max_bytes"):
        tools.read_artifact("src/mod.py", max_bytes=tools.policy.max_read_bytes + 1)
    with pytest.raises(WorkerError, match="max_entries"):
        tools.list_tree(max_entries=tools.policy.max_tool_calls * 8 + 1)
    with pytest.raises(WorkerError, match="max_matches"):
        tools.search_text("VALUE", max_matches=10_000)


def test_worker_rejects_malformed_limits_as_worker_error(tmp_path):
    """Non-integer / boolean limits must raise WorkerError, never TypeError/ValueError."""
    repo, state = _repo_fixture(tmp_path)
    tools = _toolset(repo, state)
    store = tools.store

    for bad in (True, 1.5, "64", None, object()):
        # None is allowed (means use policy default) — only exercise non-None bad values.
        if bad is None:
            continue
        with pytest.raises(WorkerError, match="max_bytes"):
            tools.read_artifact("src/mod.py", max_bytes=bad)  # type: ignore[arg-type]
        with pytest.raises(WorkerError, match="max_entries"):
            tools.list_tree(max_entries=bad)  # type: ignore[arg-type]
        with pytest.raises(WorkerError, match="max_matches"):
            tools.search_text("VALUE", max_matches=bad)  # type: ignore[arg-type]

    # Zero / negative also WorkerError (not TypeError).
    with pytest.raises(WorkerError, match="positive"):
        tools.read_artifact("src/mod.py", max_bytes=0)
    with pytest.raises(WorkerError, match="positive"):
        tools.list_tree(max_entries=-1)

    # Rejected calls are logged without raw file/query body content.
    events = list(store.list_records("worker_tool_events"))
    if not events:
        # Fall back to event stream records when dedicated kind is empty.
        events = [
            e
            for e in store.list_events()
            if e.get("event_type") == "worker_tool_call"
        ]
    assert events, "expected durable rejection logs"
    blob = json.dumps(events)
    assert "VALUE = 1" not in blob
    # Query body must not appear raw; digests/hashes only.
    assert '"query": "VALUE"' not in blob
    # Non-JSON limits (object()) must be type-marked, not repr/raw.
    digests = [
        e.get("arguments_digest") or {}
        for e in events
        if isinstance(e, dict) and e.get("arguments_digest") is not None
    ]
    type_marked = [
        d
        for d in digests
        if any(
            isinstance(v, dict) and v.get("type") == "object"
            for v in d.values()
        )
    ]
    assert type_marked, "expected object() limits logged as type markers only"
    assert "object object at 0x" not in blob.lower()


def test_sandbox_policy_hash_covers_manifest_fields(tmp_path):
    repo, state = _repo_fixture(tmp_path)
    workspace = state / "workspaces" / "hash"
    workspace.mkdir(parents=True)
    policy = build_worker_policy(
        repo_root=repo,
        state_root=state,
        attempt_workspace=workspace,
        writable_artifact="target.py",
    )
    manifest = policy.policy_manifest
    # Hash body must include every policy field that appears in the manifest
    # (except the derived policy_sha256 itself).
    for key in manifest:
        if key == "policy_sha256":
            continue
        assert key in {
            "schema_version",
            "repo_root",
            "state_root",
            "attempt_workspace",
            "writable_artifact",
            "baseline_mode",
            "max_writable_slots",
            "allow_shell",
            "allow_process_execution",
            "allow_network",
            "allow_credentials",
            "allow_state_root_access",
            "os_process_isolation",
            "os_process_isolation_note",
            "max_read_bytes",
            "max_write_bytes",
            "max_output_bytes",
            "max_tool_calls",
            "timeout_seconds",
            "allowed_tools",
            "forbidden_capabilities",
            "extra",
        }
    assert "os_process_isolation_note" in manifest
    assert policy.policy_sha256 == manifest["policy_sha256"]
    # Changing the note must change the hash.
    other = SandboxPolicy(
        repo_root=str(repo.resolve()),
        state_root=str(state.resolve()),
        attempt_workspace=str(workspace.resolve()),
        writable_artifact="target.py",
        os_process_isolation_note="different note",
    )
    assert other.policy_sha256 != policy.policy_sha256


def test_old_callers_without_read_only_paths_still_work(tmp_path):
    repo, state = _repo_fixture(tmp_path)
    store = AtomicStore(state)
    submitted = AtomicHarness(store).submit(
        request_text="Make value two.",
        objective="Set value to 2.",
        artifact_path="target.py",
        repo_root=repo,
        validators=["python_syntax"],
    )
    result = AtomicHarness(store).run_attempt(
        task_id=submitted["task_id"],
        repo_root=repo,
        client=FakeAtomicClient(),
        model="local-model",
        reasoning="off",
        temperature=0,
        worker_max_output_tokens=100,
        review_max_output_tokens=100,
        strategy_summary="legacy caller",
    )
    assert result["promotable"] is True
    manifest = store.read(
        "context_manifests",
        AtomicHarness(store)._read_attempt(result["attempt_id"])[
            "context_manifest_id"
        ],
    )
    assert manifest["role"] == "worker"
