"""Focused tests for repository inspection and context compilation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosmo.atomic import AtomicStore
from cosmo.context import (
    ContextCompiler,
    ContextError,
    RepositoryInspector,
)


def _repo_with_noise(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "mod.py").write_text(
        "import os\n\nVALUE = 1\n\ndef run():\n    return VALUE\n",
        encoding="utf-8",
    )
    (repo / "src" / "other.py").write_text("from mod import run\n", encoding="utf-8")
    (repo / "README.md").write_text("# demo\n", encoding="utf-8")
    (repo / ".git").mkdir()
    (repo / ".git" / "config").write_text("secret\n", encoding="utf-8")
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "mod.cpython-314.pyc").write_bytes(b"\x00\x01")
    (repo / ".venv").mkdir()
    (repo / ".venv" / "pyvenv.cfg").write_text("home = /tmp\n", encoding="utf-8")
    state = tmp_path / "state"
    # Place state inside repo to ensure exclusion.
    nested_state = repo / ".cosmo"
    nested_state.mkdir()
    (nested_state / "events.jsonl").write_text("{}\n", encoding="utf-8")
    return repo, state


def test_repository_map_is_deterministic_and_excludes_noise(tmp_path):
    repo, _state = _repo_with_noise(tmp_path)
    inspector = RepositoryInspector(repo, state_root=repo / ".cosmo")
    first = inspector.build_map()
    second = inspector.build_map()
    assert first.map_sha256 == second.map_sha256
    assert first.to_dict() == second.to_dict()
    paths = {entry.path for entry in first.files}
    assert "src/mod.py" in paths
    assert "README.md" in paths
    assert not any(p.startswith(".git") for p in paths)
    assert not any("__pycache__" in p for p in paths)
    assert not any(p.startswith(".venv") for p in paths)
    assert not any(p.startswith(".cosmo") for p in paths)
    mod = next(entry for entry in first.files if entry.path == "src/mod.py")
    assert mod.language == "python"
    assert "run" in mod.symbols
    assert "VALUE" in mod.symbols
    assert "os" in mod.imports


def test_path_escape_and_symlink_refused(tmp_path):
    repo, _ = _repo_with_noise(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("nope\n", encoding="utf-8")
    inspector = RepositoryInspector(repo, state_root=repo / ".cosmo")
    with pytest.raises(ContextError):
        inspector.read_artifact("../outside.txt")
    with pytest.raises(ContextError):
        inspector.read_artifact("/etc/passwd")
    with pytest.raises(ContextError):
        inspector.list_tree(prefix="../outside")
    with pytest.raises(ContextError):
        inspector.search_text("secret", prefix="/etc")
    link = repo / "escape"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported")
    with pytest.raises(ContextError):
        inspector.read_artifact("escape")
    # Symlink chain intermediate.
    chain_dir = repo / "chain"
    chain_dir.mkdir()
    mid = chain_dir / "mid"
    try:
        mid.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported")
    with pytest.raises(ContextError):
        inspector.read_artifact("chain/mid")
    # State-root content is fail-closed.
    with pytest.raises(ContextError):
        inspector.read_artifact(".cosmo/events.jsonl")
    # Nonexistent but safe prefix yields empty listing (not an open browse).
    empty = inspector.list_tree(prefix="does/not/exist")
    assert empty["entries"] == []


def test_large_file_map_sha256_is_true_content_hash(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    payload = b"cosmo-large-content\n" * 2000
    big = repo / "big.bin"
    big.write_bytes(payload)
    (repo / "small.py").write_text("X = 1\n", encoding="utf-8")
    import hashlib

    expected = hashlib.sha256(payload).hexdigest()
    inspector = RepositoryInspector(repo, max_file_bytes=64, max_map_entries=50)
    repo_map = inspector.build_map()
    entry = next(e for e in repo_map.files if e.path == "big.bin")
    assert entry.size_bytes == len(payload)
    assert entry.sha256 == expected
    assert not entry.sha256.startswith("skipped")
    # Deterministic and still limited.
    again = inspector.build_map()
    assert again.map_sha256 == repo_map.map_sha256
    assert entry.symbols == []


def test_zero_limits_fail_closed(tmp_path):
    repo, _ = _repo_with_noise(tmp_path)
    inspector = RepositoryInspector(repo)
    with pytest.raises(ContextError):
        inspector.list_tree(max_entries=0)
    with pytest.raises(ContextError):
        inspector.read_artifact("src/mod.py", max_bytes=0)
    with pytest.raises(ContextError):
        inspector.search_text("import", max_matches=0)
    with pytest.raises(ContextError):
        RepositoryInspector(repo, max_map_entries=0)


def test_bounded_read_and_search(tmp_path):
    repo, _ = _repo_with_noise(tmp_path)
    big = repo / "src" / "big.py"
    big.write_text("line0\n" + ("x" * 1000 + "\n") * 50 + "needle-here\n", encoding="utf-8")
    inspector = RepositoryInspector(
        repo,
        max_read_bytes=64,
        max_search_matches=2,
        max_search_file_bytes=200_000,
    )
    read = inspector.read_artifact("src/big.py", max_bytes=64)
    assert read["truncated"] is True
    assert len(read["excerpt"].encode("utf-8")) <= 64
    assert read["sha256"]
    ranged = inspector.read_artifact("src/mod.py", start_line=1, end_line=2)
    assert ranged["range"]["start_line"] == 1
    assert "import os" in ranged["excerpt"]

    search = inspector.search_text("needle-here", prefix="src")
    assert search["matches"]
    assert search["matches"][0]["path"] == "src/big.py"
    limited = inspector.search_text("import", max_matches=1)
    assert len(limited["matches"]) == 1
    assert limited["truncated"] is True


def test_list_tree_and_python_symbols(tmp_path):
    repo, _ = _repo_with_noise(tmp_path)
    inspector = RepositoryInspector(repo, state_root=repo / ".cosmo")
    tree = inspector.list_tree(prefix="src")
    assert all(e["path"].startswith("src") for e in tree["entries"])
    symbols = inspector.inspect_python_symbols("src/mod.py")
    assert symbols["skipped"] is False
    assert "run" in symbols["symbols"]


def test_context_compiler_planning_and_worker_manifests(tmp_path):
    repo, state = _repo_with_noise(tmp_path)
    store = AtomicStore(state)
    compiler = ContextCompiler(store)
    inspector = RepositoryInspector(repo, state_root=repo / ".cosmo")
    repo_map = inspector.build_map()
    selected = inspector.read_artifact("src/mod.py")

    request = {
        "request_id": "REQ-1",
        "original_text": "change value",
        "desired_outcome": "VALUE=2",
        "explicit_constraints": [],
        "immutable_sha256": "abc",
    }
    task = {
        "task_id": "TASK-1",
        "status": "ready",
        "current_spec_id": "SPEC-1",
        "plan_node_id": "NODE-1",
    }
    spec = {
        "spec_id": "SPEC-1",
        "objective": "Set VALUE to 2",
        "requirements": [{"requirement_id": "objective", "text": "Set VALUE to 2"}],
        "acceptance_criteria": ["exists"],
        "constraints": [],
        "artifact_path": "src/mod.py",
        "validators": ["python_syntax"],
    }
    planning = compiler.compile_for_planning(
        request=request,
        task=task,
        task_spec=spec,
        repository_map=repo_map,
        selected_files=[
            {
                "path": selected["path"],
                "sha256": selected["sha256"],
                "excerpt": selected["excerpt"],
                "selection_method": "explicit",
            }
        ],
    )
    assert planning["role"] == "planning"
    assert planning["manifest_sha256"]
    kinds = {item.get("kind") for item in planning["items"]}
    assert "repository_map" in kinds
    assert "request" in kinds
    # Manifests are immutable.
    with pytest.raises(Exception):
        store.write("context_manifests", planning["context_manifest_id"], planning)

    worker = compiler.compile_for_worker(
        task_spec=spec,
        writable_artifact={
            "path": "src/mod.py",
            "sha256": selected["sha256"],
            "excerpt": selected["excerpt"],
        },
        read_only_dependencies=[
            {
                "path": "src/other.py",
                "sha256": inspector.read_artifact("src/other.py")["sha256"],
                "excerpt": "from mod import run\n",
                "selection_method": "dependency",
            }
        ],
        tests=[
            {
                "path": "README.md",
                "sha256": inspector.read_artifact("README.md")["sha256"],
                "excerpt": "# demo\n",
                "selection_method": "explicit",
            }
        ],
        strategy_summary="direct assignment",
    )
    assert worker["role"] == "worker"
    worker_paths = {
        item.get("source_path")
        for item in worker["items"]
        if item.get("source_path")
    }
    assert "src/mod.py" in worker_paths
    assert "repository_map" not in {
        item.get("kind") for item in worker["items"]
    }
    assert "implicit whole-repository contents" in worker["excluded"]

    with pytest.raises(ContextError):
        compiler.compile_for_worker(
            task_spec=spec,
            writable_artifact={"path": "src/mod.py", "sha256": "x"},
            read_only_dependencies=[
                {"selection_method": "repository_map", "path": "ALL"}
            ],
        )


def test_retrieval_appends_event_without_transcript(tmp_path):
    repo, state = _repo_with_noise(tmp_path)
    store = AtomicStore(state)
    compiler = ContextCompiler(store)
    inspector = RepositoryInspector(repo, state_root=repo / ".cosmo")
    manifest = compiler.compile_for_planning(
        repository_map=inspector.build_map(),
    )
    read = inspector.read_artifact("README.md")
    item = inspector.retrieval_manifest_item("retrieval", read)
    recorded = compiler.record_retrieval(
        context_manifest_id=manifest["context_manifest_id"],
        item=item,
    )
    assert recorded["content_hash"]
    events = (state / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    types = [json.loads(line)["type"] for line in events]
    assert "context_manifest_compiled" in types
    assert "context_retrieval" in types
    # No prompt-like transcript keys in retrieval event.
    retrieval_events = [
        json.loads(line) for line in events if json.loads(line)["type"] == "context_retrieval"
    ]
    assert all("prompt" not in event for event in retrieval_events)


def test_review_and_synthesis_roles(tmp_path):
    state = tmp_path / "state"
    store = AtomicStore(state)
    compiler = ContextCompiler(store)
    spec = {
        "spec_id": "SPEC-1",
        "objective": "o",
        "requirements": [],
        "acceptance_criteria": [],
        "artifact_path": "a.py",
    }
    worker_manifest = compiler.compile_for_worker(
        task_spec=spec,
        writable_artifact={"path": "a.py", "sha256": "1", "excerpt": "x=1\n"},
    )
    review = compiler.compile_for_review(
        task_spec=spec,
        worker_context_manifest=worker_manifest,
        candidate={"path": "a.py", "sha256": "2", "excerpt": "x=2\n"},
        validation={
            "validation_id": "VAL-1",
            "passed": True,
            "checks": [{"name": "non_empty", "result": "pass"}],
            "candidate_sha256": "2",
        },
    )
    assert review["role"] == "review"
    synthesis = compiler.compile_for_synthesis(
        task_spec=spec,
        candidates=[
            {"path": "a.py", "sha256": "2", "excerpt": "x=2\n"},
            {"path": "a.py", "sha256": "3", "excerpt": "x=3\n"},
        ],
        rejected_strategies=["retry-same"],
    )
    assert synthesis["role"] == "synthesis"
    assert len(synthesis["items"]) >= 3
