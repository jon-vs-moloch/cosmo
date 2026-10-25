"""Parent integration isolation, ordering, approval, and promotion gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cosmo.atomic import AtomicHarness, AtomicStaleError, AtomicStore
from cosmo.authority import authenticate_operator
from cosmo.control import ControlPlane
from cosmo.integration import IntegrationError, IntegrationService
from cosmo.models import Generation


_OPERATOR = authenticate_operator("pytest-operator")


def _review(verdict="PASS", requirement_id="objective"):
    return json.dumps(
        {
            "verdict": verdict,
            "requirement_results": [
                {
                    "requirement_id": requirement_id,
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


class FakeAtomicClient:
    def __init__(self, candidate: str, requirement_id: str = "objective"):
        self.candidate = candidate
        self.requirement_id = requirement_id
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        content = (
            self.candidate
            if kwargs["role"] == "atomic_worker"
            else _review(requirement_id=self.requirement_id)
        )
        return Generation(
            role=kwargs["role"],
            content=content,
            reasoning_hash=None,
            stats={
                "input_tokens": 8,
                "total_output_tokens": 4,
                "reasoning_output_tokens": 0,
            },
            latency_seconds=0.01,
            request_hash=f"request-{len(self.calls)}",
            model_instance_id=kwargs["model"],
        )


def _parent_with_leaves(tmp_path: Path) -> tuple[Path, AtomicStore, dict[str, Any], list[dict[str, Any]]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("A = 0\n", encoding="utf-8")
    (repo / "b.py").write_text("B = 0\n", encoding="utf-8")
    state = tmp_path / "state"
    store = AtomicStore(state)
    plane = ControlPlane(store)

    parent = plane.submit_request(
        request_text="Compose a and b modules.",
        objective="Integrate a and b.",
        artifact_path="",
        repo_root=repo,
        validators=["python_syntax"],
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
    parent_node_id = parent["node_id"]

    leaves: list[dict[str, Any]] = []
    for path, content, deps in (
        ("a.py", "A = 1\n", ()),
        ("b.py", "B = 2\n", ()),
    ):
        submitted = plane.submit_request(
            request_text=f"Set {path}",
            objective=f"Write {path}",
            artifact_path=path,
            repo_root=repo,
            validators=["python_syntax"],
            plan_kind="leaf",
            parent_node_id=parent_node_id,
            dependencies=deps,
            priority_components={
                "urgency": 0.5,
                "impact": 0.5,
                "freshness": 1.0,
                "operator_boost": 0.0,
                "blocked_penalty": 0.0,
                "dependency_penalty": 0.0,
            },
        )
        result = AtomicHarness(store).run_attempt(
            task_id=submitted["task_id"],
            repo_root=repo,
            client=FakeAtomicClient(content),
            model="local-model",
            reasoning="off",
            temperature=0,
            worker_max_output_tokens=100,
            review_max_output_tokens=100,
            strategy_summary=f"write {path}",
        )
        assert result["promotable"] is True
        leaves.append({**submitted, **result, "path": path, "content": content})

    # Wire leaf-2 depends on leaf-1 for order testing.
    leaf_b = store.read("plan_nodes", leaves[1]["node_id"])
    leaf_b["dependencies"] = [leaves[0]["node_id"]]
    store.replace("plan_nodes", leaves[1]["node_id"], leaf_b)

    return repo, store, parent, leaves


def test_integration_workspace_isolation_and_no_product_mutation(tmp_path):
    repo, store, parent, leaves = _parent_with_leaves(tmp_path)
    service = IntegrationService(store)
    run = service.assemble(parent["node_id"], repo_root=repo)

    assert run["validation_passed"] is True
    workspace = Path(run["workspace"])
    assert workspace.is_dir()
    assert str(store.root) in str(workspace.resolve())
    assert (workspace / "a.py").read_text(encoding="utf-8") == "A = 1\n"
    assert (workspace / "b.py").read_text(encoding="utf-8") == "B = 2\n"
    # Product unchanged until explicit promote.
    assert (repo / "a.py").read_text(encoding="utf-8") == "A = 0\n"
    assert (repo / "b.py").read_text(encoding="utf-8") == "B = 0\n"
    assert run["status"] == "assembled"
    assert run["promotable"] is False


def test_integration_dependency_order(tmp_path):
    repo, store, parent, leaves = _parent_with_leaves(tmp_path)
    service = IntegrationService(store)
    run = service.assemble(parent["node_id"], repo_root=repo)
    order = [item["node_id"] for item in run["selected_children"]]
    assert order.index(leaves[0]["node_id"]) < order.index(leaves[1]["node_id"])
    check = next(c for c in run["checks"] if c["name"] == "dependency_order")
    assert check["result"] == "pass"
    assert check["evidence"] == order


def test_unknown_parent_validator_fails_closed(tmp_path):
    repo, store, parent, _leaves = _parent_with_leaves(tmp_path)
    parent_plan = store.read("plan_nodes", parent["node_id"])
    parent_plan["validators"] = ["compose_contract_v1"]
    store.replace("plan_nodes", parent["node_id"], parent_plan)

    run = IntegrationService(store).assemble(parent["node_id"], repo_root=repo)

    assert run["validation_passed"] is False
    check = next(
        item
        for item in run["checks"]
        if item["name"] == "parent_validator:compose_contract_v1"
    )
    assert check["result"] == "fail"
    assert "no deterministic adapter" in check["evidence"]


def test_child_approval_required(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("A = 0\n", encoding="utf-8")
    state = tmp_path / "state"
    store = AtomicStore(state)
    plane = ControlPlane(store)
    parent = plane.submit_request(
        request_text="parent",
        objective="parent",
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
    plane.submit_request(
        request_text="leaf",
        objective="leaf",
        artifact_path="a.py",
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
    with pytest.raises(IntegrationError, match="not node-approved"):
        IntegrationService(store).assemble(parent["node_id"], repo_root=repo)


def test_stale_baseline_and_duplicate_path_rejection(tmp_path):
    repo, store, parent, leaves = _parent_with_leaves(tmp_path)
    service = IntegrationService(store)
    expected = store.read("plan_nodes", parent["node_id"])
    # Capture current baseline then mutate product.
    from cosmo.atomic import _product_revision

    baseline = _product_revision(repo, excluded_root=store.root)
    (repo / "extra.txt").write_text("concurrent\n", encoding="utf-8")
    with pytest.raises(IntegrationError, match="stale baseline"):
        service.assemble(
            parent["node_id"],
            repo_root=repo,
            expected_baseline_tree_sha256=baseline,
        )

    # Duplicate path: two leaves claiming same path.
    # Reset product extra for a clean assemble attempt on path collision logic.
    (repo / "extra.txt").unlink()
    # Manually force two selected children with same logical path via selected map
    # is hard; instead corrupt second leaf's artifact path record.
    art = store.read("artifacts", leaves[1]["artifact_version_id"])
    # Point leaf plan writable to same as leaf 0 while keeping artifact path.
    leaf0 = store.read("plan_nodes", leaves[0]["node_id"])
    leaf1 = store.read("plan_nodes", leaves[1]["node_id"])
    leaf1["writable_artifact"] = leaf0["writable_artifact"]
    store.replace("plan_nodes", leaves[1]["node_id"], leaf1)
    # Also rewrite artifact logical_path to collide.
    # Artifacts are immutable — write a shadow by replacing is forbidden.
    # Use resolve_selected_children path: inject via selected with crafted store.
    # Simpler: call resolve after making both approved artifacts share path.
    content_path = Path(art["content_path"])
    # Create a second artifact version with a's path.
    from cosmo.atomic import _sha256_text, new_id, utc_now

    content = "B = 2\n"
    digest = _sha256_text(content)
    version_id = new_id("ARTV")
    cpath = store.root / "artifacts" / f"{digest}-dup.content"
    cpath.write_text(content, encoding="utf-8")
    store.write(
        "artifacts",
        version_id,
        {
            "schema_version": "cosmo.artifact-version.v1",
            "artifact_id": f"ART-{digest[:16]}",
            "logical_path": "a.py",  # collide
            "version_id": version_id,
            "content_hash": digest,
            "content_path": str(cpath),
            "created_by_attempt": leaves[1]["attempt_id"],
            "status": "candidate",
            "created_at": utc_now(),
        },
    )
    # Force attempt state to use colliding artifact and still look approved.
    attempt = store.read("attempt_states", leaves[1]["attempt_id"])
    attempt["artifact_version_id"] = version_id
    store.replace("attempt_states", leaves[1]["attempt_id"], attempt)
    # Also update validation linkage is already PASS on old version — approval
    # walk uses attempt state's artifact_version_id and original validation.
    # _node_approved_artifact reads artifact by version_id from attempt — good.
    with pytest.raises(IntegrationError, match="duplicate logical path|differs from leaf"):
        service.assemble(parent["node_id"], repo_root=repo)


def test_no_product_mutation_before_approved_promotion(tmp_path):
    repo, store, parent, leaves = _parent_with_leaves(tmp_path)
    plane = ControlPlane(store)
    run = plane.assemble_integration(parent["node_id"], repo_root=repo)
    assert (repo / "a.py").read_text(encoding="utf-8") == "A = 0\n"

    with pytest.raises(IntegrationError, match="review PASS"):
        plane.promote_integration(run["integration_id"], repo_root=repo)

    reviewed = plane.review_integration(
        run["integration_id"],
        verdict="PASS",
        notes="composition looks good",
        operator_caller=_OPERATOR,
    )
    assert reviewed["integration"]["promotable"] is True
    assert (repo / "a.py").read_text(encoding="utf-8") == "A = 0\n"

    promotion = plane.promote_integration(run["integration_id"], repo_root=repo)
    assert promotion["promotion_id"]
    assert (repo / "a.py").read_text(encoding="utf-8") == "A = 1\n"
    assert (repo / "b.py").read_text(encoding="utf-8") == "B = 2\n"

    parent_plan = store.read("plan_nodes", parent["node_id"])
    assert parent_plan["status"] == "approved"
    integ = store.read("integration_runs", run["integration_id"])
    assert integ["status"] == "promoted"


def test_stale_baseline_blocks_integration_promotion(tmp_path):
    repo, store, parent, _leaves = _parent_with_leaves(tmp_path)
    service = IntegrationService(store)
    run = service.assemble(parent["node_id"], repo_root=repo)
    service.record_review(
        run["integration_id"], verdict="PASS", operator_caller=_OPERATOR
    )
    (repo / "neighbor.txt").write_text("race\n", encoding="utf-8")
    with pytest.raises(AtomicStaleError, match="baseline changed"):
        service.promote(run["integration_id"], repo_root=repo)
    assert (repo / "a.py").read_text(encoding="utf-8") == "A = 0\n"


def test_end_to_end_fake_client_with_atomic_gates(tmp_path):
    """Leaf atomic path + parent integration still pass validation/review gates."""
    repo, store, parent, leaves = _parent_with_leaves(tmp_path)
    # Leaf product files still baseline (node-approved only).
    assert (repo / "a.py").read_text(encoding="utf-8") == "A = 0\n"

    plane = ControlPlane(store)
    run = plane.assemble_integration(parent["node_id"], repo_root=repo)
    plane.review_integration(
        run["integration_id"], verdict="PASS", operator_caller=_OPERATOR
    )
    plane.promote_integration(run["integration_id"], repo_root=repo)

    assert (repo / "a.py").read_text(encoding="utf-8") == "A = 1\n"
    assert (repo / "b.py").read_text(encoding="utf-8") == "B = 2\n"
    # Each leaf still has validation + review records from AtomicHarness.
    for leaf in leaves:
        validation = store.read("validations", leaf["validation_id"])
        review = store.read("reviews", leaf["review_id"])
        assert validation["passed"] is True
        assert review["verdict"] == "PASS"


def test_assemble_seeds_full_baseline_without_treating_baseline_as_writes(tmp_path):
    repo, store, parent, _leaves = _parent_with_leaves(tmp_path)
    (repo / "keep.txt").write_text("baseline stays\n", encoding="utf-8")
    service = IntegrationService(store)
    run = service.assemble(parent["node_id"], repo_root=repo)
    workspace = Path(run["workspace"])
    assert (workspace / "keep.txt").read_text(encoding="utf-8") == "baseline stays\n"
    assert (workspace / "a.py").read_text(encoding="utf-8") == "A = 1\n"
    # Undeclared-write check passes; baseline files are not mistaken for writes.
    check = next(c for c in run["checks"] if c["name"] == "no_undeclared_writes")
    assert check["result"] == "pass"
    assert run["diff"]["baseline_path_count"] >= 3
    # Product still unchanged.
    assert (repo / "a.py").read_text(encoding="utf-8") == "A = 0\n"
    assert (repo / "keep.txt").read_text(encoding="utf-8") == "baseline stays\n"


def test_nested_parent_not_flattened_without_approved_integration(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("A = 0\n", encoding="utf-8")
    state = tmp_path / "state"
    store = AtomicStore(state)
    plane = ControlPlane(store)
    root = plane.submit_request(
        request_text="root",
        objective="root",
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
    nested = plane.submit_request(
        request_text="nested",
        objective="nested",
        artifact_path="",
        repo_root=repo,
        plan_kind="parent",
        parent_node_id=root["node_id"],
        priority_components={
            "urgency": 0.5,
            "impact": 0.5,
            "freshness": 1.0,
            "operator_boost": 0.0,
            "blocked_penalty": 0.0,
            "dependency_penalty": 0.0,
        },
    )
    leaf = plane.submit_request(
        request_text="leaf under nested",
        objective="leaf",
        artifact_path="a.py",
        repo_root=repo,
        validators=["python_syntax"],
        plan_kind="leaf",
        parent_node_id=nested["node_id"],
        priority_components={
            "urgency": 0.5,
            "impact": 0.5,
            "freshness": 1.0,
            "operator_boost": 0.0,
            "blocked_penalty": 0.0,
            "dependency_penalty": 0.0,
        },
    )
    AtomicHarness(store).run_attempt(
        task_id=leaf["task_id"],
        repo_root=repo,
        client=FakeAtomicClient("A = 1\n"),
        model="local-model",
        reasoning="off",
        temperature=0,
        worker_max_output_tokens=100,
        review_max_output_tokens=100,
        strategy_summary="write a",
    )
    # Ancestor must not silently flatten the nested leaf without nested approval.
    with pytest.raises(IntegrationError, match="nested parent|approved IntegrationRun"):
        IntegrationService(store).assemble(root["node_id"], repo_root=repo)


def test_promotion_rollback_on_post_write_hash_failure(tmp_path, monkeypatch):
    repo, store, parent, leaves = _parent_with_leaves(tmp_path)
    service = IntegrationService(store)
    run = service.assemble(parent["node_id"], repo_root=repo)
    service.record_review(
        run["integration_id"], verdict="PASS", operator_caller=_OPERATOR
    )

    original_read = Path.read_bytes
    call_count = {"n": 0}

    def flaky_read(self):
        # Allow staging/pre-checks, then fail a post-write verification read
        # for the second product file after the first replace succeeded.
        data = original_read(self)
        path_s = str(self)
        if path_s.endswith("b.py") and ".cosmo" not in path_s and "artifacts" not in path_s:
            call_count["n"] += 1
            # First reads are baseline/stage; fail once after a write by
            # returning corrupted bytes when the product file already has new content.
            if data == b"B = 2\n" and (repo / "a.py").read_text(encoding="utf-8") == "A = 1\n":
                return b"CORRUPT\n"
        return data

    monkeypatch.setattr(Path, "read_bytes", flaky_read)
    with pytest.raises(IntegrationError, match="hash"):
        service.promote(run["integration_id"], repo_root=repo)
    # Product must not remain partially applied.
    assert (repo / "a.py").read_text(encoding="utf-8") == "A = 0\n"
    assert (repo / "b.py").read_text(encoding="utf-8") == "B = 0\n"
    # The prepared intent stays behind for inspection but nothing committed.
    intents = store.list_records("integration_promotion_intents")
    assert len(intents) == 1
    assert intents[0]["status"] == "prepared"


def test_promotion_writes_durable_intent_and_stays_idempotent(tmp_path):
    repo, store, parent, leaves = _parent_with_leaves(tmp_path)
    service = IntegrationService(store)
    run = service.assemble(parent["node_id"], repo_root=repo)
    service.record_review(
        run["integration_id"], verdict="PASS", operator_caller=_OPERATOR
    )

    promotion = service.promote(run["integration_id"], repo_root=repo)

    intents = [
        item
        for item in store.list_records("integration_promotion_intents")
        if item.get("integration_id") == run["integration_id"]
    ]
    assert len(intents) == 1
    intent = intents[0]
    assert intent["status"] == "committed"
    assert intent["promotion_id"] == promotion["promotion_id"]
    assert {item["logical_path"] for item in intent["applied_paths"]} == {
        "a.py",
        "b.py",
    }
    events = [
        json.loads(line)
        for line in (store.root / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(e["type"] == "integration_promotion_prepared" for e in events)

    # A repeat promote returns the same promotion without new intents.
    again = service.promote(run["integration_id"], repo_root=repo)
    assert again["promotion_id"] == promotion["promotion_id"]
    assert len(store.list_records("integration_promotion_intents")) == 1


class _SimulatedCrash(Exception):
    pass


def test_recovery_finalizes_crash_after_writes_before_bookkeeping(tmp_path, monkeypatch):
    """Kill -9 between the last product write and durable bookkeeping."""
    repo, store, parent, leaves = _parent_with_leaves(tmp_path)
    service = IntegrationService(store)
    run = service.assemble(parent["node_id"], repo_root=repo)
    service.record_review(
        run["integration_id"], verdict="PASS", operator_caller=_OPERATOR
    )

    real_write = store.write

    def crashing_write(kind, identifier, value):
        if kind == "promotions":
            raise _SimulatedCrash("process died after product writes")
        return real_write(kind, identifier, value)

    monkeypatch.setattr(store, "write", crashing_write)
    with pytest.raises(_SimulatedCrash):
        service.promote(run["integration_id"], repo_root=repo)
    monkeypatch.undo()

    # Product bytes are applied, but no promotion was ever recorded.
    assert (repo / "a.py").read_text(encoding="utf-8") == "A = 1\n"
    assert (repo / "b.py").read_text(encoding="utf-8") == "B = 2\n"
    assert (
        store.read("integration_runs", run["integration_id"])["status"]
        == "reviewed"
    )
    assert not store.list_records("promotions")

    promotion = service.promote(run["integration_id"], repo_root=repo)
    assert promotion["recovered"] is True
    assert promotion["intent_id"]
    assert len(store.list_records("promotions")) == 1

    integ = store.read("integration_runs", run["integration_id"])
    assert integ["status"] == "promoted"
    assert integ["promotion_id"] == promotion["promotion_id"]

    intent = store.list_records("integration_promotion_intents")[-1]
    assert intent["status"] == "committed"
    assert intent["promotion_id"] == promotion["promotion_id"]

    parent_plan = store.read("plan_nodes", parent["node_id"])
    assert parent_plan["status"] == "approved"
    for leaf in leaves:
        node = store.read("plan_nodes", leaf["node_id"])
        assert node["status"] == "completed"

    events = [
        json.loads(line)
        for line in (store.root / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(e["type"] == "integration_promotion_recovered" for e in events)


def test_recovery_reuses_existing_promotion_record(tmp_path, monkeypatch):
    """Kill -9 after the promotion record but before the run/status update."""
    repo, store, parent, leaves = _parent_with_leaves(tmp_path)
    plane = ControlPlane(store)
    run = plane.assemble_integration(parent["node_id"], repo_root=repo)
    plane.review_integration(
        run["integration_id"], verdict="PASS", operator_caller=_OPERATOR
    )

    real_replace = store.replace

    def crashing_replace(kind, identifier, value):
        if kind == "integration_runs":
            raise _SimulatedCrash("process died after promotion record")
        return real_replace(kind, identifier, value)

    monkeypatch.setattr(store, "replace", crashing_replace)
    with pytest.raises(_SimulatedCrash):
        plane.promote_integration(run["integration_id"], repo_root=repo)
    monkeypatch.undo()

    assert (repo / "a.py").read_text(encoding="utf-8") == "A = 1\n"
    assert len(store.list_records("promotions")) == 1

    promotion = plane.promote_integration(run["integration_id"], repo_root=repo)
    # The pre-crash promotion record is reused verbatim, never duplicated.
    assert len(store.list_records("promotions")) == 1
    assert not promotion.get("recovered")

    integ = store.read("integration_runs", run["integration_id"])
    assert integ["status"] == "promoted"
    assert integ["promotion_id"] == promotion["promotion_id"]


def test_torn_product_state_fails_closed_and_never_guesses(tmp_path):
    """A partially applied tree matches neither post-tree nor baseline."""
    repo, store, parent, leaves = _parent_with_leaves(tmp_path)
    service = IntegrationService(store)
    run = service.assemble(parent["node_id"], repo_root=repo)
    service.record_review(
        run["integration_id"], verdict="PASS", operator_caller=_OPERATOR
    )

    real_write = store.write

    def crashing_write(kind, identifier, value):
        if kind == "promotions":
            raise _SimulatedCrash("crash after writes")
        return real_write(kind, identifier, value)

    original_write = service.store.write
    service.store.write = crashing_write
    try:
        with pytest.raises(_SimulatedCrash):
            service.promote(run["integration_id"], repo_root=repo)
    finally:
        service.store.write = original_write
    # Simulate a torn application by reverting exactly one file.
    (repo / "b.py").write_text("B = 0\n", encoding="utf-8")

    with pytest.raises(AtomicStaleError, match="baseline changed"):
        service.promote(run["integration_id"], repo_root=repo)
    # Nothing further mutated; the gated path marks the run stale instead of
    # guessing, and the prepared intent remains for operator inspection.
    assert (repo / "b.py").read_text(encoding="utf-8") == "B = 0\n"
    integ = store.read("integration_runs", run["integration_id"])
    assert integ["status"] == "stale"
    assert integ["stale_reason"]["observed_tree"]
    assert not store.list_records("promotions")
    intents = store.list_records("integration_promotion_intents")
    assert len(intents) == 1
    assert intents[0]["status"] == "prepared"


def test_recovery_rejects_repo_root_substitution(tmp_path):
    repo, store, parent, leaves = _parent_with_leaves(tmp_path)
    service = IntegrationService(store)
    run = service.assemble(parent["node_id"], repo_root=repo)
    service.record_review(
        run["integration_id"], verdict="PASS", operator_caller=_OPERATOR
    )

    real_write = store.write

    def crashing_write(kind, identifier, value):
        if kind == "promotions":
            raise _SimulatedCrash("crash after writes")
        return real_write(kind, identifier, value)

    original_write = service.store.write
    service.store.write = crashing_write
    try:
        with pytest.raises(_SimulatedCrash):
            service.promote(run["integration_id"], repo_root=repo)
    finally:
        service.store.write = original_write

    from cosmo.atomic import utc_now

    tampered = dict(store.list_records("integration_promotion_intents")[-1])
    tampered["repo_root"] = str(tmp_path / "elsewhere")
    store.replace(
        "integration_promotion_intents", tampered["intent_id"], tampered
    )

    with pytest.raises(IntegrationError, match="prepared"):
        service.promote(run["integration_id"], repo_root=repo)
    # Product bytes were already applied before the crash; recovery refuses to
    # finalize under a different root rather than blessing a moved target.
    assert (repo / "a.py").read_text(encoding="utf-8") == "A = 1\n"
    assert (
        store.read("integration_runs", run["integration_id"])["status"]
        == "reviewed"
    )
