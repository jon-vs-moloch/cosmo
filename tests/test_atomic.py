import json
from pathlib import Path

import pytest

from cosmo.atomic import (
    AtomicHarness,
    AtomicHarnessError,
    AtomicStaleError,
    AtomicStore,
    parse_review,
)
from cosmo.models import Generation


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


def test_parse_review_accepts_coding_model_requirements_map():
    raw = {
        "review_id": "model-review",
        "result": "PASS",
        "requirements": {
            "objective": {
                "result": "PASS",
                "evidence": "python_syntax passed",
            }
        },
        "observations": {"code_quality": []},
        "hypotheses": [],
    }
    parsed = parse_review(json.dumps(raw), required_requirement_ids=["objective"])
    assert parsed["verdict"] == "PASS"
    assert parsed["requirement_results"][0]["evidence"] == ["python_syntax passed"]


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


def _submitted(tmp_path, validators=("python_syntax",)):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "target.py").write_text("value = 1\n", encoding="utf-8")
    state = tmp_path / "state"
    result = AtomicHarness(AtomicStore(state)).submit(
        request_text="Make the target value two.",
        objective="Set value to 2.",
        artifact_path="target.py",
        repo_root=repo,
        validators=validators,
    )
    return repo, state, result["task_id"]


def test_submit_creates_immutable_request_versioned_spec_and_leaf_plan(tmp_path):
    _repo, state, task_id = _submitted(tmp_path)
    store = AtomicStore(state)
    task = store.read("tasks", task_id)
    assert task["current_spec_version"] == 1
    assert store.read("requests", task["linked_request_ids"][0])["original_text"]
    plan = store.read("plan_nodes", task["plan_node_id"])
    assert plan["kind"] == "leaf"
    assert plan["writable_artifact"] == "target.py"


def test_worker_does_not_touch_product_until_explicit_promotion(tmp_path):
    repo, state, task_id = _submitted(tmp_path)
    client = FakeAtomicClient()
    result = AtomicHarness(AtomicStore(state)).run_attempt(
        task_id=task_id,
        repo_root=repo,
        client=client,
        model="local-model",
        reasoning="off",
        temperature=0,
        worker_max_output_tokens=100,
        review_max_output_tokens=100,
        strategy_summary="replace the assignment with the requested value",
    )
    assert result["promotable"] is True
    assert (repo / "target.py").read_text(encoding="utf-8") == "value = 1\n"
    store = AtomicStore(state)
    artifact = store.read("artifacts", result["artifact_version_id"])
    validation = store.read("validations", result["validation_id"])
    assert validation["candidate_sha256"] == artifact["content_hash"]
    promotion = AtomicHarness(AtomicStore(state)).promote(
        attempt_id=result["attempt_id"], repo_root=repo
    )
    assert promotion["promoted_sha256"]
    assert (repo / "target.py").read_text(encoding="utf-8") == "value = 2\n"
    assert [call["role"] for call in client.calls] == [
        "atomic_worker",
        "atomic_reviewer",
    ]


def test_worker_normalizes_one_outer_code_fence_and_records_it(tmp_path):
    repo, state, task_id = _submitted(tmp_path)
    client = FakeAtomicClient(candidate="```python\nvalue = 2\n```")
    result = AtomicHarness(AtomicStore(state)).run_attempt(
        task_id=task_id,
        repo_root=repo,
        client=client,
        model="local-model",
        reasoning="off",
        temperature=0,
        worker_max_output_tokens=100,
        review_max_output_tokens=100,
        strategy_summary="normalize a model presentation wrapper",
    )
    assert result["promotable"] is True
    store = AtomicStore(state)
    artifact = store.read("artifacts", result["artifact_version_id"])
    assert store.path("artifacts", result["artifact_version_id"])
    assert Path(artifact["content_path"]).read_text(encoding="utf-8") == "value = 2\n"
    attempt = store.read("attempt_states", result["attempt_id"])
    assert attempt["worker_protocol_normalization"]["kind"] == "outer_markdown_fence"


@pytest.mark.parametrize(
    "candidate",
    [
        "```diff\n--- a/target.txt\n+++ b/target.txt\n@@ -1 +1 @@\n-old\n+new\n```",
        "*** Begin Patch\n*** Update File: target.txt\n@@\n-old\n+new\n*** End Patch",
    ],
)
def test_worker_rejects_patch_before_persisting_candidate(tmp_path, candidate):
    repo, state, task_id = _submitted(tmp_path, validators=("non_empty", "utf8"))
    client = FakeAtomicClient(candidate=candidate)
    with pytest.raises(AtomicHarnessError, match="not a diff or patch"):
        AtomicHarness(AtomicStore(state)).run_attempt(
            task_id=task_id,
            repo_root=repo,
            client=client,
            model="local-model",
            reasoning="off",
            temperature=0,
            worker_max_output_tokens=100,
            review_max_output_tokens=100,
            strategy_summary="reject a patch-shaped worker response",
        )
    assert not list((state / "artifacts").glob("*.json"))


def test_state_root_inside_repo_is_excluded_from_product_revision(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "target.py").write_text("value = 1\n", encoding="utf-8")
    state = repo / ".cosmo"
    harness = AtomicHarness(AtomicStore(state))
    submitted = harness.submit(
        request_text="Make the target value two.",
        objective="Set value to 2.",
        artifact_path="target.py",
        repo_root=repo,
        validators=["python_syntax"],
    )
    result = harness.run_attempt(
        task_id=submitted["task_id"],
        repo_root=repo,
        client=FakeAtomicClient(),
        model="local-model",
        reasoning="off",
        temperature=0,
        worker_max_output_tokens=100,
        review_max_output_tokens=100,
        strategy_summary="replace the assignment",
    )
    harness.promote(attempt_id=result["attempt_id"], repo_root=repo)
    assert (repo / "target.py").read_text(encoding="utf-8") == "value = 2\n"


def test_state_root_ancestor_of_repo_is_rejected(tmp_path):
    state = tmp_path / "state"
    repo = state / "repo"
    repo.mkdir(parents=True)
    (repo / "target.py").write_text("value = 1\n", encoding="utf-8")
    with pytest.raises(AtomicHarnessError, match="state_root must not"):
        AtomicHarness(AtomicStore(state)).submit(
            request_text="set value",
            objective="set value",
            artifact_path="target.py",
            repo_root=repo,
            validators=["python_syntax"],
        )


def test_validation_completes_before_independent_review(tmp_path):
    repo, state, task_id = _submitted(tmp_path)
    client = FakeAtomicClient(candidate="value =\n")
    result = AtomicHarness(AtomicStore(state)).run_attempt(
        task_id=task_id,
        repo_root=repo,
        client=client,
        model="local-model",
        reasoning="off",
        temperature=0,
        worker_max_output_tokens=100,
        review_max_output_tokens=100,
        strategy_summary="try a deliberately invalid candidate",
    )
    assert result["validation_passed"] is False
    assert result["review_verdict"] == "PASS"
    events = [
        json.loads(line)["type"]
        for line in (state / "events.jsonl").read_text().splitlines()
    ]
    assert events.index("validation_completed") < events.index("review_completed")
    with pytest.raises(AtomicHarnessError, match="validation PASS"):
        AtomicHarness(AtomicStore(state)).promote(
            attempt_id=result["attempt_id"], repo_root=repo
        )


def test_write_scope_rejects_undeclared_changes(tmp_path, monkeypatch):
    repo, state, task_id = _submitted(tmp_path)
    client = FakeAtomicClient()
    calls = {"count": 0}

    def fake_snapshot(_root):
        calls["count"] += 1
        return {} if calls["count"] == 1 else {"neighbor.txt": "changed"}

    monkeypatch.setattr("cosmo.atomic._snapshot", fake_snapshot)
    with pytest.raises(AtomicHarnessError, match="undeclared paths"):
        AtomicHarness(AtomicStore(state)).run_attempt(
            task_id=task_id,
            repo_root=repo,
            client=client,
            model="local-model",
            reasoning="off",
            temperature=0,
            worker_max_output_tokens=100,
            review_max_output_tokens=100,
            strategy_summary="attempt a multi-file change",
        )


def test_no_op_candidate_is_rejected_before_review(tmp_path):
    repo, state, task_id = _submitted(tmp_path)
    client = FakeAtomicClient(candidate="value = 1\n")
    with pytest.raises(AtomicHarnessError, match="no artifact change"):
        AtomicHarness(AtomicStore(state)).run_attempt(
            task_id=task_id,
            repo_root=repo,
            client=client,
            model="local-model",
            reasoning="off",
            temperature=0,
            worker_max_output_tokens=100,
            review_max_output_tokens=100,
            strategy_summary="confirm whether the requested change is already present",
        )
    assert [call["role"] for call in client.calls] == ["atomic_worker"]


def test_retries_require_explicit_differentiation_metadata(tmp_path):
    repo, state, task_id = _submitted(tmp_path)
    harness = AtomicHarness(AtomicStore(state))
    harness.run_attempt(
        task_id=task_id,
        repo_root=repo,
        client=FakeAtomicClient(),
        model="local-model",
        reasoning="off",
        temperature=0,
        worker_max_output_tokens=100,
        review_max_output_tokens=100,
        strategy_summary="first attempt",
    )
    with pytest.raises(AtomicHarnessError, match="retries require"):
        harness.run_attempt(
            task_id=task_id,
            repo_root=repo,
            client=FakeAtomicClient(),
            model="local-model",
            reasoning="off",
            temperature=0,
            worker_max_output_tokens=100,
            review_max_output_tokens=100,
            strategy_summary="try again",
        )


def test_stale_baseline_blocks_promotion(tmp_path):
    repo, state, task_id = _submitted(tmp_path)
    result = AtomicHarness(AtomicStore(state)).run_attempt(
        task_id=task_id,
        repo_root=repo,
        client=FakeAtomicClient(),
        model="local-model",
        reasoning="off",
        temperature=0,
        worker_max_output_tokens=100,
        review_max_output_tokens=100,
        strategy_summary="first attempt",
    )
    (repo / "target.py").write_text("value = 99\n", encoding="utf-8")
    with pytest.raises(AtomicStaleError, match="baseline changed"):
        AtomicHarness(AtomicStore(state)).promote(
            attempt_id=result["attempt_id"], repo_root=repo
        )


def test_stale_neighbor_blocks_promotion(tmp_path):
    repo, state, task_id = _submitted(tmp_path)
    result = AtomicHarness(AtomicStore(state)).run_attempt(
        task_id=task_id,
        repo_root=repo,
        client=FakeAtomicClient(),
        model="local-model",
        reasoning="off",
        temperature=0,
        worker_max_output_tokens=100,
        review_max_output_tokens=100,
        strategy_summary="change only the declared target",
    )
    (repo / "neighbor.txt").write_text("concurrent edit\n", encoding="utf-8")
    with pytest.raises(AtomicStaleError, match="baseline changed"):
        AtomicHarness(AtomicStore(state)).promote(
            attempt_id=result["attempt_id"], repo_root=repo
        )


def test_retry_cannot_repeat_identical_strategy(tmp_path):
    repo, state, task_id = _submitted(tmp_path)
    harness = AtomicHarness(AtomicStore(state))
    first = harness.run_attempt(
        task_id=task_id,
        repo_root=repo,
        client=FakeAtomicClient(),
        model="local-model",
        reasoning="off",
        temperature=0,
        worker_max_output_tokens=100,
        review_max_output_tokens=100,
        strategy_summary="replace the assignment",
    )
    with pytest.raises(AtomicHarnessError, match="identical"):
        harness.run_attempt(
            task_id=task_id,
            repo_root=repo,
            client=FakeAtomicClient(),
            model="local-model",
            reasoning="off",
            temperature=0,
            worker_max_output_tokens=100,
            review_max_output_tokens=100,
            strategy_summary="replace the assignment",
            retry_contract={
                "addresses": ["the failed review"],
                "preserves": ["the original contract"],
                "differs_from": [first["attempt_id"]],
                "expected_improvement": ["satisfy the objective"],
            },
        )


def test_retry_context_includes_compact_convergence_projection(tmp_path):
    repo, state, task_id = _submitted(tmp_path)
    harness = AtomicHarness(AtomicStore(state))
    first = harness.run_attempt(
        task_id=task_id,
        repo_root=repo,
        client=FakeAtomicClient(review=_review("REVISE")),
        model="local-model",
        reasoning="off",
        temperature=0,
        worker_max_output_tokens=100,
        review_max_output_tokens=100,
        strategy_summary="use the first implementation strategy",
    )
    assert first["promotable"] is False

    second = harness.run_attempt(
        task_id=task_id,
        repo_root=repo,
        client=FakeAtomicClient(),
        model="local-model",
        reasoning="off",
        temperature=0,
        worker_max_output_tokens=100,
        review_max_output_tokens=100,
        strategy_summary="use a materially different implementation strategy",
        retry_contract={
            "addresses": [first["review_id"]],
            "preserves": ["the objective"],
            "differs_from": [first["attempt_id"]],
            "expected_improvement": ["satisfy the review"],
        },
    )
    attempt = AtomicStore(state).read("attempts", second["attempt_id"])
    manifest = AtomicStore(state).read(
        "context_manifests", attempt["context_manifest_id"]
    )
    ledger_items = [
        item for item in manifest["items"] if item.get("kind") == "ledger_projection"
    ]
    assert len(ledger_items) == 1
    projection = ledger_items[0]["included_range_or_excerpt"]
    assert any(
        item["attempt_id"] == first["attempt_id"]
        for item in projection["rejected_strategies"]
    )


def test_promotion_intent_recovers_after_receipt_write_failure(tmp_path, monkeypatch):
    repo, state, task_id = _submitted(tmp_path)
    store = AtomicStore(state)
    harness = AtomicHarness(store)
    result = harness.run_attempt(
        task_id=task_id,
        repo_root=repo,
        client=FakeAtomicClient(),
        model="local-model",
        reasoning="off",
        temperature=0,
        worker_max_output_tokens=100,
        review_max_output_tokens=100,
        strategy_summary="replace the assignment and preserve syntax",
    )
    original_write = store.write

    def fail_promotion_receipt(kind, identifier, value):
        if kind == "promotions":
            raise OSError("simulated receipt crash")
        return original_write(kind, identifier, value)

    monkeypatch.setattr(store, "write", fail_promotion_receipt)
    with pytest.raises(OSError, match="receipt crash"):
        harness.promote(attempt_id=result["attempt_id"], repo_root=repo)
    monkeypatch.setattr(store, "write", original_write)

    recovered = harness.promote(attempt_id=result["attempt_id"], repo_root=repo)
    assert recovered["recovered"] is True
    attempt = harness.show(result["attempt_id"])["record"]
    assert attempt["status"] == "promoted"
    assert (repo / "target.py").read_text(encoding="utf-8") == "value = 2\n"


def test_review_parser_rejects_unknown_verdict():
    with pytest.raises(AtomicHarnessError, match="unsupported review verdict"):
        parse_review(_review("MAYBE"))


def test_review_parser_requires_evidence_backed_requirement_results():
    value = json.loads(_review())
    value["requirement_results"][0]["evidence"] = []
    with pytest.raises(AtomicHarnessError, match="non-empty evidence"):
        parse_review(json.dumps(value), required_requirement_ids=["objective"])


def test_review_parser_accepts_fenced_canonical_json():
    assert parse_review(
        "```json\n" + _review() + "\n```",
        required_requirement_ids=["objective"],
    )["verdict"] == "PASS"


def test_review_parser_normalizes_compact_local_model_envelope():
    compact = {
        "review": {
            "status": "PASS",
            "findings": [
                {
                    "requirement_id": "objective",
                    "result": "PASS",
                    "evidence": "python_syntax validator passed",
                }
            ],
        }
    }
    parsed = parse_review(
        "```json\n" + json.dumps(compact) + "\n```",
        required_requirement_ids=["objective"],
    )
    assert parsed["verdict"] == "PASS"
    assert parsed["requirement_results"][0]["evidence"] == [
        "python_syntax validator passed"
    ]


def test_review_parser_normalizes_lmstudio_requirements_wrapper():
    compact = {
        "review": {
            "status": "PASS",
            "requirements": [
                {
                    "requirement_id": "objective",
                    "result": "PASS",
                    "evidence": "candidate passed python_syntax",
                }
            ],
            "observations": [],
            "hypotheses": [],
        }
    }
    parsed = parse_review(
        "```json\n" + json.dumps(compact) + "\n```",
        required_requirement_ids=["objective"],
    )
    assert parsed["verdict"] == "PASS"
    assert parsed["requirement_results"][0]["evidence"] == [
        "candidate passed python_syntax"
    ]


def test_record_identifier_cannot_escape_state_root(tmp_path):
    store = AtomicStore(tmp_path / "state")
    with pytest.raises(AtomicHarnessError, match="record identifier"):
        store.path("requests", "../../outside")
    with pytest.raises(AtomicHarnessError, match="record identifier"):
        AtomicHarness(store).show("../../outside")


def test_declared_artifact_cannot_escape_repo_root(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(AtomicHarnessError, match="normalized relative path"):
        AtomicHarness(AtomicStore(tmp_path / "state")).submit(
            request_text="bad path",
            objective="write outside the repository",
            artifact_path="../outside.py",
            repo_root=repo,
            validators=["python_syntax"],
        )
