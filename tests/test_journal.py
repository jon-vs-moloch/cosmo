import json

import pytest

from cosmo.cli import _reasoning_probe
from cosmo.convergence import REVISION_INSTRUCTION
from cosmo.evolution import (
    EvolutionConfig,
    EvolutionSearch,
    EvolutionWriter,
    PromptGenome,
)
from cosmo.journal import (
    InferenceJournalError,
    JournaledInferenceClient,
    RunDirectoryLock,
)
from cosmo.inference import InferenceError
from cosmo.models import BenchmarkCase, Generation
from cosmo.reporting import write_json


class FakeClient:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        role = kwargs["role"]
        content = "4"
        return Generation(
            role=role,
            content=content,
            reasoning_hash=None,
            stats={
                "input_tokens": 10,
                "total_output_tokens": 2,
                "reasoning_output_tokens": 0,
                "transport_attempts": 1,
                "provider_cost": 0,
            },
            latency_seconds=0.25,
            request_hash=f"request-{len(self.calls)}",
            model_instance_id=kwargs["model"],
        )


def _chat(client, *, operation_id="operation", prompt="prompt"):
    return client.chat(
        model="model",
        role="draft",
        system_prompt="system",
        prompt=prompt,
        reasoning="off",
        temperature=0,
        max_output_tokens=100,
        operation_id=operation_id,
    )


def test_completed_operation_is_replayed_and_accounted_once(tmp_path):
    underlying = FakeClient()
    client = JournaledInferenceClient(underlying, tmp_path / "journal")

    first = _chat(client)
    second = _chat(client)

    assert len(underlying.calls) == 1
    assert first.request_hash == second.request_hash
    assert first.stats["journal_replayed"] is False
    assert second.stats["journal_replayed"] is True
    work = client.work()
    assert work["completed_operations"] == 1
    assert work["total_tokens"] == 12
    assert work["provider_cost"] == 0
    assert work["accounting_complete"] is True


def test_operation_spec_change_fails_closed(tmp_path):
    client = JournaledInferenceClient(FakeClient(), tmp_path / "journal")
    _chat(client)

    with pytest.raises(
        InferenceJournalError, match="operation changed on resume"
    ):
        _chat(client, prompt="different")


def test_inflight_receipt_is_retried_and_marked_unknown(tmp_path):
    underlying = FakeClient()
    client = JournaledInferenceClient(underlying, tmp_path / "journal")
    operation_id = "interrupted"
    spec_hash = client._spec_hash(
        model="model",
        role="draft",
        system_prompt="system",
        prompt="prompt",
        reasoning="off",
        temperature=0,
        max_output_tokens=100,
        response_format=None,
    )
    write_json(
        client._path(operation_id),
        {
            "schema_version": "cosmo.inference-receipt.v1",
            "operation_id": operation_id,
            "spec_sha256": spec_hash,
            "state": "in_flight",
            "attempts": [
                {
                    "number": 1,
                    "started_at": "before-crash",
                    "status": "in_flight",
                }
            ],
        },
    )

    generation = _chat(client, operation_id=operation_id)

    assert generation.stats["journal_prior_unknown_attempts"] == 1
    work = client.work()
    assert work["unknown_outcome_attempts"] == 1
    assert work["accounting_complete"] is False


def test_post_response_validation_failure_is_explicitly_unaccounted(
    tmp_path,
):
    class InvalidResponseError(InferenceError):
        accounting_outcome = "response_invalid"
        observed_usage = {
            "input_tokens": 10,
            "total_output_tokens": 2,
        }

    class InvalidResponseClient:
        def chat(self, **_kwargs):
            raise InvalidResponseError("LM Studio response omitted usage")

    client = JournaledInferenceClient(
        InvalidResponseClient(), tmp_path / "journal"
    )
    with pytest.raises(InvalidResponseError, match="omitted usage"):
        _chat(client)

    receipt = json.loads(
        next((tmp_path / "journal").glob("*.json")).read_text()
    )
    attempt = receipt["attempts"][0]
    assert attempt["status"] == "outcome_unknown"
    assert attempt["accounting_outcome"] == "response_invalid"
    assert attempt["error"]["observed_usage"]["input_tokens"] == 10
    work = client.work()
    assert work["response_invalid_attempts"] == 1
    assert work["unknown_outcome_attempts"] == 0
    assert work["accounting_complete"] is False


def test_journaled_capability_probe_replays_and_enters_lifecycle_total(
    tmp_path,
):
    class ReasoningClient(FakeClient):
        def chat(self, **kwargs):
            generation = super().chat(**kwargs)
            generation.stats["reasoning_output_tokens"] = (
                3 if kwargs["reasoning"] == "on" else 0
            )
            return generation

    underlying = ReasoningClient()
    client = JournaledInferenceClient(underlying, tmp_path / "journal")

    first, first_valid = _reasoning_probe(
        client,
        "model",
        operation_scope="preflight/reasoning",
    )
    second, second_valid = _reasoning_probe(
        client,
        "model",
        operation_scope="preflight/reasoning",
    )

    assert first_valid is second_valid is True
    assert len(underlying.calls) == 2
    assert first["off"]["content_sha256"] == second["off"]["content_sha256"]
    work = client.work()
    assert work["completed_operations"] == 2
    assert work["by_role"]["capability_probe"]["completed_operations"] == 2
    assert work["total_tokens"] == 24
    assert work["accounting_complete"] is True


def test_run_directory_lock_rejects_concurrent_writer(tmp_path):
    run_dir = tmp_path / "search"
    first = RunDirectoryLock(run_dir)
    second = RunDirectoryLock(run_dir)
    first.acquire()
    try:
        with pytest.raises(
            InferenceJournalError, match="live writer"
        ):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    second.release()


def test_generation_zero_resume_reuses_seed_receipt(tmp_path):
    case = BenchmarkCase(
        case_id="case",
        family="test",
        prompt="Compute 2 + 3.",
        output_contract="Return only the integer.",
        evaluator={"kind": "exact", "expected": "5"},
    )
    run_dir = tmp_path / "search"
    underlying = FakeClient()
    first_writer = EvolutionWriter(run_dir)
    first = EvolutionSearch(
        client=JournaledInferenceClient(
            underlying, run_dir / "inference-journal"
        ),
        model="model",
        cases=[case],
        config=EvolutionConfig(generations=0, max_cycles=1),
        writer=first_writer,
    )
    first.prepare_seed_drafts()
    base_revision = PromptGenome.create(
        kind="revision",
        text=REVISION_INSTRUCTION,
        generation=0,
        hypothesis="bootstrap",
    )
    first._register(base_revision)
    first.evaluate_revision(base_revision, generation=0)
    calls_before_resume = len(underlying.calls)
    assert sum(
        call["role"] == "evolution_seed_draft"
        for call in underlying.calls
    ) == 1
    assert not (run_dir / "state.json").exists()

    resumed = EvolutionSearch(
        client=JournaledInferenceClient(
            underlying, run_dir / "inference-journal"
        ),
        model="model",
        cases=[case],
        config=EvolutionConfig(generations=0, max_cycles=1),
        writer=EvolutionWriter(run_dir, resume=True),
    )
    summary = resumed.run(resume=True)

    assert sum(
        call["role"] == "evolution_seed_draft"
        for call in underlying.calls
    ) == 1
    assert len(underlying.calls) == calls_before_resume
    assert summary["search_work"]["accounting_source"] == "inference_journal"
    assert (run_dir / "state.json").is_file()
    assert list((run_dir / "orphaned").rglob("case.json"))
    receipts = list((run_dir / "inference-journal").glob("*.json"))
    assert receipts
    assert all(
        json.loads(path.read_text())["state"] == "completed"
        for path in receipts
    )
