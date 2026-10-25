import json

import pytest

from cosmo.evolution import (
    EvolutionConfig,
    EvolutionSearch,
    EvolutionSearchError,
    EvolutionWriter,
    MutationProposal,
    MutationProtocolError,
    PromptGenome,
    RevisionEvaluation,
    meta_fitness,
    mutation_schedule,
    parse_mutation_response,
    revision_child_selection_credit,
    revision_fitness,
)
from cosmo.models import BenchmarkCase, Generation
from cosmo.lmstudio import LMStudioError


def test_revision_fitness_rewards_accuracy_and_penalizes_tokens():
    low_cost = revision_fitness(
        successes=8,
        case_count=10,
        total_tokens=1_000,
        token_penalty_per_1k_per_case=0.05,
    )
    high_cost = revision_fitness(
        successes=8,
        case_count=10,
        total_tokens=10_000,
        token_penalty_per_1k_per_case=0.05,
    )
    more_accurate = revision_fitness(
        successes=9,
        case_count=10,
        total_tokens=10_000,
        token_penalty_per_1k_per_case=0.05,
    )
    assert low_cost[0] > high_cost[0]
    assert more_accurate[0] > high_cost[0]


def _evaluation(
    *,
    successes=6,
    tokens=1_000,
    false_returns=3,
    corrected=0,
):
    return RevisionEvaluation(
        genome_id="evaluation",
        generation=0,
        successes=successes,
        case_count=12,
        accuracy=successes / 12,
        total_tokens=tokens,
        mean_tokens_per_case=tokens / 12,
        token_penalty=0,
        fitness=successes / 12,
        valid=True,
        inference_seconds=1,
        diagnostics={
            "false_return_current_decisions": false_returns,
            "corrected_runs": corrected,
            "regressed_runs": 0,
            "protocol_errors": 0,
        },
    )


def test_revision_child_credit_prioritizes_accuracy_then_behavior_then_cost():
    parent = _evaluation()
    more_accurate = _evaluation(successes=7, tokens=100_000)
    fewer_false_returns = _evaluation(
        tokens=2_000,
        false_returns=2,
    )
    merely_cheaper = _evaluation(tokens=900)

    assert revision_child_selection_credit(parent, more_accurate) > 0
    assert revision_child_selection_credit(parent, fewer_false_returns) > (
        revision_child_selection_credit(parent, merely_cheaper)
    )


def test_mutation_schedule_honors_exact_call_budget_and_rotates_revision_parents():
    revisions = ["r0", "r1", "r2", "r3"]
    revisors = ["m0", "m1", "m2"]

    first = mutation_schedule(
        revision_parents=revisions,
        revisor_parents=revisors,
        calls=3,
        generation=1,
    )
    second = mutation_schedule(
        revision_parents=revisions,
        revisor_parents=revisors,
        calls=3,
        generation=2,
    )

    assert first == [("r0", "m0"), ("r1", "m1"), ("r2", "m2")]
    assert second == [("r3", "m1"), ("r0", "m2"), ("r1", "m0")]
    assert len(first) == len(second) == 3
    assert {revision for revision, _ in first + second} == set(revisions)


def test_mutation_parser_requires_exact_complete_arrays():
    payload = json.dumps(
        {
            "revision_prompt_mutations": [
                {"prompt": "revision one", "hypothesis": "h1"},
                {"prompt": "revision two", "hypothesis": "h2"},
            ],
            "revisor_prompt_mutations": [
                {"prompt": "revisor one", "hypothesis": "m1"}
            ],
        }
    )
    revisions, revisors = parse_mutation_response(
        payload, revision_count=2, revisor_count=1
    )
    assert [item["prompt"] for item in revisions] == [
        "revision one",
        "revision two",
    ]
    assert revisors[0]["prompt"] == "revisor one"

    with pytest.raises(MutationProtocolError, match="exactly 2"):
        parse_mutation_response(
            json.dumps(
                {
                    "revision_prompt_mutations": revisions[:1],
                    "revisor_prompt_mutations": revisors,
                }
            ),
            revision_count=2,
            revisor_count=1,
        )


def test_meta_fitness_uses_signed_deltas_and_invalid_slots():
    proposal = MutationProposal(
        proposal_id="p",
        generation=1,
        revisor_genome_id="m",
        revision_parent_id="r",
        generation_record=Generation(
            role="prompt_revisor",
            content="{}",
            reasoning_hash=None,
            stats={"input_tokens": 100, "total_output_tokens": 100},
            latency_seconds=1,
            request_hash="hash",
        ),
        requested_revision_children=3,
        requested_revisor_children=1,
        child_fitness_deltas=[0.2, -0.1],
        revision_slots=[
            {"status": "created", "child_fitness_delta": 0.2},
            {"status": "created", "child_fitness_delta": -0.1},
        ],
        revisor_slots=[{"status": "created"}],
    )
    metrics = meta_fitness(
        [proposal],
        meta_token_penalty_per_1k=0.01,
        invalid_mutation_penalty=0.05,
    )
    assert metrics["offspring_slots"] == 4
    assert metrics["mean_child_delta"] == pytest.approx(
        (0.2 - 0.1 - 0.05) / 3
    )
    assert metrics["mean_slot_credit"] == pytest.approx(
        (0.2 - 0.1 - 0.05 + 0.0) / 4
    )
    assert metrics["fitness"] < metrics["mean_slot_credit"]


def test_duplicate_revision_slot_cannot_claim_existing_genome_improvement():
    proposal = MutationProposal(
        proposal_id="p",
        generation=1,
        revisor_genome_id="m",
        revision_parent_id="bad-parent",
        generation_record=Generation(
            role="prompt_revisor",
            content="{}",
            reasoning_hash=None,
            stats={"input_tokens": 0, "total_output_tokens": 0},
            latency_seconds=0,
            request_hash="hash",
        ),
        requested_revision_children=1,
        requested_revisor_children=0,
        revision_slots=[
            {
                "status": "duplicate",
                "target_genome_id": "existing-champion",
                "child_fitness_delta": 0.5,
            }
        ],
    )
    metrics = meta_fitness(
        [proposal],
        meta_token_penalty_per_1k=0,
        invalid_mutation_penalty=0.05,
    )
    assert metrics["mean_child_delta"] == -0.05
    assert metrics["fitness"] == -0.05


class FakeEvolutionClient:
    def __init__(self):
        self.calls = []
        self.proposal_index = 0

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        role = kwargs["role"]
        if role == "evolution_seed_draft":
            content = "4"
        elif role == "prompt_revisor":
            self.proposal_index += 1
            content = json.dumps(
                {
                    "revision_prompt_mutations": [
                        {
                            "prompt": (
                                "Actively recompute the task, correct it, and return "
                                f"only the full artifact. Variant {self.proposal_index}."
                            ),
                            "hypothesis": "Independent recomputation will repair errors.",
                        }
                    ],
                    "revisor_prompt_mutations": [
                        {
                            "prompt": (
                                "Generate compact, independent-recomputation prompt "
                                f"mutations. Meta variant {self.proposal_index}."
                            ),
                            "hypothesis": "Focus the mutation operator on recomputation.",
                        }
                    ],
                }
            )
        elif "correct it" in kwargs["prompt"]:
            content = "5"
        else:
            current = kwargs["prompt"].split("CURRENT DRAFT\n", 1)[1].split(
                "\n\nCRITIC PROMPT", 1
            )[0]
            content = current
        return Generation(
            role=role,
            content=content,
            reasoning_hash=None,
            stats={
                "input_tokens": 20,
                "total_output_tokens": 5,
                "reasoning_output_tokens": 0,
            },
            latency_seconds=0.1,
            request_hash=f"request-{len(self.calls)}",
            model_instance_id=kwargs["model"],
        )


def test_nested_search_uses_fixed_draft_and_later_meta_child(tmp_path):
    case = BenchmarkCase(
        case_id="case",
        family="test",
        prompt="Compute 2 + 3.",
        output_contract="Return only the integer.",
        evaluator={"kind": "exact", "expected": "5"},
    )
    client = FakeEvolutionClient()
    writer = EvolutionWriter(tmp_path / "search")
    search = EvolutionSearch(
        client=client,
        model="model",
        cases=[case],
        config=EvolutionConfig(
            generations=2,
            revision_population=2,
            revisor_population=1,
            mutation_calls_per_generation=1,
            revision_mutations_per_call=1,
            revisor_mutations_per_call=1,
            max_cycles=2,
            mutation_temperature=0,
            token_penalty_per_1k_per_case=0,
            meta_token_penalty_per_1k=0,
        ),
        writer=writer,
    )
    summary = search.run()
    assert summary["champion_evaluation"]["accuracy"] == 1.0
    assert sum(call["role"] == "evolution_seed_draft" for call in client.calls) == 1

    revision_calls = [call for call in client.calls if call["role"] == "revision"]
    assert (
        sum("CURRENT DRAFT\n4" in call["prompt"] for call in revision_calls)
        == len(search.evaluations)
    )

    revisor_calls = [call for call in client.calls if call["role"] == "prompt_revisor"]
    assert len(revisor_calls) == 2
    assert any("Meta variant 1" in call["system_prompt"] for call in revisor_calls)
    for call in revisor_calls:
        assert case.prompt not in call["prompt"]
        assert '"expected"' not in call["prompt"]
        assert '"terminal_accuracy":' not in call["prompt"]
        assert '"fitness":' not in call["prompt"]
        assert '"child_selection_credit":' not in call["prompt"]
        assert '"final_minus_best":' not in call["prompt"]
        assert '"regressed_runs":' not in call["prompt"]
        assert '"revision_regressions":' not in call["prompt"]

    root_revisor = next(
        item
        for item in summary["revisor_ranking"]
        if item["genome"]["parent_id"] is None
    )
    assert root_revisor["metrics"]["mean_child_delta"] > 0
    assert (tmp_path / "search" / "state.json").is_file()
    event_types = {
        json.loads(line)["type"]
        for line in (tmp_path / "search" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    assert "mutation_credit_assigned" in event_types
    assert "generation_selected" in event_types


def test_correctness_target_records_durable_milestone_and_stops(tmp_path):
    case = BenchmarkCase(
        case_id="case",
        family="test",
        prompt="Compute 2 + 3.",
        output_contract="Return only the integer.",
        evaluator={"kind": "exact", "expected": "5"},
    )
    run_dir = tmp_path / "search"
    search = EvolutionSearch(
        client=FakeEvolutionClient(),
        model="model",
        cases=[case],
        config=EvolutionConfig(
            generations=None,
            revision_population=2,
            revisor_population=1,
            mutation_calls_per_generation=1,
            revision_mutations_per_call=1,
            revisor_mutations_per_call=1,
            max_cycles=2,
            mutation_temperature=0,
            token_penalty_per_1k_per_case=0,
            meta_token_penalty_per_1k=0,
            target_correct=1,
        ),
        writer=EvolutionWriter(run_dir),
    )
    summary = search.run()
    assert summary["stop_reason"] == "correctness_target"
    assert summary["completed_generation"] == 1
    assert summary["accuracy_progress"]["baseline_successes"] == 0
    assert summary["accuracy_progress"]["high_water_successes"] == 1
    assert summary["accuracy_progress"]["milestones"]["1"][
        "observed_successes"
    ] == 1
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["stop_reason"] == "correctness_target"
    assert state["target_crossing"]["threshold_successes"] == 1
    event_types = [
        json.loads(line)["type"]
        for line in (run_dir / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert event_types.count("accuracy_baseline_established") == 1
    assert event_types.count("accuracy_milestone_crossed") == 1
    assert event_types.count("correctness_target_reached") == 1


def test_macos_notification_attempt_is_at_most_once(tmp_path, monkeypatch):
    search, _genome = _bare_search(tmp_path, FailingRevisionClient("unused"))
    search.notify_milestones = True
    calls = []

    class Result:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(
        "cosmo.evolution.subprocess.run",
        lambda *args, **kwargs: calls.append((args, kwargs)) or Result(),
    )
    record = {
        "event_id": "accuracy-milestone:test:1-of-1",
        "observed_successes": 1,
        "case_count": 1,
        "search_generation": 1,
        "genome_id": "revision-abc",
    }
    search._notify_crossing(record)
    search._notify_crossing(record)
    assert len(calls) == 1
    receipts = list((tmp_path / "search" / "notifications").glob("*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text(encoding="utf-8"))["status"] == "accepted"


def test_prompt_genome_identity_deduplicates_transport_whitespace():
    first = PromptGenome.create(kind="revision", text="prompt\n", generation=0)
    second = PromptGenome.create(kind="revision", text="prompt", generation=1)
    assert first.genome_id == second.genome_id


def test_valid_revision_always_ranks_above_invalid_candidate(tmp_path):
    search, _genome = _bare_search(tmp_path, FailingRevisionClient("unused"))
    valid = PromptGenome.create(kind="revision", text="valid", generation=0)
    invalid = PromptGenome.create(kind="revision", text="invalid", generation=0)
    search._register(valid)
    search._register(invalid)
    common = {
        "generation": 0,
        "successes": 0,
        "case_count": 1,
        "accuracy": 0,
        "total_tokens": 100,
        "mean_tokens_per_case": 100,
        "token_penalty": 0.1,
        "inference_seconds": 1,
        "diagnostics": {},
    }
    search.evaluations[valid.genome_id] = RevisionEvaluation(
        genome_id=valid.genome_id,
        fitness=-0.1,
        valid=True,
        **common,
    )
    search.evaluations[invalid.genome_id] = RevisionEvaluation(
        genome_id=invalid.genome_id,
        fitness=0.0,
        valid=False,
        **common,
    )
    assert search._ranked_revisions()[0] == valid.genome_id


def test_revision_quality_key_prioritizes_correctness_over_scalar_fitness(
    tmp_path,
):
    search, _genome = _bare_search(tmp_path, FailingRevisionClient("unused"))
    lower_cost = PromptGenome.create(
        kind="revision", text="lower cost", generation=0
    )
    more_correct = PromptGenome.create(
        kind="revision", text="more correct", generation=1
    )
    search._register(lower_cost)
    search._register(more_correct)
    common = {
        "case_count": 12,
        "accuracy": 0.5,
        "mean_tokens_per_case": 100,
        "token_penalty": 0,
        "valid": True,
        "inference_seconds": 1,
        "diagnostics": {
            "false_return_current_decisions": 0,
            "regressed_runs": 0,
            "protocol_errors": 0,
            "corrected_runs": 0,
        },
    }
    search.evaluations[lower_cost.genome_id] = RevisionEvaluation(
        genome_id=lower_cost.genome_id,
        generation=0,
        successes=6,
        total_tokens=100,
        fitness=0.9,
        **common,
    )
    search.evaluations[more_correct.genome_id] = RevisionEvaluation(
        genome_id=more_correct.genome_id,
        generation=1,
        successes=7,
        total_tokens=100_000,
        fitness=-1.0,
        **common,
    )

    assert (
        search._revision_quality_key(more_correct.genome_id)
        < search._revision_quality_key(lower_cost.genome_id)
    )
    assert search._ranked_revisions()[0] == more_correct.genome_id


def test_invalid_revision_cannot_satisfy_correctness_target(tmp_path):
    search, genome = _bare_search(tmp_path, FailingRevisionClient("unused"))
    search.config = EvolutionConfig(
        generations=None,
        target_correct=1,
    )
    search.evaluations[genome.genome_id] = RevisionEvaluation(
        genome_id=genome.genome_id,
        generation=0,
        successes=1,
        case_count=1,
        accuracy=1.0,
        total_tokens=100,
        mean_tokens_per_case=100,
        token_penalty=0,
        fitness=1.0,
        valid=False,
        inference_seconds=1,
        diagnostics={"protocol_failures": 1},
    )
    milestones, target = search._record_accuracy_progress(0)
    assert milestones == []
    assert target is None
    assert search.accuracy_leader_genome_id is None


class FailingRevisionClient:
    def __init__(self, message):
        self.message = message

    def chat(self, **_kwargs):
        raise LMStudioError(self.message)


def _bare_search(tmp_path, client):
    case = BenchmarkCase(
        case_id="case",
        family="test",
        prompt="Compute 2 + 3.",
        output_contract="Return only the integer.",
        evaluator={"kind": "exact", "expected": "5"},
    )
    search = EvolutionSearch(
        client=client,
        model="model",
        cases=[case],
        config=EvolutionConfig(
            generations=1,
            max_cycles=2,
            revision_max_output_tokens=100,
            token_penalty_per_1k_per_case=0.05,
        ),
        writer=EvolutionWriter(tmp_path / "search"),
    )
    search.seed_candidates[case.case_id] = "4"
    genome = PromptGenome.create(
        kind="revision", text="Try to correct the draft.", generation=0
    )
    search._register(genome)
    return search, genome


def test_transport_outage_aborts_instead_of_becoming_cheap_candidate(tmp_path):
    search, genome = _bare_search(
        tmp_path,
        FailingRevisionClient("LM Studio unavailable at http://localhost"),
    )
    with pytest.raises(EvolutionSearchError, match="transport failure"):
        search.evaluate_revision(genome, generation=0)


def test_candidate_request_rejection_receives_full_cap_penalty(tmp_path):
    search, genome = _bare_search(
        tmp_path,
        FailingRevisionClient("LM Studio HTTP 400: prompt too long"),
    )
    evaluation = search.evaluate_revision(genome, generation=0)
    assert evaluation.accuracy == 0
    assert evaluation.total_tokens == 200
    assert evaluation.fitness < 0
    assert evaluation.diagnostics["candidate_request_failures"] == 1


class InvalidLocalRequestError(LMStudioError):
    status_code = 400


class InvalidLocalRequestClient:
    def chat(self, **_kwargs):
        raise InvalidLocalRequestError(
            "LM Studio HTTP 400: invalid candidate prompt",
        )


def test_local_candidate_request_rejection_is_invalid_not_fatal(tmp_path):
    search, genome = _bare_search(tmp_path, InvalidLocalRequestClient())
    evaluation = search.evaluate_revision(genome, generation=0)

    assert evaluation.valid is False
    assert evaluation.total_tokens == 200
    assert evaluation.diagnostics["candidate_request_failures"] == 1
    result = search._evaluation_results[genome.genome_id][0]
    assert result.error_status_code == 400


def test_resume_restores_checkpoint_without_regenerating_seed_drafts(tmp_path):
    case = BenchmarkCase(
        case_id="case",
        family="test",
        prompt="Compute 2 + 3.",
        output_contract="Return only the integer.",
        evaluator={"kind": "exact", "expected": "5"},
    )
    client = FakeEvolutionClient()
    run_dir = tmp_path / "search"
    config = EvolutionConfig(
        generations=1,
        revision_population=2,
        revisor_population=1,
        mutation_calls_per_generation=1,
        revision_mutations_per_call=1,
        revisor_mutations_per_call=1,
        max_cycles=2,
        token_penalty_per_1k_per_case=0,
        meta_token_penalty_per_1k=0,
    )
    EvolutionSearch(
        client=client,
        model="model",
        cases=[case],
        config=config,
        writer=EvolutionWriter(run_dir),
    ).run()
    initial_seed_calls = sum(
        call["role"] == "evolution_seed_draft" for call in client.calls
    )
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state_path.write_text(json.dumps(state), encoding="utf-8")
    (run_dir / "summary.json").unlink()
    (run_dir / "proposals" / "orphan.json").write_text("{}", encoding="utf-8")

    resumed = EvolutionSearch(
        client=client,
        model="model",
        cases=[case],
        config=config,
        writer=EvolutionWriter(run_dir, resume=True),
    )
    resumed.run(resume=True)
    assert (
        sum(call["role"] == "evolution_seed_draft" for call in client.calls)
        == initial_seed_calls
    )
    assert list((run_dir / "orphaned").rglob("orphan.json"))
