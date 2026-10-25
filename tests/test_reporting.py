from cosmo.models import ArmResult
from cosmo.reporting import summarize


def test_summary_exposes_false_model_return_current():
    result = ArmResult(
        case_id="c",
        family="f",
        arm_id="instant_recursive",
        reasoning="off",
        recursive=True,
        stop_reason="model_return_current",
        analysis={
            "terminal_passed": False,
            "calls": 2,
            "input_tokens": 25,
            "output_tokens": 10,
            "reasoning_tokens": 0,
            "inference_seconds": 1.0,
            "self_assessed_return_current": True,
            "return_current_false_positive": True,
        },
    )
    arm = summarize([result])["arms"]["instant_recursive"]
    assert arm["return_current_decisions"] == 1
    assert arm["false_return_current_decisions"] == 1
    assert arm["input_tokens"] == 25


def test_budget_exhaustion_is_separate_from_thrash():
    result = ArmResult(
        case_id="c",
        family="f",
        arm_id="instant_recursive",
        reasoning="off",
        recursive=True,
        stop_reason="cycle_cap",
        analysis={
            "terminal_passed": True,
            "calls": 3,
            "input_tokens": 30,
            "output_tokens": 6,
            "reasoning_tokens": 0,
            "inference_seconds": 1.0,
        },
    )
    arm = summarize([result])["arms"]["instant_recursive"]
    assert arm["budget_exhausted_runs"] == 1
    assert arm["thrash_runs"] == 0


def test_hidden_score_regression_does_not_enter_observable_thrash_metric():
    result = ArmResult(
        case_id="c",
        family="f",
        arm_id="instant_recursive",
        reasoning="off",
        recursive=True,
        stop_reason="model_return_current",
        analysis={
            "terminal_passed": False,
            "final_minus_best": -1.0,
            "duplicate_candidate": False,
            "two_cycle": False,
        },
    )

    arm = summarize([result])["arms"]["instant_recursive"]

    assert arm["thrash_runs"] == 0


def test_protocol_error_does_not_count_a_correct_checkpoint_as_treatment_success():
    result = ArmResult(
        case_id="c",
        family="f",
        arm_id="instant_recursive",
        reasoning="off",
        recursive=True,
        stop_reason="error",
        error="RevisionProtocolError: invalid JSON",
        error_kind="model_protocol",
        analysis={"terminal_passed": True, "protocol_error": True},
    )

    arm = summarize([result])["arms"]["instant_recursive"]

    assert arm["terminal_checkpoint_successes"] == 1
    assert arm["terminal_successes"] == 0
    assert arm["terminal_success_rate"] is None


def test_cost_cache_and_retry_metrics_are_aggregated():
    result = ArmResult(
        case_id="c",
        family="f",
        arm_id="instant_once",
        reasoning="off",
        recursive=False,
        stop_reason="one_shot",
        analysis={
            "terminal_passed": True,
            "calls": 2,
            "input_tokens": 100,
            "output_tokens": 20,
            "charged_tokens": 120,
            "cached_input_tokens": 30,
            "provider_cost": 0.00125,
            "provider_cost_observations": 2,
            "transport_attempts": 3,
            "transport_retries": 1,
            "inference_seconds": 1.0,
        },
    )

    arm = summarize([result])["arms"]["instant_once"]

    assert arm["charged_tokens"] == 120
    assert arm["cached_input_tokens"] == 30
    assert arm["provider_cost"] == 0.00125
    assert arm["provider_cost_complete"] is True
    assert arm["transport_retries"] == 1


def test_regression_counts_any_lost_checkpoint_not_only_bad_terminal_from_good_seed():
    result = ArmResult(
        case_id="c",
        family="f",
        arm_id="instant_recursive",
        reasoning="off",
        recursive=True,
        stop_reason="model_return_current",
        analysis={
            "terminal_passed": False,
            "initial_passed": False,
            "regressed_by_loop": False,
            "revision_regressions": 1,
        },
    )

    arm = summarize([result])["arms"]["instant_recursive"]

    assert arm["regressed_runs"] == 1
    assert arm["revision_regressions"] == 1
