import json

import pytest

from cosmo.convergence import (
    REVISION_INSTRUCTION,
    REVISION_SYSTEM,
    WITNESSED_REVISION_SYSTEM,
    WITNESSED_RESPONSE_FORMAT,
    RevisionProtocolError,
    _revision_prompt,
    content_hash,
    is_return_current,
    parse_witnessed_revision,
)
from cosmo.models import ARMS, BenchmarkCase, Generation
from cosmo.convergence import run_arm


def test_candidate_hash_normalizes_line_endings_only():
    assert content_hash("FINAL: 42\r\n") == content_hash("FINAL: 42\n")
    assert content_hash("FINAL: 42") != content_hash("FINAL:  42")


def test_return_current_is_textual_identity():
    assert is_return_current("draft", "draft")
    assert is_return_current("draft\r\n", "draft\n")
    assert not is_return_current("draft", "`draft`")
    assert not is_return_current("draft", "Draft")


def test_revision_prompt_contains_only_task_contract_and_current_draft():
    case = _case()
    prompt = _revision_prompt(case, "current")
    assert case.prompt in prompt
    assert case.output_contract in prompt
    assert "current" in prompt
    assert prompt.endswith(REVISION_INSTRUCTION)
    assert "LATEST CRITIQUE" not in prompt
    assert "reference" not in prompt.lower()
    assert "evaluator" not in prompt.lower()


def test_revision_prompt_accepts_an_evolved_instruction():
    case = _case()
    prompt = _revision_prompt(case, "current", "MUTATED INSTRUCTION")
    assert prompt.endswith("MUTATED INSTRUCTION")
    assert REVISION_INSTRUCTION not in prompt


class FakeClient:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        content = next(self.outputs)
        return Generation(
            role=kwargs["role"],
            content=content,
            reasoning_hash=None,
            stats={
                "input_tokens": 10,
                "total_output_tokens": 2,
                "reasoning_output_tokens": 0,
            },
            latency_seconds=0.25,
            request_hash="request",
            model_instance_id=kwargs["model"],
        )


def _case():
    return BenchmarkCase(
        case_id="test",
        family="test",
        prompt="Compute 2 + 3.",
        output_contract="Return only the integer.",
        evaluator={"kind": "exact", "expected": "5"},
    )


def _run(
    outputs,
    *,
    max_cycles=4,
    initial_candidate=None,
    initial_generation=None,
    revision_mode="direct",
):
    client = FakeClient(outputs)
    result = run_arm(
        client=client,
        model="model",
        case=_case(),
        arm=ARMS["instant_recursive"],
        max_cycles=max_cycles,
        temperature=0,
        max_output_tokens=100,
        revision_max_output_tokens=100,
        revision_mode=revision_mode,
        initial_candidate=initial_candidate,
        initial_generation=initial_generation,
    )
    return client, result


def test_replacement_then_return_current_is_model_owned_convergence():
    client, result = _run(["4", "5", "5"])
    assert [item.candidate for item in result.checkpoints] == ["4", "5"]
    assert result.stop_reason == "model_return_current"
    assert result.analysis["terminal_passed"] is True
    assert result.analysis["corrected_by_loop"] is True
    assert result.analysis["calls"] == 3
    assert client.calls[1]["system_prompt"] == REVISION_SYSTEM
    assert "CURRENT DRAFT\n4" in client.calls[1]["prompt"]
    assert "CURRENT DRAFT\n5" in client.calls[2]["prompt"]
    assert "CURRENT DRAFT\n4" not in client.calls[2]["prompt"]


def test_unchanged_revision_is_model_owned_return_current():
    _client, result = _run(["4", "4"], max_cycles=2)
    assert result.stop_reason == "model_return_current"
    assert len(result.checkpoints) == 1
    assert result.analysis["self_assessed_return_current"] is True


def test_two_cycle_does_not_stop_before_budget():
    _client, result = _run(["4", "5", "4"], max_cycles=2)
    assert result.stop_reason == "cycle_cap"
    assert result.analysis["two_cycle"] is True
    assert result.analysis["calls"] == 3


def test_empty_revision_is_an_excluded_protocol_error():
    _client, result = _run(["5", ""])
    assert result.stop_reason == "error"
    assert result.error and "RevisionProtocolError" in result.error
    assert result.error_kind == "model_protocol"
    assert result.analysis["protocol_error"] is True


def test_terminal_control_call_counts_as_work_after_best_checkpoint():
    _client, result = _run(["5", "5"])
    assert result.analysis["terminal_passed"] is True
    assert result.analysis["calls_after_best_checkpoint"] == 1
    assert result.analysis["tokens_after_best_checkpoint"] == 2


def test_fixed_initial_candidate_is_reused_without_charging_a_draft_call():
    client, result = _run(["5", "5"], initial_candidate="4")
    assert [item.candidate for item in result.checkpoints] == ["4", "5"]
    assert [item["role"] for item in client.calls] == ["revision", "revision"]
    assert result.analysis["calls"] == 2
    assert result.checkpoints[0].generation_sequence == -1


def test_shared_initial_generation_is_reused_with_identical_draft_accounting():
    initial = Generation(
        role="draft",
        content="4",
        reasoning_hash=None,
        stats={
            "input_tokens": 7,
            "total_output_tokens": 1,
            "reasoning_output_tokens": 0,
            "transport_attempts": 1,
        },
        latency_seconds=0.5,
        request_hash="shared-request",
        sequence=19,
        model_instance_id="model",
    )
    client, result = _run(
        ["5", "5"],
        initial_generation=initial,
    )

    assert [call["role"] for call in client.calls] == ["revision", "revision"]
    assert result.generations[0].request_hash == "shared-request"
    assert result.generations[0].sequence == 0
    assert result.checkpoints[0].generation_sequence == 0
    assert result.analysis["calls"] == 3
    assert result.analysis["input_tokens"] == 27


def _witnessed(candidate, verification=None):
    return json.dumps(
        {
            "verification": verification or "independent check completed",
            "candidate": candidate,
        }
    )


def test_parse_witnessed_revision_preserves_verification_and_candidate():
    parsed = parse_witnessed_revision(
        _witnessed("5", "3 + 2 = 5")
    )
    assert parsed.verification == "3 + 2 = 5"
    assert parsed.candidate == "5"


def test_parse_witnessed_revision_serializes_nested_json_artifact_in_key_order():
    parsed = parse_witnessed_revision(
        json.dumps(
            {
                "verification": "record fields checked",
                "candidate": {
                    "id": "A-17",
                    "active": True,
                    "tags": ["local", "test"],
                },
            }
        )
    )

    assert parsed.candidate == (
        '{"id":"A-17","active":true,"tags":["local","test"]}'
    )


@pytest.mark.parametrize(
    "content",
    [
        "not JSON",
        json.dumps(
            {
                "verification": ["check"],
                "candidate": "5",
                "extra": True,
            }
        ),
        json.dumps(
            {
                "verification": "",
                "candidate": "5",
            }
        ),
        json.dumps(
            {
                "verification": ["check"],
                "candidate": "4",
            }
        ),
    ],
)
def test_invalid_witnessed_revision_is_a_model_protocol_error(content):
    _client, result = _run(
        ["4", content],
        revision_mode="witnessed",
    )
    assert result.stop_reason == "error"
    assert result.error_kind == "model_protocol"
    assert result.error and "RevisionProtocolError" in result.error
    assert result.analysis["protocol_error"] is True
    assert result.analysis["calls"] == 2


def test_witnessed_candidate_identity_is_the_model_owned_stop_decision():
    _client, result = _run(
        ["4", _witnessed("4", "independent check agrees")],
        revision_mode="witnessed",
    )
    assert result.stop_reason == "model_return_current"
    assert len(result.checkpoints) == 1


def test_witnessed_replacement_then_return_current_uses_extracted_trajectory():
    client, result = _run(
        [
            "4",
            _witnessed("5", "independently recomputed 3 + 2"),
            _witnessed("5", "recomputation agrees"),
        ],
        revision_mode="witnessed",
    )
    assert [item.candidate for item in result.checkpoints] == ["4", "5"]
    assert result.stop_reason == "model_return_current"
    assert result.analysis["terminal_passed"] is True
    assert result.analysis["corrected_by_loop"] is True
    assert result.analysis["calls"] == 3
    assert result.analysis["output_tokens"] == 6
    assert client.calls[1]["system_prompt"] == WITNESSED_REVISION_SYSTEM
    assert client.calls[1]["reasoning"] == "off"
    assert client.calls[1]["response_format"] == WITNESSED_RESPONSE_FORMAT
    assert "CURRENT DRAFT\n4" in client.calls[1]["prompt"]
    assert "CURRENT DRAFT\n5" in client.calls[2]["prompt"]
    assert "CURRENT DRAFT\n4" not in client.calls[2]["prompt"]
    assert "independently recomputed" not in client.calls[2]["prompt"]
