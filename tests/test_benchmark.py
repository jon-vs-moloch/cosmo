import json

from cosmo.benchmark import evaluate, load_cases
from cosmo.models import BenchmarkCase


def test_loads_bundled_cases():
    cases = load_cases(
        __import__("pathlib").Path(
            "benchmarks/iterative-self-verification/v1/cases.jsonl"
        )
    )
    assert len(cases) == 12
    assert len({case.case_id for case in cases}) == 12


def test_exact_evaluator_extracts_final_line():
    case = BenchmarkCase(
        case_id="x",
        family="test",
        prompt="",
        evaluator={
            "kind": "exact",
            "expected": "42",
            "extract": r"(?im)^FINAL:\s*([^\n]+)\s*$",
        },
    )
    assert evaluate(case, "FINAL: 42")["passed"] is True
    assert evaluate(case, "FINAL: 41")["passed"] is False
    assert evaluate(case, "commentary\nFINAL: 42")["passed"] is False
    assert evaluate(case, "```text\nFINAL: 42\n```")["passed"] is False


def test_json_evaluator_is_structural():
    case = BenchmarkCase(
        case_id="x",
        family="test",
        prompt="",
        evaluator={"kind": "json_exact", "expected": {"a": [1, 2]}},
    )
    assert evaluate(case, '{"a": [1, 2]}')["passed"] is True
    assert evaluate(case, '{"a": [2, 1]}')["passed"] is False
    assert evaluate(case, '```json\n{"a": [1, 2]}\n```')["passed"] is False
    assert evaluate(case, '{"a": [9], "a": [1, 2]}')["passed"] is False


def test_json_evaluator_enforces_top_level_key_order():
    case = BenchmarkCase(
        case_id="x",
        family="test",
        prompt="",
        evaluator={"kind": "json_exact", "expected": {"a": 1, "b": 2}},
    )
    assert evaluate(case, '{"a": 1, "b": 2}')["passed"] is True
    assert evaluate(case, '{"b": 2, "a": 1}')["passed"] is False


def test_json_evaluator_is_type_exact():
    boolean_case = BenchmarkCase(
        case_id="bool",
        family="test",
        prompt="",
        evaluator={"kind": "json_exact", "expected": {"value": True}},
    )
    integer_case = BenchmarkCase(
        case_id="int",
        family="test",
        prompt="",
        evaluator={"kind": "json_exact", "expected": {"value": 1}},
    )
    assert evaluate(boolean_case, '{"value": 1}')["passed"] is False
    assert evaluate(integer_case, '{"value": true}')["passed"] is False
    assert evaluate(integer_case, '{"value": 1.0}')["passed"] is False
