from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import BenchmarkCase


def load_cases(path: Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            case = BenchmarkCase.from_dict(raw)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid benchmark case: {exc}") from exc
        if case.case_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate case_id {case.case_id}")
        seen.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError(f"Benchmark contains no cases: {path}")
    return cases


def _normal(value: Any, case_sensitive: bool = False) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if case_sensitive else text.casefold()


def _json_type_exact(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return list(actual) == list(expected) and all(
            _json_type_exact(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _json_type_exact(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def evaluate(case: BenchmarkCase, candidate: str) -> dict[str, Any]:
    config = case.evaluator
    kind = str(config.get("kind", "exact"))
    if kind == "exact":
        actual = candidate.strip()
        extract = config.get("extract")
        if extract:
            match = re.fullmatch(str(extract), actual, flags=re.MULTILINE | re.DOTALL)
            actual = match.group(1).strip() if match else ""
        case_sensitive = bool(config.get("case_sensitive", False))
        expected = config["expected"]
        passed = _normal(actual, case_sensitive) == _normal(expected, case_sensitive)
        return {
            "passed": passed,
            "score": 1.0 if passed else 0.0,
            "actual": actual,
            "expected": expected,
        }
    if kind == "json_exact":
        raw = candidate.strip()
        duplicate_keys: list[str] = []

        def strict_object(items):
            result = {}
            for key, value in items:
                if key in result:
                    duplicate_keys.append(key)
                result[key] = value
            return result

        try:
            actual_json = json.loads(raw, object_pairs_hook=strict_object)
            parse_error = None
        except (json.JSONDecodeError, ValueError) as exc:
            actual_json = None
            parse_error = str(exc)
        expected_json = config["expected"]
        expected_keys = list(expected_json) if isinstance(expected_json, dict) else None
        actual_keys = list(actual_json) if isinstance(actual_json, dict) else None
        passed = (
            _json_type_exact(actual_json, expected_json)
            and not duplicate_keys
            and (expected_keys is None or actual_keys == expected_keys)
        )
        return {
            "passed": passed,
            "score": 1.0 if passed else 0.0,
            "actual": actual_json,
            "expected": expected_json,
            "parse_error": parse_error,
            "duplicate_keys": duplicate_keys,
            "actual_keys": actual_keys,
            "expected_keys": expected_keys,
        }
    raise ValueError(f"Unsupported evaluator kind for {case.case_id}: {kind}")
