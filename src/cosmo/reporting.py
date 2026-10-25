from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import ArmResult


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


class RunWriter:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.events_path = run_dir / "events.jsonl"
        (run_dir / "results").mkdir(parents=True, exist_ok=False)

    def event(self, event: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")

    def result(self, result: ArmResult) -> None:
        write_json(
            self.run_dir / "results" / f"{result.case_id}--{result.arm_id}.json",
            result.to_dict(),
        )


def summarize(results: list[ArmResult]) -> dict[str, Any]:
    arms: dict[str, dict[str, Any]] = {}
    for result in results:
        bucket = arms.setdefault(
            result.arm_id,
            {
                "runs": 0,
                "errors": 0,
                "terminal_successes": 0,
                "terminal_checkpoint_successes": 0,
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "charged_tokens": 0,
                "reasoning_tokens": 0,
                "cached_input_tokens": 0,
                "provider_cost": 0.0,
                "provider_cost_observations": 0,
                "transport_attempts": 0,
                "transport_retries": 0,
                "inference_seconds": 0.0,
                "thrash_runs": 0,
                "budget_exhausted_runs": 0,
                "corrected_runs": 0,
                "regressed_runs": 0,
                "revision_regressions": 0,
                "return_current_decisions": 0,
                "false_return_current_decisions": 0,
                "protocol_errors": 0,
            },
        )
        bucket["runs"] += 1
        if result.error:
            bucket["errors"] += 1
        analysis = result.analysis
        checkpoint_passed = bool(analysis.get("terminal_passed"))
        bucket["terminal_checkpoint_successes"] += int(checkpoint_passed)
        bucket["terminal_successes"] += int(
            checkpoint_passed and not result.error
        )
        bucket["calls"] += int(analysis.get("calls", 0))
        bucket["input_tokens"] += int(analysis.get("input_tokens", 0))
        bucket["output_tokens"] += int(analysis.get("output_tokens", 0))
        bucket["charged_tokens"] += int(analysis.get("charged_tokens", 0))
        bucket["reasoning_tokens"] += int(analysis.get("reasoning_tokens", 0))
        bucket["cached_input_tokens"] += int(
            analysis.get("cached_input_tokens", 0)
        )
        observed_cost = analysis.get("provider_cost")
        if isinstance(observed_cost, (int, float)):
            bucket["provider_cost"] += float(observed_cost)
        bucket["provider_cost_observations"] += int(
            analysis.get("provider_cost_observations", 0)
        )
        bucket["transport_attempts"] += int(
            analysis.get("transport_attempts", 0)
        )
        bucket["transport_retries"] += int(
            analysis.get("transport_retries", 0)
        )
        bucket["inference_seconds"] += float(analysis.get("inference_seconds", 0.0))
        bucket["thrash_runs"] += int(
            bool(
                analysis.get("duplicate_candidate")
                or analysis.get("two_cycle")
            )
        )
        bucket["budget_exhausted_runs"] += int(result.stop_reason == "cycle_cap")
        bucket["corrected_runs"] += int(bool(analysis.get("corrected_by_loop")))
        revision_regressions = int(
            analysis.get("revision_regressions", 0)
        )
        bucket["regressed_runs"] += int(revision_regressions > 0)
        bucket["revision_regressions"] += revision_regressions
        bucket["return_current_decisions"] += int(
            bool(analysis.get("self_assessed_return_current"))
        )
        bucket["false_return_current_decisions"] += int(
            bool(analysis.get("return_current_false_positive"))
        )
        bucket["protocol_errors"] += int(bool(analysis.get("protocol_error")))
    for bucket in arms.values():
        valid = bucket["runs"] - bucket["errors"]
        bucket["terminal_success_rate"] = (
            bucket["terminal_successes"] / valid if valid else None
        )
        bucket["inference_seconds"] = round(bucket["inference_seconds"], 6)
        bucket["provider_cost"] = round(bucket["provider_cost"], 12)
        bucket["provider_cost_complete"] = (
            bucket["provider_cost_observations"] == bucket["calls"]
        )

    by_case: dict[str, dict[str, bool | None]] = {}
    for result in results:
        by_case.setdefault(result.case_id, {})[result.arm_id] = (
            None if result.error else bool(result.analysis.get("terminal_passed"))
        )
    return {"arms": arms, "paired_terminal_outcomes": by_case}
