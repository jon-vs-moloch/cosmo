from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .inference import InferenceClient, InferenceError
from .models import Generation
from .reporting import canonical_hash, write_json


class InferenceJournalError(InferenceError):
    """A durable inference receipt is missing, corrupt, or inconsistent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunDirectoryLock:
    """Hold one nonblocking writer lease for an evolution directory."""

    def __init__(self, run_dir: Path):
        self.path = run_dir.parent / f".{run_dir.name}.writer.lock"
        self._stream = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.close()
            raise InferenceJournalError(
                f"Evolution directory already has a live writer: "
                f"{self.path.parent}"
            ) from exc
        stream.seek(0)
        stream.truncate()
        stream.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "acquired_at": _utc_now(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        stream.flush()
        self._stream = stream

    def release(self) -> None:
        if self._stream is None:
            return
        fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        self._stream.close()
        self._stream = None

    def __enter__(self) -> "RunDirectoryLock":
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()


class JournaledInferenceClient:
    """Persist one validated Generation per stable logical operation.

    The journal writes an in-flight receipt before calling the provider and a
    completed receipt before returning the generation to search logic. Replayed
    operations reuse the completed generation without another inference call.
    Prompts are represented only by a canonical hash; the benchmark and prompt
    genomes remain authoritative in their existing artifacts.
    """

    def __init__(self, client: InferenceClient, root: Path):
        self.client = client
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _receipt_name(operation_id: str) -> str:
        digest = hashlib.sha256(operation_id.encode("utf-8")).hexdigest()
        return f"{digest}.json"

    def _path(self, operation_id: str) -> Path:
        return self.root / self._receipt_name(operation_id)

    @staticmethod
    def _spec_hash(
        *,
        model: str,
        role: str,
        system_prompt: str,
        prompt: str,
        reasoning: str,
        temperature: float,
        max_output_tokens: int,
        response_format: dict[str, Any] | None,
    ) -> str:
        return canonical_hash(
            {
                "model": model,
                "role": role,
                "system_prompt": system_prompt,
                "prompt": prompt,
                "reasoning": reasoning,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "response_format": response_format,
            }
        )

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InferenceJournalError(
                f"Unreadable inference journal receipt: {path}"
            ) from exc
        if not isinstance(value, dict):
            raise InferenceJournalError(
                f"Inference journal receipt is not an object: {path}"
            )
        return value

    def chat(
        self,
        *,
        model: str,
        role: str,
        system_prompt: str,
        prompt: str,
        reasoning: str,
        temperature: float,
        max_output_tokens: int,
        response_format: dict[str, Any] | None = None,
        operation_id: str | None = None,
    ) -> Generation:
        if not operation_id:
            raise InferenceJournalError(
                "Journaled inference requires a stable operation_id"
            )
        spec_hash = self._spec_hash(
            model=model,
            role=role,
            system_prompt=system_prompt,
            prompt=prompt,
            reasoning=reasoning,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
        )
        path = self._path(operation_id)
        if path.exists():
            receipt = self._load(path)
            if (
                receipt.get("operation_id") != operation_id
                or receipt.get("spec_sha256") != spec_hash
            ):
                raise InferenceJournalError(
                    f"Inference journal operation changed on resume: "
                    f"{operation_id}"
                )
            if receipt.get("state") == "completed":
                raw_generation = receipt.get("generation")
                if not isinstance(raw_generation, dict):
                    raise InferenceJournalError(
                        f"Completed inference receipt has no generation: "
                        f"{operation_id}"
                    )
                generation = Generation(**raw_generation)
                generation.stats = {
                    **generation.stats,
                    "journal_replayed": True,
                }
                return generation
        else:
            receipt = {
                "schema_version": "cosmo.inference-receipt.v1",
                "operation_id": operation_id,
                "spec_sha256": spec_hash,
                "state": "new",
                "attempts": [],
                "created_at": _utc_now(),
            }

        attempts = receipt.get("attempts")
        if not isinstance(attempts, list):
            raise InferenceJournalError(
                f"Inference receipt has invalid attempts: {operation_id}"
            )
        for prior in attempts:
            if isinstance(prior, dict) and prior.get("status") == "in_flight":
                prior["status"] = "outcome_unknown"
                prior["accounting_outcome"] = "outcome_unknown"
                prior["finished_at"] = _utc_now()
        prior_unknown = sum(
            isinstance(item, dict)
            and (
                (
                    item.get("status") == "outcome_unknown"
                    and item.get("accounting_outcome")
                    in {None, "outcome_unknown"}
                )
                or item.get("accounting_outcome") == "outcome_unknown"
            )
            for item in attempts
        )
        prior_response_invalid = sum(
            isinstance(item, dict)
            and item.get("accounting_outcome") == "response_invalid"
            for item in attempts
        )
        attempt = {
            "number": len(attempts) + 1,
            "started_at": _utc_now(),
            "status": "in_flight",
        }
        attempts.append(attempt)
        receipt["state"] = "in_flight"
        receipt["updated_at"] = _utc_now()
        write_json(path, receipt)

        try:
            generation = self.client.chat(
                model=model,
                role=role,
                system_prompt=system_prompt,
                prompt=prompt,
                reasoning=reasoning,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_format=response_format,
                operation_id=operation_id,
            )
        except Exception as exc:
            accounting_outcome = str(
                getattr(exc, "accounting_outcome", "outcome_unknown")
            )
            if accounting_outcome not in {
                "not_dispatched",
                "known_rejected",
                "outcome_unknown",
                "response_invalid",
            }:
                accounting_outcome = "outcome_unknown"
            attempt["status"] = (
                "failed"
                if accounting_outcome in {"not_dispatched", "known_rejected"}
                else "outcome_unknown"
            )
            attempt["accounting_outcome"] = accounting_outcome
            attempt["finished_at"] = _utc_now()
            attempt["error"] = {
                "kind": type(exc).__name__,
                "message": str(exc),
                "status_code": getattr(exc, "status_code", None),
                "accounting_outcome": accounting_outcome,
            }
            observed_usage = getattr(exc, "observed_usage", None)
            if isinstance(observed_usage, dict):
                attempt["error"]["observed_usage"] = observed_usage
            receipt["state"] = "failed"
            receipt["updated_at"] = _utc_now()
            write_json(path, receipt)
            raise

        generation.stats = {
            **generation.stats,
            "journal_operation_id": operation_id,
            "journal_attempt_number": attempt["number"],
            "journal_prior_unknown_attempts": prior_unknown,
            "journal_prior_response_invalid_attempts": (
                prior_response_invalid
            ),
            "journal_replayed": False,
        }
        attempt["status"] = "completed"
        attempt["accounting_outcome"] = "completed_with_usage"
        attempt["finished_at"] = _utc_now()
        receipt["state"] = "completed"
        receipt["generation"] = asdict(generation)
        receipt["completed_at"] = _utc_now()
        receipt["updated_at"] = receipt["completed_at"]
        write_json(path, receipt)
        return generation

    def work(self) -> dict[str, Any]:
        completed: list[Generation] = []
        unknown_attempts = 0
        response_invalid_attempts = 0
        known_rejected_attempts = 0
        not_dispatched_attempts = 0
        failed_attempts = 0
        logical_attempts = 0
        for path in sorted(self.root.glob("*.json")):
            receipt = self._load(path)
            attempts = receipt.get("attempts")
            if not isinstance(attempts, list):
                raise InferenceJournalError(
                    f"Inference receipt has invalid attempts: {path}"
                )
            logical_attempts += len(attempts)
            unknown_attempts += sum(
                isinstance(item, dict)
                and (
                    item.get("status") == "in_flight"
                    or item.get("accounting_outcome") == "outcome_unknown"
                    or (
                        item.get("status") == "outcome_unknown"
                        and item.get("accounting_outcome") is None
                    )
                    or (
                        item.get("status") == "failed"
                        and item.get("accounting_outcome") is None
                    )
                )
                for item in attempts
            )
            response_invalid_attempts += sum(
                isinstance(item, dict)
                and item.get("accounting_outcome") == "response_invalid"
                for item in attempts
            )
            known_rejected_attempts += sum(
                isinstance(item, dict)
                and item.get("accounting_outcome") == "known_rejected"
                for item in attempts
            )
            not_dispatched_attempts += sum(
                isinstance(item, dict)
                and item.get("accounting_outcome") == "not_dispatched"
                for item in attempts
            )
            failed_attempts += sum(
                isinstance(item, dict) and item.get("status") == "failed"
                for item in attempts
            )
            if receipt.get("state") != "completed":
                continue
            raw_generation = receipt.get("generation")
            if not isinstance(raw_generation, dict):
                raise InferenceJournalError(
                    f"Completed inference receipt has no generation: {path}"
                )
            completed.append(Generation(**raw_generation))

        provider_costs: list[float] = []
        for generation in completed:
            raw_cost = generation.stats.get("provider_cost")
            try:
                cost = float(raw_cost)
            except (TypeError, ValueError):
                continue
            if math.isfinite(cost) and cost >= 0:
                provider_costs.append(cost)
        input_tokens = sum(
            int(item.stats.get("input_tokens", 0)) for item in completed
        )
        output_tokens = sum(
            int(item.stats.get("total_output_tokens", 0))
            for item in completed
        )
        transport_attempts = sum(
            max(1, int(item.stats.get("transport_attempts", 1)))
            for item in completed
        )
        by_role: dict[str, dict[str, int]] = {}
        for generation in completed:
            bucket = by_role.setdefault(
                generation.role,
                {
                    "completed_operations": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            )
            generation_input = int(
                generation.stats.get("input_tokens", 0)
            )
            generation_output = int(
                generation.stats.get("total_output_tokens", 0)
            )
            bucket["completed_operations"] += 1
            bucket["input_tokens"] += generation_input
            bucket["output_tokens"] += generation_output
            bucket["total_tokens"] += generation_input + generation_output
        return {
            "accounting_scope": "all journaled evolution lifecycle inference",
            "completed_operations": len(completed),
            "logical_attempts": logical_attempts,
            "unknown_outcome_attempts": unknown_attempts,
            "response_invalid_attempts": response_invalid_attempts,
            "known_rejected_attempts": known_rejected_attempts,
            "not_dispatched_attempts": not_dispatched_attempts,
            "failed_attempts": failed_attempts,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "reasoning_tokens": sum(
                int(item.stats.get("reasoning_output_tokens", 0))
                for item in completed
            ),
            "cached_input_tokens": sum(
                int(item.stats.get("cached_input_tokens", 0))
                for item in completed
            ),
            "transport_attempts": transport_attempts,
            "transport_retries": transport_attempts - len(completed),
            "inference_seconds": sum(
                item.latency_seconds for item in completed
            ),
            "provider_cost": (
                sum(provider_costs)
                if len(provider_costs) == len(completed)
                else None
            ),
            "provider_cost_observations": len(provider_costs),
            "by_role": by_role,
            "accounting_complete": (
                unknown_attempts == 0 and response_invalid_attempts == 0
            ),
        }
