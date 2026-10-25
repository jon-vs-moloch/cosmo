"""Deterministic offline replay transport.

`ReplayClient` satisfies the same stateless `InferenceClient` surface as
`LMStudioClient`, but answers every request from a JSON fixture instead of a
model server. It exists so the harness can be exercised end to end (tests,
smoke runs, demos) without LM Studio. A replay run is never experiment
evidence: its manifest records `transport: "replay"` and the fixture hash.

Fixture schema (`cosmo.replay-fixture.v1`)::

    {
      "schema_version": "cosmo.replay-fixture.v1",
      "model_record": { ... an LM Studio catalog record with loaded_instances
                        and capabilities.reasoning.allowed_options ... },
      "responses": [
        {
          "match": {"role": "draft", "reasoning": "off",
                    "prompt_contains": ["CURRENT DRAFT\\n7\\n"],
                    "request_sha256": "..."},
          "content": "5",
          "reasoning": "optional native reasoning text (hashed, never stored)",
          "stats": {"input_tokens": 40, "total_output_tokens": 1}
        }
      ]
    }

Every `match` key is optional; all present keys must hold. Rules are tried in
order and the first match answers, so the same request always receives the
same response. An unmatched request is an `InferenceError`, never a silent
default.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .inference import InferenceError
from .lmstudio import (
    canonical_request_hash,
    chat_request_body,
    select_loaded_reasoning_model,
)
from .models import Generation

REPLAY_FIXTURE_SCHEMA = "cosmo.replay-fixture.v1"
_MATCH_KEYS = {"role", "reasoning", "prompt_contains", "request_sha256"}


class ReplayError(InferenceError):
    pass


@dataclass
class ReplayClient:
    model_record: dict[str, Any]
    responses: list[dict[str, Any]]
    fixture_sha256: str
    calls: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> "ReplayClient":
        raw = Path(path).read_bytes()
        try:
            fixture = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ReplayError(f"replay fixture is not valid JSON: {path}") from exc
        if not isinstance(fixture, dict) or (
            fixture.get("schema_version") != REPLAY_FIXTURE_SCHEMA
        ):
            raise ReplayError(
                f"replay fixture must declare schema_version {REPLAY_FIXTURE_SCHEMA}"
            )
        model_record = fixture.get("model_record")
        responses = fixture.get("responses")
        if not isinstance(model_record, dict) or not isinstance(responses, list):
            raise ReplayError("replay fixture needs model_record and responses")
        for index, rule in enumerate(responses):
            if not isinstance(rule, dict) or not isinstance(rule.get("content"), str):
                raise ReplayError(f"replay response {index} needs string content")
            unknown = set(rule.get("match") or {}) - _MATCH_KEYS
            if unknown:
                raise ReplayError(
                    f"replay response {index} has unknown match keys: {sorted(unknown)}"
                )
        return cls(
            model_record=model_record,
            responses=responses,
            fixture_sha256=hashlib.sha256(raw).hexdigest(),
        )

    def models(self) -> list[dict[str, Any]]:
        return [self.model_record]

    def require_loaded_reasoning_model(self, model_key: str) -> dict[str, Any]:
        return select_loaded_reasoning_model(self.models(), model_key)

    @staticmethod
    def _matches(
        match: dict[str, Any], *, role: str, reasoning: str, prompt: str, request_sha256: str
    ) -> bool:
        if "role" in match and match["role"] != role:
            return False
        if "reasoning" in match and match["reasoning"] != reasoning:
            return False
        if "request_sha256" in match and match["request_sha256"] != request_sha256:
            return False
        needles = match.get("prompt_contains") or []
        if isinstance(needles, str):
            needles = [needles]
        return all(needle in prompt for needle in needles)

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
        del operation_id
        request_sha256 = canonical_request_hash(
            chat_request_body(
                model=model,
                system_prompt=system_prompt,
                prompt=prompt,
                reasoning=reasoning,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_format=response_format,
            )
        )
        rule = next(
            (
                item
                for item in self.responses
                if self._matches(
                    item.get("match") or {},
                    role=role,
                    reasoning=reasoning,
                    prompt=prompt,
                    request_sha256=request_sha256,
                )
            ),
            None,
        )
        if rule is None:
            raise ReplayError(
                f"no replay response matches role={role} reasoning={reasoning} "
                f"request_sha256={request_sha256}"
            )
        self.calls.append(
            {"role": role, "reasoning": reasoning, "request_sha256": request_sha256}
        )
        reasoning_text = str(rule.get("reasoning") or "")
        stats = dict(rule.get("stats") or {})
        stats["cosmo_replay"] = True
        return Generation(
            role=role,
            content=rule["content"].strip(),
            reasoning_hash=(
                hashlib.sha256(reasoning_text.encode("utf-8")).hexdigest()
                if reasoning_text
                else None
            ),
            stats=stats,
            latency_seconds=0.0,
            request_hash=request_sha256,
            model_instance_id=model,
        )
