from __future__ import annotations

from typing import Any, Protocol

from .models import Generation


class InferenceError(RuntimeError):
    """Transport or response-protocol failure from an inference provider."""


class InferenceClient(Protocol):
    """The stateless chat surface consumed by Cosmo convergence loops."""

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
    ) -> Generation: ...
