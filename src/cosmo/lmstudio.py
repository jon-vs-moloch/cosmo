from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from .inference import InferenceError
from .models import Generation


class LMStudioError(InferenceError):
    pass


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class LMStudioRedirectRefused(LMStudioError):
    pass


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Never follow HTTP redirects from the LM Studio endpoint.

    `--base-url` is validated as loopback, but urllib's default opener would
    follow a 3xx to any host and re-send the Authorization header there. The
    LM Studio API never needs a redirect, so every one is refused before any
    connection to the target is attempted.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise LMStudioRedirectRefused(
            f"LM Studio responded with HTTP {code} redirect to {newurl}; "
            "Cosmo is loopback-only and refuses to follow redirects."
        )


# No environment proxies (a loopback request must not leave the host via
# HTTP(S)_PROXY with its Authorization header) and no redirect following.
_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), _RefuseRedirects()
)


def select_loaded_reasoning_model(
    models: list[dict[str, Any]], model_key: str
) -> dict[str, Any]:
    """Return the catalog record for a loaded model with off/on reasoning.

    Shared by the live LM Studio client and the offline replay transport so
    both apply the same loaded-instance and reasoning-capability preflight.
    """
    match = next(
        (
            model
            for model in models
            if model.get("key") == model_key
            or any(
                instance.get("id") == model_key
                for instance in model.get("loaded_instances") or []
            )
        ),
        None,
    )
    if match is None:
        raise LMStudioError(f"Model is not installed in LM Studio: {model_key}")
    instances = list(match.get("loaded_instances") or [])
    if not instances:
        raise LMStudioError(
            f"Model is installed but not loaded: {model_key}. "
            "Load it in LM Studio or with `lms load` before running Cosmo."
        )
    reasoning = (match.get("capabilities") or {}).get("reasoning") or {}
    allowed = set(reasoning.get("allowed_options") or [])
    if not {"off", "on"}.issubset(allowed):
        raise LMStudioError(
            f"Model does not advertise both reasoning modes: {model_key} ({sorted(allowed)})"
        )
    return match


def chat_request_body(
    *,
    model: str,
    system_prompt: str,
    prompt: str,
    reasoning: str,
    temperature: float,
    max_output_tokens: int,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The stateless LM Studio `/api/v1/chat` request Cosmo sends.

    Its canonical JSON hash is `Generation.request_hash`; the replay transport
    uses the same body so recorded fixtures can match on that hash.
    """
    body: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "system_prompt": system_prompt,
        "reasoning": reasoning,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "store": False,
        "stream": False,
    }
    if response_format is not None:
        body["response_format"] = response_format
    return body


def canonical_request_hash(body: dict[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class LMStudioClient:
    base_url: str = "http://127.0.0.1:1234"
    timeout_seconds: float = 300.0
    api_token: str | None = None

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in _LOCAL_HOSTS
        ):
            raise LMStudioError(
                "Cosmo is local-only; --base-url must target localhost, "
                "127.0.0.1, or ::1."
            )

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with _OPENER.open(request, timeout=self.timeout_seconds) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LMStudioError(f"LM Studio HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LMStudioError(f"LM Studio unavailable at {self.base_url}: {exc}") from exc
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise LMStudioError("LM Studio returned invalid JSON") from exc

    def models(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/api/v1/models").get("models", []))

    def require_loaded_reasoning_model(self, model_key: str) -> dict[str, Any]:
        return select_loaded_reasoning_model(self.models(), model_key)

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
        body = chat_request_body(
            model=model,
            system_prompt=system_prompt,
            prompt=prompt,
            reasoning=reasoning,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
        )
        requested_response_format = response_format is not None
        started = time.monotonic()
        response_format_fallback = False
        try:
            response = self._request("POST", "/api/v1/chat", body)
        except LMStudioError as exc:
            # LM Studio's native `/api/v1/chat` endpoint (unlike its
            # OpenAI-compatible `/v1/chat/completions` endpoint) may reject
            # the optional OpenAI structured-output hint.  The Cosmo prompts
            # already carry the required JSON contract and parse/validate the
            # returned content, so retry once without the unsupported hint.
            # Keep this explicit in Generation.stats; it is a transport
            # compatibility fallback, not an invisible acceptance bypass.
            detail = str(exc).lower()
            if not (
                requested_response_format
                and "response_format" in detail
                and ("unrecognized" in detail or "unsupported" in detail)
            ):
                raise
            body.pop("response_format", None)
            response_format_fallback = True
            response = self._request("POST", "/api/v1/chat", body)
        latency = time.monotonic() - started
        response_model = response.get("model_instance_id")
        if response_model and response_model != model:
            raise LMStudioError(
                f"LM Studio responded from {response_model}, expected fixed instance {model}"
            )
        messages = [
            str(item.get("content", ""))
            for item in response.get("output", [])
            if item.get("type") == "message"
        ]
        reasoning_parts = [
            str(item.get("content", ""))
            for item in response.get("output", [])
            if item.get("type") == "reasoning"
        ]
        content = "\n".join(part for part in messages if part).strip()
        reasoning_text = "\n".join(reasoning_parts)
        reasoning_hash = (
            hashlib.sha256(reasoning_text.encode("utf-8")).hexdigest()
            if reasoning_text
            else None
        )
        stats = dict(response.get("stats") or {})
        if response_format_fallback:
            stats["cosmo_response_format_fallback"] = "prompt_only_after_lmstudio_rejection"
        return Generation(
            role=role,
            content=content,
            reasoning_hash=reasoning_hash,
            stats=stats,
            latency_seconds=latency,
            request_hash=canonical_request_hash(body),
            model_instance_id=str(response_model) if response_model else None,
        )
