"""Ephemeral local chat gateway over Cosmo tools and LM Studio inference.

The chat transcript is in-memory only. Durable task/attempt/ledger state lives
in AtomicStore via ControlPlane tools. Model calls use the local
InferenceClient surface only — no external providers, shell, or agent CLIs.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from .authority import authenticate_model
from .help import format_harness_help
from .inference import InferenceClient, InferenceError
from .models import Generation
from .tools import ToolDispatcher, ToolResult, tool_definitions_json


class ChatError(RuntimeError):
    """Chat loop protocol or configuration failure."""


DEFAULT_MAX_TOOL_TURNS = 8

# Compact system prompt: tool contracts + response protocol. No hidden evaluator data.
CHAT_SYSTEM_PROMPT_HEADER = """You are Cosmo, the primary local chat control-plane assistant.
Architecture (product-facing):
  chat (this surface) -> strict tools -> ControlPlane / PlanningService /
  ContextCompiler / AtomicHarness / IntegrationService
  -> durable AtomicStore state; product files change only on explicit promotion.

You may call tools from the catalog below. Tools act only on the durable Cosmo
control plane, planning, context inspection, atomic leaf attempts, and parent
integration. You have no shell, no network providers, no cloud inference, and
no ability to invent Validation/Review PASS evidence.

Primary flows:
- Broad user outcomes: submit_request/ingest_request (immutable Request only)
  -> propose_triage -> independent review_triage -> apply_triage
  (creates/links Task; does NOT force a leaf artifact).
- Single-artifact work follows the same triage/review/apply path; the direct
  leaf shortcut is operator-only and absent from this catalog.
- Multi-artifact work after triage: build_planning_pack/propose_plan ->
  independent review_plan (full planning context) -> apply_approved_plan
  before any leaf dispatch. Never silently apply plans.
- Dispatch: ready_queue / scheduler_tick / dispatch_scheduler_batch, leases,
  reassess_priority, refresh_dependency_readiness, inspect_lifecycle,
  run_attempt/dispatch_ready/retry_attempt, promote_attempt.
- Parent integration: assemble_integration -> review_integration ->
  promote_integration (product unchanged until promote succeeds).
- Repository inspection: repository_map, list_tree, read_artifact, search_text,
  inspect_python_symbols (approved repo root only).
- If a lifecycle, plan shape, or output contract is unclear, call
  harness_help. The human can also type /help for the same compact guide.

Response protocol (every model turn must be exactly one of):
1) Final answer for the user as plain text (no tool envelope), OR
2) A single JSON object with tool calls:
   {"tool_calls":[{"name":"<tool>","arguments":{...}}, ...]}

Rules:
- Prefer one tool call per turn when sequencing matters.
- After tool results are appended, continue until you can answer the user.
- Never claim product promotion or PASS review without tool evidence.
- Do not force multi-file work into a single leaf artifact; use triage + planning.
- Sequence: triage propose -> independent review -> apply; then plan propose ->
  independent review -> apply; then leaf dispatch; integration assemble ->
  review -> promote.
- Operator recovery, operator-supplied proposals/reviews, and arbitrary
  repository roots are absent from this model catalog. Repository tools use
  only the immutable root capability approved before this session began.
- Malformed or unknown tools will return structured errors; correct and retry
  within the turn budget.
"""


def build_system_prompt(extra: str | None = None) -> str:
    """Build the compact system prompt including JSON tool definitions."""
    catalog = json.dumps(
        tool_definitions_json(catalog="model", compact=True),
        indent=2,
        sort_keys=True,
    )
    parts = [
        CHAT_SYSTEM_PROMPT_HEADER.strip(),
        "",
        "Tool catalog (compact JSON; call harness_help for field/lifecycle prose):",
        catalog,
    ]
    if extra and extra.strip():
        parts.extend(["", "Operator context:", extra.strip()])
    return "\n".join(parts)


_TOOL_CALLS_FENCE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)


def parse_model_turn(content: str) -> dict[str, Any]:
    """Parse a model turn into either final text or structured tool_calls.

    Returns:
      {"kind": "final", "text": str}
      {"kind": "tool_calls", "calls": [{"name": str, "arguments": Any}, ...]}
      {"kind": "error", "error": str, "raw": str}
    """
    text = (content or "").strip()
    if not text:
        return {"kind": "error", "error": "empty model response", "raw": content}

    candidates: list[str] = [text]
    fenced = _TOOL_CALLS_FENCE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    # If the model wraps JSON with prose, try the first {...} span.
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            candidates.insert(0, text[start : end + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        if "tool_calls" in payload:
            raw_calls = payload.get("tool_calls")
            if not isinstance(raw_calls, list) or not raw_calls:
                return {
                    "kind": "error",
                    "error": "tool_calls must be a non-empty array",
                    "raw": content,
                }
            calls: list[dict[str, Any]] = []
            for index, item in enumerate(raw_calls):
                if not isinstance(item, Mapping):
                    return {
                        "kind": "error",
                        "error": f"tool_calls[{index}] must be an object",
                        "raw": content,
                    }
                name = item.get("name")
                if not isinstance(name, str) or not name.strip():
                    return {
                        "kind": "error",
                        "error": f"tool_calls[{index}].name must be a non-empty string",
                        "raw": content,
                    }
                arguments = item.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments) if arguments.strip() else {}
                    except json.JSONDecodeError:
                        return {
                            "kind": "error",
                            "error": (
                                f"tool_calls[{index}].arguments is not valid JSON"
                            ),
                            "raw": content,
                        }
                calls.append({"name": name.strip(), "arguments": arguments})
            return {"kind": "tool_calls", "calls": calls}
        # Explicit final envelope.
        if "final" in payload and isinstance(payload.get("final"), str):
            return {"kind": "final", "text": payload["final"]}

    # No tool envelope: treat as final user-facing text.
    return {"kind": "final", "text": text}


def format_tool_results_message(results: list[ToolResult]) -> str:
    """Serialize tool results for the next model prompt turn."""
    body = [item.to_json_dict() for item in results]
    return (
        "Tool results (JSON; structured only — no shell side effects):\n"
        + json.dumps(body, indent=2, sort_keys=True)
    )


@dataclass
class ChatTurn:
    """One recorded chat/model/tool step (ephemeral transcript entry)."""

    role: str  # user | assistant | tool | system
    content: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatSessionResult:
    """Outcome of one user message through the tool loop."""

    final_text: str
    transcript: list[ChatTurn]
    tool_turns: int
    stop_reason: str
    generations: list[Generation] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "final_text": self.final_text,
            "tool_turns": self.tool_turns,
            "stop_reason": self.stop_reason,
            "transcript": [
                {"role": t.role, "content": t.content, "meta": t.meta}
                for t in self.transcript
            ],
            "generation_count": len(self.generations),
        }


@dataclass
class ChatSession:
    """Ephemeral multi-turn chat that may invoke control-plane tools."""

    client: InferenceClient
    dispatcher: ToolDispatcher
    model: str
    system_prompt: str | None = None
    max_tool_turns: int = DEFAULT_MAX_TOOL_TURNS
    reasoning: str = "off"
    temperature: float = 0.0
    max_output_tokens: int = 2048
    transcript: list[ChatTurn] = field(default_factory=list)
    _turn_counter: int = 0
    # A fresh durable namespace per ChatSession prevents a new CLI invocation
    # from reusing ``chat/1/...`` receipts from an older process on the same
    # state root.
    _session_nonce: str = field(
        default_factory=lambda: uuid.uuid4().hex[:16], repr=False
    )

    def __post_init__(self) -> None:
        caller = self.dispatcher.caller
        if caller is None or caller.principal_type != "model":
            # ChatSession itself is the trusted transition into inference.  It
            # never carries an operator dispatcher through to model output.
            self.dispatcher = ToolDispatcher(
                self.dispatcher.plane,
                client=self.dispatcher.client,
                model=self.dispatcher.model or self.model,
                caller=authenticate_model(
                    f"lmstudio:{self.model}",
                    authenticated_by="cosmo.chat-session",
                ),
            )
        if self.system_prompt is None:
            self.system_prompt = build_system_prompt()
        if self.max_tool_turns < 0:
            raise ChatError("max_tool_turns must be non-negative")

    def _operation_id(self, label: str) -> str | None:
        """Stable operation id for journaling and model calls."""
        # JournaledInferenceClient requires operation_id; plain clients ignore it.
        self._turn_counter += 1
        return f"chat/{self._session_nonce}/{self._turn_counter}/{label}"

    def _journal_tool_result(
        self, result: ToolResult, *, operation_id: str | None
    ) -> None:
        """Append a durable control event for one tool call when possible.

        Does not persist prompt/transcript bodies. Failures to journal must not
        crash the chat loop or affect product state.
        """
        plane = getattr(self.dispatcher, "plane", None)
        if plane is None or not hasattr(plane, "record_chat_tool_event"):
            return
        task_id: str | None = None
        attempt_id: str | None = None
        payload = result.result
        if isinstance(payload, Mapping):
            raw_task = payload.get("task_id")
            if isinstance(raw_task, str) and raw_task.strip():
                task_id = raw_task
            raw_attempt = payload.get("attempt_id")
            if isinstance(raw_attempt, str) and raw_attempt.strip():
                attempt_id = raw_attempt
            # Nested shapes from inspect_* tools.
            nested_task = payload.get("task")
            if task_id is None and isinstance(nested_task, Mapping):
                nested_id = nested_task.get("task_id")
                if isinstance(nested_id, str) and nested_id.strip():
                    task_id = nested_id
        try:
            plane.record_chat_tool_event(
                tool=result.tool,
                ok=result.ok,
                operation_id=operation_id,
                error=result.error,
                task_id=task_id,
                attempt_id=attempt_id,
                caller_provenance=result.caller_provenance,
                repo_capability_id=result.repo_capability_id,
            )
        except Exception:  # noqa: BLE001 — journaling is best-effort
            return

    def _model_chat(self, *, prompt: str, role: str = "chat") -> Generation:
        return self.client.chat(
            model=self.model,
            role=role,
            system_prompt=self.system_prompt or build_system_prompt(),
            prompt=prompt,
            reasoning=self.reasoning,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            operation_id=self._operation_id(role),
        )

    def _build_prompt(self) -> str:
        """Flatten the ephemeral transcript into a single user prompt.

        InferenceClient presents system_prompt + prompt only (no multi-message
        chat array). We keep history compact and structured.
        """
        parts: list[str] = []
        for turn in self.transcript:
            if turn.role == "user":
                parts.append(f"User:\n{turn.content}")
            elif turn.role == "assistant":
                parts.append(f"Assistant:\n{turn.content}")
            elif turn.role == "tool":
                parts.append(turn.content)
            elif turn.role == "system":
                parts.append(f"Note:\n{turn.content}")
        parts.append(
            "Respond with either final user-facing text or a tool_calls JSON object."
        )
        return "\n\n".join(parts)

    def handle_user_message(self, message: str) -> ChatSessionResult:
        """Run the tool loop for one user message until final text or bound."""
        if not isinstance(message, str) or not message.strip():
            raise ChatError("user message must be a non-empty string")

        normalized_message = message.strip()
        self.transcript.append(ChatTurn(role="user", content=normalized_message))
        if normalized_message == "/help" or normalized_message.startswith("/help "):
            # Keep the shortcut deterministic and inference-free so it remains
            # useful when a local model is unavailable or context-starved.
            text = format_harness_help()
            self.transcript.append(
                ChatTurn(role="assistant", content=text, meta={"stop": "help"})
            )
            return ChatSessionResult(
                final_text=text,
                transcript=list(self.transcript),
                tool_turns=0,
                stop_reason="help",
                generations=[],
            )
        generations: list[Generation] = []
        tool_turns = 0

        while True:
            try:
                generation = self._model_chat(
                    prompt=self._build_prompt(),
                    role="chat",
                )
            except InferenceError as exc:
                err = f"inference error: {exc}"
                self.transcript.append(
                    ChatTurn(role="assistant", content=err, meta={"error": True})
                )
                return ChatSessionResult(
                    final_text=err,
                    transcript=list(self.transcript),
                    tool_turns=tool_turns,
                    stop_reason="inference_error",
                    generations=generations,
                )
            generations.append(generation)
            parsed = parse_model_turn(generation.content)

            if parsed["kind"] == "final":
                text = str(parsed["text"])
                self.transcript.append(
                    ChatTurn(
                        role="assistant",
                        content=text,
                        meta={"stop": "final"},
                    )
                )
                return ChatSessionResult(
                    final_text=text,
                    transcript=list(self.transcript),
                    tool_turns=tool_turns,
                    stop_reason="final",
                    generations=generations,
                )

            if parsed["kind"] == "error":
                # Feed structured parse error back so the model can recover once.
                error_payload = {
                    "ok": False,
                    "error": parsed["error"],
                    "raw": parsed.get("raw"),
                }
                tool_msg = (
                    "Model response parse error (JSON):\n"
                    + json.dumps(error_payload, indent=2, sort_keys=True)
                )
                self.transcript.append(
                    ChatTurn(
                        role="assistant",
                        content=generation.content,
                        meta={"parse_error": parsed["error"]},
                    )
                )
                self.transcript.append(
                    ChatTurn(role="tool", content=tool_msg, meta={"parse_error": True})
                )
                tool_turns += 1
                if tool_turns > self.max_tool_turns:
                    final = (
                        "Stopped: exceeded max tool turns while recovering from "
                        f"malformed model output ({parsed['error']})."
                    )
                    self.transcript.append(
                        ChatTurn(role="assistant", content=final, meta={"stop": "max_tool_turns"})
                    )
                    return ChatSessionResult(
                        final_text=final,
                        transcript=list(self.transcript),
                        tool_turns=tool_turns,
                        stop_reason="max_tool_turns",
                        generations=generations,
                    )
                continue

            # tool_calls branch
            if tool_turns >= self.max_tool_turns:
                final = (
                    f"Stopped: max tool turns ({self.max_tool_turns}) reached "
                    "before a final answer."
                )
                self.transcript.append(
                    ChatTurn(
                        role="assistant",
                        content=generation.content,
                        meta={"stop": "max_tool_turns"},
                    )
                )
                self.transcript.append(
                    ChatTurn(role="assistant", content=final, meta={"stop": "max_tool_turns"})
                )
                return ChatSessionResult(
                    final_text=final,
                    transcript=list(self.transcript),
                    tool_turns=tool_turns,
                    stop_reason="max_tool_turns",
                    generations=generations,
                )

            calls = list(parsed["calls"])
            self.transcript.append(
                ChatTurn(
                    role="assistant",
                    content=generation.content,
                    meta={
                        "tool_calls": [
                            {"name": c["name"], "arguments": c.get("arguments")}
                            for c in calls
                        ]
                    },
                )
            )
            results: list[ToolResult] = []
            for call in calls:
                result = self.dispatcher.execute(
                    call["name"], call.get("arguments")
                )
                results.append(result)
                self._journal_tool_result(
                    result,
                    operation_id=self._operation_id(f"tool/{call['name']}"),
                )
            tool_msg = format_tool_results_message(results)
            self.transcript.append(
                ChatTurn(
                    role="tool",
                    content=tool_msg,
                    meta={
                        "results": [r.to_json_dict() for r in results],
                    },
                )
            )
            tool_turns += 1


def run_chat_turn(
    *,
    client: InferenceClient,
    dispatcher: ToolDispatcher,
    model: str,
    message: str,
    system_prompt: str | None = None,
    max_tool_turns: int = DEFAULT_MAX_TOOL_TURNS,
    reasoning: str = "off",
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
) -> ChatSessionResult:
    """Convenience: one-shot user message on a fresh ephemeral session."""
    session = ChatSession(
        client=client,
        dispatcher=dispatcher,
        model=model,
        system_prompt=system_prompt,
        max_tool_turns=max_tool_turns,
        reasoning=reasoning,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return session.handle_user_message(message)
