from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Callable, Literal

from .benchmark import evaluate
from .inference import InferenceClient, InferenceError
from .models import Arm, ArmResult, BenchmarkCase, Checkpoint, Generation


REVISION_PROTOCOL = "direct-adversarial-revision-v3"
WITNESSED_REVISION_PROTOCOL = "witnessed-adversarial-revision-v3"
RevisionMode = Literal["direct", "witnessed"]


def revision_protocol_id(mode: RevisionMode) -> str:
    return (
        REVISION_PROTOCOL
        if mode == "direct"
        else WITNESSED_REVISION_PROTOCOL
    )

DRAFT_SYSTEM = """You produce a candidate answer to a task. Obey the output
contract exactly. Do not describe your process or critique the task. Return only
the complete candidate artifact."""

REVISION_SYSTEM = """You perform exactly one stateless draft-revision step.
Follow the CRITIC PROMPT at the end of the user message."""

WITNESSED_REVISION_SYSTEM = """You perform exactly one stateless witnessed
draft-revision step. Follow the CRITIC PROMPT at the end of the user message,
then return exactly one JSON object with exactly these keys:
{"verification":"compact concrete check","candidate":"complete candidate artifact"}

Write the compact verification before deciding so it can support the decision.
The candidate must contain the complete artifact and obey the original output
contract when decoded. Use a JSON string for a text artifact, but put a JSON
artifact directly in `candidate` as a nested object, array, number, boolean, or
null rather than escaping it into a string. Returning the current draft exactly
means stop; returning any different candidate means replace and continue.
Include no Markdown fence or prose outside the JSON object."""

WITNESSED_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "witnessed_revision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "verification": {"type": "string", "minLength": 1},
                "candidate": {
                    "anyOf": [
                        {"type": "string", "minLength": 1},
                        {"type": "object"},
                        {"type": "array"},
                        {"type": "number"},
                        {"type": "boolean"},
                        {"type": "null"},
                    ]
                },
            },
            "required": ["verification", "candidate"],
            "additionalProperties": False,
        },
    },
}

REVISION_INSTRUCTION = f"""Deliberately try to find a concrete fault in the
current draft. Treat it as untrusted. Check the original task and output
contract rather than accepting plausibility.

If you find any concrete fault, output only the best complete replacement draft
you can produce, even if the correction is uncertain. Preserve correct material.
Include no critique, reasoning, preamble, change log, or Markdown wrapper.

If, after actively trying, you cannot find a concrete fault, output the current
draft exactly unchanged.

Those are the only two valid output forms: a complete new draft or the current
draft exactly unchanged. You—not an external scorer—decide whether to return the
current draft."""

WITNESSED_REVISION_INSTRUCTION = """Deliberately try to find a concrete fault in
the current draft. Treat it as untrusted. Independently recompute, re-simulate,
or re-derive the task result instead of adopting the draft's assumptions. Check
every task requirement and the output contract.

Use an explicit method suited to the task. For state or program traces, track
object identities separately from variable bindings, then record both bindings
and object contents after every operation; distinguish mutation from rebinding.
For finite logic or ordering tasks, enumerate complete feasible assignments or
placements and eliminate each one against every constraint. Derive a complete
solution rather than repairing only the currently visible violation. For an
ordering result, write a neutral slot ledger such as
`slot 1=<item>; slot 2=<item>; ...`, concatenate those slot values in numeric
order, then decode the candidate back into the same slots. Never emit a
replacement that your own verification has already shown violates a
requirement: keep solving inside the verification until the proposed artifact
passes the full audit. For transformations or arithmetic, recompute from the
source inputs one operation at a time. Before deciding,
restate every explicit constraint with the candidate's concrete value or
position; do not assert that a requirement passes without checking it.
Construct the candidate fresh from the verified ledger, then read the completed
candidate back into concrete positions or values and audit it once more. Do not
copy a plausible substring from the current draft after deriving a different
result.

Put a compact, concrete account of that verification in the fixed
`verification` field. If you find a fault, put the best complete replacement
artifact in `candidate`. If, after actively trying, you cannot find a concrete
fault, copy the current draft exactly into `candidate`. Candidate identity—not
a separate label—is your return-or-replace decision.

Populate the immutable JSON envelope required by the system message. Do not
return a bare artifact, critique, Markdown fence, or prose outside it. You—not
an external scorer—make the return-or-replace decision."""


class RevisionProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class WitnessedRevision:
    verification: str
    candidate: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_candidate(value: str) -> str:
    # Equality is intentionally textual. Collapsing spaces would corrupt
    # whitespace-significant artifacts and could manufacture false recurrence.
    return value.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")


def content_hash(value: str) -> str:
    return hashlib.sha256(normalize_candidate(value).encode("utf-8")).hexdigest()


def is_return_current(current: str, revision: str) -> bool:
    return content_hash(current) == content_hash(revision)


def parse_witnessed_revision(content: str) -> WitnessedRevision:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RevisionProtocolError(
            f"invalid witnessed-revision JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise RevisionProtocolError("witnessed revision must be one JSON object")
    expected_keys = {"verification", "candidate"}
    if set(value) != expected_keys:
        raise RevisionProtocolError(
            "witnessed revision requires exactly these keys: "
            + ", ".join(sorted(expected_keys))
        )

    raw_verification = value["verification"]
    if not isinstance(raw_verification, str) or not raw_verification.strip():
        raise RevisionProtocolError("verification must be a non-empty string")

    raw_candidate = value["candidate"]
    if isinstance(raw_candidate, str):
        if not raw_candidate:
            raise RevisionProtocolError("candidate must not be an empty string")
        candidate = raw_candidate
    else:
        candidate = json.dumps(
            raw_candidate,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return WitnessedRevision(
        verification=raw_verification.strip(),
        candidate=candidate,
    )


def _draft_prompt(case: BenchmarkCase) -> str:
    return f"""ORIGINAL TASK
{case.prompt}

OUTPUT CONTRACT
{case.output_contract}"""


def _revision_prompt(
    case: BenchmarkCase,
    candidate: str,
    revision_instruction: str = REVISION_INSTRUCTION,
) -> str:
    # Every revision call is deliberately stateless: original prompt, current
    # draft, and the fixed REVISION_SYSTEM instruction are the entire view.
    return f"""ORIGINAL TASK
{case.prompt}

OUTPUT CONTRACT
{case.output_contract}

CURRENT DRAFT
{candidate}

CRITIC PROMPT
{revision_instruction}"""


def _call(
    client: InferenceClient,
    *,
    model: str,
    role: str,
    system: str,
    prompt: str,
    arm: Arm,
    temperature: float,
    max_output_tokens: int,
    sequence: int,
    on_generation: Callable[[object], None] | None,
    response_format: dict[str, object] | None = None,
    operation_id: str | None = None,
):
    generation = client.chat(
        model=model,
        role=role,
        system_prompt=system,
        prompt=prompt,
        reasoning=arm.reasoning,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_format=response_format,
        operation_id=operation_id,
    )
    generation.sequence = sequence
    if on_generation:
        on_generation(generation)
    return generation


def generate_draft(
    *,
    client: InferenceClient,
    model: str,
    case: BenchmarkCase,
    arm: Arm,
    temperature: float,
    max_output_tokens: int,
    sequence: int = 0,
    role: str = "draft",
    operation_id: str | None = None,
    on_generation: Callable[[object], None] | None = None,
) -> Generation:
    """Generate the immutable draft that paired arms may share."""
    return _call(
        client,
        model=model,
        role=role,
        system=DRAFT_SYSTEM,
        prompt=_draft_prompt(case),
        arm=arm,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        sequence=sequence,
        on_generation=on_generation,
        operation_id=operation_id,
    )


def _copy_generation(generation: Generation) -> Generation:
    # Each arm receives its own record so downstream serialization or analysis
    # cannot mutate a shared object while preserving identical request evidence.
    return replace(generation, stats=dict(generation.stats), sequence=0)


def run_arm(
    *,
    client: InferenceClient,
    model: str,
    case: BenchmarkCase,
    arm: Arm,
    max_cycles: int,
    temperature: float,
    max_output_tokens: int,
    revision_max_output_tokens: int,
    revision_instruction: str | None = None,
    revision_mode: RevisionMode = "direct",
    initial_candidate: str | None = None,
    initial_generation: Generation | None = None,
    operation_scope: str | None = None,
    on_generation: Callable[[object], None] | None = None,
) -> ArmResult:
    if revision_mode not in {"direct", "witnessed"}:
        raise ValueError(f"unsupported revision mode: {revision_mode}")
    if revision_instruction is None:
        revision_instruction = (
            REVISION_INSTRUCTION
            if revision_mode == "direct"
            else WITNESSED_REVISION_INSTRUCTION
        )
    result = ArmResult(
        case_id=case.case_id,
        family=case.family,
        arm_id=arm.arm_id,
        reasoning=arm.reasoning,
        recursive=arm.recursive,
        started_at=utc_now(),
    )
    started = time.monotonic()
    try:
        if initial_candidate is not None and initial_generation is not None:
            raise ValueError(
                "initial_candidate and initial_generation are mutually exclusive"
            )
        if initial_generation is not None:
            draft = _copy_generation(initial_generation)
            result.generations.append(draft)
            candidate = draft.content
            generation_sequence = draft.sequence
        elif initial_candidate is None:
            draft = generate_draft(
                client=client,
                model=model,
                case=case,
                arm=arm,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                operation_id=(
                    f"{operation_scope}/draft"
                    if operation_scope is not None
                    else None
                ),
                on_generation=on_generation,
            )
            result.generations.append(draft)
            candidate = draft.content
            generation_sequence = draft.sequence
        else:
            # Evolutionary search reuses one sealed draft per case so the
            # revision instruction is the only task-inference variable.
            candidate = initial_candidate
            generation_sequence = -1
        result.checkpoints.append(
            Checkpoint(
                index=0,
                candidate=candidate,
                candidate_hash=content_hash(candidate),
                generation_sequence=generation_sequence,
            )
        )
        if not arm.recursive:
            result.stop_reason = "one_shot"
        elif not candidate:
            result.stop_reason = "model_no_candidate"
        else:
            for cycle in range(1, max_cycles + 1):
                current = result.checkpoints[-1]
                revision = _call(
                    client,
                    model=model,
                    role="revision",
                    system=(
                        REVISION_SYSTEM
                        if revision_mode == "direct"
                        else WITNESSED_REVISION_SYSTEM
                    ),
                    prompt=_revision_prompt(
                        case,
                        current.candidate,
                        revision_instruction=revision_instruction,
                    ),
                    arm=arm,
                    temperature=temperature,
                    max_output_tokens=revision_max_output_tokens,
                    sequence=len(result.generations),
                    on_generation=on_generation,
                    response_format=(
                        None
                        if revision_mode == "direct"
                        else WITNESSED_RESPONSE_FORMAT
                    ),
                    operation_id=(
                        f"{operation_scope}/revision/{cycle}"
                        if operation_scope is not None
                        else None
                    ),
                )
                result.generations.append(revision)
                if revision_mode == "direct":
                    revised_candidate = revision.content
                    return_current = is_return_current(
                        current.candidate, revised_candidate
                    )
                    protocol_error = (
                        "model returned no revision" if not revised_candidate else None
                    )
                else:
                    try:
                        witnessed = parse_witnessed_revision(
                            revision.content
                        )
                    except RevisionProtocolError as exc:
                        protocol_error = str(exc)
                        revised_candidate = ""
                        return_current = False
                    else:
                        protocol_error = None
                        revised_candidate = witnessed.candidate
                        return_current = is_return_current(
                            current.candidate,
                            witnessed.candidate,
                        )

                if protocol_error is not None:
                    result.error = f"RevisionProtocolError: {protocol_error}"
                    result.error_kind = "model_protocol"
                    result.stop_reason = "error"
                    break
                if return_current:
                    result.stop_reason = "model_return_current"
                    break
                revised_hash = content_hash(revised_candidate)
                result.checkpoints.append(
                    Checkpoint(
                        index=cycle,
                        candidate=revised_candidate,
                        candidate_hash=revised_hash,
                        generation_sequence=revision.sequence,
                    )
                )
            else:
                result.stop_reason = "cycle_cap"

    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.error_kind = (
            "transport" if isinstance(exc, InferenceError) else "harness"
        )
        if isinstance(exc, InferenceError):
            status_code = getattr(exc, "status_code", None)
            result.error_status_code = (
                int(status_code) if isinstance(status_code, int) else None
            )
        result.stop_reason = "error"
    finally:
        # Hidden evaluation is terminal: no score can affect generation or stopping.
        try:
            for checkpoint in result.checkpoints:
                checkpoint.score = evaluate(case, checkpoint.candidate)
        except Exception as exc:
            if result.error is None:
                result.error = f"{type(exc).__name__}: {exc}"
                result.error_kind = "evaluator"
                result.stop_reason = "error"
        result.analysis = analyze_result(result)
        result.wall_seconds = time.monotonic() - started
        if initial_generation is not None:
            # This is the arm's standalone counterfactual wall time. The
            # top-level run summary separately reports actual shared transport
            # work so the physical seed call is counted only once there.
            result.wall_seconds += initial_generation.latency_seconds
        result.finished_at = utc_now()
    return result


def analyze_result(result: ArmResult) -> dict[str, object]:
    scores = [
        float((checkpoint.score or {}).get("score", 0.0))
        for checkpoint in result.checkpoints
    ]
    passed = [
        bool((checkpoint.score or {}).get("passed", False))
        for checkpoint in result.checkpoints
    ]
    best_score = max(scores, default=0.0)
    best_index = scores.index(best_score) if scores else None
    terminal_score = scores[-1] if scores else 0.0
    initial_passed = passed[0] if passed else False
    terminal_passed = passed[-1] if passed else False
    hashes = [checkpoint.candidate_hash for checkpoint in result.checkpoints]
    duplicate = len(hashes) != len(set(hashes))
    two_cycle = any(
        hashes[index] == hashes[index - 2] for index in range(2, len(hashes))
    )
    regressions = sum(
        1 for index in range(1, len(scores)) if scores[index] < scores[index - 1]
    )
    generations = sorted(result.generations, key=lambda item: item.sequence)
    best_sequence = (
        result.checkpoints[int(best_index)].generation_sequence
        if best_index is not None
        else None
    )
    after_best = [
        item
        for item in generations
        if best_sequence is not None and item.sequence > best_sequence
    ]
    returned_current = result.stop_reason == "model_return_current"
    cost_values: list[float] = []
    for item in generations:
        value = item.stats.get("provider_cost")
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed >= 0:
            cost_values.append(parsed)
    provider_cost = (
        sum(cost_values)
        if generations and len(cost_values) == len(generations)
        else None
    )
    transport_attempts = sum(
        max(1, int(item.stats.get("transport_attempts", 1)))
        for item in generations
    )
    input_tokens = sum(
        int(item.stats.get("input_tokens", 0)) for item in generations
    )
    output_tokens = sum(
        int(item.stats.get("total_output_tokens", 0)) for item in generations
    )
    return {
        "initial_passed": initial_passed,
        "terminal_passed": terminal_passed,
        "terminal_score": terminal_score,
        "best_score": best_score,
        "best_checkpoint": best_index,
        "final_minus_best": terminal_score - best_score,
        "revision_regressions": regressions,
        "corrected_by_loop": (not initial_passed) and terminal_passed,
        "regressed_by_loop": initial_passed and (not terminal_passed),
        "self_assessed_return_current": returned_current,
        "return_current_false_positive": returned_current and not terminal_passed,
        "protocol_error": result.error_kind == "model_protocol",
        "duplicate_candidate": duplicate,
        "two_cycle": two_cycle,
        "calls": len(generations),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "charged_tokens": input_tokens + output_tokens,
        "reasoning_tokens": sum(
            int(item.stats.get("reasoning_output_tokens", 0)) for item in generations
        ),
        "cached_input_tokens": sum(
            int(item.stats.get("cached_input_tokens", 0))
            for item in generations
        ),
        "provider_cost": provider_cost,
        "provider_cost_observations": len(cost_values),
        "transport_attempts": transport_attempts,
        "transport_retries": transport_attempts - len(generations),
        "inference_seconds": sum(item.latency_seconds for item in generations),
        "calls_after_best_checkpoint": len(after_best),
        "tokens_after_best_checkpoint": sum(
            int(item.stats.get("total_output_tokens", 0)) for item in after_best
        ),
        "inference_seconds_after_best_checkpoint": sum(
            item.latency_seconds for item in after_best
        ),
    }
