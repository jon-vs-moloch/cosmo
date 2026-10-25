from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    family: str
    prompt: str
    evaluator: dict[str, Any]
    output_contract: str = "Return only the requested final artifact."

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BenchmarkCase":
        return cls(
            case_id=str(value["case_id"]),
            family=str(value["family"]),
            prompt=str(value["prompt"]),
            evaluator=dict(value["evaluator"]),
            output_contract=str(
                value.get("output_contract", "Return only the requested final artifact.")
            ),
        )


@dataclass(frozen=True)
class Arm:
    arm_id: str
    reasoning: str
    recursive: bool


ARMS: dict[str, Arm] = {
    "instant_once": Arm("instant_once", "off", False),
    "thinking_once": Arm("thinking_once", "on", False),
    "instant_recursive": Arm("instant_recursive", "off", True),
    "thinking_recursive": Arm("thinking_recursive", "on", True),
}


@dataclass
class Generation:
    role: str
    content: str
    reasoning_hash: str | None
    stats: dict[str, Any]
    latency_seconds: float
    request_hash: str
    sequence: int = 0
    model_instance_id: str | None = None


@dataclass
class Checkpoint:
    index: int
    candidate: str
    candidate_hash: str
    generation_sequence: int
    score: dict[str, Any] | None = None


@dataclass
class ArmResult:
    case_id: str
    family: str
    arm_id: str
    reasoning: str
    recursive: bool
    checkpoints: list[Checkpoint] = field(default_factory=list)
    generations: list[Generation] = field(default_factory=list)
    stop_reason: str = "not_started"
    started_at: str = ""
    finished_at: str = ""
    wall_seconds: float = 0.0
    error: str | None = None
    error_kind: str | None = None
    error_status_code: int | None = None
    analysis: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
