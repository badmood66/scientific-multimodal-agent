from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ReasoningType(str, Enum):
    LOOKUP = "lookup"
    COUNT = "count"
    COMPARISON = "comparison"
    ARITHMETIC = "arithmetic"
    TREND = "trend"
    OTHER = "other"


@dataclass(slots=True)
class ChartSample:
    sample_id: str
    image_path: Path
    question: str
    answer: str
    source_type: str
    caption: str = ""
    answer_options: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReasoningPlan:
    reasoning_type: ReasoningType
    steps: list[str]
    needs_structured_extraction: bool = True
    needs_calculator: bool = False


@dataclass(slots=True)
class Evidence:
    claim: str
    source: str
    confidence: float


@dataclass(slots=True)
class AgentTrace:
    sample_id: str
    question: str
    gold_answer: str
    prediction: str = ""
    mode: str = "full"
    plan: ReasoningPlan | None = None
    memory: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    verified: bool = False
    confidence: float = 0.0
    latency_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.plan:
            data["plan"]["reasoning_type"] = self.plan.reasoning_type.value
        return data
