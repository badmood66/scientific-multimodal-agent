from __future__ import annotations

import re

from scimultimodal_agent.schemas import ReasoningPlan, ReasoningType


class QuestionPlanner:
    """Deterministic routing keeps ablations reproducible and auditable."""

    _count = re.compile(r"\bhow many|number of|count\b", re.IGNORECASE)
    _comparison = re.compile(
        r"\bmore|less|higher|highest|lower|lowest|largest|smallest|maximum|minimum\b",
        re.IGNORECASE,
    )
    _arithmetic = re.compile(
        r"\bdifference|sum|total|average|mean|percent|percentage|ratio|combined\b", re.IGNORECASE
    )
    _trend = re.compile(r"\btrend|increase|decrease|change|over time|between .* and\b", re.IGNORECASE)

    def plan(self, question: str) -> ReasoningPlan:
        if self._count.search(question):
            kind = ReasoningType.COUNT
        elif self._arithmetic.search(question):
            kind = ReasoningType.ARITHMETIC
        elif self._comparison.search(question):
            kind = ReasoningType.COMPARISON
        elif self._trend.search(question):
            kind = ReasoningType.TREND
        elif question.strip():
            kind = ReasoningType.LOOKUP
        else:
            kind = ReasoningType.OTHER

        needs_calculator = kind == ReasoningType.ARITHMETIC
        steps = ["inspect_chart", "extract_relevant_values"]
        if needs_calculator:
            steps.append("calculate")
        steps.extend(["answer_concisely", "verify_against_evidence"])
        return ReasoningPlan(kind, steps, True, needs_calculator)
