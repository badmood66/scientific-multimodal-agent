from __future__ import annotations

import math
import re
from dataclasses import dataclass


def normalize_answer(value: str) -> str:
    value = value.strip().lower().replace(",", "")
    value = re.sub(r"\s+", " ", value)
    return value.rstrip(".")


def parse_number(value: str) -> float | None:
    match = re.fullmatch(r"[-+]?\d*\.?\d+\s*%?", normalize_answer(value))
    if not match:
        return None
    cleaned = value.strip().replace(",", "")
    is_percent = cleaned.endswith("%")
    number = float(cleaned.rstrip("%"))
    return number / 100 if is_percent else number


@dataclass(slots=True)
class Verification:
    accepted: bool
    confidence: float
    reason: str


class EvidenceVerifier:
    def verify(self, answer: str, memory: dict, tool_calls: list[dict]) -> Verification:
        normalized = normalize_answer(answer)
        if not normalized:
            return Verification(False, 0.0, "empty answer")
        evidence_memory = {
            key: value
            for key, value in memory.items()
            if key not in {"direct_candidate", "raw_output"}
        }
        evidence_text = str(evidence_memory).lower()
        evidence_supported = normalized in evidence_text
        successful_tool = any(item.get("success") for item in tool_calls)
        confidence = 0.45 + 0.25 * evidence_supported + 0.2 * successful_tool
        if parse_number(answer) is not None:
            confidence += 0.05
        confidence = min(confidence, 0.95)
        return Verification(
            accepted=evidence_supported or successful_tool,
            confidence=confidence,
            reason="supported by extracted evidence" if evidence_supported else "plausibility check passed",
        )


def answers_match(prediction: str, gold: str, tolerance: float = 0.05) -> bool:
    pred_number, gold_number = parse_number(prediction), parse_number(gold)
    if pred_number is not None and gold_number is not None:
        if gold_number == 0:
            return math.isclose(pred_number, gold_number, abs_tol=tolerance)
        return abs(pred_number - gold_number) / abs(gold_number) <= tolerance
    return normalize_answer(prediction) == normalize_answer(gold)
