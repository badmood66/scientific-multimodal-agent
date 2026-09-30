from __future__ import annotations

from collections import defaultdict
from statistics import mean

from scimultimodal_agent.agent.verifier import answers_match
from scimultimodal_agent.schemas import AgentTrace


def expected_calibration_error(traces: list[AgentTrace], bins: int = 10) -> float:
    if not traces:
        return 0.0
    bucketed: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for trace in traces:
        confidence = min(max(trace.confidence, 0.0), 1.0)
        bucket = min(int(confidence * bins), bins - 1)
        correct = float(answers_match(trace.prediction, trace.gold_answer))
        bucketed[bucket].append((confidence, correct))
    ece = 0.0
    for values in bucketed.values():
        weight = len(values) / len(traces)
        ece += weight * abs(mean(x[0] for x in values) - mean(x[1] for x in values))
    return ece


def summarize(traces: list[AgentTrace], tolerance: float = 0.05) -> dict:
    successful = [trace for trace in traces if not trace.error]
    correct = [answers_match(t.prediction, t.gold_answer, tolerance) for t in successful]
    tool_calls = [call for trace in traces for call in trace.tool_calls]
    return {
        "samples": len(traces),
        "completed": len(successful),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "verified_rate": mean(t.verified for t in successful) if successful else 0.0,
        "tool_success_rate": (
            mean(bool(call.get("success")) for call in tool_calls) if tool_calls else None
        ),
        "mean_latency_seconds": mean(t.latency_seconds for t in traces) if traces else 0.0,
        "ece": expected_calibration_error(successful),
        "errors": len(traces) - len(successful),
    }

