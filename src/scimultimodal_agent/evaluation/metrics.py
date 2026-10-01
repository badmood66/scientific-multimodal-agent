from __future__ import annotations

import re
from collections import Counter, defaultdict
from statistics import mean

from scimultimodal_agent.agent.verifier import answers_match
from scimultimodal_agent.schemas import AgentTrace


def _tokens(value: str) -> list[str]:
    return re.findall(r"\w+", value.lower(), flags=re.UNICODE)


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens, gold_tokens = _tokens(prediction), _tokens(gold)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred_tokens), overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(prediction: str, gold: str) -> float:
    pred_tokens, gold_tokens = _tokens(prediction), _tokens(gold)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    previous = [0] * (len(gold_tokens) + 1)
    for pred in pred_tokens:
        current = [0]
        for index, gold_token in enumerate(gold_tokens, start=1):
            if pred == gold_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    lcs = previous[-1]
    precision, recall = lcs / len(pred_tokens), lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall) if lcs else 0.0


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
        "token_f1": mean(token_f1(t.prediction, t.gold_answer) for t in successful)
        if successful
        else 0.0,
        "rouge_l_f1": mean(rouge_l_f1(t.prediction, t.gold_answer) for t in successful)
        if successful
        else 0.0,
        "verified_rate": mean(t.verified for t in successful) if successful else 0.0,
        "tool_success_rate": (
            mean(bool(call.get("success")) for call in tool_calls) if tool_calls else None
        ),
        "mean_latency_seconds": mean(t.latency_seconds for t in traces) if traces else 0.0,
        "ece": expected_calibration_error(successful),
        "errors": len(traces) - len(successful),
    }
