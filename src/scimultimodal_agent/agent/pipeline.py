from __future__ import annotations

import re
import time
from typing import ClassVar, Protocol

from scimultimodal_agent.agent.memory import compact_memory, extract_json_object
from scimultimodal_agent.agent.planner import QuestionPlanner
from scimultimodal_agent.agent.tools import CalculatorTool
from scimultimodal_agent.agent.verifier import EvidenceVerifier, parse_number
from scimultimodal_agent.schemas import AgentTrace, ChartSample, Evidence


class VisionLanguageBackend(Protocol):
    def generate(self, image_path, prompt: str, max_new_tokens: int | None = None) -> str: ...


def clean_short_answer(text: str) -> str:
    memory = extract_json_object(text)
    for key in ("final_answer", "answer", "answer_candidate"):
        if key in memory and str(memory[key]).strip():
            return str(memory[key]).strip()
    match = re.search(r"(?:final answer|answer)\s*[:：]\s*(.+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().splitlines()[0]
    return text.strip().splitlines()[-1] if text.strip() else ""


def expected_answer_type(question: str) -> str:
    if re.search(r"\brepresents? what|what does .* represent", question, re.IGNORECASE):
        return "label_or_concept"
    if re.search(
        r"\bhow many|what percentage|bar value|what (?:is the )?"
        r"(?:value|ratio|percentage|percent|difference|average|sum)|"
        r"what was the .*\b(?:income|value|number|rate|amount)\b",
        question,
        re.IGNORECASE,
    ):
        return "number"
    if re.search(
        r"\bwhich|what (?:company|country|category|brand|game|year|color)\b",
        question,
        re.IGNORECASE,
    ):
        return "label_or_concept"
    return "short_text_or_number"


def sample_context(sample: ChartSample) -> str:
    parts = []
    if sample.caption:
        parts.append(f"Figure caption: {sample.caption}")
    if sample.answer_options:
        parts.append(
            "Answer options (return only the correct option letter or comma-separated letters): "
            + "; ".join(sample.answer_options)
        )
    return "\n".join(parts)


class ScientificChartAgent:
    MODES: ClassVar[set[str]] = {
        "direct",
        "cot",
        "full",
        "no_memory",
        "no_tools",
        "no_verifier",
    }

    def __init__(self, backend: VisionLanguageBackend) -> None:
        self.backend = backend
        self.planner = QuestionPlanner()
        self.calculator = CalculatorTool()
        self.verifier = EvidenceVerifier()

    def run(self, sample: ChartSample, mode: str = "full") -> AgentTrace:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        started = time.perf_counter()
        trace = AgentTrace(
            sample_id=sample.sample_id,
            question=sample.question,
            gold_answer=sample.answer,
            mode=mode,
        )
        try:
            if mode in {"direct", "cot"}:
                trace.prediction = self._baseline(sample, chain_of_thought=mode == "cot")
                trace.confidence = 0.5
                trace.verified = False
                return trace

            trace.plan = self.planner.plan(sample.question)
            direct_candidate = self._baseline(sample, chain_of_thought=False)
            memory = self._perceive(sample, trace.plan.reasoning_type.value)
            memory["direct_candidate"] = direct_candidate
            if mode == "no_memory":
                memory = {"direct_candidate": direct_candidate}
            trace.memory = memory
            trace.evidence = self._collect_evidence(memory)

            answer_type = expected_answer_type(sample.question)
            should_calculate = trace.plan.needs_calculator or answer_type == "number"
            if mode != "no_tools" and should_calculate:
                expression = self._find_expression(memory)
                if expression:
                    result = self.calculator(expression)
                    trace.tool_calls.append(result.to_dict())

            trace.prediction = self._resolve_answer(
                direct_candidate,
                memory,
                trace.tool_calls,
                answer_type,
                sample.question,
                bool(sample.answer_options),
            )
            if mode != "no_verifier":
                result = self.verifier.verify(trace.prediction, memory, trace.tool_calls)
                trace.verified = result.accepted
                trace.confidence = result.confidence
            else:
                trace.confidence = 0.5
            return trace
        except Exception as exc:  # noqa: BLE001 - preserve failures in per-sample research traces
            trace.error = f"{type(exc).__name__}: {exc}"
            return trace
        finally:
            trace.latency_seconds = time.perf_counter() - started

    def _baseline(self, sample: ChartSample, chain_of_thought: bool) -> str:
        if chain_of_thought:
            prompt = (
                "Study the chart, identify the relevant visual evidence, and reason briefly. "
                "Return one compact JSON object with keys `reasoning` and `final_answer`; "
                "the final answer must be short. Do not use markdown.\n"
                f"Expected answer type: {expected_answer_type(sample.question)}\n"
                f"{sample_context(sample)}\n"
                f"Question: {sample.question}"
            )
        else:
            prompt = (
                "Answer the scientific-figure question using the image and supplied caption. "
                "Return only the short answer, "
                "without explanation.\n"
                f"{sample_context(sample)}\n"
                f"Question: {sample.question}"
            )
        return clean_short_answer(self.backend.generate(sample.image_path, prompt, 160))

    def _perceive(self, sample: ChartSample, reasoning_type: str) -> dict:
        prompt = f"""You are the perception module of a scientific-figure reasoning agent.
Question: {sample.question}
Planned reasoning type: {reasoning_type}
Expected answer type: {expected_answer_type(sample.question)}
{sample_context(sample)}

Extract only evidence strictly needed for this question. Never transcribe or enumerate the full
figure, even when it contains many panels. Return one compact valid JSON object under 100 words:
{{
  "answer_type": "number or label_or_concept",
  "relevant_entities": ["at most two question-relevant labels, components, or concepts"],
  "relevant_values": [{{"label": "at most two relevant labels", "value": 0, "unit": "..."}}],
  "visual_evidence": "one short supporting observation",
  "calculation": {{"expression": "numeric arithmetic or empty", "purpose": "short"}},
  "answer_candidate": "short answer"
}}
For a ratio of X and Y, calculate X divided by Y in the written order. Include no more than two
values. If only one value is requested, include exactly that value and no other categories.
For questions asking what a bar, line, or color represents, answer with its legend or concept
label, not its numeric value. Do not infer unavailable values. Do not wrap the JSON in markdown."""
        raw = self.backend.generate(sample.image_path, prompt, 256)
        memory = extract_json_object(raw)
        memory.setdefault("raw_output", raw)
        return memory

    def _answer(self, sample: ChartSample, memory: dict, tool_calls: list[dict]) -> str:
        prompt = f"""You are the answer module of a scientific-chart agent.
Question: {sample.question}
Expected answer type: {expected_answer_type(sample.question)}
{sample_context(sample)}
Structured visual memory: {compact_memory(memory)}
Tool results: {compact_memory({"calls": tool_calls})}

Recheck the evidence against the image. If a successful calculator result answers the question,
use it. Otherwise preserve `answer_candidate` unless the image clearly contradicts it.
Return exactly one JSON object: {{"final_answer":"short answer","evidence":"brief"}}.
For a question asking what something represents, return the legend or concept, not the number.
Do not include markdown."""
        raw = self.backend.generate(sample.image_path, prompt, 160)
        return clean_short_answer(raw)

    @staticmethod
    def _resolve_answer(
        direct_candidate: str,
        memory: dict,
        tool_calls: list[dict],
        answer_type: str,
        question: str,
        has_answer_options: bool,
    ) -> str:
        """Resolve candidates with deterministic, auditable precedence rules."""
        successful = [call for call in tool_calls if call.get("success")]
        memory_candidate = str(memory.get("answer_candidate", "")).strip()
        if has_answer_options and re.fullmatch(r"[A-D](?:\s*,\s*[A-D])*", memory_candidate):
            return re.sub(r"\s+", "", memory_candidate)
        if re.search(r"\brepresents? what|what does .* represent", question, re.IGNORECASE) and memory_candidate:
            return memory_candidate
        direct_number = parse_number(direct_candidate)
        if answer_type == "number" and successful and direct_number is not None:
            tool_value = float(successful[-1]["output"])
            scale = max(abs(direct_number), abs(tool_value), 1e-9)
            if abs(direct_number - tool_value) / scale <= 0.1:
                return str(int(tool_value)) if tool_value.is_integer() else f"{tool_value:.10g}"
        if direct_candidate:
            return direct_candidate
        if answer_type == "number" and successful:
            tool_value = float(successful[-1]["output"])
            return str(int(tool_value)) if tool_value.is_integer() else f"{tool_value:.10g}"
        return memory_candidate

    @staticmethod
    def _find_expression(memory: dict) -> str:
        calculation = memory.get("calculation", {})
        if isinstance(calculation, dict):
            expression = calculation.get("expression", "")
            if isinstance(expression, str) and re.fullmatch(r"[\d\s.+\-*/()]+", expression):
                return expression.strip()
        if isinstance(calculation, str) and re.fullmatch(r"[\d\s.+\-*/()]+", calculation):
            return calculation.strip()
        return ""

    @staticmethod
    def _collect_evidence(memory: dict) -> list[Evidence]:
        output: list[Evidence] = []
        if not isinstance(memory, dict):
            return output
        evidence = memory.get("visual_evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        for item in evidence:
            output.append(Evidence(str(item), "chart_image", 0.7))
        return output
