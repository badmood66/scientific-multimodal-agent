from pathlib import Path

from scimultimodal_agent.agent.pipeline import ScientificChartAgent, expected_answer_type
from scimultimodal_agent.schemas import ChartSample


class FakeBackend:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, image_path, prompt: str, max_new_tokens=None) -> str:
        self.calls += 1
        if "perception module" in prompt:
            return (
                '{"chart_type":"bar","values":[{"label":"A","value":12},'
                '{"label":"B","value":5}],"visual_evidence":["bars A and B"],'
                '"calculation":{"expression":"12-5","purpose":"difference"},'
                '"answer_candidate":"7"}'
            )
        if "answer module" in prompt:
            return '{"final_answer":"7","evidence":"12 minus 5"}'
        return "7"


def sample() -> ChartSample:
    return ChartSample("x", Path("unused.png"), "What is the difference?", "7", "human")


def test_full_pipeline_uses_tool_and_verifier() -> None:
    trace = ScientificChartAgent(FakeBackend()).run(sample(), "full")
    assert trace.prediction == "7"
    assert trace.tool_calls[0]["output"] == 7.0
    assert trace.verified
    assert trace.error is None


def test_label_answer_preserves_direct_candidate() -> None:
    item = ChartSample("x", Path("unused.png"), "Which brand is highest?", "7", "human")
    trace = ScientificChartAgent(FakeBackend()).run(item, "full")
    assert trace.prediction == "7"


def test_conflicting_tool_does_not_override_numeric_candidate() -> None:
    result = ScientificChartAgent._resolve_answer(
        "30",
        {"answer_candidate": "74"},
        [{"success": True, "output": 74.0}],
        "number",
        "What is the average?",
    )
    assert result == "30"


def test_direct_baseline_is_single_call() -> None:
    backend = FakeBackend()
    trace = ScientificChartAgent(backend).run(sample(), "direct")
    assert trace.prediction == "7"
    assert backend.calls == 1


def test_expected_answer_type_detects_legend_question() -> None:
    question = "The blue bar represents what (value 55, 70)?"
    assert expected_answer_type(question) == "label_or_concept"
