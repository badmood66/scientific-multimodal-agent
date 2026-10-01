from scimultimodal_agent.agent.memory import extract_json_object
from scimultimodal_agent.agent.verifier import answers_match
from scimultimodal_agent.evaluation.metrics import rouge_l_f1, token_f1


def test_extracts_json_from_fenced_output() -> None:
    result = extract_json_object('```json\n{"answer_candidate": "14"}\n```')
    assert result["answer_candidate"] == "14"


def test_does_not_treat_nested_object_as_complete_memory() -> None:
    truncated = '{"values":[{"label":"A"}'
    result = extract_json_object(truncated)
    assert result == {"raw_output": truncated}


def test_relaxed_numeric_match() -> None:
    assert answers_match("104", "100", tolerance=0.05)
    assert not answers_match("110", "100", tolerance=0.05)
    assert answers_match("10%", "10%")


def test_text_overlap_metrics() -> None:
    assert token_f1("blue solid line", "the blue line") > 0.6
    assert rouge_l_f1("blue solid line", "the blue line") > 0.6
    assert token_f1("red", "blue") == 0.0
