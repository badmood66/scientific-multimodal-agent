from scimultimodal_agent.agent.planner import QuestionPlanner
from scimultimodal_agent.schemas import ReasoningType


def test_routes_count_question() -> None:
    plan = QuestionPlanner().plan("How many categories are shown?")
    assert plan.reasoning_type is ReasoningType.COUNT
    assert not plan.needs_calculator


def test_routes_arithmetic_question() -> None:
    plan = QuestionPlanner().plan("What is the difference between A and B?")
    assert plan.reasoning_type is ReasoningType.ARITHMETIC
    assert plan.needs_calculator


def test_routes_superlative_question_as_comparison() -> None:
    plan = QuestionPlanner().plan("Which company has the highest revenue?")
    assert plan.reasoning_type is ReasoningType.COMPARISON
