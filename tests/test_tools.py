import pytest

from scimultimodal_agent.agent.tools import CalculatorTool, safe_calculate


def test_safe_calculator_handles_arithmetic() -> None:
    assert safe_calculate("(12.5 - 2.5) / 2") == 5.0


@pytest.mark.parametrize("expression", ["__import__('os')", "x + 1", "[1][0]"])
def test_safe_calculator_rejects_code(expression: str) -> None:
    with pytest.raises((ValueError, SyntaxError)):
        safe_calculate(expression)


def test_tool_records_failure() -> None:
    result = CalculatorTool()("1 / 0")
    assert not result.success
    assert result.error

