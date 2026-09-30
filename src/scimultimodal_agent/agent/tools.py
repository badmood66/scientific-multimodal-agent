from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Any

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_calculate(expression: str) -> float:
    """Evaluate arithmetic only; names, calls, attributes, and indexing are rejected."""
    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponent is outside the safe range")
            return float(_OPS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return float(_OPS[type(node.op)](evaluate(node.operand)))
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    return evaluate(tree)


@dataclass(slots=True)
class ToolResult:
    name: str
    input: dict[str, Any]
    output: Any
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input": self.input,
            "output": self.output,
            "success": self.success,
            "error": self.error,
        }


class CalculatorTool:
    name = "calculator"

    def __call__(self, expression: str) -> ToolResult:
        try:
            value = safe_calculate(expression)
            return ToolResult(self.name, {"expression": expression}, value, True)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
            return ToolResult(self.name, {"expression": expression}, None, False, str(exc))
