"""Exact arithmetic tool.

A language model this small guesses arithmetic; a calculator doesn't.
`try_calculate` recognizes messages that are (or contain) a plain
arithmetic expression and evaluates them exactly via an AST whitelist —
`eval()` is never used, and anything outside literal numbers and
+ - * / % ** ( ) is rejected, so there is no code-execution surface.
"""

from __future__ import annotations

import ast
import operator
import re

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# Leading phrases to strip before checking whether what remains is an
# arithmetic expression ("What is 12 * 8?" / "Quanto é 12 * 8?").
_LEADING = re.compile(
    r"^\s*(what\s+is|what's|calculate|compute|how\s+much\s+is|quanto\s+é|quanto\s+e|calcule)\s+",
    re.IGNORECASE,
)
_EXPRESSION = re.compile(r"^[\d\s\.\+\-\*/%\(\)]+$")

# Word-form operators, normalized before parsing.
_WORD_OPS = [
    (re.compile(r"\b(plus|mais)\b", re.I), "+"),
    (re.compile(r"\b(minus|menos)\b", re.I), "-"),
    (re.compile(r"\b(times|multiplied\s+by|vezes|multiplicado\s+por)\b", re.I), "*"),
    (re.compile(r"\b(divided\s+by|dividido\s+por)\b", re.I), "/"),
]

MAX_POW_EXPONENT = 1000
MAX_EXPRESSION_LENGTH = 100


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POW_EXPONENT:
            raise ValueError("exponent too large")
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"disallowed expression element: {type(node).__name__}")


def _format(value: float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def try_calculate(message: str) -> str | None:
    """If `message` is an arithmetic question, return the formatted exact
    result; otherwise None (not an arithmetic question — not an error)."""
    text = (message or "").strip().rstrip("?!.")
    if not text or len(text) > MAX_EXPRESSION_LENGTH:
        return None

    text = _LEADING.sub("", text)
    for pattern, symbol in _WORD_OPS:
        text = pattern.sub(symbol, text)
    text = text.strip()

    # Must look like pure arithmetic, contain a digit, and an operator.
    if not _EXPRESSION.match(text):
        return None
    if not re.search(r"\d", text) or not re.search(r"[\+\-\*/%]", text):
        return None
    # A lone negative number ("-5") is not a calculation request.
    if re.fullmatch(r"[\s\-\+]*[\d\.]+", text):
        return None

    try:
        tree = ast.parse(text, mode="eval")
        result = _safe_eval(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return _format(result)
