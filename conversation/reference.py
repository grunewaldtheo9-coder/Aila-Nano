"""ReferenceResolver: turn a short, context-dependent message into the
thing it refers to.

Conversation is full of messages that mean nothing on their own — "the
second one", "number 2", "yes", "the last one". Their meaning lives in the
turn before them. This resolver reads the recent turns and, when it can,
resolves the reference to a concrete value with a confidence level:

- high   : an unambiguous ordinal against a list the assistant just gave
           ("the second one" -> the 2nd item)
- medium : a best-guess from context
- low    : genuinely ambiguous -> the caller should ask a natural
           clarification rather than invent an answer (spec §18, §107)

It is deterministic and CPU-only; it never calls the model and never
invents a value. When it can't resolve, it says so (`kind="none"`), so the
system can ask instead of guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ordinal words -> 1-based index (-1 = last). Deliberately NOT the cardinal
# numbers ("one", "two"): "the second one" and "the last one" use "one" as
# the noun, not the ordinal, so treating "one" as 1 mis-resolves them.
_ORDINALS = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    "last": -1, "final": -1,
    # Portuguese
    "primeiro": 1, "primeira": 1, "segundo": 2, "segunda": 2,
    "terceiro": 3, "terceira": 3, "quarto": 4, "quarta": 4,
    "quinto": 5, "quinta": 5, "último": -1, "ultima": -1, "última": -1,
}

_AFFIRM = frozenset({
    "yes", "yeah", "yep", "yup", "sure", "exactly", "correct", "right",
    "definitely", "ok", "okay", "do it", "go ahead", "please do",
    "sim", "isso", "exato", "claro", "pode", "manda",
})
_NEGATE = frozenset({
    "no", "nope", "nah", "not really", "don't", "dont",
    "não", "nao", "melhor não",
})

_NUM_LINE = re.compile(r"^\s*(\d+)\s*[.)\-:]\s*(.+?)\s*$")


@dataclass
class Resolution:
    kind: str  # "list_item" | "affirmation" | "negation" | "none"
    value: str | None = None
    confidence: str = "low"  # "high" | "medium" | "low"
    options: list[str] = field(default_factory=list)  # candidates when ambiguous


def parse_options(text: str) -> list[str]:
    """Extract the list the assistant offered, from either a numbered list
    ("1. SQLite\n2. PostgreSQL") or an inline "A, B, or C" sentence."""
    # 1) numbered / bulleted lines
    items: list[str] = []
    for line in text.splitlines():
        m = _NUM_LINE.match(line)
        if m:
            items.append(m.group(2).strip().rstrip("."))
    if len(items) >= 2:
        return items

    # 2) inline "X, Y, or Z" (or Portuguese "ou"). Look at the sentence that
    #    contains the enumeration.
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if " or " in sentence or " ou " in sentence:
            # take the clause after a colon or verb-ish lead if present
            clause = sentence.split(":", 1)[-1]
            # split on commas and the final "or"/"ou"
            parts = re.split(r",\s*|\s+\bor\b\s+|\s+\bou\b\s+", clause)
            cleaned = [p.strip().strip(".?!").strip() for p in parts if p.strip()]
            # drop a leading lead-in fragment ("You could use SQLite" -> keep
            # only the capitalised/option tokens) by keeping short items
            cleaned = [_last_noun(c) for c in cleaned if c]
            cleaned = [c for c in cleaned if c and len(c.split()) <= 4]
            if len(cleaned) >= 2:
                return cleaned
    return []


def _last_noun(fragment: str) -> str:
    """For an inline option fragment like "You could use SQLite", keep the
    trailing option token ("SQLite"). Heuristic: the last capitalised or
    all-caps token, else the last word."""
    words = fragment.split()
    if not words:
        return fragment
    for w in reversed(words):
        if w[:1].isupper() or w.isupper():
            return w.strip(".,?!")
    return words[-1].strip(".,?!")


def _last_assistant(recent_turns: list[dict]) -> str | None:
    for t in reversed(recent_turns):
        if t.get("role") == "assistant":
            return t.get("content", "")
    return None


def _ordinal_index(message: str) -> int | None:
    low = message.lower().strip().rstrip("?.!")
    # explicit digit: "number 2", "option 2", "the 2", "2"
    m = re.search(r"\b(?:number|option|item|n[uú]mero|op[cç][aã]o)?\s*#?\s*(\d+)\b", low)
    if m and (m.group(1) in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}):
        # avoid matching years etc. — only when the message is short
        if len(low.split()) <= 4:
            return int(m.group(1))
    for word, idx in _ORDINALS.items():
        if re.search(rf"\b{re.escape(word)}\b", low):
            # require the message to be a short reference, not a full sentence
            if len(low.split()) <= 4:
                return idx
    return None


def resolve_reference(message: str, recent_turns: list[dict]) -> Resolution:
    """Resolve a short reference against the recent turns."""
    text = (message or "").strip()
    if not text:
        return Resolution(kind="none")

    low = text.lower().rstrip("?.! ")

    # 1) List-item reference: "the second one", "number 2", "the last".
    idx = _ordinal_index(text)
    if idx is not None:
        assistant = _last_assistant(recent_turns)
        options = parse_options(assistant) if assistant else []
        if options:
            if idx == -1:
                return Resolution("list_item", options[-1], "high", options)
            if 1 <= idx <= len(options):
                return Resolution("list_item", options[idx - 1], "high", options)
            # asked for an item beyond the list -> ambiguous
            return Resolution("none", None, "low", options)
        # An ordinal with no list to resolve against.
        return Resolution("none", None, "low")

    # 2) "the other one" — only unambiguous when exactly two options exist.
    if low in {"the other one", "the other", "o outro", "a outra"}:
        assistant = _last_assistant(recent_turns)
        options = parse_options(assistant) if assistant else []
        if len(options) == 2:
            return Resolution("list_item", options[1], "medium", options)
        return Resolution("none", None, "low", options)

    # 3) Affirmation / negation — meaningful only as an answer to the
    #    assistant's previous message.
    if low in _AFFIRM:
        return Resolution("affirmation", None, "high")
    if low in _NEGATE:
        return Resolution("negation", None, "high")

    return Resolution(kind="none")
