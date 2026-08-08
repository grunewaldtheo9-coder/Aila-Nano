"""Turn a stored memory into a natural first-person reply.

Why this exists: memories are stored in the user's own words ("my name
is Theo"). Echoing that verbatim as an answer reads wrong ("What is my
name?" → "my name is Theo"), but handing the fact to a ~20M-parameter
model to rephrase reliably garbles it (measured: the model produced
"oovano o Grxtwaations ecameters" from a correctly-injected memory).

So: a small, deterministic pronoun flip for the overwhelmingly common
"my X is Y" shape, with an honest, never-wrong fallback ("You told me:
...") for everything else. Rules, not generation — this path must never
be able to invent or corrupt a remembered fact.
"""

from __future__ import annotations

import re

# (pattern, replacement) applied to the *start* of a memory only, so a
# mid-sentence "my" (e.g. "Ana is my sister") is left alone.
_FIRST_TO_SECOND_PERSON = [
    (re.compile(r"^my\b", re.IGNORECASE), "your"),
    (re.compile(r"^i\s+am\b", re.IGNORECASE), "you are"),
    (re.compile(r"^i'm\b", re.IGNORECASE), "you are"),
    (re.compile(r"^i\s+", re.IGNORECASE), "you "),
    # Portuguese
    (re.compile(r"^meu\b", re.IGNORECASE), "seu"),
    (re.compile(r"^minha\b", re.IGNORECASE), "sua"),
    (re.compile(r"^eu\s+sou\b", re.IGNORECASE), "você é"),
    (re.compile(r"^eu\s+", re.IGNORECASE), "você "),
]

_FALLBACK_EN = "You told me: {memory}"
_FALLBACK_PT = "Você me disse: {memory}"


def memory_to_answer(memory: str, language: str = "en") -> str:
    """Render a stored memory as a reply to a question about it.

    Never paraphrases beyond a leading-pronoun flip: whatever the user
    stored is what comes back, so this path cannot fabricate.
    """
    text = (memory or "").strip().rstrip(".")
    if not text:
        return ""

    for pattern, replacement in _FIRST_TO_SECOND_PERSON:
        flipped, n = pattern.subn(replacement, text, count=1)
        if n:
            return flipped[0].upper() + flipped[1:] + "."

    template = _FALLBACK_PT if language == "pt" else _FALLBACK_EN
    return template.format(memory=text) + "."
