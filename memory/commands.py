"""Natural-language memory commands: "remember that X", "forget that X",
"what do you remember about me?".

Deliberately rule-based (regex), not model-driven: the whole point of
having explicit commands is a *guaranteed*, zero-hallucination path for
storing/removing/listing memories — asking an ~11M-parameter model to
reliably emit a structured "store this fact" decision would reintroduce
exactly the unreliability this architecture change is meant to avoid.
`agents/base.py` checks `parse_memory_command` before touching the model
at all; a match is handled deterministically and never reaches
generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

CommandKind = Literal["remember", "forget", "list"]


@dataclass
class MemoryCommand:
    kind: CommandKind
    content: str | None = None  # the fact text for remember/forget; unused for list


# Order matters: more specific patterns first. Each is matched against the
# *stripped* user message with IGNORECASE; `.+` bodies are captured as the
# fact text.
_REMEMBER_RE = re.compile(r"^(?:please\s+)?remember\s+(?:that\s+)?(.+)$", re.IGNORECASE)
_FORGET_RE = re.compile(r"^(?:please\s+)?forget\s+(?:that|about)?\s*(.+)$", re.IGNORECASE)
_LIST_RE = re.compile(
    r"^what\s+do\s+you\s+remember(?:\s+about\s+me)?\??$|^what\s+have\s+i\s+told\s+you\s+to\s+remember\??$",
    re.IGNORECASE,
)


def parse_memory_command(text: str) -> MemoryCommand | None:
    """Returns a MemoryCommand if `text` matches one of the recognized
    patterns, else None (meaning: treat it as an ordinary message)."""
    stripped = text.strip()
    if not stripped:
        return None

    if _LIST_RE.match(stripped):
        return MemoryCommand(kind="list")

    m = _FORGET_RE.match(stripped)
    if m:
        content = m.group(1).strip().rstrip(".!?")
        if content:
            return MemoryCommand(kind="forget", content=content)

    m = _REMEMBER_RE.match(stripped)
    if m:
        content = m.group(1).strip().rstrip(".!?")
        if content:
            return MemoryCommand(kind="remember", content=content)

    return None


# -- category guessing --------------------------------------------------

_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("identity", ("my name is", "i am called", "i'm called", "call me", "you can call me")),
    ("preference", ("favorite", "favourite", "i like", "i love", "i prefer", "i hate", "i dislike", "i enjoy")),
    ("project", ("project", "working on", "building", "repo", "repository", "codebase")),
    ("instruction", ("always", "never", "please always", "please never", "from now on", "in the future")),
]


def guess_category(content: str) -> str:
    """Cheap keyword heuristic for `long_term_facts.category` when a
    memory is stored without an explicit category (e.g. via the natural-
    language "remember that ..." command). Good enough to route obvious
    cases; falls back to 'personal_fact' for anything else, since most
    ad-hoc "remember that ..." statements are exactly that."""
    lowered = content.lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(kw in lowered for kw in keywords):
            return category
    return "personal_fact"
