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

# Filler words that are never a memory on their own. "remember that" (no
# actual content) otherwise captures the literal word "that" as the fact,
# because the optional `(?:that\s+)?` group needs trailing whitespace to
# match and falls through to the `(.+)` body instead.
_CONTENT_FILLER = {"that", "about", "it", "this", "the"}

# Upper bound on a single stored memory. Memories are injected verbatim
# into the prompt, so an unbounded one could crowd out the entire system
# prompt (or the user's actual question) inside the context window.
MAX_MEMORY_CHARS = 500


def _clean_content(raw: str) -> str | None:
    """Normalize a captured fact body, or None if it isn't real content."""
    content = raw.strip().rstrip(".!?").strip()
    if not content:
        return None
    if content.lower() in _CONTENT_FILLER:
        return None
    return content[:MAX_MEMORY_CHARS]


# Fallback for a command that doesn't START the message, e.g. a typo or
# false start: "rembember remember that my name is Theo". Requires the
# explicit "remember/forget that" form, and the message must not be a
# question — so "Do you remember that my name is Theo?" stays a question
# rather than silently storing something.
_EMBEDDED_REMEMBER_RE = re.compile(r"\bremember\s+that\s+(.+)$", re.IGNORECASE)
_EMBEDDED_FORGET_RE = re.compile(r"\bforget\s+(?:that|about)\s+(.+)$", re.IGNORECASE)


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
        content = _clean_content(m.group(1))
        if content:
            return MemoryCommand(kind="forget", content=content)

    m = _REMEMBER_RE.match(stripped)
    if m:
        content = _clean_content(m.group(1))
        if content:
            return MemoryCommand(kind="remember", content=content)

    # Anchored patterns missed. Try the embedded form, but only for
    # non-questions — a real user typo ("rembember remember that ...")
    # shouldn't be silently dropped just because the message didn't
    # start cleanly.
    if not stripped.rstrip().endswith("?"):
        m = _EMBEDDED_FORGET_RE.search(stripped)
        if m:
            content = _clean_content(m.group(1))
            if content:
                return MemoryCommand(kind="forget", content=content)

        m = _EMBEDDED_REMEMBER_RE.search(stripped)
        if m:
            content = _clean_content(m.group(1))
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
