"""Natural-language memory commands: "remember that X", "forget that X",
"what do you remember about me?" — in English *and* Portuguese.

Deliberately rule-based (regex), not model-driven: the whole point of
having explicit commands is a *guaranteed*, zero-hallucination path for
storing/removing/listing memories — asking an ~20M-parameter model to
reliably emit a structured "store this fact" decision would reintroduce
exactly the unreliability this architecture change is meant to avoid.
`agents/base.py` checks `parse_memory_command` before touching the model
at all; a match is handled deterministically and never reaches
generation.

Portuguese is a first-class citizen here: the user talks to Aila in
Portuguese, so "lembre que meu nome é Theo" / "esqueça meu nome" must
store and delete just like their English twins — a Portuguese speaker
should never be unable to save a memory in their own language.
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
# *stripped* user message with IGNORECASE; the `(.+)` body is captured as
# the fact text. A shared verb+connector prefix keeps a single capture
# group across both languages.
#
# Remember verbs: EN "remember"; PT "lembre"/"lembre-se"/"lembra" (recall),
# "anote" (note down), "guarde" (keep). Connectors that introduce the fact
# — "that" / "que" / "de que" / "de" — are optional, so both "remember I
# like tea" and "lembre que eu gosto de chá" work.
_REMEMBER_RE = re.compile(
    r"^(?:please\s+|por\s+favor,?\s+)?"
    r"(?:remember|lembre(?:-se)?|lembra|anote|guarde)"
    r"\s+(?:that\s+|de\s+que\s+|que\s+|de\s+)?(.+)$",
    re.IGNORECASE,
)
# Forget verbs: EN "forget"; PT "esqueça"/"esquece" (forget), "apague"
# (erase), "remova" (remove). Connector (that/about/que/de/o/a) optional so
# "forget my name" and "esqueça meu nome" both work.
_FORGET_RE = re.compile(
    r"^(?:please\s+|por\s+favor,?\s+)?"
    r"(?:forget|esque[çc]a|esquece|apague|remova)"
    r"\s+(?:(?:that|about|de\s+que|que|o|a|de)\s+)?(.+)$",
    re.IGNORECASE,
)
_LIST_RE = re.compile(
    r"^what\s+do\s+you\s+remember(?:\s+about\s+me)?\??$"
    r"|^what\s+have\s+i\s+told\s+you\s+to\s+remember\??$"
    r"|^o\s+que\s+voc[eê]\s+(?:se\s+)?lembra(?:\s+(?:sobre|de)\s+mim)?\??$"
    r"|^do\s+que\s+voc[eê]\s+se\s+lembra(?:\s+sobre\s+mim)?\??$"
    r"|^o\s+que\s+voc[eê]\s+sabe\s+sobre\s+mim\??$",
    re.IGNORECASE,
)

# Filler words that are never a memory on their own. "remember that" (no
# actual content) otherwise captures the literal word "that" as the fact,
# because the optional connector group needs trailing whitespace to match
# and falls through to the `(.+)` body instead.
_CONTENT_FILLER = {"that", "about", "it", "this", "the", "que", "isso", "aquilo", "de"}

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
# explicit "remember/forget that" (or Portuguese "que") form.
_EMBEDDED_REMEMBER_RE = re.compile(
    r"\b(?:remember\s+that|lembre(?:-se)?\s+(?:de\s+)?que|lembra\s+que|anote\s+que)\s+(.+)$",
    re.IGNORECASE,
)
_EMBEDDED_FORGET_RE = re.compile(
    r"\b(?:forget\s+(?:that|about)|esque[çc]a\s+que|esquece\s+que)\s+(.+)$",
    re.IGNORECASE,
)


def parse_memory_command(text: str) -> MemoryCommand | None:
    """Returns a MemoryCommand if `text` matches one of the recognized
    patterns, else None (meaning: treat it as an ordinary message)."""
    stripped = text.strip()
    if not stripped:
        return None

    # List queries are the one command that legitimately ends in "?", so
    # they are checked before the question guard below.
    if _LIST_RE.match(stripped):
        return MemoryCommand(kind="list")

    # A trailing "?" marks a question — "Remember when we met?", "Lembra do
    # filme?", "Do you remember my name?" — not an imperative to store or
    # delete. Real store/forget commands never end in "?", so treating a
    # question as one would silently save the wrong thing.
    if stripped.rstrip().endswith("?"):
        return None

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

    # Anchored patterns missed. Try the embedded form — a real typo/false
    # start ("rembember remember that ...") shouldn't be silently dropped.
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
    (
        "identity",
        (
            "my name is", "i am called", "i'm called", "call me", "you can call me",
            "meu nome é", "meu nome e", "me chamo", "pode me chamar", "podem me chamar",
        ),
    ),
    (
        "preference",
        (
            "favorite", "favourite", "i like", "i love", "i prefer", "i hate", "i dislike", "i enjoy",
            "favorito", "favorita", "eu gosto", "eu amo", "eu prefiro", "eu odeio", "eu adoro", "eu curto",
        ),
    ),
    (
        "project",
        (
            "project", "working on", "building", "repo", "repository", "codebase",
            "projeto", "trabalhando em", "trabalhando no", "construindo",
        ),
    ),
    (
        "instruction",
        (
            "always", "never", "please always", "please never", "from now on", "in the future",
            "sempre", "nunca", "de agora em diante", "a partir de agora",
        ),
    ),
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
