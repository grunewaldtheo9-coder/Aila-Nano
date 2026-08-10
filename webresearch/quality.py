"""Source quality ranking and untrusted-content sanitization for web
research results.

Everything a search API returns — titles, snippets, answer boxes — is
untrusted external data. Two defenses live here:

1. `sanitize_snippet`: strips control characters, collapses whitespace,
   truncates, and drops text that pattern-matches prompt-injection
   attempts ("ignore previous instructions", "system prompt:", role-tag
   lookalikes). Injection text in a *snippet* can't execute anything by
   itself — the real defense is that snippets are only ever placed in a
   clearly-delimited data block and never parsed as commands — but
   refusing to store/serve recognizable injection strings removes the
   cheapest attack and keeps the knowledge base clean.

2. `rank_sources`: prefers reputable/primary domains over SEO spam.
   Deliberately a small, transparent tier list rather than a scraped
   "authority" metric — auditable, testable, easy to extend.
"""

from __future__ import annotations

import re

from webresearch.serper import SearchResult

# Domains with an editorial/institutional review process, or that are
# primary sources for their own subject matter. Tier 1 outranks tier 2
# outranks unknown. Substring match against the registrable domain.
TIER_1_DOMAINS = (
    "wikipedia.org",
    "britannica.com",
    ".gov",
    ".edu",
    "nature.com",
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
)
TIER_2_DOMAINS = (
    "nytimes.com",
    "theguardian.com",
    "washingtonpost.com",
    "economist.com",
    "forbes.com",
    "bloomberg.com",
    "cnn.com",
    "nasa.gov",
    "who.int",
    "un.org",
    "stackoverflow.com",
    "github.com",
    "docs.python.org",
    "developer.mozilla.org",
)

# Patterns that mark a snippet as a likely prompt-injection attempt.
# Checked case-insensitively against the whole snippet.
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+|any\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(all\s+|any\s+)?(previous|prior|above)", re.I),
    re.compile(r"system\s*prompt", re.I),
    re.compile(r"you\s+are\s+now\s+(?:a|an|the)\b", re.I),
    re.compile(r"<\|?(system|user|assistant|end)\|?>", re.I),
    re.compile(r"\[\s*/?(SYSTEM|INST)\s*\]", re.I),
    re.compile(r"api[_\s-]?key", re.I),
    re.compile(r"reveal\s+your\s+(instructions|prompt|rules)", re.I),
]

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Upper bound on one piece of retrieved text. Raised from 400 when
# Wikipedia became a source: encyclopedia summaries are typically
# 300–800 characters of finished prose, and capping at 400 chopped the
# best answers mid-sentence. Web text is served directly rather than
# injected into the model's prompt, so the old prompt-budget reason for a
# tight cap no longer applies; this is now just a sanity bound on
# untrusted input.
MAX_SNIPPET_CHARS = 600

# Search engines return snippets already cut off mid-thought, marked with
# a trailing ellipsis ("... insurance, securities, ..."). Serving that
# verbatim is what made Aila's answers read as unfinished — reported from
# real use ("she doesn't say everything, making it incomplete").
_TRAILING_ELLIPSIS = re.compile(r"[\s,;:]*(?:\.\s*\.\s*\.|…)\s*$")

# A sentence ends at . ! or ? optionally followed by a closing quote or
# bracket, and then whitespace or end-of-string. The negative lookbehind
# skips the most common abbreviations so "Inc. was founded" isn't treated
# as a sentence boundary.
_SENTENCE_END = re.compile(
    r"(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bSt)(?<!\bInc)(?<!\bLtd)(?<!\bvs)"
    r"[.!?][\"'”’)\]]?(?=\s|$)"
)

# Dangling connectives left at the end after an ellipsis is removed. Cut
# these too — "…and" or "…including" reads more unfinished than the
# clause that precedes them.
_DANGLING_TAIL = re.compile(
    r"\s+(?:and|or|but|including|such\s+as|like|with|for|from|to|of|in|on|at|by|as|"
    r"e|ou|mas|incluindo|como|com|para|de|da|do|em|no|na|por)\s*$",
    re.IGNORECASE,
)

# Below this, a "complete sentence" prefix is too short to be a useful
# answer — better to repair the full text than to serve three words.
MIN_COMPLETE_ANSWER_CHARS = 40


def looks_truncated(text: str) -> bool:
    """True when `text` visibly stops mid-thought — it ends in an
    ellipsis, a comma, or a dangling connective ("... including").

    Deliberately NOT "has no final period": a perfectly good answer-box
    result ("428 million subscribers") has no terminal punctuation and is
    complete. Only evidence of an actual cut-off counts, because this is
    what decides whether a *different* candidate answer is preferred.
    """
    if not text:
        return False
    stripped = text.strip()
    if _TRAILING_ELLIPSIS.search(stripped):
        return True
    if stripped.endswith((",", ";", ":")):
        return True
    return bool(_DANGLING_TAIL.search(stripped))


def complete_sentence(text: str) -> str:
    """Make a piece of web text read as finished.

    For visibly cut-off text, two strategies in order:
    1. If it contains a complete sentence long enough to stand on its own,
       keep everything up to the last sentence end and drop the cut-off
       remainder.
    2. Otherwise (very common — a snippet is often one long unfinished
       sentence) strip the trailing ellipsis and any dangling connective,
       then close it with a period.

    Text that isn't cut off is returned as-is apart from a final period
    when it has no closing punctuation at all.

    Never invents words: only ever removes text and adds a final '.'.
    """
    if not text:
        return text
    stripped = text.strip()
    if not looks_truncated(stripped):
        return _ensure_final_period(stripped)

    without_ellipsis = _TRAILING_ELLIPSIS.sub("", stripped).strip()
    if not without_ellipsis:
        return ""

    ends = list(_SENTENCE_END.finditer(without_ellipsis))
    if ends:
        candidate = without_ellipsis[: ends[-1].end()].strip()
        if len(candidate) >= MIN_COMPLETE_ANSWER_CHARS:
            return candidate

    repaired = without_ellipsis
    while True:
        trimmed = _DANGLING_TAIL.sub("", repaired).rstrip(" ,;:")
        if trimmed == repaired:
            break
        repaired = trimmed
    repaired = repaired.rstrip(" ,;:").strip()
    if not repaired:
        return ""
    return _ensure_final_period(repaired)


def _ensure_final_period(text: str) -> str:
    if text and not re.search(r"[.!?][\"'”’)\]]?$", text):
        return text + "."
    return text


def _truncate_on_word_boundary(text: str, limit: int) -> str:
    """Hard cap at `limit` characters without slicing a word in half.

    Marks the result with an ellipsis when it actually cut something.
    Without the marker, `looks_truncated` cannot tell that our own cap
    stopped the text mid-sentence, `complete_sentence` leaves it alone,
    and the user reads "...founded in 1938 and The." — the same
    unfinished-answer complaint, produced by us rather than by the search
    engine.
    """
    if len(text) <= limit:
        return text
    marker = "..."
    cut = text[: max(1, limit - len(marker))]
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:") + marker


def sanitize_snippet(text: str) -> str | None:
    """Clean one piece of untrusted web text for storage/serving.
    Returns None if the text should be discarded entirely (injection
    attempt or nothing left after cleaning)."""
    if not text:
        return None
    cleaned = _CONTROL_CHARS.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            return None
    return _truncate_on_word_boundary(cleaned, MAX_SNIPPET_CHARS)


def domain_tier(domain: str) -> int:
    """1 = most trusted, 2 = reputable, 3 = unknown."""
    d = (domain or "").lower()
    if any(t in d for t in TIER_1_DOMAINS):
        return 1
    if any(t in d for t in TIER_2_DOMAINS):
        return 2
    return 3


def rank_sources(results: list[SearchResult]) -> list[SearchResult]:
    """Order results by (domain tier, search position). Search position
    still matters within a tier — the engine's own ranking carries
    signal — but a tier-1 domain at position 4 beats an unknown blog at
    position 1."""
    return sorted(results, key=lambda r: (domain_tier(r.domain), r.position))
