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

MAX_SNIPPET_CHARS = 400


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
    return cleaned[:MAX_SNIPPET_CHARS]


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
