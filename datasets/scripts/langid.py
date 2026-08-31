"""Lightweight, dependency-free language identification for the pretraining
pipeline — enough to route documents into English / Portuguese / other and
to measure the language mixture of a corpus.

This is deliberately a heuristic (function-word frequency + Portuguese
accented/diacritic signal), not a trained classifier: it needs no model
download, runs on CPU in microseconds per document, and is deterministic
and testable. For a nano bilingual (EN+PT) corpus that is the right
trade-off; a fastText/langdetect model can be swapped in behind the same
`identify()` interface later without changing callers.
"""

from __future__ import annotations

import re
import unicodedata

# High-frequency function words that are strongly language-specific. Kept
# short and non-overlapping (e.g. no "a"/"e" which exist in both).
_EN_WORDS = frozenset(
    "the and that with have this from they which would there their what about"
    " your not you for was are but his her she him".split()
)
_PT_WORDS = frozenset(
    "que não uma com você para por como mais mas ele ela nós eles isso está"
    " são também então porque quando muito então obrigado".split()
)
# Portuguese-specific letters/diacritics (ã õ ç plus common accents).
_PT_CHARS = set("ãõçáàâéêíóôúü")

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def language_scores(text: str) -> dict[str, float]:
    """Return normalized evidence scores for 'en' and 'pt' in [0, 1]-ish.
    Combines function-word hit rate with the Portuguese-diacritic rate."""
    toks = _tokens(text)
    if not toks:
        return {"en": 0.0, "pt": 0.0}
    n = len(toks)
    en_hits = sum(1 for t in toks if t in _EN_WORDS)
    pt_hits = sum(1 for t in toks if t in _PT_WORDS)
    # diacritic signal: fraction of characters that are PT-specific letters
    letters = [c for c in text.lower() if c.isalpha()]
    pt_char_rate = (sum(1 for c in letters if c in _PT_CHARS) / len(letters)) if letters else 0.0
    return {
        "en": en_hits / n,
        "pt": pt_hits / n + pt_char_rate,  # diacritics reinforce PT
    }


def identify(text: str, min_confidence: float = 0.02) -> str:
    """Best-guess language label: 'en', 'pt', or 'unknown'.

    'unknown' when neither language clears `min_confidence` (very short or
    non-EN/PT text), so callers can drop or bucket it rather than
    mislabelling. Deterministic."""
    s = language_scores(text)
    best = max(s, key=s.get)
    if s[best] < min_confidence:
        return "unknown"
    # Require a small margin so mixed/ambiguous snippets fall to the stronger
    # signal rather than flapping on ties.
    other = "pt" if best == "en" else "en"
    if s[best] - s[other] < 1e-9:
        return "unknown"
    return best


def is_probably_portuguese(text: str) -> bool:
    return identify(text) == "pt"


def corpus_language_distribution(texts) -> dict[str, int]:
    """Count documents by identified language. Returns a dict with 'en',
    'pt', 'unknown' keys (documents, not tokens)."""
    dist = {"en": 0, "pt": 0, "unknown": 0}
    for t in texts:
        dist[identify(t)] += 1
    return dist
