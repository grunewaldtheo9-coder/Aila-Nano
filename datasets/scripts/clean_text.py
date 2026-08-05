"""Text cleaning/normalization used before tokenizer training and
pretraining data preparation.

Kept deliberately simple and inspectable — this is a nano model, and
overly aggressive filtering can throw away more signal than noise at this
scale.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_document(text: str) -> str:
    """Normalize a single document: Unicode NFKC normalize, strip control
    characters, collapse repeated whitespace/blank lines, trim edges.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def is_low_quality(
    text: str,
    min_chars: int = 20,
    max_chars: int = 200_000,
    min_alpha_ratio: float = 0.5,
) -> bool:
    """Heuristic filter for junk documents: too short, absurdly long, or
    mostly non-alphabetic (e.g. markup soup, tables of numbers).
    """
    n = len(text)
    if n < min_chars or n > max_chars:
        return True
    alpha = sum(1 for c in text if c.isalpha())
    if n > 0 and (alpha / n) < min_alpha_ratio:
        return True
    return False


def clean_corpus(documents: list[str], min_chars: int = 20, max_chars: int = 200_000) -> list[str]:
    cleaned = []
    for doc in documents:
        c = clean_document(doc)
        if c and not is_low_quality(c, min_chars=min_chars, max_chars=max_chars):
            cleaned.append(c)
    return cleaned
