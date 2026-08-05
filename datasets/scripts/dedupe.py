"""Deduplication for pretraining corpora.

Two passes:
  1. Exact dedup — hash each normalized document, drop repeats.
  2. Near-duplicate dedup — MinHash/Jaccard over shingles would be the
     "correct" large-scale approach, but at nano-corpus scale a cheaper
     n-gram-overlap check catches the common case (near-identical
     boilerplate/templated text) without extra dependencies.
"""

from __future__ import annotations

import hashlib
import re


def _normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def exact_dedupe(documents: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for doc in documents:
        h = hashlib.sha256(_normalize_for_hash(doc).encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(doc)
    return out


def _shingles(text: str, n: int = 5) -> set[str]:
    words = _normalize_for_hash(text).split()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def near_dedupe(documents: list[str], threshold: float = 0.8, shingle_size: int = 5) -> list[str]:
    """Drop documents whose word-shingle Jaccard similarity to any
    previously kept document exceeds `threshold`. O(n^2) — fine for
    corpora up to a few tens of thousands of documents; swap in MinHash
    LSH if the corpus grows much larger.
    """
    kept: list[str] = []
    kept_shingles: list[set[str]] = []
    for doc in documents:
        s = _shingles(doc, n=shingle_size)
        is_dup = False
        for other in kept_shingles:
            union = s | other
            if not union:
                continue
            jaccard = len(s & other) / len(union)
            if jaccard >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(doc)
            kept_shingles.append(s)
    return kept


def dedupe_corpus(documents: list[str], near_threshold: float | None = 0.8) -> list[str]:
    docs = exact_dedupe(documents)
    if near_threshold is not None:
        docs = near_dedupe(docs, threshold=near_threshold)
    return docs
