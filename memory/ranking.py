"""Memory ranking: combine semantic relevance, recency, importance, and
access frequency into a single score used to decide which memories to
surface into a prompt's context window.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class RankingWeights:
    relevance: float = 0.6
    recency: float = 0.25
    importance: float = 0.15
    recency_half_life_days: float = 14.0


def recency_score(created_at: float, now: float | None = None, half_life_days: float = 14.0) -> float:
    """Exponential decay: 1.0 for "just now", 0.5 after one half-life, etc."""
    now = now or time.time()
    age_days = max(0.0, (now - created_at) / 86400.0)
    return 0.5 ** (age_days / max(half_life_days, 1e-6))


def rank_memories(
    candidates: list[dict],
    weights: RankingWeights | None = None,
    now: float | None = None,
) -> list[dict]:
    """`candidates` items must have 'score' (semantic similarity, roughly
    [-1, 1] for cosine), 'created_at' (unix timestamp), and optionally
    'importance' ([0, 1], default 0.5). Returns the list sorted descending
    by combined score, with a 'combined_score' field added to each item.
    """
    weights = weights or RankingWeights()
    now = now or time.time()

    ranked = []
    for item in candidates:
        relevance = max(0.0, min(1.0, (item.get("score", 0.0) + 1) / 2))  # cosine -> [0,1]
        recency = recency_score(
            item.get("created_at", now), now=now, half_life_days=weights.recency_half_life_days
        )
        importance = item.get("importance", 0.5)

        combined = (
            weights.relevance * relevance
            + weights.recency * recency
            + weights.importance * importance
        )
        ranked.append({**item, "combined_score": combined})

    ranked.sort(key=lambda x: x["combined_score"], reverse=True)
    return ranked
