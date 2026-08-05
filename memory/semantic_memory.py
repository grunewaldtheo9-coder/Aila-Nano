"""Semantic memory: retrieve long-term facts by meaning rather than by
recency or exact keyword match.

Wraps a FaissIndex keyed by the *same* integer ids as
`LongTermMemory`/`MemoryStore.long_term_facts`, so there is exactly one
copy of each fact's text (in SQLite) and one vector per fact (in FAISS) —
no duplicated storage between the two.
"""

from __future__ import annotations

import numpy as np

from memory.long_term_memory import LongTermMemory
from memory.ranking import RankingWeights, rank_memories
from vectordb.embedder import AilaEmbedder
from vectordb.faiss_index import FaissIndex


class SemanticMemory:
    def __init__(
        self,
        embedder: AilaEmbedder,
        long_term: LongTermMemory,
        faiss_path: str | None = None,
    ):
        self.embedder = embedder
        self.long_term = long_term
        self.faiss_path = faiss_path
        self.index = (
            FaissIndex.load(faiss_path, dim=embedder.dim)
            if faiss_path and _exists(faiss_path)
            else FaissIndex(dim=embedder.dim)
        )

    def remember(
        self, content: str, importance: float = 0.5, conversation_id: str | None = None
    ) -> int:
        fact_id = self.long_term.remember(content, importance=importance, conversation_id=conversation_id)
        vec = self.embedder.embed(content)
        self.index.add(vec, np.array([fact_id]))
        return fact_id

    def forget(self, fact_id: int) -> None:
        self.long_term.forget(fact_id)
        self.index.remove(np.array([fact_id]))

    def recall(
        self, query: str, k: int = 5, weights: RankingWeights | None = None
    ) -> list[dict]:
        """Semantic search over remembered facts, re-ranked by
        relevance + recency + importance (memory/ranking.py)."""
        if self.index.ntotal == 0:
            return []
        query_vec = self.embedder.embed(query)
        scores, ids = self.index.search(query_vec, k=k)
        scores, ids = scores[0], ids[0]

        valid_ids = [int(i) for i in ids if i != -1]
        facts = self.long_term.store.get_facts(valid_ids)

        candidates = []
        for fact_id, score in zip(ids, scores):
            fact_id = int(fact_id)
            if fact_id == -1 or fact_id not in facts:
                continue
            fact = facts[fact_id]
            candidates.append(
                {
                    "id": fact_id,
                    "content": fact["content"],
                    "score": float(score),
                    "created_at": fact["created_at"],
                    "importance": fact["importance"],
                }
            )
            self.long_term.touch(fact_id)

        return rank_memories(candidates, weights=weights)

    def save(self) -> None:
        if self.faiss_path:
            self.index.save(self.faiss_path)


def _exists(path: str) -> bool:
    from pathlib import Path

    return Path(path).exists()
