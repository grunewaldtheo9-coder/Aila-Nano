"""Semantic + lexical memory retrieval.

Wraps a FaissIndex keyed by the *same* integer ids as
`LongTermMemory`/`MemoryStore.long_term_facts`, so there is exactly one
copy of each fact's text (in SQLite) and one vector per fact (in FAISS) —
no duplicated storage between the two.

Two retrieval methods, deliberately different in how strict they are:

- `search_memories`: broad, semantic (embedding) search — good for an
  explicit "search my memories for X" utility, ranked by relevance +
  recency + importance, but not gated by any hard relevance cutoff.
- `get_relevant_memories`: the one that feeds prompt injection
  (agents/base.py's [MEMORY] block). Gated by deterministic lexical
  word-overlap (memory/lexical.py) rather than the embedding's cosine
  score — see that module's docstring for why. A fact that doesn't clear
  the threshold is never returned, so "no relevant memory exists" (empty
  result) is a real, reachable state the caller can act on instead of
  always getting *something* back.
"""

from __future__ import annotations

import time

import numpy as np

from memory.lexical import lexical_overlap_score
from memory.long_term_memory import LongTermMemory
from memory.ranking import RankingWeights, rank_memories, recency_score
from memory.store import DEFAULT_CATEGORY
from vectordb.embedder import AilaEmbedder
from vectordb.faiss_index import FaissIndex

# Overlap-coefficient cutoff (see memory/lexical.py) below which a fact is
# considered irrelevant to the current query and never injected. Tuned so
# a single shared significant word between a short question and a fact
# ("name" in both "What's my name?" and "The user's name is Theo.")
# clears it, while genuinely unrelated text (no shared significant words)
# scores exactly 0 and is always excluded.
DEFAULT_RELEVANCE_THRESHOLD = 0.2


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

    # -- CRUD (keeps the FAISS index in sync with SQLite) ------------------

    def add_memory(
        self,
        content: str,
        category: str = DEFAULT_CATEGORY,
        importance: float = 0.5,
        session_id: str | None = None,
        conversation_id: str | None = None,
    ) -> int:
        fact_id = self.long_term.add_memory(
            content,
            category=category,
            importance=importance,
            session_id=session_id,
            conversation_id=conversation_id,
        )
        vec = self.embedder.embed(content)
        self.index.add(vec, np.array([fact_id]))
        return fact_id

    def update_memory(
        self,
        fact_id: int,
        content: str | None = None,
        category: str | None = None,
        importance: float | None = None,
    ) -> bool:
        ok = self.long_term.update_memory(fact_id, content=content, category=category, importance=importance)
        if ok and content is not None:
            # The stored vector was embedded from the *old* content —
            # re-embed so future semantic search reflects the update.
            self.index.remove(np.array([fact_id]))
            vec = self.embedder.embed(content)
            self.index.add(vec, np.array([fact_id]))
        return ok

    def delete_memory(self, fact_id: int) -> bool:
        ok = self.long_term.delete_memory(fact_id)
        self.index.remove(np.array([fact_id]))
        return ok

    def clear_memories(self, session_id: str | None = None) -> int:
        facts = self.long_term.all_memories(session_id=session_id)
        n = self.long_term.clear_memories(session_id=session_id)
        if facts:
            self.index.remove(np.array([f["id"] for f in facts]))
        return n

    def get_memory(self, fact_id: int) -> dict | None:
        return self.long_term.get_memory(fact_id)

    def all_memories(self, session_id: str | None = None) -> list[dict]:
        return self.long_term.all_memories(session_id=session_id)

    # -- retrieval ----------------------------------------------------------

    def search_memories(
        self, query: str, k: int = 5, session_id: str | None = None, weights: RankingWeights | None = None
    ) -> list[dict]:
        """Broad semantic search, ranked by relevance + recency +
        importance. Not relevance-gated — use `get_relevant_memories` for
        anything that gets auto-injected into a prompt."""
        if self.index.ntotal == 0:
            return []
        query_vec = self.embedder.embed(query)
        scores, ids = self.index.search(query_vec, k=k)
        scores, ids = scores[0], ids[0]

        valid_ids = [int(i) for i in ids if i != -1]
        facts = self.long_term.store.get_facts(valid_ids)
        if session_id is not None:
            facts = {
                fid: f for fid, f in facts.items()
                if f.get("session_id") in (None, session_id)
            }

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
                    "category": fact.get("category"),
                }
            )
            self.long_term.touch(fact_id)

        return rank_memories(candidates, weights=weights)

    def get_relevant_memories(
        self,
        query: str,
        k: int = 3,
        session_id: str | None = None,
        threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        weights: RankingWeights | None = None,
    ) -> list[dict]:
        """Deterministic, lexically-gated retrieval — see module
        docstring. Returns [] (not "the closest thing anyway") when
        nothing clears `threshold`, which is exactly the "no relevant
        memory exists" signal callers need to avoid ever injecting an
        unrelated fact."""
        weights = weights or RankingWeights()
        now = time.time()

        candidates = []
        for fact in self.long_term.all_memories(session_id=session_id):
            relevance = lexical_overlap_score(query, fact["content"])
            if relevance < threshold:
                continue
            recency = recency_score(
                fact["created_at"], now=now, half_life_days=weights.recency_half_life_days
            )
            importance = fact.get("importance", 0.5)
            combined = (
                weights.relevance * relevance
                + weights.recency * recency
                + weights.importance * importance
            )
            candidates.append(
                {
                    "id": fact["id"],
                    "content": fact["content"],
                    "score": relevance,
                    "combined_score": combined,
                    "created_at": fact["created_at"],
                    "importance": importance,
                    "category": fact.get("category"),
                }
            )

        candidates.sort(key=lambda c: c["combined_score"], reverse=True)
        result = candidates[:k]
        for c in result:
            self.long_term.touch(c["id"])
        return result

    # -- backward-compatible aliases ---------------------------------------

    def remember(
        self, content: str, importance: float = 0.5, conversation_id: str | None = None
    ) -> int:
        return self.add_memory(content, importance=importance, conversation_id=conversation_id)

    def forget(self, fact_id: int) -> None:
        self.delete_memory(fact_id)

    def recall(self, query: str, k: int = 5, weights: RankingWeights | None = None) -> list[dict]:
        return self.search_memories(query, k=k, weights=weights)

    def save(self) -> None:
        if self.faiss_path:
            self.index.save(self.faiss_path)


def _exists(path: str) -> bool:
    from pathlib import Path

    return Path(path).exists()
