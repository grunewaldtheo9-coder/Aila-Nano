"""Unified memory API used by the agents/ and web/ layers.

Combines short-term conversation history with lexically-gated long-term
facts into a single "context package" ready to hand to the model as chat
turns plus a small set of relevant background facts — see
memory/semantic_memory.py for why relevance gating uses deterministic
word-overlap rather than raw embedding similarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memory.conversation_memory import ConversationMemory
from memory.long_term_memory import LongTermMemory
from memory.ranking import RankingWeights
from memory.semantic_memory import DEFAULT_RELEVANCE_THRESHOLD, SemanticMemory
from memory.store import DEFAULT_CATEGORY, MemoryStore
from vectordb.embedder import AilaEmbedder


@dataclass
class MemoryContext:
    history: list[dict] = field(default_factory=list)  # [{"role": ..., "content": ...}]
    relevant_facts: list[dict] = field(default_factory=list)


class MemoryManager:
    def __init__(
        self,
        embedder: AilaEmbedder,
        db_path: str = "memory/data/aila_memory.db",
        faiss_path: str = "memory/data/aila_memory.faiss",
        max_facts: int = 3,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ):
        self.store = MemoryStore(db_path)
        self.conversation = ConversationMemory(self.store)
        self.long_term = LongTermMemory(self.store)
        self.semantic = SemanticMemory(embedder, self.long_term, faiss_path=faiss_path)
        # Defaults for `build_context`, so the engine can honour
        # AILA_RETRIEVAL_TOP_K / AILA_RELEVANCE_THRESHOLD. Both were
        # documented in docs/CONFIGURATION.md while being read by nothing
        # at all — setting either did exactly nothing.
        self.max_facts = max_facts
        self.relevance_threshold = relevance_threshold

    # -- conversation (short-term) -----------------------------------------

    def add_turn(
        self, conversation_id: str, role: str, content: str, agent_type: str | None = None
    ) -> None:
        self.conversation.add_turn(conversation_id, role, content, agent_type=agent_type)

    # -- long-term memory CRUD ----------------------------------------------
    # Thin passthroughs to memory.semantic_memory.SemanticMemory, which
    # keeps SQLite (text) and FAISS (vectors) in sync. This is the surface
    # agents/base.py and chat.py (and any future interface) use.

    def add_memory(
        self,
        content: str,
        category: str = DEFAULT_CATEGORY,
        importance: float = 0.5,
        session_id: str | None = None,
    ) -> int:
        return self.semantic.add_memory(content, category=category, importance=importance, session_id=session_id)

    def get_memory(self, fact_id: int) -> dict | None:
        return self.semantic.get_memory(fact_id)

    def update_memory(
        self,
        fact_id: int,
        content: str | None = None,
        category: str | None = None,
        importance: float | None = None,
    ) -> bool:
        return self.semantic.update_memory(fact_id, content=content, category=category, importance=importance)

    def delete_memory(self, fact_id: int) -> bool:
        return self.semantic.delete_memory(fact_id)

    def clear_memories(self, session_id: str | None = None) -> int:
        return self.semantic.clear_memories(session_id=session_id)

    def all_memories(self, session_id: str | None = None) -> list[dict]:
        return self.semantic.all_memories(session_id=session_id)

    def search_memories(
        self, query: str, k: int = 5, session_id: str | None = None, weights: RankingWeights | None = None
    ) -> list[dict]:
        return self.semantic.search_memories(query, k=k, session_id=session_id, weights=weights)

    def get_relevant_memories(
        self,
        query: str,
        k: int = 3,
        session_id: str | None = None,
        threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        weights: RankingWeights | None = None,
    ) -> list[dict]:
        return self.semantic.get_relevant_memories(
            query, k=k, session_id=session_id, threshold=threshold, weights=weights
        )

    def remember_fact(self, content: str, importance: float = 0.6) -> int:
        """Back-compat alias for the pre-memory-architecture API (still
        used by chat.py's `/remember <text>` slash command)."""
        return self.add_memory(content, importance=importance)

    # -- context building ----------------------------------------------------

    def build_context(
        self,
        conversation_id: str,
        query: str,
        max_turns: int = 10,
        max_facts: int | None = None,
        session_id: str | None = None,
        threshold: float | None = None,
        weights: RankingWeights | None = None,
    ) -> MemoryContext:
        """`max_facts`/`threshold` default to the values this manager was
        constructed with (which the engine takes from configuration), so
        an explicit argument still wins for callers that need one."""
        history = self.conversation.render_for_prompt(conversation_id, max_turns=max_turns)
        facts = self.get_relevant_memories(
            query,
            k=self.max_facts if max_facts is None else max_facts,
            session_id=session_id,
            threshold=self.relevance_threshold if threshold is None else threshold,
            weights=weights,
        )
        return MemoryContext(history=history, relevant_facts=facts)

    def save(self) -> None:
        self.semantic.save()

    def close(self) -> None:
        self.save()
        self.store.close()
