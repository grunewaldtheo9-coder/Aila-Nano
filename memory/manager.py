"""Unified memory API used by the agents/ and web/ layers.

Combines short-term conversation history with semantically-retrieved
long-term facts into a single "context package" ready to hand to the
model as chat turns plus a small set of relevant background facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memory.conversation_memory import ConversationMemory
from memory.long_term_memory import LongTermMemory
from memory.ranking import RankingWeights
from memory.semantic_memory import SemanticMemory
from memory.store import MemoryStore
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
    ):
        self.store = MemoryStore(db_path)
        self.conversation = ConversationMemory(self.store)
        self.long_term = LongTermMemory(self.store)
        self.semantic = SemanticMemory(embedder, self.long_term, faiss_path=faiss_path)

    def add_turn(
        self, conversation_id: str, role: str, content: str, agent_type: str | None = None
    ) -> None:
        self.conversation.add_turn(conversation_id, role, content, agent_type=agent_type)

    def remember_fact(self, content: str, importance: float = 0.6) -> int:
        return self.semantic.remember(content, importance=importance)

    def build_context(
        self,
        conversation_id: str,
        query: str,
        max_turns: int = 10,
        max_facts: int = 3,
        weights: RankingWeights | None = None,
    ) -> MemoryContext:
        history = self.conversation.render_for_prompt(conversation_id, max_turns=max_turns)
        facts = self.semantic.recall(query, k=max_facts, weights=weights)
        return MemoryContext(history=history, relevant_facts=facts)

    def save(self) -> None:
        self.semantic.save()

    def close(self) -> None:
        self.save()
        self.store.close()
