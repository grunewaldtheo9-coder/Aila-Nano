"""Long-term memory: durable facts that persist across conversations.

Unlike ConversationMemory (scoped to one conversation_id and typically
just the last N turns), facts stored here are meant to survive indefinitely
— user preferences, recurring context, explicitly "remembered" statements
— and are retrieved by semantic relevance (see memory/semantic_memory.py)
rather than by conversation id.
"""

from __future__ import annotations

from memory.store import MemoryStore


class LongTermMemory:
    def __init__(self, store: MemoryStore):
        self.store = store

    def remember(
        self, content: str, importance: float = 0.5, conversation_id: str | None = None
    ) -> int:
        """Store a fact. `importance` in [0, 1] influences ranking (see
        memory/ranking.py) — callers can set it higher for facts the user
        explicitly asked to be remembered.
        """
        importance = max(0.0, min(1.0, importance))
        return self.store.add_fact(content, importance=importance, conversation_id=conversation_id)

    def forget(self, fact_id: int) -> None:
        self.store.delete_fact(fact_id)

    def all_facts(self) -> list[dict]:
        return self.store.get_all_facts()

    def touch(self, fact_id: int) -> None:
        """Mark a fact as accessed, boosting its recency for future ranking."""
        self.store.touch_fact(fact_id)
