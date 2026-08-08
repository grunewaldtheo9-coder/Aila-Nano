"""Long-term memory: durable facts that persist across conversations.

Unlike ConversationMemory (scoped to one conversation_id and typically
just the last N turns), facts stored here are meant to survive indefinitely
— user preferences, recurring context, explicitly "remembered" statements
— and are retrieved by relevance (see memory/semantic_memory.py) rather
than by conversation id.

This is the pure-SQL CRUD layer (no vectors) — `memory/semantic_memory.py`
wraps it to keep a FAISS index in sync for retrieval, and
`memory/manager.py` is the public API most callers (agents/, chat.py)
actually use.
"""

from __future__ import annotations

from memory.store import DEFAULT_CATEGORY, MemoryStore


class LongTermMemory:
    def __init__(self, store: MemoryStore):
        self.store = store

    # -- CRUD -------------------------------------------------------------

    def add_memory(
        self,
        content: str,
        category: str = DEFAULT_CATEGORY,
        importance: float = 0.5,
        session_id: str | None = None,
        conversation_id: str | None = None,
    ) -> int:
        """Store a fact. `importance` in [0, 1] influences ranking (see
        memory/ranking.py) — callers can set it higher for facts the user
        explicitly asked to be remembered."""
        importance = max(0.0, min(1.0, importance))
        return self.store.add_fact(
            content,
            importance=importance,
            conversation_id=conversation_id,
            session_id=session_id,
            category=category,
        )

    def get_memory(self, fact_id: int) -> dict | None:
        return self.store.get_fact(fact_id)

    def update_memory(
        self,
        fact_id: int,
        content: str | None = None,
        category: str | None = None,
        importance: float | None = None,
    ) -> bool:
        if importance is not None:
            importance = max(0.0, min(1.0, importance))
        return self.store.update_fact(fact_id, content=content, importance=importance, category=category)

    def delete_memory(self, fact_id: int) -> bool:
        return self.store.delete_fact(fact_id)

    def clear_memories(self, session_id: str | None = None) -> int:
        return self.store.clear_facts(session_id=session_id)

    def all_memories(self, session_id: str | None = None) -> list[dict]:
        return self.store.get_all_facts(session_id=session_id)

    def touch(self, fact_id: int) -> None:
        """Mark a fact as accessed, boosting its recency for future ranking."""
        self.store.touch_fact(fact_id)

    # -- backward-compatible aliases ---------------------------------------
    # Kept because memory/semantic_memory.py and existing callers/tests use
    # these names; new code should prefer add_memory/delete_memory/all_memories.

    def remember(
        self, content: str, importance: float = 0.5, conversation_id: str | None = None
    ) -> int:
        return self.add_memory(content, importance=importance, conversation_id=conversation_id)

    def forget(self, fact_id: int) -> None:
        self.delete_memory(fact_id)

    def all_facts(self) -> list[dict]:
        return self.all_memories()
