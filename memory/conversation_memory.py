"""Short-term (session) conversation memory.

Holds the running back-and-forth of a single conversation and formats it
back into the chat token format (see finetuning/format.py) so the
transformer can condition on it directly.
"""

from __future__ import annotations

from memory.store import MemoryStore


class ConversationMemory:
    def __init__(self, store: MemoryStore):
        self.store = store

    def add_turn(
        self, conversation_id: str, role: str, content: str, agent_type: str | None = None
    ) -> int:
        return self.store.add_message(conversation_id, role, content, agent_type)

    def get_history(self, conversation_id: str, max_turns: int | None = 20) -> list[dict]:
        return self.store.get_messages(conversation_id, limit=max_turns)

    def clear(self, conversation_id: str) -> None:
        self.store.clear_conversation(conversation_id)

    def list_conversations(self) -> list[str]:
        return self.store.list_conversations()

    def render_for_prompt(self, conversation_id: str, max_turns: int = 10) -> list[dict]:
        """Return the last `max_turns` (role, content) pairs, oldest first,
        excluding any system messages (the active agent supplies its own
        system prompt at call time — see agents/base.py).
        """
        history = self.get_history(conversation_id, max_turns=max_turns)
        return [{"role": h["role"], "content": h["content"]} for h in history if h["role"] != "system"]
