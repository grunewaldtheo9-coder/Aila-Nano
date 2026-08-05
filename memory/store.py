"""SQLite persistence for conversation turns and long-term memory facts.

Two tables, one file: `messages` (short-term conversation turns, scoped by
conversation_id) and `long_term_facts` (durable facts/summaries that
outlive any single conversation, optionally still tagged with the
conversation they were learned in).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    agent_type TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS long_term_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT,
    content TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    created_at REAL NOT NULL,
    last_accessed_at REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0
);
"""


class MemoryStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # -- conversation (short-term) messages --------------------------------

    def add_message(
        self, conversation_id: str, role: str, content: str, agent_type: str | None = None
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO messages (conversation_id, role, content, agent_type, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, content, agent_type, time.time()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get_messages(self, conversation_id: str, limit: int | None = None) -> list[dict]:
        query = (
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC"
        )
        rows = self._conn.execute(query, (conversation_id,)).fetchall()
        rows = [dict(r) for r in rows]
        return rows[-limit:] if limit else rows

    def list_conversations(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT conversation_id FROM messages ORDER BY conversation_id"
        ).fetchall()
        return [r["conversation_id"] for r in rows]

    def clear_conversation(self, conversation_id: str) -> None:
        self._conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        self._conn.commit()

    # -- long-term facts ------------------------------------------------

    def add_fact(
        self, content: str, importance: float = 0.5, conversation_id: str | None = None
    ) -> int:
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO long_term_facts "
            "(conversation_id, content, importance, created_at, last_accessed_at, access_count) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            (conversation_id, content, importance, now, now),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def touch_fact(self, fact_id: int) -> None:
        self._conn.execute(
            "UPDATE long_term_facts SET last_accessed_at = ?, access_count = access_count + 1 "
            "WHERE id = ?",
            (time.time(), fact_id),
        )
        self._conn.commit()

    def get_all_facts(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM long_term_facts").fetchall()
        return [dict(r) for r in rows]

    def get_facts(self, fact_ids: list[int]) -> dict[int, dict]:
        if not fact_ids:
            return {}
        placeholders = ",".join("?" for _ in fact_ids)
        rows = self._conn.execute(
            f"SELECT * FROM long_term_facts WHERE id IN ({placeholders})", fact_ids
        ).fetchall()
        return {r["id"]: dict(r) for r in rows}

    def delete_fact(self, fact_id: int) -> None:
        self._conn.execute("DELETE FROM long_term_facts WHERE id = ?", (fact_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
