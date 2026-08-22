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

DEFAULT_CATEGORY = "other"

# Valid long_term_facts.category values (see memory/commands.py's
# guess_category for how a category gets picked automatically). Kept as a
# plain tuple rather than a DB-level CHECK constraint so adding a new
# category later doesn't require a migration.
MEMORY_CATEGORIES = ("identity", "preference", "personal_fact", "project", "instruction", "other")

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

# Columns added after the original schema above shipped. Added via
# ALTER TABLE (nullable, app-level defaults) rather than folded into
# SCHEMA/CREATE TABLE, so a database created by an older version of Aila
# Nano upgrades in place instead of erroring on first use.
#   session_id: an opaque caller-chosen id (e.g. a user id) memories can be
#     scoped to; NULL means "global" — visible regardless of which session
#     is asking (the right default for a single-user terminal app: a fact
#     remembered in one conversation should still be known in the next).
#   category:   one of MEMORY_CATEGORIES, defaults to 'other'.
#   updated_at: last time the fact's content/importance/category changed
#     (distinct from last_accessed_at, which tracks *reads* via touch_fact).
#   status:     'active' (default / NULL), 'superseded' (a newer value for
#     the same attribute replaced it), or 'deleted' (the user forgot it).
#     Only active facts are retrieved normally; NULL is treated as active so
#     databases written before this column upgrade in place.
#   source:     where the fact came from — 'explicit_user_request',
#     'conversation', 'system', 'tool', 'import'.
#   confidence: [0,1]; ~0.95 for explicit "remember that ...", lower for
#     facts merely mentioned in passing.
#   attribute_key: a normalized key ('favorite_game', 'name', 'project')
#     that ties successive values of the same fact together, so a new value
#     supersedes the old one (last-writer-wins) instead of coexisting.
#   superseded_by: the id of the newer fact that replaced this one.
#   version:    1, 2, 3, ... within an attribute_key's history.
_MIGRATIONS: list[tuple[str, str]] = [
    ("session_id", "ALTER TABLE long_term_facts ADD COLUMN session_id TEXT"),
    ("category", "ALTER TABLE long_term_facts ADD COLUMN category TEXT"),
    ("updated_at", "ALTER TABLE long_term_facts ADD COLUMN updated_at REAL"),
    ("status", "ALTER TABLE long_term_facts ADD COLUMN status TEXT"),
    ("source", "ALTER TABLE long_term_facts ADD COLUMN source TEXT"),
    ("confidence", "ALTER TABLE long_term_facts ADD COLUMN confidence REAL"),
    ("attribute_key", "ALTER TABLE long_term_facts ADD COLUMN attribute_key TEXT"),
    ("superseded_by", "ALTER TABLE long_term_facts ADD COLUMN superseded_by INTEGER"),
    ("version", "ALTER TABLE long_term_facts ADD COLUMN version INTEGER"),
]

# Facts with these statuses are hidden from normal retrieval.
_INACTIVE_STATUSES = ("superseded", "deleted")
# SQL fragment: a fact is "active" when its status is NULL (pre-upgrade) or
# the literal 'active'.
_ACTIVE_CLAUSE = "(status IS NULL OR status = 'active')"


class MemoryStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(long_term_facts)")}
        for column, ddl in _MIGRATIONS:
            if column not in existing:
                self._conn.execute(ddl)

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
        self,
        content: str,
        importance: float = 0.5,
        conversation_id: str | None = None,
        session_id: str | None = None,
        category: str = DEFAULT_CATEGORY,
        source: str | None = None,
        confidence: float | None = None,
        attribute_key: str | None = None,
        version: int = 1,
    ) -> int:
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO long_term_facts "
            "(conversation_id, content, importance, created_at, last_accessed_at, "
            " access_count, session_id, category, updated_at, "
            " status, source, confidence, attribute_key, version) "
            "VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, 'active', ?, ?, ?, ?)",
            (conversation_id, content, importance, now, now, session_id, category, now,
             source, confidence, attribute_key, version),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def supersede_fact(self, old_id: int, new_id: int) -> None:
        """Mark `old_id` as superseded by `new_id` (last-writer-wins). The
        old fact is preserved for history but excluded from normal
        retrieval."""
        self._conn.execute(
            "UPDATE long_term_facts SET status = 'superseded', superseded_by = ?, updated_at = ? "
            "WHERE id = ?",
            (new_id, time.time(), old_id),
        )
        self._conn.commit()

    def set_fact_status(self, fact_id: int, status: str) -> bool:
        cur = self._conn.execute(
            "UPDATE long_term_facts SET status = ?, updated_at = ? WHERE id = ?",
            (status, time.time(), fact_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get_active_facts_by_attribute(
        self, attribute_key: str, session_id: str | None = None
    ) -> list[dict]:
        """Active facts for an attribute key (usually 0 or 1 — the current
        value). Scoped to the session plus global facts."""
        if session_id is None:
            rows = self._conn.execute(
                f"SELECT * FROM long_term_facts WHERE attribute_key = ? AND {_ACTIVE_CLAUSE}",
                (attribute_key,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT * FROM long_term_facts WHERE attribute_key = ? AND {_ACTIVE_CLAUSE} "
                "AND (session_id = ? OR session_id IS NULL)",
                (attribute_key, session_id),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_attribute_history(
        self, attribute_key: str, session_id: str | None = None
    ) -> list[dict]:
        """Every value ever stored for an attribute (active + superseded +
        deleted), newest version first — for history/audit queries."""
        rows = self._conn.execute(
            "SELECT * FROM long_term_facts WHERE attribute_key = ? "
            "ORDER BY version DESC, created_at DESC",
            (attribute_key,),
        ).fetchall()
        facts = [dict(r) for r in rows]
        if session_id is not None:
            facts = [f for f in facts if f["session_id"] in (session_id, None)]
        return facts

    def max_attribute_version(self, attribute_key: str) -> int:
        row = self._conn.execute(
            "SELECT MAX(version) AS v FROM long_term_facts WHERE attribute_key = ?",
            (attribute_key,),
        ).fetchone()
        return int(row["v"]) if row and row["v"] is not None else 0

    def count_by_status(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT COALESCE(status, 'active') AS s, COUNT(*) AS n FROM long_term_facts GROUP BY s"
        ).fetchall()
        return {r["s"]: r["n"] for r in rows}

    def touch_fact(self, fact_id: int) -> None:
        self._conn.execute(
            "UPDATE long_term_facts SET last_accessed_at = ?, access_count = access_count + 1 "
            "WHERE id = ?",
            (time.time(), fact_id),
        )
        self._conn.commit()

    def update_fact(
        self,
        fact_id: int,
        content: str | None = None,
        importance: float | None = None,
        category: str | None = None,
    ) -> bool:
        """Update whichever fields are given (None = leave unchanged).
        Returns False if no fact with that id exists."""
        if content is None and importance is None and category is None:
            return fact_id in self.get_facts([fact_id])

        fields, values = [], []
        if content is not None:
            fields.append("content = ?")
            values.append(content)
        if importance is not None:
            fields.append("importance = ?")
            values.append(importance)
        if category is not None:
            fields.append("category = ?")
            values.append(category)
        fields.append("updated_at = ?")
        values.append(time.time())
        values.append(fact_id)

        cur = self._conn.execute(
            f"UPDATE long_term_facts SET {', '.join(fields)} WHERE id = ?", values
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get_all_facts(
        self, session_id: str | None = None, only_global: bool = False, include_inactive: bool = False
    ) -> list[dict]:
        """`session_id=None, only_global=False` (the default): every fact,
        regardless of session — the right default for a single-user app.
        `session_id=<id>`: that session's facts plus global (session_id
        IS NULL) ones. `only_global=True`: only session_id IS NULL facts,
        ignoring `session_id`.

        Superseded and deleted facts are excluded unless
        `include_inactive=True` — so normal retrieval only ever sees the
        current value of a fact, never an outdated or forgotten one.
        """
        active = "" if include_inactive else f" AND {_ACTIVE_CLAUSE}"
        active_where = "" if include_inactive else f" WHERE {_ACTIVE_CLAUSE}"
        if only_global:
            rows = self._conn.execute(
                f"SELECT * FROM long_term_facts WHERE session_id IS NULL{active}"
            ).fetchall()
        elif session_id is None:
            rows = self._conn.execute(
                f"SELECT * FROM long_term_facts{active_where}"
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT * FROM long_term_facts WHERE (session_id = ? OR session_id IS NULL){active}",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_fact(self, fact_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM long_term_facts WHERE id = ?", (fact_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_facts(self, fact_ids: list[int]) -> dict[int, dict]:
        if not fact_ids:
            return {}
        placeholders = ",".join("?" for _ in fact_ids)
        rows = self._conn.execute(
            f"SELECT * FROM long_term_facts WHERE id IN ({placeholders})", fact_ids
        ).fetchall()
        return {r["id"]: dict(r) for r in rows}

    def delete_fact(self, fact_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM long_term_facts WHERE id = ?", (fact_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def clear_facts(self, session_id: str | None = None) -> int:
        """Delete facts and return how many were removed. `session_id=None`
        clears every fact (all sessions + global); a concrete id clears
        only that session's own facts (global facts are left alone —
        clearing one session's memory shouldn't erase shared identity
        facts another session/user may depend on)."""
        if session_id is None:
            cur = self._conn.execute("DELETE FROM long_term_facts")
        else:
            cur = self._conn.execute(
                "DELETE FROM long_term_facts WHERE session_id = ?", (session_id,)
            )
        self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        self._conn.close()
