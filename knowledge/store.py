"""SQLite persistence for Aila's global knowledge, knowledge candidates,
and the web-research cache.

Distinct from memory/store.py on purpose — the memory DB holds *per-user*
state (conversations, user memories, scoped by session), while this DB
holds *global*, user-independent knowledge:

- `knowledge`            — validated facts any user/session may reuse.
- `knowledge_candidates` — extracted-but-not-yet-validated information
                           (never served to users until promoted).
- `web_cache`            — raw web search results keyed by normalized
                           query, so repeated/similar questions don't
                           re-hit the search API inside the TTL window.

User memories must never flow into these tables automatically — see
docs/ARCHITECTURE.md's privacy section; nothing in this module accepts a
session/user id at all, which makes the leak structurally impossible at
this layer.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'en',
    category TEXT NOT NULL DEFAULT 'general',
    confidence REAL NOT NULL DEFAULT 0.5,
    source_urls TEXT NOT NULL DEFAULT '[]',   -- JSON list
    source_titles TEXT NOT NULL DEFAULT '[]', -- JSON list
    verification TEXT NOT NULL DEFAULT 'unverified', -- unverified|corroborated|conflicted
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_verified_at REAL,
    use_count INTEGER NOT NULL DEFAULT 0,
    version INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS knowledge_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    source_urls TEXT NOT NULL DEFAULT '[]',
    source_titles TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.0,
    reason TEXT,                    -- why it's still a candidate (e.g. 'single_source')
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS web_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_normalized TEXT NOT NULL UNIQUE,
    results_json TEXT NOT NULL,
    fetched_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_web_cache_query ON web_cache(query_normalized);
"""


class KnowledgeStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # -- knowledge ---------------------------------------------------------

    def add_knowledge(
        self,
        question: str,
        answer: str,
        language: str = "en",
        category: str = "general",
        confidence: float = 0.5,
        source_urls: list[str] | None = None,
        source_titles: list[str] | None = None,
        verification: str = "unverified",
    ) -> int:
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO knowledge (question, answer, language, category, confidence, "
            "source_urls, source_titles, verification, created_at, updated_at, last_verified_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                question,
                answer,
                language,
                category,
                confidence,
                json.dumps(source_urls or []),
                json.dumps(source_titles or []),
                verification,
                now,
                now,
                now if verification == "corroborated" else None,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get_knowledge(self, knowledge_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM knowledge WHERE id = ?", (knowledge_id,)
        ).fetchone()
        return self._decode(row) if row else None

    def all_knowledge(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM knowledge").fetchall()
        return [self._decode(r) for r in rows]

    def update_knowledge(self, knowledge_id: int, **fields) -> bool:
        allowed = {
            "question", "answer", "language", "category", "confidence",
            "verification", "last_verified_at",
        }
        updates, values = [], []
        for key, value in fields.items():
            if key not in allowed:
                raise ValueError(f"Cannot update field {key!r}")
            updates.append(f"{key} = ?")
            values.append(value)
        if not updates:
            return self.get_knowledge(knowledge_id) is not None
        updates.append("updated_at = ?")
        values.append(time.time())
        updates.append("version = version + 1")
        values.append(knowledge_id)
        cur = self._conn.execute(
            f"UPDATE knowledge SET {', '.join(updates)} WHERE id = ?", values
        )
        self._conn.commit()
        return cur.rowcount > 0

    def touch_knowledge(self, knowledge_id: int, verified: bool = False) -> None:
        if verified:
            self._conn.execute(
                "UPDATE knowledge SET use_count = use_count + 1, last_verified_at = ? WHERE id = ?",
                (time.time(), knowledge_id),
            )
        else:
            self._conn.execute(
                "UPDATE knowledge SET use_count = use_count + 1 WHERE id = ?",
                (knowledge_id,),
            )
        self._conn.commit()

    def delete_knowledge(self, knowledge_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM knowledge WHERE id = ?", (knowledge_id,))
        self._conn.commit()
        return cur.rowcount > 0

    # -- candidates --------------------------------------------------------

    def add_candidate(
        self,
        question: str,
        answer: str,
        confidence: float = 0.0,
        reason: str | None = None,
        source_urls: list[str] | None = None,
        source_titles: list[str] | None = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO knowledge_candidates "
            "(question, answer, source_urls, source_titles, confidence, reason, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                question,
                answer,
                json.dumps(source_urls or []),
                json.dumps(source_titles or []),
                confidence,
                reason,
                time.time(),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def all_candidates(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM knowledge_candidates").fetchall()
        return [self._decode(r) for r in rows]

    def delete_candidate(self, candidate_id: int) -> bool:
        cur = self._conn.execute(
            "DELETE FROM knowledge_candidates WHERE id = ?", (candidate_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    # -- web cache ---------------------------------------------------------

    def cache_web_results(self, query_normalized: str, results: list[dict]) -> None:
        self._conn.execute(
            "INSERT INTO web_cache (query_normalized, results_json, fetched_at) VALUES (?, ?, ?) "
            "ON CONFLICT(query_normalized) DO UPDATE SET results_json = excluded.results_json, "
            "fetched_at = excluded.fetched_at",
            (query_normalized, json.dumps(results), time.time()),
        )
        self._conn.commit()

    def get_cached_web_results(
        self, query_normalized: str, max_age_seconds: float
    ) -> list[dict] | None:
        row = self._conn.execute(
            "SELECT * FROM web_cache WHERE query_normalized = ?", (query_normalized,)
        ).fetchone()
        if row is None:
            return None
        if time.time() - row["fetched_at"] > max_age_seconds:
            return None
        return json.loads(row["results_json"])

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict:
        d = dict(row)
        for key in ("source_urls", "source_titles"):
            if key in d and isinstance(d[key], str):
                d[key] = json.loads(d[key])
        return d
