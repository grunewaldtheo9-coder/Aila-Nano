"""SQLite-backed metadata store for indexed documents.

The FAISS index (vectordb/faiss_index.py) only knows about vectors and
integer ids; this store holds the actual text and JSON metadata for each
id, so a search hit can be turned back into something useful to show a
user or feed back into the model as context.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
"""


class DocumentStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(SCHEMA)
        self._conn.commit()

    def add(self, text: str, metadata: dict | None = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO documents (text, metadata, created_at) VALUES (?, ?, ?)",
            (text, json.dumps(metadata or {}), time.time()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get(self, doc_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT id, text, metadata, created_at FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        return _row_to_dict(row) if row else None

    def get_many(self, doc_ids: list[int]) -> dict[int, dict]:
        if not doc_ids:
            return {}
        placeholders = ",".join("?" for _ in doc_ids)
        rows = self._conn.execute(
            f"SELECT id, text, metadata, created_at FROM documents WHERE id IN ({placeholders})",
            doc_ids,
        ).fetchall()
        return {row["id"]: _row_to_dict(row) for row in rows}

    def delete(self, doc_id: int) -> None:
        self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self._conn.commit()

    def all_ids(self) -> list[int]:
        rows = self._conn.execute("SELECT id FROM documents").fetchall()
        return [row["id"] for row in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"]

    def close(self) -> None:
        self._conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "text": row["text"],
        "metadata": json.loads(row["metadata"]),
        "created_at": row["created_at"],
    }
