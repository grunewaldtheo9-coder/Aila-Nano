"""Plain-text chunking for indexing documents into a SemanticIndex.

Kept interface-agnostic on purpose: this used to live behind the (now
removed) HTTP `/upload` endpoint, but chunking a document into
retrievable pieces has nothing to do with HTTP — it's used the same way
by `chat.py`'s terminal `/learn` command today, and will be reused
identically by a future PDF reader / file-reading tool (see
`docs/ARCHITECTURE.md`'s roadmap section) without change.
"""

from __future__ import annotations

ALLOWED_SUFFIXES = {".txt", ".md", ".jsonl", ".csv", ".log"}
MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB — generous for a nano-scale model's context needs
CHUNK_CHARS = 800
CHUNK_OVERLAP = 100


def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split `text` into overlapping character-window chunks, small enough
    for a nano-scale model's context window to reason about individually.
    """
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]
