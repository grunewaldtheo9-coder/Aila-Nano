"""File upload: index a plain-text (or markdown) document's content into
the knowledge base (vectordb.SemanticIndex) so agents can retrieve
relevant chunks as context. This is intentionally separate from
memory.LongTermMemory — uploaded documents are shared knowledge, not
per-conversation facts.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from web.backend.app.deps import AilaState, get_state
from web.backend.app.schemas import UploadResponse

router = APIRouter(prefix="/upload", tags=["upload"])

ALLOWED_SUFFIXES = {".txt", ".md", ".jsonl", ".csv", ".log"}
MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB — generous for a nano-scale model's context needs
CHUNK_CHARS = 800
CHUNK_OVERLAP = 100


def _chunk_text(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
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


@router.post("", response_model=UploadResponse)
async def upload_file(
    file: UploadFile, state: AilaState = Depends(get_state)
) -> UploadResponse:
    from pathlib import Path

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )

    raw = await file.read()
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 5 MB).")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text.")

    chunks = _chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="File contained no text to index.")

    ids = state.knowledge.add_documents(
        chunks, metadatas=[{"source": file.filename} for _ in chunks]
    )
    state.knowledge.save()

    return UploadResponse(filename=file.filename or "unknown", chunks_indexed=len(chunks), document_ids=ids)
