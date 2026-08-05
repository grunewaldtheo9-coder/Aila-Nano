"""High-level semantic search API: document indexing + retrieval, built
on AilaEmbedder (embeddings) + FaissIndex (vector search) + DocumentStore
(text/metadata persistence).

This is the module other parts of the project (memory/, agents/) import.
"""

from __future__ import annotations

import numpy as np

from vectordb.document_store import DocumentStore
from vectordb.embedder import AilaEmbedder
from vectordb.faiss_index import FaissIndex


class SemanticIndex:
    def __init__(self, embedder: AilaEmbedder, db_path: str, faiss_path: str | None = None):
        self.embedder = embedder
        self.store = DocumentStore(db_path)
        self.faiss_path = faiss_path

        self.index = (
            FaissIndex.load(faiss_path, dim=embedder.dim)
            if faiss_path and _path_exists(faiss_path)
            else FaissIndex(dim=embedder.dim)
        )

    def add_document(self, text: str, metadata: dict | None = None) -> int:
        doc_id = self.store.add(text, metadata)
        vec = self.embedder.embed(text)
        self.index.add(vec, np.array([doc_id]))
        return doc_id

    def add_documents(self, texts: list[str], metadatas: list[dict] | None = None) -> list[int]:
        metadatas = metadatas or [{} for _ in texts]
        ids = [self.store.add(t, m) for t, m in zip(texts, metadatas)]
        vecs = self.embedder.embed(texts)
        self.index.add(vecs, np.array(ids))
        return ids

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        query_vec = self.embedder.embed(query)
        scores, ids = self.index.search(query_vec, k=k)
        scores, ids = scores[0], ids[0]

        valid = [(i, s) for i, s in zip(ids, scores) if i != -1]
        docs = self.store.get_many([int(i) for i, _ in valid])

        results = []
        for doc_id, score in valid:
            doc = docs.get(int(doc_id))
            if doc is None:
                continue
            results.append({**doc, "score": float(score)})
        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    def delete(self, doc_id: int) -> None:
        self.store.delete(doc_id)
        self.index.remove(np.array([doc_id]))

    def save(self) -> None:
        if self.faiss_path:
            self.index.save(self.faiss_path)

    def close(self) -> None:
        self.save()
        self.store.close()

    def __len__(self) -> int:
        return self.store.count()


def _path_exists(path: str) -> bool:
    from pathlib import Path

    return Path(path).exists()
