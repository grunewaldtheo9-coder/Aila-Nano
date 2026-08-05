"""A thin, typed wrapper around a FAISS flat inner-product index.

Design decisions:
- IndexFlatIP (exact search) over an approximate index (HNSW/IVF): at the
  scale a nano-model deployment realistically indexes (thousands to low
  millions of memory/document chunks), exact search is fast enough and
  removes an entire class of recall-tuning problems. Swap to IndexHNSWFlat
  if the index grows past ~1M vectors — the wrapper's public API doesn't
  need to change.
- Wrapped in IndexIDMap2 so callers can use their own stable integer ids
  (matching primary keys in the document store) instead of relying on
  FAISS's implicit insertion order, and so vectors can be removed by id.
- Vectors are expected pre-normalized (see vectordb/embedder.py), so inner
  product == cosine similarity.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class FaissIndex:
    def __init__(self, dim: int):
        self.dim = dim
        base = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap2(base)

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vectors.shape[1]}")
        self.index.add_with_ids(vectors, ids)

    def remove(self, ids: np.ndarray) -> int:
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        return self.index.remove_ids(ids)

    def search(self, query: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scores, ids), each shape (n_queries, k). Missing
        results (fewer than k total vectors indexed) are id == -1."""
        query = np.ascontiguousarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query[None, :]
        k = min(k, max(1, self.index.ntotal))
        scores, ids = self.index.search(query, k)
        return scores, ids

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: str, dim: int) -> FaissIndex:
        obj = cls(dim)
        obj.index = faiss.read_index(str(path))
        return obj
