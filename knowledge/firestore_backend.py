"""Firestore adapter for the global knowledge store.

STATUS: adapter implementation only — NOT verified against a live
Firestore project. Verifying it requires a Firebase *service-account*
credential (GOOGLE_APPLICATION_CREDENTIALS pointing at the JSON key
file), which is intentionally not part of this repository. The browser
`firebaseConfig` snippet (apiKey/authDomain/projectId/...) is a web-
client configuration and cannot authenticate a Python backend — do not
put it here; create a service account in the Firebase console instead.

Design: same public surface as knowledge.store.KnowledgeStore, so
engine/state.py can construct either backend behind the same
KnowledgeBase. Collections mirror the SQLite tables:

    knowledge/{autoId}              — validated global facts
    knowledge_candidates/{autoId}   — extracted, not yet validated
    web_cache/{queryHash}           — cached search results with TTL

Security notes for whoever deploys this:
- Server-side only. Never expose these writes directly to browser
  clients; Firestore security rules must deny arbitrary client writes to
  `knowledge/*` (global knowledge would otherwise be poisonable by any
  user).
- User memories (memory/) are deliberately NOT part of this adapter —
  keeping user-scoped data and global knowledge in separate storage
  paths is what makes cross-user leaks structurally impossible.

Requires: pip install firebase-admin
"""

from __future__ import annotations

import hashlib
import json
import time


class FirestoreUnavailableError(RuntimeError):
    """firebase-admin missing or no usable credential."""


class FirestoreKnowledgeStore:
    """Drop-in (interface-compatible) alternative to KnowledgeStore.

    Construction fails fast with a clear error when firebase-admin or
    credentials are missing — engine/state.py catches this and falls
    back to SQLite with a logged warning.
    """

    def __init__(self, project_id: str | None = None):
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore
        except ImportError as e:
            raise FirestoreUnavailableError(
                "firebase-admin is not installed. Run: pip install firebase-admin"
            ) from e

        try:
            if not firebase_admin._apps:
                cred = credentials.ApplicationDefault()
                options = {"projectId": project_id} if project_id else None
                firebase_admin.initialize_app(cred, options)
            self._db = firestore.client()
        except Exception as e:  # noqa: BLE001 — credential errors vary widely
            raise FirestoreUnavailableError(
                "Could not initialize Firebase. Set GOOGLE_APPLICATION_CREDENTIALS "
                "to a service-account JSON file (Firebase console -> Project "
                "settings -> Service accounts). The browser firebaseConfig "
                "snippet cannot authenticate a Python backend."
            ) from e

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
    ) -> str:
        now = time.time()
        doc = {
            "question": question,
            "answer": answer,
            "language": language,
            "category": category,
            "confidence": confidence,
            "source_urls": source_urls or [],
            "source_titles": source_titles or [],
            "verification": verification,
            "created_at": now,
            "updated_at": now,
            "last_verified_at": now if verification == "corroborated" else None,
            "use_count": 0,
            "version": 1,
        }
        _, ref = self._db.collection("knowledge").add(doc)
        return ref.id

    def get_knowledge(self, knowledge_id: str) -> dict | None:
        snap = self._db.collection("knowledge").document(str(knowledge_id)).get()
        if not snap.exists:
            return None
        d = snap.to_dict()
        d["id"] = snap.id
        return d

    def all_knowledge(self) -> list[dict]:
        out = []
        for snap in self._db.collection("knowledge").stream():
            d = snap.to_dict()
            d["id"] = snap.id
            out.append(d)
        return out

    def update_knowledge(self, knowledge_id: str, **fields) -> bool:
        allowed = {
            "question", "answer", "language", "category", "confidence",
            "verification", "last_verified_at",
        }
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"Cannot update field(s) {sorted(bad)!r}")
        ref = self._db.collection("knowledge").document(str(knowledge_id))
        if not ref.get().exists:
            return False
        from firebase_admin import firestore

        fields["updated_at"] = time.time()
        fields["version"] = firestore.Increment(1)
        ref.update(fields)
        return True

    def touch_knowledge(self, knowledge_id: str, verified: bool = False) -> None:
        from firebase_admin import firestore

        updates = {"use_count": firestore.Increment(1)}
        if verified:
            updates["last_verified_at"] = time.time()
        self._db.collection("knowledge").document(str(knowledge_id)).update(updates)

    def delete_knowledge(self, knowledge_id: str) -> bool:
        ref = self._db.collection("knowledge").document(str(knowledge_id))
        if not ref.get().exists:
            return False
        ref.delete()
        return True

    # -- candidates --------------------------------------------------------

    def add_candidate(
        self,
        question: str,
        answer: str,
        confidence: float = 0.0,
        reason: str | None = None,
        source_urls: list[str] | None = None,
        source_titles: list[str] | None = None,
    ) -> str:
        doc = {
            "question": question,
            "answer": answer,
            "source_urls": source_urls or [],
            "source_titles": source_titles or [],
            "confidence": confidence,
            "reason": reason,
            "created_at": time.time(),
        }
        _, ref = self._db.collection("knowledge_candidates").add(doc)
        return ref.id

    def all_candidates(self) -> list[dict]:
        out = []
        for snap in self._db.collection("knowledge_candidates").stream():
            d = snap.to_dict()
            d["id"] = snap.id
            out.append(d)
        return out

    def delete_candidate(self, candidate_id: str) -> bool:
        ref = self._db.collection("knowledge_candidates").document(str(candidate_id))
        if not ref.get().exists:
            return False
        ref.delete()
        return True

    # -- web cache ---------------------------------------------------------

    @staticmethod
    def _cache_doc_id(query_normalized: str) -> str:
        return hashlib.sha256(query_normalized.encode("utf-8")).hexdigest()[:32]

    def cache_web_results(self, query_normalized: str, results: list[dict]) -> None:
        self._db.collection("web_cache").document(self._cache_doc_id(query_normalized)).set(
            {
                "query_normalized": query_normalized,
                "results_json": json.dumps(results),
                "fetched_at": time.time(),
            }
        )

    def get_cached_web_results(
        self, query_normalized: str, max_age_seconds: float
    ) -> list[dict] | None:
        snap = (
            self._db.collection("web_cache")
            .document(self._cache_doc_id(query_normalized))
            .get()
        )
        if not snap.exists:
            return None
        d = snap.to_dict()
        if time.time() - d.get("fetched_at", 0) > max_age_seconds:
            return None
        return json.loads(d["results_json"])

    def close(self) -> None:  # interface parity with KnowledgeStore
        pass
