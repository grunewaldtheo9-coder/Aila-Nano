"""KnowledgeBase: global, user-independent question/answer knowledge with
deduplication, conflict marking, and relevance-gated retrieval.

Retrieval uses the same deterministic lexical-overlap gate as user
memory (memory/lexical.py) and for the same reason: what gets *served or
injected* must never be "the nearest vector anyway" — an empty result
("Aila doesn't know this yet") has to be a real, reachable state so the
web-research tool knows when to fire and answers are never fabricated
from irrelevant near-matches. Semantic embeddings can be layered on for
broader recall later without changing this gate.

Dedup/conflict policy (see `remember_answer`):
- A new fact whose *question* matches an existing item and whose
  *answer* substantially agrees → update metadata (re-verify timestamp,
  bump use count), don't create a duplicate row.
- Question matches but answers disagree → mark the existing item
  'conflicted' and store the newcomer as a *candidate*, never silently
  overwrite. A conflicted item stops being served deterministically
  until re-verified.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from knowledge.store import KnowledgeStore
from memory.lexical import lexical_overlap_score

logger = logging.getLogger(__name__)

# Question-to-question overlap needed to treat two entries as "the same
# question" for dedup purposes. Higher than the serving threshold: dedup
# merges rows, so a false positive destroys information.
DEDUP_QUESTION_THRESHOLD = 0.6

# Answer-to-answer overlap above which two answers are considered to
# agree (below it, same-question different-answer = conflict).
ANSWER_AGREEMENT_THRESHOLD = 0.5

# Minimum question-relevance for an item to be *served* for a query.
SERVE_THRESHOLD = 0.45

# Confidence at or above which a knowledge item's answer may be used
# directly in a reply (vs. only as background context).
DIRECT_ANSWER_CONFIDENCE = 0.7


@dataclass
class KnowledgeItem:
    id: int
    question: str
    answer: str
    confidence: float
    verification: str
    relevance: float  # lexical overlap vs. the current query
    source_urls: list[str]


class KnowledgeBase:
    def __init__(self, store: KnowledgeStore):
        self.store = store

    # -- retrieval ---------------------------------------------------------

    def lookup(self, query: str, k: int = 3, threshold: float = SERVE_THRESHOLD) -> list[KnowledgeItem]:
        """Relevance-gated lookup. Returns [] when nothing clears the
        threshold — the signal that web research (or a plain model
        answer) is needed. Conflicted items are excluded: a fact under
        active contradiction must not be served as truth."""
        items = []
        for row in self.store.all_knowledge():
            if row["verification"] == "conflicted":
                continue
            relevance = lexical_overlap_score(query, row["question"])
            if relevance < threshold:
                continue
            items.append(
                KnowledgeItem(
                    id=row["id"],
                    question=row["question"],
                    answer=row["answer"],
                    confidence=row["confidence"],
                    verification=row["verification"],
                    relevance=relevance,
                    source_urls=row["source_urls"],
                )
            )
        items.sort(key=lambda i: (i.relevance, i.confidence), reverse=True)
        for item in items[:k]:
            self.store.touch_knowledge(item.id)
        return items[:k]

    def best_direct_answer(self, query: str) -> KnowledgeItem | None:
        """The single best item usable as a *direct* answer: relevant,
        not conflicted, and confident enough. None otherwise."""
        items = self.lookup(query, k=1)
        if items and items[0].confidence >= DIRECT_ANSWER_CONFIDENCE:
            return items[0]
        return None

    # -- writing -----------------------------------------------------------

    def remember_answer(
        self,
        question: str,
        answer: str,
        confidence: float,
        source_urls: list[str] | None = None,
        source_titles: list[str] | None = None,
        language: str = "en",
        category: str = "general",
        verification: str = "unverified",
    ) -> tuple[str, int]:
        """Store new knowledge with dedup + conflict handling.

        Returns (outcome, id) where outcome is one of:
        'created'   — new knowledge row
        'updated'   — merged into an existing row (metadata refreshed)
        'conflict'  — existing row marked conflicted; newcomer stored as
                      a candidate (id is the candidate id)
        """
        for row in self.store.all_knowledge():
            if lexical_overlap_score(question, row["question"]) < DEDUP_QUESTION_THRESHOLD:
                continue
            agreement = lexical_overlap_score(answer, row["answer"])
            if agreement >= ANSWER_AGREEMENT_THRESHOLD:
                # Same question, agreeing answer: refresh, don't duplicate.
                import time as _time

                self.store.update_knowledge(
                    row["id"],
                    confidence=max(row["confidence"], confidence),
                    verification="corroborated",
                    last_verified_at=_time.time(),
                )
                self.store.touch_knowledge(row["id"], verified=True)
                logger.info("knowledge dedup: merged into existing id=%s", row["id"])
                return ("updated", row["id"])
            # Same question, disagreeing answer: conflict.
            self.store.update_knowledge(row["id"], verification="conflicted")
            candidate_id = self.store.add_candidate(
                question,
                answer,
                confidence=confidence,
                reason=f"conflicts_with_knowledge_id={row['id']}",
                source_urls=source_urls,
                source_titles=source_titles,
            )
            logger.warning(
                "knowledge conflict: existing id=%s marked conflicted; newcomer stored "
                "as candidate id=%s",
                row["id"],
                candidate_id,
            )
            return ("conflict", candidate_id)

        knowledge_id = self.store.add_knowledge(
            question,
            answer,
            language=language,
            category=category,
            confidence=confidence,
            source_urls=source_urls,
            source_titles=source_titles,
            verification=verification,
        )
        return ("created", knowledge_id)
