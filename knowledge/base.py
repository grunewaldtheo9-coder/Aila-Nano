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
from memory.lexical import has_distinctive_overlap, lexical_overlap_score, same_subject

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
    language: str = "en"


class KnowledgeBase:
    def __init__(self, store: KnowledgeStore):
        self.store = store

    # -- retrieval ---------------------------------------------------------

    def lookup(
        self, query: str, k: int = 3, threshold: float = SERVE_THRESHOLD, language: str | None = None
    ) -> list[KnowledgeItem]:
        """Relevance-gated lookup. Returns [] when nothing clears the
        threshold — the signal that web research (or a plain model
        answer) is needed. Conflicted items are excluded: a fact under
        active contradiction must not be served as truth.

        `language` is a *tiebreaker*, not a hard filter: among equally
        relevant facts the one in the question's language wins, so a
        Portuguese question gets the Portuguese copy of a fact rather than
        the English one. It is deliberately not a hard filter — language
        detection is imperfect ("Quantos continentes existem?" has no
        accented letters and reads as English to the detector), and a hard
        filter would then drop the very fact that answers the question.
        Relevance already separates the right fact from a wrong-language
        near-match, because the same-language question shares more words
        with the query."""
        items = []
        for row in self.store.all_knowledge():
            if row["verification"] == "conflicted":
                continue
            relevance = lexical_overlap_score(query, row["question"])
            if relevance < threshold:
                continue
            # Score alone is not enough to *serve* an answer: two short
            # questions about different subjects can clear the threshold
            # on a shared asking verb alone ("Who created Samsung?" vs
            # "Who created Hames Eventos" scored 0.5 on "created" and
            # returned the Samsung fact). Require a shared term that
            # actually identifies the subject.
            if not has_distinctive_overlap(query, row["question"]):
                logger.debug(
                    "knowledge id=%s scored %.2f but shares no distinctive term; skipping",
                    row["id"],
                    relevance,
                )
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
                    language=row["language"],
                )
            )
        # Relevance first, then confidence; language is the final
        # tiebreaker so that when two facts are otherwise equally good, the
        # one in the question's language is served.
        items.sort(
            key=lambda i: (i.relevance, i.confidence, i.language == language),
            reverse=True,
        )
        for item in items[:k]:
            self.store.touch_knowledge(item.id)
        return items[:k]

    def best_direct_answer(self, query: str, language: str | None = None) -> KnowledgeItem | None:
        """The single best item usable as a *direct* answer: relevant,
        not conflicted, confident enough, and — when `language` is given —
        in that language. None otherwise."""
        items = self.lookup(query, k=1, language=language)
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
        'rejected'  — input was empty/unusable; nothing stored (id is -1)
        """
        question = (question or "").strip()
        answer = (answer or "").strip()
        # An empty question or answer is never usable knowledge — it can
        # never be retrieved (lexical_overlap_score returns 0.0 for empty
        # text) and would only pollute the store. Reject rather than
        # store an unreachable row.
        if not question or not answer:
            logger.info("knowledge rejected: empty question or answer")
            return ("rejected", -1)

        for row in self.store.all_knowledge():
            # Only the same-language copy of a fact can be its duplicate: a
            # Portuguese and an English phrasing of "the capital of Portugal"
            # overlap perfectly but are two facts we must keep, one per
            # language — not a conflict.
            if row["language"] != language:
                continue
            if lexical_overlap_score(question, row["question"]) < DEDUP_QUESTION_THRESHOLD:
                continue
            # A shared question *template* is not a shared question. "The
            # largest planet?" and "The smallest planet?" overlap on
            # "planet/solar/system" but ask about different subjects, so
            # they are neither duplicates (don't merge) nor contradictions
            # (don't mark either conflicted) — they simply coexist.
            if not same_subject(question, row["question"]):
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
