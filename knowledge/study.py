"""Self-directed study: Aila looks things up on her own so she knows
more tomorrow than she does today.

The knowledge base already grows passively — every question researched
is stored and answered offline forever after. Study makes that growth
active, and it draws its topics from the best possible source: the
questions the user actually asked that Aila *failed* to answer well.
Those are recorded as `knowledge_candidates` by the research pipeline,
so the study session is literally "go back and learn the things you got
stuck on".

Design constraints, all of which exist to keep this from being annoying:

- **Bounded.** At most `max_topics` lookups per session, one network
  request budget each. A study session cannot turn startup into a
  minute-long wait.
- **Once a day.** The last run is recorded in the knowledge store, so
  restarting the chat ten times studies once, and a laptop left closed
  for a week doesn't try to catch up seven times over.
- **Never fatal.** Every failure is caught and reported as a count. A
  study session that cannot reach the network must not stop Aila from
  starting.
- **Never invents.** Study is exactly the normal research path, so
  everything it stores went through the same extraction, on-topic
  gating, confidence scoring and conflict detection as any answer given
  to the user directly. There is no "learned" text that didn't come from
  a real source.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

LAST_STUDY_KEY = "last_study_at"

SECONDS_PER_DAY = 24 * 3600

# Fallback topics for a fresh install with no unanswered questions yet.
# Deliberately broad, common subjects — the things a general assistant is
# asked about first — rather than a long tail nobody will ask for.
SEED_TOPICS: tuple[str, ...] = (
    "What is artificial intelligence?",
    "What is a computer?",
    "What is the Internet?",
    "What is electricity?",
    "What is photosynthesis?",
    "What is gravity?",
    "What is the Solar System?",
    "What is DNA?",
    "What is climate change?",
    "What is mathematics?",
    "What is history?",
    "What is medicine?",
    "What is music?",
    "What is Brazil?",
    "What is the human brain?",
    "What is language?",
    "What is energy?",
    "What is a vaccine?",
    "What is the ocean?",
    "What is evolution?",
)


@dataclass
class StudyReport:
    learned: list[str] = field(default_factory=list)
    failed: int = 0
    skipped: bool = False  # True when the session wasn't due

    @property
    def studied(self) -> int:
        return len(self.learned)

    def summary(self) -> str:
        if self.skipped:
            return ""
        if self.learned:
            topics = ", ".join(self.learned[:3])
            more = f" (+{self.studied - 3} more)" if self.studied > 3 else ""
            return f"Studied {self.studied} new topic(s): {topics}{more}."
        if self.failed:
            return "Tried to study but couldn't reach the web this time."
        return ""


class StudySession:
    """Runs one bounded round of self-directed research."""

    def __init__(self, store, research, max_topics: int = 3, interval_seconds: float = SECONDS_PER_DAY):
        self.store = store
        self.research = research
        self.max_topics = max_topics
        self.interval_seconds = interval_seconds

    # -- scheduling ---------------------------------------------------------

    def due(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        raw = self.store.get_meta(LAST_STUDY_KEY)
        if raw is None:
            return True
        try:
            last = float(raw)
        except (TypeError, ValueError):
            # Corrupted value — treat as never run rather than crashing
            # on a field that is only bookkeeping.
            logger.warning("ignoring unreadable %s value", LAST_STUDY_KEY)
            return True
        # A clock moved backwards (timezone change, VM snapshot) would
        # otherwise block study forever.
        if last > now:
            return True
        return (now - last) >= self.interval_seconds

    def mark_ran(self, now: float | None = None) -> None:
        self.store.set_meta(LAST_STUDY_KEY, str(time.time() if now is None else now))

    # -- topic selection ----------------------------------------------------

    def pick_topics(self) -> list[str]:
        """Unanswered user questions first, then seed topics — skipping
        anything already answered confidently."""
        known = {
            (row.get("question") or "").strip().lower()
            for row in self.store.all_knowledge()
        }

        topics: list[str] = []
        seen: set[str] = set()

        def offer(question: str) -> None:
            question = (question or "").strip()
            key = question.lower()
            if not question or key in seen or key in known:
                return
            seen.add(key)
            topics.append(question)

        # Oldest candidates first: a question the user asked and didn't
        # get a good answer to has been waiting the longest.
        for row in sorted(self.store.all_candidates(), key=lambda r: r.get("created_at") or 0):
            offer(row.get("question", ""))

        for seed in SEED_TOPICS:
            offer(seed)

        return topics[: self.max_topics]

    # -- run ----------------------------------------------------------------

    def run(self, force: bool = False) -> StudyReport:
        """One study round. Never raises."""
        if not force and not self.due():
            return StudyReport(skipped=True)

        report = StudyReport()
        topics = self.pick_topics()
        if not topics:
            self.mark_ran()
            return report

        for topic in topics:
            try:
                outcome = self.research.research(topic)
            except Exception:  # noqa: BLE001 — study must never break startup
                logger.exception("study lookup crashed for %r", topic)
                report.failed += 1
                continue

            if outcome.ok and outcome.stored in ("created", "updated"):
                report.learned.append(_topic_label(topic))
            else:
                report.failed += 1
                logger.info("study: nothing learned for %r (%s)", topic, outcome.reason)

            # The pipeline's offline breaker has tripped — every further
            # lookup this round would just wait and fail.
            if getattr(self.research, "offline", False):
                logger.info("study stopping early: network unavailable")
                break

        self.mark_ran()
        return report


def _topic_label(question: str) -> str:
    """Short human label for a studied question ("What is DNA?" -> "DNA")."""
    text = question.strip().rstrip("?").strip()
    for prefix in ("What is the ", "What is a ", "What is an ", "What is ", "Who is ", "Who created ", "Who founded "):
        if text.lower().startswith(prefix.lower()):
            return text[len(prefix):].strip() or text
    return text
