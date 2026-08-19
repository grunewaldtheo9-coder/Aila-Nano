"""Seed the knowledge base with a curated set of stable, general facts.

At ~20M parameters Aila cannot reliably *generate* facts, so the honest
way to make her "know" more is to give her facts she can serve directly
and correctly — the same deterministic path used for web-research
results, but shipped in the repository so every install has them offline,
with no API key and no network. The files live in `knowledge/seed/*.jsonl`
(one JSON object per line: question, answer, optional language/category).

Design choices:
- Facts are deliberately *timeless* (capitals, science, maths, geography).
  Volatile facts ("who is the president") would go stale in the repo, so
  those are left to live web research instead.
- Loading is idempotent and cheap to repeat: `KnowledgeBase.remember_answer`
  already deduplicates by question, and a content hash stored in the
  knowledge store's meta table skips the whole pass when nothing changed,
  so startup pays the cost only when the seed files actually change.
- Seeding must never break startup. Any error (a malformed line, a missing
  directory) is logged and skipped; a partial seed is better than a crash.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# High enough to be served as a direct answer (KnowledgeBase's
# DIRECT_ANSWER_CONFIDENCE is 0.7) without claiming the certainty of a
# corroborated, multi-source web fact.
SEED_CONFIDENCE = 0.9

# Marks a fact as hand-curated rather than web-verified. `KnowledgeBase`
# serves anything that is not "conflicted", so this is served normally;
# the label just keeps its provenance honest in the store.
SEED_VERIFICATION = "curated"

_SEED_DIR = Path(__file__).resolve().parent / "seed"
_META_KEY = "seed_version"


def _seed_files(seed_dir: Path) -> list[Path]:
    return sorted(seed_dir.glob("*.jsonl")) if seed_dir.is_dir() else []


def _content_hash(files: list[Path]) -> str:
    """A hash of every seed file's bytes, so any edit (add, remove, or
    change a fact) forces a re-seed and an unchanged set skips it."""
    h = hashlib.sha256()
    for path in files:
        h.update(path.name.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _language_for(path: Path, record: dict) -> str:
    """Explicit per-record language wins; otherwise infer from the file
    name (`general_knowledge.pt.jsonl` -> pt), defaulting to English."""
    lang = (record.get("language") or "").strip().lower()
    if lang:
        return lang
    # ".pt.jsonl" -> suffixes [".pt", ".jsonl"]; check the second-to-last.
    parts = path.suffixes
    if len(parts) >= 2 and parts[-2].lstrip(".") in ("pt", "en"):
        return parts[-2].lstrip(".")
    return "en"


def seed_knowledge_base(knowledge_base, *, seed_dir: Path | None = None, force: bool = False) -> int:
    """Load the curated seed facts into `knowledge_base`, returning the
    number of facts newly created. Idempotent: skips entirely when the
    seed files are unchanged since the last run (unless `force`). Never
    raises — a seeding problem must not stop Aila from starting.
    """
    directory = seed_dir or _SEED_DIR
    try:
        files = _seed_files(directory)
        if not files:
            return 0

        store = getattr(knowledge_base, "store", None)
        version = _content_hash(files)
        if not force and store is not None:
            try:
                if store.get_meta(_META_KEY) == version:
                    return 0  # already seeded this exact set
            except Exception:  # noqa: BLE001 — meta is an optimization, never fatal
                pass

        created = 0
        for path in files:
            created += _seed_one_file(knowledge_base, path)

        if store is not None:
            try:
                store.set_meta(_META_KEY, version)
            except Exception:  # noqa: BLE001
                logger.debug("could not record seed version (non-fatal)", exc_info=True)

        if created:
            logger.info("seeded %d curated knowledge fact(s)", created)
        return created
    except Exception:  # noqa: BLE001 — seeding is best-effort, never a startup blocker
        logger.warning("knowledge seeding skipped due to an error", exc_info=True)
        return 0


def _seed_one_file(knowledge_base, path: Path) -> int:
    created = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("skipping malformed seed line %s:%d", path.name, line_no)
            continue
        question = (record.get("question") or "").strip()
        answer = (record.get("answer") or "").strip()
        if not question or not answer:
            continue
        outcome, _ = knowledge_base.remember_answer(
            question,
            answer,
            confidence=SEED_CONFIDENCE,
            language=_language_for(path, record),
            category=(record.get("category") or "general").strip() or "general",
            verification=SEED_VERIFICATION,
        )
        if outcome == "created":
            created += 1
    return created
