"""Curated knowledge seeding, and the knowledge-base changes that let a
dense set of templated facts coexist:

- the seed loads (idempotently) and is served through the router,
- templated questions that share a shape but not a subject ("largest" vs
  "smallest planet") do not conflict each other out of the store,
- a Portuguese question gets the Portuguese copy of a fact, an English
  question the English copy, even though the two overlap heavily.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from knowledge.base import KnowledgeBase
from knowledge.seed_loader import seed_knowledge_base
from knowledge.store import KnowledgeStore
from memory.lexical import distinctive_terms, same_subject
from tools.router import ToolRouter

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = REPO_ROOT / "knowledge" / "seed"


@pytest.fixture
def seeded(tmp_path):
    store = KnowledgeStore(str(tmp_path / "k.db"))
    base = KnowledgeBase(store)
    created = seed_knowledge_base(base)
    yield base, store, created
    store.close()


# -- the seed files themselves ------------------------------------------------


def test_seed_files_are_well_formed():
    files = list(SEED_DIR.glob("*.jsonl"))
    assert files, "expected curated seed files to exist"
    for path in files:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)  # must parse
            assert record.get("question"), f"{path.name}:{line_no} missing question"
            assert record.get("answer"), f"{path.name}:{line_no} missing answer"


# -- loading ------------------------------------------------------------------


def test_seeding_loads_facts_and_is_idempotent(seeded):
    base, store, created = seeded
    assert created > 50  # a meaningful body of knowledge
    total = len(store.all_knowledge())
    # A second pass adds nothing new and touches nothing.
    assert seed_knowledge_base(base) == 0
    assert len(store.all_knowledge()) == total


def test_seeding_creates_no_conflicts(seeded):
    """Templated facts ("capital of X", "largest/smallest planet") must not
    knock each other into the 'conflicted' state during loading — that was
    the bug that made a whole family of facts unservable."""
    _, store, _ = seeded
    conflicted = [r for r in store.all_knowledge() if r["verification"] == "conflicted"]
    assert conflicted == []


# -- serving through the router ----------------------------------------------


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What is the capital of France?", "Paris"),
        ("What is the largest planet in the solar system?", "Jupiter"),
        ("What is the smallest planet in the solar system?", "Mercury"),
        ("How many continents are there?", "seven"),
        ("What is the freezing point of water?", "0"),
        ("What is the boiling point of water?", "100"),
        # Portuguese must get the Portuguese copy, not the English one.
        ("Qual é a capital do Brasil?", "Brasília"),
        ("Qual é a capital de Portugal?", "Lisboa"),
        ("Qual é o menor planeta do sistema solar?", "Mercúrio"),
        ("Quantos continentes existem?", "sete"),
        ("Qual é a moeda do Brasil?", "real"),
    ],
)
def test_seeded_facts_are_served(seeded, question, expected):
    base, _, _ = seeded
    router = ToolRouter(knowledge=base)
    result = router.route(question)
    assert result.tool_used == "knowledge"
    assert expected.lower() in result.direct_reply.lower()


def test_a_portuguese_question_is_answered_in_portuguese(seeded):
    """The English and Portuguese copies of "the capital of Portugal"
    overlap perfectly, but the language tiebreaker must serve the one that
    matches the question."""
    base, _, _ = seeded
    router = ToolRouter(knowledge=base)

    en = router.route("What is the capital of Portugal?")
    pt = router.route("Qual é a capital de Portugal?")
    assert en.direct_reply == "The capital of Portugal is Lisbon."
    assert pt.direct_reply == "A capital de Portugal é Lisboa."


# -- the lexical helpers ------------------------------------------------------


def test_same_subject_separates_a_shared_template_from_a_shared_subject():
    # Same template, different subject -> not the same question.
    assert not same_subject("the largest planet", "the smallest planet")
    assert not same_subject("the capital of France", "the capital of Brazil")
    # Same subject, reworded -> the same question.
    assert same_subject("the capital of France", "France's capital")


def test_distinctive_terms_drops_generic_words():
    # "capital" is a subject word here (not in the generic set); the asking
    # scaffolding is gone.
    assert distinctive_terms("What is the capital of France?") == {"capital", "france"}
