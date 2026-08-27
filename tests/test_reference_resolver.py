"""ReferenceResolver: short contextual messages ("the second one", "yes")
resolved against recent turns, and the ambiguity/confidence behaviour."""

from __future__ import annotations

import pytest

from conversation.reference import Resolution, parse_options, resolve_reference

NUMBERED = [
    {"role": "user", "content": "Which database should I use?"},
    {"role": "assistant", "content": "You have a few options:\n1. SQLite\n2. PostgreSQL\n3. JSON"},
]
INLINE = [
    {"role": "assistant", "content": "You could use SQLite, PostgreSQL, or JSON."},
]
TWO = [
    {"role": "assistant", "content": "You can use SQLite or PostgreSQL."},
]


# -- option parsing -----------------------------------------------------------


def test_parses_a_numbered_list():
    assert parse_options(NUMBERED[1]["content"]) == ["SQLite", "PostgreSQL", "JSON"]


def test_parses_an_inline_list():
    assert parse_options(INLINE[0]["content"]) == ["SQLite", "PostgreSQL", "JSON"]


# -- ordinal / list-item references (spec §153, §229) -------------------------


@pytest.mark.parametrize(
    "message,expected",
    [
        ("The second one", "PostgreSQL"),
        ("the second", "PostgreSQL"),
        ("number 2", "PostgreSQL"),
        ("option 2", "PostgreSQL"),
        ("2", "PostgreSQL"),
        ("the first", "SQLite"),
        ("the first one", "SQLite"),
        ("the third", "JSON"),
        ("the last one", "JSON"),
        ("o segundo", "PostgreSQL"),  # Portuguese
    ],
)
def test_ordinal_references_resolve_high_confidence(message, expected):
    r = resolve_reference(message, NUMBERED)
    assert r.kind == "list_item"
    assert r.value == expected
    assert r.confidence == "high"


def test_inline_list_ordinal_resolves():
    assert resolve_reference("the second one", INLINE).value == "PostgreSQL"


def test_the_other_one_resolves_only_with_two_options():
    # Two options: "the other one" is the second, medium confidence.
    r2 = resolve_reference("the other one", TWO)
    assert r2.kind == "list_item" and r2.value == "PostgreSQL" and r2.confidence == "medium"
    # Three options: genuinely ambiguous -> no resolution (ask instead).
    r3 = resolve_reference("the other one", NUMBERED)
    assert r3.kind == "none"


def test_an_out_of_range_ordinal_is_not_invented():
    r = resolve_reference("number 9", NUMBERED)
    assert r.kind == "none"
    assert r.value is None
    # The candidates are still available for a clarification.
    assert r.options == ["SQLite", "PostgreSQL", "JSON"]


def test_an_ordinal_with_no_list_does_not_resolve():
    turns = [{"role": "assistant", "content": "SQLite is a good choice."}]
    assert resolve_reference("the second one", turns).kind == "none"


# -- affirmation / negation ---------------------------------------------------


@pytest.mark.parametrize("msg", ["yes", "yeah", "exactly", "sure", "sim", "isso"])
def test_affirmations(msg):
    assert resolve_reference(msg, NUMBERED).kind == "affirmation"


@pytest.mark.parametrize("msg", ["no", "nope", "nah", "não"])
def test_negations(msg):
    assert resolve_reference(msg, NUMBERED).kind == "negation"


# -- non-references ------------------------------------------------------------


@pytest.mark.parametrize("msg", ["why?", "How do I install it?", "", "What is the capital of France?"])
def test_non_references_return_none(msg):
    assert resolve_reference(msg, NUMBERED).kind == "none"


def test_a_full_sentence_with_a_number_is_not_a_list_reference():
    # "I have 2 databases" is not "the 2nd item".
    r = resolve_reference("I have 2 databases running", NUMBERED)
    assert r.kind == "none"


# -- through the ConversationManager ------------------------------------------


class _FakeStore:
    def __init__(self):
        self.t: dict[str, list[dict]] = {}

    def add_turn(self, c, r, ct, agent_type=None):
        self.t.setdefault(c, []).append({"role": r, "content": ct})

    def render_for_prompt(self, c, max_turns=1000):
        return [t for t in self.t.get(c, []) if t["role"] != "system"][-max_turns:]


class _FakeMemory:
    def __init__(self):
        self.conversation = _FakeStore()

    def add_turn(self, c, r, ct):
        self.conversation.add_turn(c, r, ct)

    def get_relevant_memories(self, q, k=5):
        return []


def test_manager_resolves_reference_against_conversation():
    from conversation import ConversationManager

    cm = ConversationManager(_FakeMemory())
    cm.add_turn("c1", "user", "Which database?")
    cm.add_turn("c1", "assistant", "Options:\n1. SQLite\n2. PostgreSQL\n3. JSON")
    r = cm.resolve_reference("c1", "the second one")
    assert isinstance(r, Resolution)
    assert r.value == "PostgreSQL"
