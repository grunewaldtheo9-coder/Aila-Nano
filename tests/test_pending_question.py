"""PendingQuestion: detect an assistant clarification/confirmation and
resolve the user's short reply against exactly the options it offered.

Deterministic, offline, English + Portuguese."""

from __future__ import annotations

from conversation.pending import (
    PendingQuestion,
    PendingResolution,
    detect_pending_question,
    resolve_pending,
)


# -- detection ------------------------------------------------------------

def test_detects_two_option_clarification():
    pq = detect_pending_question("Which one do you mean, SQLite or PostgreSQL?")
    assert pq is not None
    assert pq.kind == "clarification"
    assert pq.options == ["SQLite", "PostgreSQL"]
    # The question's lead-in ("Which one") must not leak in as an option.
    assert "Which" not in pq.options
    assert "one" not in pq.options


def test_detects_single_option_confirmation():
    pq = detect_pending_question("Do you mean PostgreSQL?")
    assert pq is not None
    assert pq.options == ["PostgreSQL"]


def test_detects_numbered_list_question():
    pq = detect_pending_question(
        "Which would you like?\n1. SQLite\n2. PostgreSQL\n3. MongoDB"
    )
    assert pq is not None
    assert pq.options == ["SQLite", "PostgreSQL", "MongoDB"]


def test_records_the_turn_it_was_asked():
    pq = detect_pending_question("Do you mean PostgreSQL?", turn=7)
    assert pq is not None
    assert pq.asked_turn == 7


def test_non_question_is_not_pending():
    assert detect_pending_question("PostgreSQL is a solid choice.") is None


def test_plain_question_without_choice_is_not_pending():
    # A question that doesn't ask the user to choose/confirm is not a
    # pending clarification we can resolve a short reply against.
    assert detect_pending_question("How is your project going?") is None


def test_empty_text_is_not_pending():
    assert detect_pending_question("") is None
    assert detect_pending_question("   ") is None


def test_detects_portuguese_clarification():
    pq = detect_pending_question("Você quer dizer SQLite ou PostgreSQL?")
    assert pq is not None
    assert pq.options == ["SQLite", "PostgreSQL"]


# -- resolution -----------------------------------------------------------

def test_resolves_by_naming_the_option():
    pq = detect_pending_question("Which one, SQLite or PostgreSQL?")
    res = resolve_pending(pq, "PostgreSQL")
    assert res.resolved == "PostgreSQL"
    assert res.confirmed is True


def test_resolves_by_ordinal():
    pq = detect_pending_question("Which one, SQLite or PostgreSQL?")
    assert resolve_pending(pq, "the second one").resolved == "PostgreSQL"
    assert resolve_pending(pq, "the first").resolved == "SQLite"
    assert resolve_pending(pq, "the last one").resolved == "PostgreSQL"


def test_yes_confirms_single_option_proposition():
    pq = detect_pending_question("Do you mean PostgreSQL?")
    res = resolve_pending(pq, "yes")
    assert res.resolved == "PostgreSQL"
    assert res.confirmed is True


def test_no_rejects_the_proposition():
    pq = detect_pending_question("Do you mean PostgreSQL?")
    res = resolve_pending(pq, "no")
    assert res.confirmed is False
    assert res.resolved is None


def test_bare_yes_without_single_option_does_not_invent():
    # "yes" to a two-way choice doesn't pick one — it must not guess.
    pq = detect_pending_question("Which one, SQLite or PostgreSQL?")
    res = resolve_pending(pq, "yes")
    assert res.resolved is None
    assert res.confirmed is True
    assert res.reason == "affirmed_without_single_option"


def test_unresolvable_reply_reports_candidates():
    pq = detect_pending_question("Which one, SQLite or PostgreSQL?")
    res = resolve_pending(pq, "actually let's talk about something else")
    assert res.resolved is None
    assert res.reason == "unresolved"
    assert res.candidates == ["SQLite", "PostgreSQL"]


def test_out_of_range_ordinal_is_ambiguous():
    pq = detect_pending_question("Which one, SQLite or PostgreSQL?")
    res = resolve_pending(pq, "the fifth")
    assert res.resolved is None
    assert res.ambiguous is True
    assert res.reason == "out_of_range"


def test_portuguese_affirmation_confirms_single_option():
    pq = detect_pending_question("Você quer dizer PostgreSQL?")
    res = resolve_pending(pq, "sim")
    assert res.resolved == "PostgreSQL"


def test_dataclasses_have_sensible_defaults():
    pq = PendingQuestion(text="?")
    assert pq.options == [] and pq.kind == "clarification" and pq.asked_turn == -1
    pr = PendingResolution()
    assert pr.resolved is None and pr.confirmed is None and pr.ambiguous is False
