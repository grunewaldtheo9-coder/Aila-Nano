"""EntityTracker: entity extraction/typing, and pronoun resolution by
recency with ambiguity safety (English + Portuguese)."""

from __future__ import annotations

import pytest

from conversation.entities import EntityTracker, extract_entities


def test_extracts_and_types_known_entities():
    ents = extract_entities("I'm deciding between SQLite and PostgreSQL")
    by_canon = {c: (s, t) for s, c, t in ents}
    assert by_canon["sqlite"][1] == "technology"
    assert by_canon["postgresql"][1] == "technology"


def test_types_games_and_projects():
    ents = {c: t for _s, c, t in extract_entities("I play Minecraft and build Aila Nano")}
    assert ents["minecraft"] == "game"
    assert ents["aila nano"] == "project"


def test_unknown_capitalised_word_is_a_proper_noun():
    ents = {c: t for _s, c, t in extract_entities("My friend Gandalf helped")}
    assert ents.get("gandalf") == "proper_noun"


def test_common_sentence_starts_are_not_entities():
    ents = [c for _s, c, _t in extract_entities("The project is going well")]
    assert "the" not in ents


# -- pronoun resolution -------------------------------------------------------


def test_most_recent_entity_wins():
    t = EntityTracker()
    t.observe("I'm deciding between SQLite and PostgreSQL.", 1)
    t.observe("PostgreSQL seems better.", 2)
    r = t.resolve_pronoun("it")
    assert r.entity is not None and r.entity.text == "PostgreSQL"
    assert r.confidence >= 0.8


def test_a_tie_is_ambiguous_not_invented():
    t = EntityTracker()
    t.observe("I'm deciding between SQLite and PostgreSQL.", 1)
    r = t.resolve_pronoun("it")
    assert r.entity is None
    assert r.ambiguous is True
    assert {c.text for c in r.candidates} == {"SQLite", "PostgreSQL"}
    assert r.reason == "multiple_active_entities"


def test_no_entities_no_resolution():
    r = EntityTracker().resolve_pronoun("it")
    assert r.entity is None and r.reason == "no_active_entity"


@pytest.mark.parametrize("pronoun", ["it", "this", "that", "isso", "ele", "ela", "essa opção"])
def test_singular_pronouns_english_and_portuguese(pronoun):
    t = EntityTracker()
    t.observe("PostgreSQL is nice.", 1)
    assert t.resolve_pronoun(pronoun).entity.text == "PostgreSQL"


def test_a_non_pronoun_is_rejected():
    t = EntityTracker()
    t.observe("PostgreSQL is nice.", 1)
    assert t.resolve_pronoun("PostgreSQL").reason == "not_a_pronoun"


def test_resolve_pronoun_inside_a_sentence():
    t = EntityTracker()
    t.observe("Let's use PostgreSQL.", 1)
    r = t.resolve_in_text("why is it faster?")
    assert r.entity is not None and r.entity.text == "PostgreSQL"


def test_entity_recency_metadata():
    t = EntityTracker()
    t.observe("SQLite is small.", 1)
    t.observe("SQLite again.", 3)
    ent = t.entities["sqlite"]
    assert ent.first_seen_turn == 1 and ent.last_seen_turn == 3 and ent.mentions == 2
