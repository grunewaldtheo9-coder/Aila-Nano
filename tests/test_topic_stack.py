"""TopicStack: current/previous/dormant topics, conservative switching, and
returning to an earlier topic (English + Portuguese)."""

from __future__ import annotations

import pytest

from conversation.topics import TopicStack


def test_creates_and_switches_topics():
    ts = TopicStack()
    assert ts.note("Let's talk about Aila Nano", 1, ["Aila Nano"]) == "switch"
    assert ts.current.name == "Aila Nano"
    assert ts.note("What about Minecraft?", 2, ["Minecraft"]) == "switch"
    assert ts.current.name == "Minecraft"
    assert [t.name for t in ts.dormant] == ["Aila Nano"]


def test_follow_ups_do_not_switch_topic():
    ts = TopicStack()
    ts.note("How does PostgreSQL work?", 1, ["PostgreSQL"])
    assert ts.note("And how fast is it?", 2, []) == "continue"
    assert ts.note("What about on a laptop?", 3, []) == "continue"
    assert ts.current.name == "PostgreSQL"


def test_returns_to_a_named_earlier_topic():
    ts = TopicStack()
    ts.note("I'm building Aila Nano", 1, ["Aila Nano"])
    ts.note("by the way, Minecraft addons?", 2, ["Minecraft"])
    assert ts.current.name == "Minecraft"
    assert ts.note("going back to Aila Nano", 3, ["Aila Nano"]) == "return"
    assert ts.current.name == "Aila Nano"
    assert [t.name for t in ts.dormant] == ["Minecraft"]


def test_returns_to_the_previous_topic_by_meta_phrase():
    ts = TopicStack()
    ts.note("Let's talk about Aila Nano", 1, ["Aila Nano"])
    ts.note("by the way, Minecraft?", 2, ["Minecraft"])
    assert ts.note("anyway, back to the previous topic", 3, []) == "return"
    assert ts.current.name == "Aila Nano"


def test_transition_marker_switches():
    ts = TopicStack()
    ts.note("I'm improving Aila Nano's memory", 1, ["Aila Nano"])
    assert ts.note("also, how do I install a Minecraft addon?", 2, ["Minecraft"]) == "switch"
    assert ts.current.name == "Minecraft"


@pytest.mark.parametrize(
    "message,target",
    [
        ("back to Aila Nano", "Aila Nano"),
        ("voltando para Aila Nano", "Aila Nano"),
        ("vamos voltar para Minecraft", "Minecraft"),
    ],
)
def test_detect_return_targets(message, target):
    ts = TopicStack()
    assert ts.detect_return(message) == target


def test_dormant_topics_are_preserved():
    ts = TopicStack()
    ts.note("Let's talk about A", 1, ["A"])
    ts.note("what about B", 2, ["B"])
    ts.note("what about C", 3, ["C"])
    assert ts.current.name == "C"
    assert [t.name for t in ts.dormant] == ["B", "A"]
    # Returning to A moves it to current, keeps the rest dormant.
    ts.note("back to A", 4, ["A"])
    assert ts.current.name == "A"
    assert set(t.name for t in ts.dormant) == {"B", "C"}
