"""Aila's personality/preference layer, and the search-routing matrix
(spec §78): opinions, greetings, identity, and basic knowledge must be
answered without a web search; only genuinely current information may
trigger one.
"""

from __future__ import annotations

import logging
import os
import tempfile

import pytest

from knowledge.base import KnowledgeBase
from knowledge.seed_loader import seed_knowledge_base
from knowledge.store import KnowledgeStore
from tools.personality import PERSONALITY, match_personality_question
from tools.router import ToolRouter


# -- the matcher --------------------------------------------------------------


@pytest.mark.parametrize(
    "message,intent",
    [
        ("Do you like Minecraft?", "like_topic"),
        ("Do you like games?", "like_topic"),
        ("Do you like technology?", "like_topic"),
        ("Do you like space?", "like_topic"),
        ("What's your favorite game?", "favorite_game"),
        ("What do you like?", "likes_general"),
        ("What do you like talking about?", "likes_general"),
        ("Are you human?", "human"),
        ("Are you a robot?", "human"),
        ("Do you have feelings?", "feelings"),
        ("Do you search everything?", "search_everything"),
        ("What do you do when you don't know?", "dont_know"),
        # Portuguese
        ("Você gosta de Minecraft?", "like_topic"),
        ("Você gosta de jogos?", "like_topic"),
        ("Você é humana?", "human"),
        ("Qual é o seu jogo favorito?", "favorite_game"),
        ("O que você gosta?", "likes_general"),
    ],
)
def test_personality_questions_are_recognized(message, intent):
    match = match_personality_question(message, language="pt" if _looks_pt(message) else "en")
    assert match is not None, message
    assert match[0] == intent


def _looks_pt(message: str) -> bool:
    return "você" in message.lower()


@pytest.mark.parametrize(
    "message",
    [
        "What is the capital of France?",  # knowledge, not personality
        "Do you like my idea?",            # about the user
        "What is 2 + 2?",                  # arithmetic
        "Who created you?",                # identity
        "Tell me about the Roman Empire",  # a topic, not a preference
    ],
)
def test_personality_does_not_capture_other_questions(message):
    assert match_personality_question(message) is None


def test_minecraft_answer_mentions_minecraft_and_asks_a_follow_up():
    _, answer = match_personality_question("Do you like Minecraft?")
    assert "Minecraft" in answer
    assert "?" in answer  # a natural follow-up question


def test_aila_never_claims_human_experience():
    """The honesty rule (spec §39): Aila enjoys *talking about* things; she
    must not claim to have played/built/watched them."""
    banned = ("i played", "i built", "i watched", "i went", "yesterday i", "i spent")
    for lang in ("en", "pt"):
        for msg in ("Do you like Minecraft?", "Do you like games?", "What's your favorite game?"):
            match = match_personality_question(msg, language=lang)
            if match:
                assert not any(b in match[1].lower() for b in banned), match[1]


def test_personality_config_is_explicit():
    # The identity/preferences live in one readable place, not scattered.
    assert PERSONALITY["favorite_game"]
    assert "technology" in PERSONALITY["likes"]
    assert PERSONALITY["traits"]


# -- routing through the full router -----------------------------------------


class _RecordingResearch:
    """Stands in for the web-research pipeline, recording every call so a
    test can assert whether a search happened."""

    def __init__(self):
        self.queries: list[str] = []

    def research(self, query):
        self.queries.append(query)
        from webresearch.pipeline import ResearchOutcome

        return ResearchOutcome(ok=False, reason="no_extractable_answer")


@pytest.fixture(scope="module")
def router_and_research():
    logging.disable(logging.WARNING)
    tmp = tempfile.mkdtemp()
    store = KnowledgeStore(os.path.join(tmp, "k.db"))
    base = KnowledgeBase(store)
    seed_knowledge_base(base)
    research = _RecordingResearch()
    router = ToolRouter(knowledge=base, research=research)
    yield router, research
    store.close()
    logging.disable(logging.NOTSET)


NO_SEARCH = [
    "Hi",
    "Hello!",
    "What's your name?",
    "Who are you?",
    "What can you do?",
    "Do you like games?",
    "Do you like Minecraft?",
    "What is gravity?",
    "What is the capital of France?",
    "What is 2 + 2?",
    "Are you human?",
]

SEARCH = [
    "What is the latest news about Nintendo?",
    "What is the weather today?",
    "What is the current price of Bitcoin?",
]


@pytest.mark.parametrize("message", NO_SEARCH)
def test_these_messages_never_trigger_a_search(router_and_research, message):
    router, research = router_and_research
    before = len(research.queries)
    router.route(message)
    assert len(research.queries) == before, f"{message!r} should not have searched"


@pytest.mark.parametrize("message", SEARCH)
def test_current_information_questions_do_search(router_and_research, message):
    router, research = router_and_research
    before = len(research.queries)
    router.route(message)
    assert len(research.queries) == before + 1, f"{message!r} should have searched"
