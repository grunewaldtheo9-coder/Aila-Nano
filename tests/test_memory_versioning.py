"""Memory versioning: correction (last-writer-wins), supersession, history,
confidence/source, attribute-based forget, and the audit — all through the
real MemoryManager with a tiny embedder (no GPU)."""

from __future__ import annotations

import pytest

from memory.attributes import extract_attribute, extract_attribute_key
from memory.manager import MemoryManager
from vectordb.embedder import AilaEmbedder


@pytest.fixture
def memory(tiny_model, tokenizer, tmp_path) -> MemoryManager:
    embedder = AilaEmbedder(tiny_model, tokenizer)
    mm = MemoryManager(
        embedder, db_path=str(tmp_path / "mem.db"), faiss_path=str(tmp_path / "mem.faiss")
    )
    yield mm
    mm.close()


# -- the extractor ------------------------------------------------------------


@pytest.mark.parametrize(
    "content,key,value",
    [
        ("my favorite game is Minecraft", "favorite_game", "Minecraft"),
        ("My favourite colour is blue", "favorite_colour", "blue"),
        ("my name is Theo", "name", "Theo"),
        ("I'm building an Arduino robot", "project", "an Arduino robot"),
        ("meu jogo favorito é Zelda", "favorite_jogo", "Zelda"),
        ("meu nome é Théo", "name", "Théo"),
        ("estou construindo um robô", "project", "um robô"),
    ],
)
def test_extract_attribute(content, key, value):
    assert extract_attribute(content) == (key, value)


def test_extract_attribute_returns_none_for_plain_statements():
    assert extract_attribute("I have a dog named Max") is None
    assert extract_attribute("the sky is blue today") is None


def test_extract_attribute_key_for_forget_phrases():
    assert extract_attribute_key("my favorite game") == "favorite_game"
    assert extract_attribute_key("my name") == "name"
    assert extract_attribute_key("my project") == "project"


# -- correction through MemoryManager -----------------------------------------


def test_new_value_supersedes_old_and_retrieval_returns_current(memory):
    memory.add_memory("my favorite game is Minecraft", source="explicit_user_request")
    memory.add_memory("my favorite game is Zelda", source="explicit_user_request")

    cur = memory.current_attribute("favorite_game")
    assert cur["content"] == "my favorite game is Zelda"
    assert cur["version"] == 2

    # Retrieval returns the current value only — never the superseded one.
    hits = [f["content"] for f in memory.get_relevant_memories("what is my favorite game", k=5)]
    assert "my favorite game is Zelda" in hits
    assert "my favorite game is Minecraft" not in hits


def test_three_versions_keep_correct_current_and_history(memory):
    for game in ("Minecraft", "Zelda", "Hollow Knight"):
        memory.add_memory(f"my favorite game is {game}", source="explicit_user_request")
    assert memory.current_attribute("favorite_game")["content"].endswith("Hollow Knight")

    hist = memory.attribute_history("favorite_game")
    # Newest first, all three preserved, versions descending.
    assert [f["version"] for f in hist] == [3, 2, 1]
    assert hist[0]["status"] == "active"
    assert all(f["status"] == "superseded" for f in hist[1:])


def test_confidence_and_source_are_stored(memory):
    memory.add_memory("my name is Theo", source="explicit_user_request", confidence=0.97)
    cur = memory.current_attribute("name")
    assert cur["source"] == "explicit_user_request"
    assert cur["confidence"] == 0.97


def test_forget_attribute_deactivates_current_value(memory):
    memory.add_memory("my favorite game is Zelda", source="explicit_user_request")
    assert memory.forget_attribute("favorite_game") == 1
    assert memory.current_attribute("favorite_game") is None
    hits = [f["content"] for f in memory.get_relevant_memories("favorite game", k=5)]
    assert hits == [] or all("favorite game" not in h for h in hits)


def test_plain_memories_are_not_versioned(memory):
    fid = memory.add_memory("I have a dog named Max")
    fact = memory.get_memory(fid)
    assert fact["attribute_key"] is None
    # Still retrievable normally.
    hits = [f["content"] for f in memory.get_relevant_memories("dog", k=5)]
    assert "I have a dog named Max" in hits


def test_audit_counts_active_superseded_deleted(memory):
    memory.add_memory("my favorite game is Minecraft", source="explicit_user_request")
    memory.add_memory("my favorite game is Zelda", source="explicit_user_request")  # supersedes
    memory.add_memory("my name is Theo", source="explicit_user_request")
    memory.forget_attribute("name")  # deletes
    audit = memory.memory_audit()
    assert audit["active"] == 1  # Zelda
    assert audit["superseded"] == 1  # Minecraft
    assert audit["deleted"] == 1  # name


# -- through the agent's memory commands --------------------------------------


def test_remember_then_correct_through_the_agent(tiny_model, tokenizer, memory):
    from agents.registry import get_agent

    agent = get_agent("general", tiny_model, tokenizer, memory=memory)
    agent._handle_memory_command("remember that my favorite game is Minecraft")
    agent._handle_memory_command("remember that my favorite game is Zelda")
    assert memory.current_attribute("favorite_game")["content"].endswith("Zelda")


def test_forget_my_favorite_game_through_the_agent(tiny_model, tokenizer, memory):
    from agents.registry import get_agent

    agent = get_agent("general", tiny_model, tokenizer, memory=memory)
    agent._handle_memory_command("remember that my favorite game is Zelda")
    reply = agent._handle_memory_command("forget my favorite game")
    assert "forgotten" in reply.lower()
    assert memory.current_attribute("favorite_game") is None


def test_forgetting_an_unstored_attribute_does_not_delete_a_different_one(tiny_model, tokenizer, memory):
    """"forget my favorite movie" shares only the word "favorite" with a
    stored "favorite game" — it must not delete the game."""
    from agents.registry import get_agent

    agent = get_agent("general", tiny_model, tokenizer, memory=memory)
    agent._handle_memory_command("remember that my favorite game is Zelda")
    reply = agent._handle_memory_command("forget my favorite movie")
    assert "don't have" in reply.lower() or "forgotten" not in reply.lower()
    # The game survives.
    assert memory.current_attribute("favorite_game")["content"].endswith("Zelda")
