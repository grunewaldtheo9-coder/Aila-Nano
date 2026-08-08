"""Tests for engine.AilaEngine — the interface-independent AI core that
chat.py (and any future interface) is built on. No terminal/IO involved
here; see tests/test_chat.py for the terminal layer itself.
"""

from __future__ import annotations

import pytest

from engine import AilaEngine, EngineSettings
from tests.conftest import SAMPLE_FINETUNE


@pytest.fixture
def engine(tokenizer, tmp_path, monkeypatch):
    monkeypatch.setenv("AILA_CHECKPOINT", str(tmp_path / "does-not-exist.pt"))
    monkeypatch.setenv("AILA_FALLBACK_CHECKPOINT", str(tmp_path / "also-missing.pt"))
    monkeypatch.setenv("AILA_TOKENIZER", tokenizer.model_path)
    monkeypatch.setenv("AILA_MEMORY_DB", str(tmp_path / "mem.db"))
    monkeypatch.setenv("AILA_MEMORY_FAISS", str(tmp_path / "mem.faiss"))
    monkeypatch.setenv("AILA_KNOWLEDGE_DB", str(tmp_path / "kb.db"))
    monkeypatch.setenv("AILA_KNOWLEDGE_FAISS", str(tmp_path / "kb.faiss"))
    monkeypatch.setenv("AILA_KNOWLEDGE_STORE_DB", str(tmp_path / "kstore.db"))
    monkeypatch.setenv("AILA_DEVICE", "cpu")

    eng = AilaEngine(EngineSettings())
    yield eng
    eng.close()


def test_engine_loads_with_untrained_fallback_matching_tokenizer_vocab(engine, tokenizer):
    assert engine.is_trained is False
    assert engine.model.cfg.vocab_size == tokenizer.vocab_size


def test_engine_lists_and_constructs_all_agents(engine):
    names = engine.available_agents()
    assert set(names) == {"general", "programming", "research", "writing"}
    for name in names:
        agent = engine.get_agent(name)
        assert agent.model is engine.model  # every agent shares one model


def test_progress_callback_reports_each_loading_stage(tokenizer, tmp_path, monkeypatch):
    monkeypatch.setenv("AILA_CHECKPOINT", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("AILA_FALLBACK_CHECKPOINT", str(tmp_path / "missing2.pt"))
    monkeypatch.setenv("AILA_TOKENIZER", tokenizer.model_path)
    monkeypatch.setenv("AILA_MEMORY_DB", str(tmp_path / "mem.db"))
    monkeypatch.setenv("AILA_MEMORY_FAISS", str(tmp_path / "mem.faiss"))
    monkeypatch.setenv("AILA_KNOWLEDGE_DB", str(tmp_path / "kb.db"))
    monkeypatch.setenv("AILA_KNOWLEDGE_FAISS", str(tmp_path / "kb.faiss"))
    monkeypatch.setenv("AILA_KNOWLEDGE_STORE_DB", str(tmp_path / "kstore.db"))
    monkeypatch.setenv("AILA_DEVICE", "cpu")

    messages: list[str] = []
    eng = AilaEngine(EngineSettings(), on_progress=messages.append)
    eng.close()

    for stage in (
        "Loading tokenizer...",
        "Loading model...",
        "Loading memory...",
        "Loading FAISS...",
        "Loading knowledge base...",
        "Loading agents...",
    ):
        assert stage in messages
    assert messages.count("OK") == 6


def test_chat_roundtrip_and_history(engine):
    reply = engine.chat("t1", "Hello!", agent_name="general")
    assert isinstance(reply, str)
    history = engine.memory.conversation.get_history("t1")
    assert len(history) == 2


def test_chat_stream_yields_text_and_stores_memory(engine):
    chunks = list(engine.chat_stream("t2", "Hi there", agent_name="research"))
    assert all(isinstance(c, str) for c in chunks)
    history = engine.memory.conversation.get_history("t2")
    assert len(history) == 2


def test_chat_unknown_agent_raises(engine):
    with pytest.raises(ValueError):
        engine.chat("t1", "hi", agent_name="not-a-real-agent")


def test_remember_and_recall(engine):
    fact_id = engine.memory.remember_fact("The user likes tea.", importance=0.7)
    assert isinstance(fact_id, int)
    results = engine.memory.semantic.recall("What drink does the user like?")
    assert len(results) >= 1


def test_learn_file_indexes_into_knowledge_base(engine):
    n = engine.learn_file(str(SAMPLE_FINETUNE))
    assert n >= 1
    assert len(engine.knowledge) == n


def test_learn_file_rejects_disallowed_extension(engine, tmp_path):
    bad_file = tmp_path / "evil.exe"
    bad_file.write_bytes(b"binary")
    with pytest.raises(ValueError):
        engine.learn_file(str(bad_file))


def test_learn_file_rejects_missing_file(engine, tmp_path):
    with pytest.raises(FileNotFoundError):
        engine.learn_file(str(tmp_path / "nope.txt"))


def test_knowledge_base_is_used_as_context_by_agents(engine, tmp_path):
    note = tmp_path / "note.txt"
    note.write_text(
        "Aila Nano's favorite color is famously always turquoise, a fact found nowhere else.",
        encoding="utf-8",
    )
    engine.learn_file(str(note))

    agent = engine.get_agent("general")
    prompt = agent.prompt_preview("t3", "What is Aila Nano's favorite color?")
    assert "turquoise" in prompt.lower() or "knowledge base" in prompt.lower()
