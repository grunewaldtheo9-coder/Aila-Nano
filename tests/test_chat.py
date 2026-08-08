"""Tests for chat.py's terminal-layer logic (command handling), exercised
directly against a real AilaEngine — no subprocess/stdin plumbing needed
since handle_command() takes plain strings in and mutates a plain dict.
"""

from __future__ import annotations

import re

import pytest

import chat
from engine import AilaEngine, EngineSettings
from tests.conftest import SAMPLE_FINETUNE


@pytest.fixture
def engine(tokenizer, tmp_path, monkeypatch):
    monkeypatch.setenv("AILA_CHECKPOINT", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("AILA_FALLBACK_CHECKPOINT", str(tmp_path / "missing2.pt"))
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


@pytest.fixture
def state(engine):
    return {"conversation_id": chat.new_conversation_id(), "agent": "general"}


def test_new_conversation_id_is_unique():
    a, b = chat.new_conversation_id(), chat.new_conversation_id()
    assert a != b
    assert re.match(r"^session-[0-9a-f]{8}$", a)


def test_exit_and_quit_end_session(engine, state):
    assert chat.handle_command(engine, "exit", state) is False
    assert chat.handle_command(engine, "quit", state) is False
    assert chat.handle_command(engine, "QUIT", state) is False


def test_unknown_command_does_not_end_session(engine, state, capsys):
    assert chat.handle_command(engine, "/bogus", state) is True
    assert "Unknown command" in capsys.readouterr().out


def test_agent_switch_command(engine, state, capsys):
    chat.handle_command(engine, "/agent programming", state)
    assert state["agent"] == "programming"
    assert "Switched" in capsys.readouterr().out

    chat.handle_command(engine, "/agent not-a-real-agent", state)
    assert state["agent"] == "programming"  # unchanged
    assert "Unknown agent" in capsys.readouterr().out


def test_agents_command_lists_all_and_marks_current(engine, state, capsys):
    chat.handle_command(engine, "/agents", state)
    out = capsys.readouterr().out
    for name in ("general", "programming", "research", "writing"):
        assert name in out
    assert "*" in out  # current agent is marked


def test_new_command_resets_conversation_id(engine, state):
    old_id = state["conversation_id"]
    chat.handle_command(engine, "/new", state)
    assert state["conversation_id"] != old_id


def test_remember_command_stores_a_fact(engine, state, capsys):
    chat.handle_command(engine, "/remember the sky is blue", state)
    assert "Remembered" in capsys.readouterr().out
    assert len(engine.memory.long_term.all_facts()) == 1


def test_history_command_shows_prior_turns(engine, state, capsys):
    engine.chat(state["conversation_id"], "Hello", agent_name="general")
    chat.handle_command(engine, "/history", state)
    out = capsys.readouterr().out
    assert "user: Hello" in out


def test_learn_command_indexes_a_file(engine, state, capsys):
    chat.handle_command(engine, f"/learn {SAMPLE_FINETUNE}", state)
    out = capsys.readouterr().out
    assert "Indexed" in out
    assert len(engine.knowledge) >= 1


def test_learn_command_reports_missing_file(engine, state, capsys):
    chat.handle_command(engine, "/learn /no/such/file.txt", state)
    assert "Could not learn" in capsys.readouterr().out


def test_help_command_prints_commands(engine, state, capsys):
    chat.handle_command(engine, "/help", state)
    assert "Commands:" in capsys.readouterr().out
