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
    # Same confirmation wording as typing "remember that ..." — the slash
    # command is a shortcut for that path, not a second implementation.
    assert "I'll remember that the sky is blue" in capsys.readouterr().out
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


def test_support_command_prints_the_address_and_a_diagnostic_report(engine, capsys):
    import chat

    state = {"conversation_id": "c1", "agent": "general"}
    assert chat.handle_command(engine, "/support", state) is True
    out = capsys.readouterr().out

    assert "mailailacompanysolutions@gmail.com" in out
    # The report is what makes a bug findable — it must carry the basics.
    for expected in ("Aila Nano version", "Python", "Trained model", "Web search"):
        assert expected in out
    # Nothing is sent on the user's behalf.
    assert "Nothing is sent automatically" in out


def test_feedback_command_includes_the_users_message(engine, capsys):
    import chat

    state = {"conversation_id": "c1", "agent": "general"}
    assert chat.handle_command(engine, "/feedback he repeats himself", state) is True
    out = capsys.readouterr().out
    assert "he repeats himself" in out
    assert "mailailacompanysolutions@gmail.com" in out


def test_support_report_survives_a_broken_engine():
    """A support command that crashes is worse than useless — it fires
    exactly when things are already broken."""
    from engine.support import build_support_report

    class Broken:
        def __getattr__(self, name):
            raise RuntimeError("everything is on fire")

    report = build_support_report(Broken(), version="2.0")
    assert "Aila Nano — support report" in report
    assert "unavailable" in report


def test_slash_commands_never_print_none_for_unusable_input(engine, capsys):
    """`/forget it` used to print the literal word "None": the slash
    command bypassed the empty-content guard that the typed form has."""
    import chat

    state = {"conversation_id": "c1", "agent": "general"}
    for command in ("/forget it", "/remember that", "/forget that"):
        capsys.readouterr()
        chat.handle_command(engine, command, state)
        out = capsys.readouterr().out.strip()
        assert out and out != "None", f"{command!r} printed {out!r}"
        assert "couldn't tell" in out


def test_slash_remember_applies_the_memory_length_cap(engine, capsys):
    """The slash command used to call the memory API directly, skipping
    MAX_MEMORY_CHARS — an unbounded memory is injected verbatim into a
    512-token context and crowds out the question itself."""
    import chat
    from memory.commands import MAX_MEMORY_CHARS

    state = {"conversation_id": "c1", "agent": "general"}
    chat.handle_command(engine, "/remember " + ("my lucky number is seven " * 60), state)
    capsys.readouterr()

    facts = engine.memory.all_memories()
    assert facts
    assert all(len(f["content"]) <= MAX_MEMORY_CHARS for f in facts)


def test_slash_remember_is_readable_back(engine, capsys):
    import chat

    state = {"conversation_id": "c1", "agent": "general"}
    chat.handle_command(engine, "/remember my name is Theo", state)
    capsys.readouterr()
    chat.handle_command(engine, "/memories", state)
    assert "my name is Theo" in capsys.readouterr().out
