"""Prove the ConversationManager is on the REAL generation path: the
summary and corrected memories it assembles actually reach the tokens the
model is asked to generate from — not just an isolated build_context call.
Uses the tiny model with freeform generation on and a recorder patched over
`generate`; no GPU.
"""

from __future__ import annotations

import pytest

import agents.base as base
from agents.registry import get_agent
from conversation import ConversationManager
from memory.manager import MemoryManager
from model.config import GPTConfig
from model.transformer import AilaNanoGPT
from vectordb.embedder import AilaEmbedder


@pytest.fixture
def wide_model(tokenizer):
    """A small model but with a realistic 512-token context, so a summary +
    system prompt isn't truncated the way the 64-token tiny fixture would be."""
    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size, max_seq_len=512, n_layers=2, d_model=32,
        n_heads=4, n_kv_heads=2, mlp_hidden_mult=2.0, dropout=0.0, tie_embeddings=True, bias=False,
    )
    return AilaNanoGPT(cfg)


def _agent_with_manager(model, tokenizer, tmp_path):
    embedder = AilaEmbedder(model, tokenizer)
    memory = MemoryManager(
        embedder, db_path=str(tmp_path / "m.db"), faiss_path=str(tmp_path / "m.faiss")
    )
    cm = ConversationManager(memory, recent_turns=4, summarize_after=8)
    agent = get_agent(
        "general", model, tokenizer, memory=memory,
        conversation_manager=cm, allow_freeform=True,
    )
    return agent, memory, cm


def _long_robot_conversation(memory, cid):
    turns = [
        ("user", "I'm building a robot."), ("assistant", "Cool!"),
        ("user", "It uses an Arduino Mega."), ("assistant", "Nice."),
        ("user", "It has ultrasonic sensors."), ("assistant", "Good for obstacles."),
        ("user", "I also want an OLED display."), ("assistant", "Great choice."),
        ("user", "The weather is nice today."), ("assistant", "Nice!"),
    ]
    for role, content in turns:
        memory.add_turn(cid, role, content)


def test_summary_reaches_the_generation_prompt(wide_model, tokenizer, tmp_path):
    agent, memory, cm = _agent_with_manager(wide_model, tokenizer, tmp_path)
    cid = "c1"
    _long_robot_conversation(memory, cid)

    # The prompt the model would be asked to generate from.
    _, prompt_ids = agent._prepare_turn(cid, "What was I building?")
    decoded = tokenizer.decode(prompt_ids)
    assert "[SUMMARY]" in decoded
    # A project fact from earlier survives into the model's context...
    assert "robot" in decoded.lower() or "arduino" in decoded.lower()
    # ...and the pure filler does not.
    assert "weather" not in decoded.lower()
    memory.close()


def test_corrected_memory_reaches_the_generation_prompt(wide_model, tokenizer, tmp_path):
    agent, memory, cm = _agent_with_manager(wide_model, tokenizer, tmp_path)
    memory.add_memory("my favorite game is Minecraft", source="explicit_user_request")
    memory.add_memory("my favorite game is Zelda", source="explicit_user_request")

    _, prompt_ids = agent._prepare_turn("c1", "what is my favorite game")
    decoded = tokenizer.decode(prompt_ids)
    # The current value is injected; the superseded one is not.
    assert "Zelda" in decoded
    assert "Minecraft" not in decoded
    memory.close()


def test_the_recorded_model_input_carries_the_context(wide_model, tokenizer, tmp_path, monkeypatch):
    """End-to-end: patch `generate` to record the exact input tensor respond
    feeds the model, and confirm the ConversationManager context is in it."""
    agent, memory, cm = _agent_with_manager(wide_model, tokenizer, tmp_path)
    _long_robot_conversation(memory, "c1")

    recorded = {}

    def _recording_generate(model, input_ids, *args, **kwargs):
        recorded["ids"] = input_ids[0].tolist()
        return input_ids  # return something the caller can decode into a reply

    monkeypatch.setattr(base, "generate", _recording_generate)
    # A message the router can't answer, so freeform generation actually runs.
    agent.respond("c1", "tell me a little story about it")

    assert "ids" in recorded, "the model was never called"
    decoded = tokenizer.decode(recorded["ids"])
    assert "[SUMMARY]" in decoded
    memory.close()
