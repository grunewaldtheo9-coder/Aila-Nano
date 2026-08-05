import pytest

from agents.base import GenerationSettings
from agents.registry import AGENT_REGISTRY, get_agent, list_agents
from memory.manager import MemoryManager
from vectordb.embedder import AilaEmbedder
from vectordb.semantic_index import SemanticIndex


@pytest.fixture
def memory(tiny_model, tokenizer, tmp_path):
    embedder = AilaEmbedder(tiny_model, tokenizer)
    mm = MemoryManager(
        embedder, db_path=str(tmp_path / "mem.db"), faiss_path=str(tmp_path / "mem.faiss")
    )
    yield mm
    mm.close()


def test_registry_lists_all_four_personas():
    names = list_agents()
    assert set(names) == {"general", "programming", "research", "writing"}
    assert len(AGENT_REGISTRY) == 4


def test_get_agent_unknown_name_raises(tiny_model, tokenizer):
    with pytest.raises(ValueError):
        get_agent("not-a-real-agent", tiny_model, tokenizer)


def test_agents_share_the_same_model_instance(tiny_model, tokenizer):
    a1 = get_agent("general", tiny_model, tokenizer)
    a2 = get_agent("programming", tiny_model, tokenizer)
    assert a1.model is a2.model is tiny_model
    assert a1.system_prompt != a2.system_prompt


def test_respond_stores_conversation_turns(tiny_model, tokenizer, memory):
    agent = get_agent("general", tiny_model, tokenizer, memory=memory)
    reply = agent.respond("c1", "Hello!", settings=GenerationSettings(max_new_tokens=8))
    assert isinstance(reply, str)
    history = memory.conversation.get_history("c1")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_respond_without_memory_does_not_crash(tiny_model, tokenizer):
    agent = get_agent("writing", tiny_model, tokenizer, memory=None)
    reply = agent.respond("c1", "Write something.", settings=GenerationSettings(max_new_tokens=8))
    assert isinstance(reply, str)


def test_respond_handles_generation_budget_larger_than_context(tiny_model, tokenizer):
    # Regression test: max_new_tokens >= max_seq_len used to silently skip
    # prompt truncation and crash deep inside model.forward().
    agent = get_agent("general", tiny_model, tokenizer)
    big_settings = GenerationSettings(max_new_tokens=tiny_model.cfg.max_seq_len * 4)
    reply = agent.respond("c1", "Hello there, how are you today?", settings=big_settings)
    assert isinstance(reply, str)


def test_respond_stream_yields_text_and_stores_memory(tiny_model, tokenizer, memory):
    agent = get_agent("research", tiny_model, tokenizer, memory=memory)
    chunks = list(
        agent.respond_stream("c2", "What is gravity?", settings=GenerationSettings(max_new_tokens=8))
    )
    assert all(isinstance(c, str) for c in chunks)
    history = memory.conversation.get_history("c2")
    assert len(history) == 2


def test_knowledge_base_hits_are_surfaced_in_the_prompt(tiny_model, tokenizer, tmp_path):
    embedder = AilaEmbedder(tiny_model, tokenizer)
    knowledge = SemanticIndex(
        embedder, db_path=str(tmp_path / "kb.db"), faiss_path=str(tmp_path / "kb.faiss")
    )
    knowledge.add_document("The launch code is nine four two seven.")

    agent = get_agent("general", tiny_model, tokenizer, knowledge=knowledge)
    prompt = agent.prompt_preview("c1", "What is the launch code?")
    assert "nine four two seven" in prompt
    assert "knowledge base" in prompt.lower()


def test_empty_knowledge_base_does_not_affect_prompt(tiny_model, tokenizer, tmp_path):
    embedder = AilaEmbedder(tiny_model, tokenizer)
    knowledge = SemanticIndex(
        embedder, db_path=str(tmp_path / "kb.db"), faiss_path=str(tmp_path / "kb.faiss")
    )
    agent = get_agent("general", tiny_model, tokenizer, knowledge=knowledge)
    prompt = agent.prompt_preview("c1", "Hello")
    assert "knowledge base" not in prompt.lower()
