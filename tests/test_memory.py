from memory.manager import MemoryManager
from memory.ranking import RankingWeights, rank_memories
from memory.store import MemoryStore
from vectordb.embedder import AilaEmbedder


def test_memory_store_conversation_crud(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))
    store.add_message("c1", "user", "hi")
    store.add_message("c1", "assistant", "hello!")
    history = store.get_messages("c1")
    assert [m["role"] for m in history] == ["user", "assistant"]

    store.clear_conversation("c1")
    assert store.get_messages("c1") == []


def test_memory_store_facts(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))
    fid = store.add_fact("The sky is blue.", importance=0.7)
    facts = store.get_facts([fid])
    assert facts[fid]["content"] == "The sky is blue."
    store.touch_fact(fid)
    assert store.get_facts([fid])[fid]["access_count"] == 1
    store.delete_fact(fid)
    assert store.get_facts([fid]) == {}


def test_rank_memories_orders_by_combined_score():
    now = 1000.0
    candidates = [
        {"content": "old but very relevant", "score": 0.95, "created_at": now - 100 * 86400, "importance": 0.5},
        {"content": "recent and relevant", "score": 0.9, "created_at": now, "importance": 0.5},
    ]
    ranked = rank_memories(candidates, weights=RankingWeights(), now=now)
    assert ranked[0]["content"] == "recent and relevant"
    assert ranked[0]["combined_score"] >= ranked[1]["combined_score"]


def test_memory_manager_build_context(tiny_model, tokenizer, tmp_path):
    embedder = AilaEmbedder(tiny_model, tokenizer)
    mm = MemoryManager(
        embedder,
        db_path=str(tmp_path / "mem.db"),
        faiss_path=str(tmp_path / "mem.faiss"),
    )
    mm.add_turn("c1", "user", "My favorite color is blue.")
    mm.add_turn("c1", "assistant", "Got it!")
    mm.remember_fact("The user's favorite color is blue.", importance=0.8)

    ctx = mm.build_context("c1", query="What color does the user like?", max_turns=5, max_facts=1)
    assert len(ctx.history) == 2
    assert len(ctx.relevant_facts) == 1
    assert "blue" in ctx.relevant_facts[0]["content"]
    mm.close()
