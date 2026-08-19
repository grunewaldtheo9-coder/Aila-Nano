"""Regression tests for the external long-term memory architecture
(memory/lexical.py, memory/store.py, memory/long_term_memory.py,
memory/semantic_memory.py, memory/manager.py, memory/commands.py, and
their integration into agents/base.py).

This replaces the earlier approach of fine-tuning the model to use
injected facts (which didn't reliably work — see git history) with a
retrieval layer that stores memories outside the model's weights and
only injects the ones a deterministic relevance check considers actually
relevant to the current question. These tests exist specifically to
catch the two failure modes that mattered most while designing it:
irrelevant memories leaking into unrelated answers, and memories being
silently dropped or fabricated.
"""

from __future__ import annotations

import pytest

from agents.base import FORGET_MATCH_THRESHOLD, Agent
from agents.registry import get_agent
from memory.commands import guess_category, parse_memory_command
from memory.lexical import lexical_overlap_score, tokenize
from memory.manager import MemoryContext, MemoryManager
from memory.semantic_memory import DEFAULT_RELEVANCE_THRESHOLD
from memory.store import MemoryStore
from vectordb.embedder import AilaEmbedder


@pytest.fixture
def memory(tiny_model, tokenizer, tmp_path) -> MemoryManager:
    embedder = AilaEmbedder(tiny_model, tokenizer)
    mm = MemoryManager(
        embedder, db_path=str(tmp_path / "mem.db"), faiss_path=str(tmp_path / "mem.faiss")
    )
    yield mm
    mm.close()


# -- 1. save a name and retrieve it -----------------------------------------


def test_save_name_and_retrieve_it(memory):
    memory.add_memory("The user's name is Theo.", category="identity", importance=0.9)
    results = memory.get_relevant_memories("What's my name?")
    assert len(results) == 1
    assert "Theo" in results[0]["content"]


# -- 2. save a preference and retrieve it -----------------------------------


def test_save_preference_and_retrieve_it(memory):
    memory.add_memory("The user's favorite color is blue.", category="preference")
    results = memory.get_relevant_memories("What's my favorite color?")
    assert len(results) == 1
    assert "blue" in results[0]["content"]


# -- 3. retrieve the correct memory for a related question ------------------


def test_retrieves_the_correct_memory_among_several(memory):
    memory.add_memory("The user's name is Theo.", category="identity")
    memory.add_memory("The user's favorite color is blue.", category="preference")
    memory.add_memory("The user has a dog named Max.", category="personal_fact")

    # Deliberately avoids the word "name" — "what's my dog's *name*" would
    # legitimately also share a word with "The user's *name* is Theo.",
    # which is a fair ambiguous case for lexical overlap, just not the one
    # this test is checking.
    results = memory.get_relevant_memories("What is my dog called?")
    assert len(results) == 1
    assert "Max" in results[0]["content"]

    results = memory.get_relevant_memories("What color do I like?")
    assert len(results) == 1
    assert "blue" in results[0]["content"]


# -- 4. do not retrieve unrelated memories -----------------------------------


def test_does_not_retrieve_unrelated_memories(memory):
    memory.add_memory("The user's name is Theo.", category="identity")
    memory.add_memory("The user's favorite color is blue.", category="preference")

    # Neither stored memory shares any significant word with this query.
    results = memory.get_relevant_memories("What is photosynthesis?")
    assert results == []


def test_unrelated_query_is_not_injected_into_system_prompt():
    ctx_with_unrelated_fact = MemoryContext(relevant_facts=[])  # gate already excluded it
    agent = Agent.__new__(Agent)
    agent.system_prompt = "Base prompt."
    agent.knowledge = None
    prompt = Agent._build_system_prompt(agent, "What is photosynthesis?", ctx_with_unrelated_fact)
    assert prompt == "Base prompt."
    assert "[MEMORY]" not in prompt


# -- 5. forget a memory and verify it is gone --------------------------------


def test_forget_a_memory_removes_it(memory):
    fact_id = memory.add_memory("The user's name is Theo.", category="identity")
    assert memory.get_relevant_memories("What's my name?") != []

    assert memory.delete_memory(fact_id) is True
    assert memory.get_memory(fact_id) is None
    assert memory.get_relevant_memories("What's my name?") == []


def test_delete_nonexistent_memory_returns_false(memory):
    assert memory.delete_memory(999999) is False


# -- 6. persistence after restart --------------------------------------------


def test_memory_persists_across_manager_restart(tiny_model, tokenizer, tmp_path):
    db_path = str(tmp_path / "mem.db")
    faiss_path = str(tmp_path / "mem.faiss")
    embedder = AilaEmbedder(tiny_model, tokenizer)

    mm1 = MemoryManager(embedder, db_path=db_path, faiss_path=faiss_path)
    mm1.add_memory("The user's name is Theo.", category="identity")
    mm1.close()  # persists SQLite (already durable) and saves the FAISS index

    mm2 = MemoryManager(embedder, db_path=db_path, faiss_path=faiss_path)
    try:
        results = mm2.get_relevant_memories("What's my name?")
        assert len(results) == 1
        assert "Theo" in results[0]["content"]
        # And the FAISS index itself was reloaded, not rebuilt empty.
        assert mm2.semantic.index.ntotal == 1
    finally:
        mm2.close()


# -- 7. empty memory database -------------------------------------------------


def test_empty_database_returns_no_memories(memory):
    assert memory.get_relevant_memories("What's my name?") == []
    assert memory.search_memories("anything") == []
    assert memory.all_memories() == []


def test_empty_database_does_not_add_memory_block_to_prompt(tiny_model, tokenizer, memory):
    agent = get_agent("general", tiny_model, tokenizer, memory=memory)
    prompt = agent.prompt_preview("c1", "What's my name?")
    assert "[MEMORY]" not in prompt


# -- 8. multiple users/sessions ----------------------------------------------


def test_session_scoped_memories_are_isolated(memory):
    memory.add_memory("The user's name is Theo.", category="identity", session_id="alice")
    memory.add_memory("The user's name is Maria.", category="identity", session_id="bob")

    alice_results = memory.get_relevant_memories("What's my name?", session_id="alice")
    assert len(alice_results) == 1
    assert "Theo" in alice_results[0]["content"]

    bob_results = memory.get_relevant_memories("What's my name?", session_id="bob")
    assert len(bob_results) == 1
    assert "Maria" in bob_results[0]["content"]


def test_global_memory_is_visible_to_every_session(memory):
    # No session_id => global, visible regardless of which session asks
    # (e.g. a fact remembered via chat.py's single-user /remember command).
    memory.add_memory("The user's name is Theo.", category="identity", session_id=None)
    assert len(memory.get_relevant_memories("What's my name?", session_id="alice")) == 1
    assert len(memory.get_relevant_memories("What's my name?", session_id="bob")) == 1


def test_clear_memories_scoped_to_one_session_leaves_others_and_global(memory):
    memory.add_memory("Alice likes tea.", session_id="alice")
    memory.add_memory("Bob likes coffee.", session_id="bob")
    memory.add_memory("The user's name is Theo.", session_id=None)  # global

    removed = memory.clear_memories(session_id="alice")
    assert removed == 1
    assert memory.get_relevant_memories("What does bob like?", session_id="bob") != []
    assert memory.get_relevant_memories("What's my name?", session_id="bob") != []  # global survives


# -- 9. memory retrieval must never fabricate information --------------------


def test_no_memories_stored_means_normal_answer_not_a_fabrication(tiny_model, tokenizer, memory):
    agent = get_agent("general", tiny_model, tokenizer, memory=memory)
    # Nothing has ever been remembered — the system prompt must not claim
    # otherwise by inventing a [MEMORY] block. (Not checking for the
    # absence of any specific name here: the persona's own system prompt
    # legitimately mentions its creators' names regardless of memory —
    # see test_empty_database_does_not_add_memory_block_to_prompt and
    # test_unrelated_query_is_not_injected_into_system_prompt for the
    # no-[MEMORY]-block assertion itself.)
    prompt = agent.prompt_preview("c1", "What's my name?")
    assert "[MEMORY]" not in prompt


def test_relevant_memories_never_exceed_the_requested_limit(memory):
    for i in range(10):
        memory.add_memory(f"The user's favorite number is {i}.", category="preference")
    results = memory.get_relevant_memories("What's my favorite number?", k=3)
    assert len(results) <= 3


def test_forget_command_does_not_delete_an_unrelated_memory(tiny_model, tokenizer, memory):
    fact_id = memory.add_memory("The user's name is Theo.", category="identity")
    agent = get_agent("general", tiny_model, tokenizer, memory=memory)

    reply = agent._handle_memory_command("forget that I like pizza")
    assert "don't have anything" in reply.lower()
    # The unrelated memory must survive an unmatched forget command.
    assert memory.get_memory(fact_id) is not None


# -- 10. existing general-answer tests must remain unchanged -----------------
# (enforced by running the full suite — tests/test_agents.py's existing
# cases are untouched by this change and still pass; nothing here
# duplicates or replaces them.)


# -- explicit memory commands -------------------------------------------------


def test_remember_command_stores_a_memory_without_calling_the_model(tiny_model, tokenizer, memory):
    agent = get_agent("general", tiny_model, tokenizer, memory=memory)
    reply = agent._handle_memory_command("remember that my name is Theo")
    assert reply == "Got it — I'll remember that my name is Theo."
    assert len(memory.all_memories()) == 1
    assert memory.all_memories()[0]["content"] == "my name is Theo"


def test_remember_command_via_respond_returns_deterministic_reply(tiny_model, tokenizer, memory):
    agent = get_agent("general", tiny_model, tokenizer, memory=memory)
    reply = agent.respond("c1", "Please remember that I work as a teacher.")
    assert reply == "Got it — I'll remember that I work as a teacher."
    assert len(memory.all_memories()) == 1


def test_forget_command_removes_the_matching_memory(tiny_model, tokenizer, memory):
    memory.add_memory("The user's name is Theo.", category="identity")
    agent = get_agent("general", tiny_model, tokenizer, memory=memory)

    reply = agent._handle_memory_command("forget that my name is Theo")
    assert "forgotten" in reply.lower()
    assert memory.all_memories() == []


def test_list_memories_command_reports_stored_facts(tiny_model, tokenizer, memory):
    memory.add_memory("The user's name is Theo.", category="identity")
    memory.add_memory("The user's favorite color is blue.", category="preference")
    agent = get_agent("general", tiny_model, tokenizer, memory=memory)

    reply = agent._handle_memory_command("What do you remember about me?")
    assert "Theo" in reply
    assert "blue" in reply


def test_list_memories_command_when_nothing_is_stored(tiny_model, tokenizer, memory):
    agent = get_agent("general", tiny_model, tokenizer, memory=memory)
    reply = agent._handle_memory_command("What do you remember about me?")
    assert "don't have anything" in reply.lower()


def test_ordinary_message_is_not_treated_as_a_command(tiny_model, tokenizer, memory):
    agent = get_agent("general", tiny_model, tokenizer, memory=memory)
    # "Do you remember my name?" is a *question*, not a "remember X"
    # command — it must not be intercepted (and therefore must not store
    # "my name?" as a new memory).
    assert agent._handle_memory_command("Do you remember my name?") is None
    assert agent._handle_memory_command("Remembering is important to me.") is None


def test_respond_stream_handles_remember_command(tiny_model, tokenizer, memory):
    agent = get_agent("general", tiny_model, tokenizer, memory=memory)
    chunks = list(agent.respond_stream("c1", "remember that my favorite food is pizza"))
    assert "".join(chunks) == "Got it — I'll remember that my favorite food is pizza."
    assert len(memory.all_memories()) == 1


def test_a_portuguese_command_stores_and_confirms_in_portuguese(tiny_model, tokenizer, memory):
    """A Portuguese speaker must be able to save a memory in their own
    language — and get a Portuguese confirmation, not an English one."""
    agent = get_agent("general", tiny_model, tokenizer, memory=memory)

    reply = agent._handle_memory_command("lembre que meu nome é Theo")
    assert reply == "Entendi — vou lembrar que meu nome é Theo."
    assert len(memory.all_memories()) == 1

    # And the matching Portuguese forget removes it, confirming in Portuguese.
    forget = agent._handle_memory_command("esqueça meu nome")
    assert forget.startswith("Pronto — esqueci")
    assert memory.all_memories() == []


# -- [MEMORY] block formatting -------------------------------------------------


def test_memory_block_uses_bracket_format_when_facts_are_relevant():
    agent = Agent.__new__(Agent)
    agent.system_prompt = "Base prompt."
    agent.knowledge = None
    ctx = MemoryContext(relevant_facts=[{"content": "The user's name is Theo."}])
    prompt = Agent._build_system_prompt(agent, "What's my name?", ctx)
    assert prompt == "Base prompt.\n\n[MEMORY]\n- The user's name is Theo.\n[/MEMORY]"


def test_no_memory_block_when_no_facts_are_relevant():
    agent = Agent.__new__(Agent)
    agent.system_prompt = "Base prompt."
    agent.knowledge = None
    prompt = Agent._build_system_prompt(agent, "What's my name?", MemoryContext(relevant_facts=[]))
    assert prompt == "Base prompt."


# -- lexical relevance module (unit-level) ------------------------------------


@pytest.mark.parametrize(
    "query,fact,expect_relevant",
    [
        ("What's my name?", "The user's name is Theo.", True),
        ("What is my name again?", "The user's name is Theo.", True),
        ("What's my favorite color?", "The user's favorite color is blue.", True),
        ("What is photosynthesis?", "The user's name is Theo.", False),
        ("What is 9 plus 10?", "The user's favorite color is blue.", False),
        ("Hello!", "The user's name is Theo.", False),
    ],
)
def test_lexical_overlap_score_gates_relevance(query, fact, expect_relevant):
    score = lexical_overlap_score(query, fact)
    is_relevant = score >= DEFAULT_RELEVANCE_THRESHOLD
    assert is_relevant == expect_relevant


def test_tokenize_strips_stopwords_and_punctuation():
    tokens = tokenize("What's my favorite color?")
    assert tokens == {"favorite", "color"}


def test_lexical_overlap_score_is_zero_for_empty_input():
    assert lexical_overlap_score("", "The user's name is Theo.") == 0.0
    assert lexical_overlap_score("What's my name?", "") == 0.0
    assert lexical_overlap_score("", "") == 0.0


# -- command parsing (unit-level) ---------------------------------------------


@pytest.mark.parametrize(
    "text,expected_kind,expected_content",
    [
        ("remember that my name is Theo", "remember", "my name is Theo"),
        ("Remember my name is Theo", "remember", "my name is Theo"),
        ("please remember that I like tea", "remember", "I like tea"),
        ("forget that my name is Theo", "forget", "my name is Theo"),
        ("forget about my name", "forget", "my name"),
        ("Forget my name", "forget", "my name"),
        ("what do you remember about me?", "list", None),
        ("What do you remember?", "list", None),
        # Portuguese — the user talks to Aila in Portuguese, so storing a
        # memory in Portuguese must work exactly like the English form.
        ("lembre que meu nome é Theo", "remember", "meu nome é Theo"),
        ("lembra que eu gosto de azul", "remember", "eu gosto de azul"),
        ("lembre-se de que eu moro em Berlim", "remember", "eu moro em Berlim"),
        ("anote que eu gosto de café", "remember", "eu gosto de café"),
        ("esqueça meu nome", "forget", "meu nome"),
        ("esqueça que eu gosto de azul", "forget", "eu gosto de azul"),
        ("apague meu nome", "forget", "meu nome"),
        ("o que você lembra sobre mim?", "list", None),
        ("do que você se lembra?", "list", None),
        ("o que você sabe sobre mim?", "list", None),
    ],
)
def test_parse_memory_command_matches_expected_patterns(text, expected_kind, expected_content):
    command = parse_memory_command(text)
    assert command is not None
    assert command.kind == expected_kind
    if expected_content is not None:
        assert command.content == expected_content


@pytest.mark.parametrize(
    "text",
    [
        "What's my name?",
        "Do you remember my name?",
        "Hello, how are you?",
        "",
        "   ",
        # A trailing "?" marks a question, never a store/delete command —
        # these used to be captured and stored as (wrong) memories.
        "Remember when we met?",
        "Lembra do filme que vimos?",
        "Você lembra meu nome?",
    ],
)
def test_parse_memory_command_returns_none_for_non_commands(text):
    assert parse_memory_command(text) is None


def test_guess_category_heuristics():
    assert guess_category("my name is Theo") == "identity"
    assert guess_category("I love pizza") == "preference"
    assert guess_category("I'm working on a new project") == "project"
    assert guess_category("always answer in French") == "instruction"
    assert guess_category("I have a dog named Max") == "personal_fact"


# -- store-level CRUD (unit-level, no embedder needed) ------------------------


def test_store_update_fact_changes_content_and_updated_at(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))
    fid = store.add_fact("Old content.", category="other")
    before = store.get_fact(fid)

    ok = store.update_fact(fid, content="New content.")
    assert ok is True
    after = store.get_fact(fid)
    assert after["content"] == "New content."
    assert after["updated_at"] >= before["updated_at"]


def test_store_update_nonexistent_fact_returns_false(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))
    assert store.update_fact(999999, content="X") is False


def test_store_clear_facts_all_vs_scoped(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))
    store.add_fact("a", session_id="s1")
    store.add_fact("b", session_id="s2")
    store.add_fact("c", session_id=None)

    removed = store.clear_facts(session_id="s1")
    assert removed == 1
    remaining = {f["content"] for f in store.get_all_facts()}
    assert remaining == {"b", "c"}

    removed_all = store.clear_facts(session_id=None)
    assert removed_all == 2
    assert store.get_all_facts() == []


def test_semantic_memory_update_reembeds_content(tiny_model, tokenizer, memory):
    fid = memory.add_memory("The user's name is Theo.", category="identity")
    assert memory.update_memory(fid, content="The user's name is Maria.") is True

    assert memory.get_relevant_memories("What's my name?")[0]["content"] == "The user's name is Maria."
    # FAISS should still have exactly one vector for this id, not two.
    assert memory.semantic.index.ntotal == 1


def test_forget_match_threshold_is_reasonably_strict():
    # A destructive command like "forget" shouldn't fire on the faintest
    # possible overlap — sanity-check the threshold isn't set to ~0.
    assert FORGET_MATCH_THRESHOLD > DEFAULT_RELEVANCE_THRESHOLD
