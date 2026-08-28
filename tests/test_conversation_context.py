"""ConversationContext: intent classification and the single, prioritised
context object the response layer conditions on.

Deterministic, offline, English + Portuguese. Also exercises the
ConversationManager.build_conversation_context() integration end to end.
"""

from __future__ import annotations

from conversation.context import ConversationContext, classify_intent
from conversation.manager import ConversationManager


# -- intent classification ------------------------------------------------

def test_empty_message_is_empty():
    assert classify_intent("") == "empty"
    assert classify_intent("   ") == "empty"


def test_correction_is_detected():
    assert classify_intent("actually my favorite game is Zelda") == "correction"
    assert classify_intent("No, I meant PostgreSQL") == "correction"
    assert classify_intent("na verdade eu quis dizer Zelda") == "correction"


def test_bare_yes_is_an_affirmation_without_pending():
    assert classify_intent("yes") == "affirmation"
    assert classify_intent("no") == "negation"


def test_yes_answering_a_pending_question_is_a_clarification_response():
    # The same "yes" means something different when a question is pending.
    assert classify_intent("yes", has_pending=True) == "clarification_response"
    assert classify_intent("PostgreSQL", has_pending=True,
                           reference_kind="entity") == "clarification_response"


def test_continuation_markers():
    assert classify_intent("continue") == "continuation"
    assert classify_intent("go on") == "continuation"
    assert classify_intent("and then?") == "continuation"
    assert classify_intent("continua") == "continuation"


def test_topic_events_win_over_plain_statement():
    assert classify_intent("back to the robot", topic_event="return") == "topic_return"
    assert classify_intent("let's talk about databases", topic_event="switch") == "topic_switch"


def test_greeting_and_farewell_and_smalltalk():
    assert classify_intent("hello") == "greeting"
    assert classify_intent("bye") == "farewell"


def test_reference_without_pending_is_a_reference():
    assert classify_intent("the second one", reference_kind="list_item") == "reference"


def test_plain_question_and_statement():
    assert classify_intent("what is postgresql?") == "question"
    assert classify_intent("I'm building a robot") == "statement"


def test_correction_beats_pending_affirmation():
    # A correction is unambiguous even while a question is pending.
    assert classify_intent("actually no", has_pending=True) == "correction"


# -- ConversationContext dataclass + render -------------------------------

def test_render_drops_empty_sections():
    ctx = ConversationContext(conversation_id="c", current_message="hi")
    assert ctx.render() == ""  # nothing to show


def test_render_includes_populated_sections_in_priority_order():
    ctx = ConversationContext(
        conversation_id="c",
        current_message="why is it better?",
        current_topic="databases",
        active_entities=["PostgreSQL", "SQLite"],
        resolved_reference="PostgreSQL",
        pending_question="Which one, SQLite or PostgreSQL?",
        summary="Earlier in this conversation: the user is choosing a database.",
        relevant_memories=["The user's favorite game is Zelda"],
    )
    block = ctx.render()
    assert block.startswith("[CONTEXT]") and block.endswith("[/CONTEXT]")
    # Reference is prioritised above the entity list.
    assert block.index("referring to: PostgreSQL") < block.index("Active things")
    assert "Current topic: databases" in block
    assert "waiting on an answer" in block
    assert "favorite game is Zelda" in block


def test_active_entities_are_capped_at_five():
    ctx = ConversationContext(
        conversation_id="c", current_message="x",
        active_entities=[f"E{i}" for i in range(9)],
    )
    line = [ln for ln in ctx.render().splitlines() if ln.startswith("Active things")][0]
    assert line.count(",") == 4  # five entities => four separators


# -- manager integration --------------------------------------------------

class _FakeConversationStore:
    def __init__(self):
        self._turns: dict[str, list[dict]] = {}

    def add_turn(self, cid, role, content, agent_type=None):
        self._turns.setdefault(cid, []).append({"role": role, "content": content})

    def render_for_prompt(self, cid, max_turns=1000):
        turns = [t for t in self._turns.get(cid, []) if t["role"] != "system"]
        return turns[-max_turns:]


class _FakeMemory:
    def __init__(self, facts=None):
        self.conversation = _FakeConversationStore()
        self._facts = facts or []

    def add_turn(self, cid, role, content):
        self.conversation.add_turn(cid, role, content)

    def get_relevant_memories(self, query, k=5):
        q = query.lower()
        scored = [f for f in self._facts if any(w in f["content"].lower() for w in q.split())]
        return (scored or self._facts)[:k]


def test_build_context_resolves_pending_clarification():
    cm = ConversationManager(_FakeMemory())
    cid = "c1"
    cm.add_turn(cid, "user", "Which database should I use?")
    cm.add_turn(cid, "assistant", "Which one do you mean, SQLite or PostgreSQL?")

    ctx = cm.build_conversation_context(cid, "the second one")
    assert ctx.pending_question is not None
    assert ctx.resolved_reference == "PostgreSQL"
    assert ctx.intent == "clarification_response"


def test_build_context_answers_pending_by_name():
    cm = ConversationManager(_FakeMemory())
    cid = "c2"
    cm.add_turn(cid, "assistant", "Do you mean PostgreSQL?")
    ctx = cm.build_conversation_context(cid, "yes")
    assert ctx.resolved_reference == "PostgreSQL"
    assert ctx.intent == "clarification_response"


def test_build_context_no_pending_when_user_already_replied():
    cm = ConversationManager(_FakeMemory())
    cid = "c3"
    cm.add_turn(cid, "assistant", "Do you mean PostgreSQL?")
    cm.add_turn(cid, "user", "yes")
    cm.add_turn(cid, "assistant", "Great, PostgreSQL it is.")
    ctx = cm.build_conversation_context(cid, "how do I install it?")
    assert ctx.pending_question is None


def test_build_context_carries_active_entities_and_memories():
    mem = _FakeMemory(facts=[{"content": "The user's favorite game is Zelda"}])
    cm = ConversationManager(mem)
    cid = "c4"
    cm.add_turn(cid, "user", "I'm comparing PostgreSQL and SQLite.")
    ctx = cm.build_conversation_context(cid, "tell me about my favorite game")
    assert "PostgreSQL" in ctx.active_entities
    assert any("Zelda" in m for m in ctx.relevant_memories)
    # render() surfaces them in the prompt block.
    assert "Zelda" in ctx.render()


def test_build_context_classifies_a_correction():
    cm = ConversationManager(_FakeMemory())
    cid = "c5"
    cm.add_turn(cid, "user", "my favorite game is Minecraft")
    cm.add_turn(cid, "assistant", "Nice, Minecraft is great.")
    ctx = cm.build_conversation_context(cid, "actually my favorite game is Zelda")
    assert ctx.intent == "correction"
