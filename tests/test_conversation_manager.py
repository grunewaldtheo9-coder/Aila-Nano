"""ConversationManager: topic tracking, summarisation of older turns,
recent-context preservation, and prioritised context assembly."""

from __future__ import annotations

from conversation.manager import ConversationManager


class _FakeConversationStore:
    def __init__(self):
        self._turns: dict[str, list[dict]] = {}

    def add_turn(self, cid, role, content, agent_type=None):
        self._turns.setdefault(cid, []).append({"role": role, "content": content})

    def render_for_prompt(self, cid, max_turns=1000):
        turns = [t for t in self._turns.get(cid, []) if t["role"] != "system"]
        return turns[-max_turns:]


class _FakeMemory:
    """Minimal stand-in for MemoryManager: a conversation store plus a
    trivial relevance search over stored facts."""

    def __init__(self, facts=None):
        self.conversation = _FakeConversationStore()
        self._facts = facts or []

    def add_turn(self, cid, role, content):
        self.conversation.add_turn(cid, role, content)

    def get_relevant_memories(self, query, k=5):
        q = query.lower()
        scored = [f for f in self._facts if any(w in f["content"].lower() for w in q.split())]
        return (scored or self._facts)[:k]


def _robot_conversation(cm, cid):
    turns = [
        ("user", "Hey!"),
        ("assistant", "Hey! What's up?"),
        ("user", "I'm building a robot."),
        ("assistant", "Cool! What kind?"),
        ("user", "It uses an Arduino Mega."),
        ("assistant", "Nice, lots of I/O."),
        ("user", "It also has ultrasonic sensors."),
        ("assistant", "Good for obstacle detection."),
        ("user", "And I want an OLED screen."),
        ("assistant", "Great for status messages."),
        ("user", "The weather is nice today."),
        ("assistant", "Nice!"),
        ("user", "I had a sandwich for lunch."),
        ("assistant", "Hope it was good."),
        ("user", "What am I building?"),
    ]
    for role, content in turns:
        cm.add_turn(cid, role, content)


def test_extracts_project_topics():
    cm = ConversationManager(_FakeMemory())
    _robot_conversation(cm, "c1")
    topics = cm.state("c1").active_topics
    # The salient project terms should surface above filler like "sandwich".
    assert "arduino" in topics or "robot" in topics
    assert "ultrasonic" in topics or "oled" in topics


def test_long_conversation_gets_a_summary_that_keeps_project_facts():
    cm = ConversationManager(_FakeMemory(), recent_turns=4, summarize_after=8)
    _robot_conversation(cm, "c1")
    st = cm.state("c1")
    assert st.turn_count > 8
    assert st.summary  # a summary was created for the older turns
    low = st.summary.lower()
    # Important project facts survive...
    assert "robot" in low or "arduino" in low
    # ...and pure filler is dropped.
    assert "sandwich" not in low


def test_recent_turns_are_preserved_verbatim():
    cm = ConversationManager(_FakeMemory(), recent_turns=3)
    _robot_conversation(cm, "c1")
    st = cm.state("c1")
    assert len(st.recent) == 3
    assert st.recent[-1]["content"] == "What am I building?"


def test_short_conversation_has_no_summary():
    cm = ConversationManager(_FakeMemory(), summarize_after=12)
    for role, content in [("user", "Hi"), ("assistant", "Hey!"), ("user", "How are you?")]:
        cm.add_turn("c1", role, content)
    assert cm.state("c1").summary == ""


def test_context_block_includes_summary_and_relevant_memories():
    mem = _FakeMemory(facts=[{"content": "The user is building an Arduino robot."}])
    cm = ConversationManager(mem, recent_turns=4, summarize_after=8)
    _robot_conversation(cm, "c1")
    block = cm.build_context_block("c1", query="robot project")
    assert "Relevant memories:" in block
    assert "Arduino robot" in block
