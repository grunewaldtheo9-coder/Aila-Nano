"""ConversationManager: owns the *shape* of a conversation.

The stores already keep the raw turns (memory/conversation_memory.py) and
the long-term facts (memory/manager.py). This layer sits on top and answers
the questions a conversational assistant needs but a raw log can't:

- What are we actually talking about? (active topics)
- What happened earlier that I should keep once the log gets long?
  (an extractive summary of older turns)
- What context should the model see this turn, in priority order?
  (recent turns > summary > relevant memories)

It is deliberately lightweight and CPU-only: topic and summary extraction
are rule-based over the turn text, not model-generated — the ~20M model
can't be trusted to summarise reliably, and a future 50M model can replace
these methods without changing the interface. Nothing here trains or calls
the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Words that never make a useful "topic" on their own.
_TOPIC_STOP = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and", "or",
        "my", "your", "it", "this", "that", "i", "you", "we", "they", "he", "she",
        "with", "for", "on", "in", "at", "do", "does", "did", "have", "has", "want",
        "like", "im", " im", "gonna", "just", "really", "also", "about", "what",
        "how", "why", "when", "where", "which", "who", "can", "could", "would",
        "um", "uma", "o", "os", "as", "de", "da", "do", "que", "e", "eu", "você",
        "vou", "estou", "meu", "minha", "um", "para", "com", "por", "na", "no",
        # conversational filler — never a useful topic
        "nice", "good", "cool", "great", "hey", "hi", "hello", "yeah", "yes", "no",
        "ok", "okay", "lots", "kind", "hope", "will", "get", "got", "some", "one",
        "legal", "massa", "boa", "oi", "olá", "sim", "não", "tudo", "bem",
    }
)

# Capitalised product-ish words and known project terms score highest as
# topics; this small lexicon boosts recall of the things people build/discuss.
_TOPIC_HINTS = frozenset(
    {
        "arduino", "raspberry", "robot", "website", "app", "drone", "sensor",
        "sensors", "ultrasonic", "oled", "screen", "display", "minecraft", "python",
        "esp32", "mega", "uno", "camera", "speaker", "motor", "battery", "game",
        "server", "database", "api", "led", "bluetooth", "wifi",
    }
)

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#-]*")


@dataclass
class ConversationState:
    conversation_id: str
    turn_count: int = 0
    active_topics: list[str] = field(default_factory=list)
    summary: str = ""
    recent: list[dict] = field(default_factory=list)  # recent {role,content} turns


@dataclass
class ContextBundle:
    """Everything the model should condition on this turn, assembled in one
    place so the generation path has a single canonical context source."""
    summary: str = ""
    relevant_facts: list[dict] = field(default_factory=list)
    recent: list[dict] = field(default_factory=list)
    active_topics: list[str] = field(default_factory=list)


class ConversationManager:
    """Wraps the existing memory manager (which owns both the conversation
    store and the long-term facts). Adds topic tracking, summarisation, and
    prioritised context assembly."""

    def __init__(self, memory_manager, recent_turns: int = 8, summarize_after: int = 12):
        self.memory = memory_manager
        self.recent_turns = recent_turns
        self.summarize_after = summarize_after

    # -- history ------------------------------------------------------------

    def add_turn(self, conversation_id: str, role: str, content: str) -> None:
        self.memory.add_turn(conversation_id, role, content)

    def history(self, conversation_id: str, max_turns: int | None = None) -> list[dict]:
        return self.memory.conversation.render_for_prompt(
            conversation_id, max_turns=max_turns or 1000
        )

    # -- topics -------------------------------------------------------------

    def extract_topics(self, turns: list[dict], limit: int = 6) -> list[str]:
        """Rank content words across the conversation by frequency, boosting
        known project/tech terms. Returns the most salient distinct topics."""
        scores: dict[str, float] = {}
        for t in turns:
            for raw in _WORD_RE.findall(t.get("content", "").lower()):
                if len(raw) < 3 or raw in _TOPIC_STOP:
                    continue
                weight = 2.0 if raw in _TOPIC_HINTS else 1.0
                scores[raw] = scores.get(raw, 0.0) + weight
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [w for w, _ in ranked[:limit]]

    # -- summarisation ------------------------------------------------------

    def summarize(self, turns: list[dict], max_points: int = 6) -> str:
        """Extractive, rule-based summary of the *user's* salient statements
        — the facts, decisions and preferences worth carrying forward. Picks
        user turns that look declarative (mention a project/preference or a
        topic hint), most recent first, de-duplicated.

        Not model-generated: this is a deterministic bridge until a trained
        model can summarise; the interface stays the same either way.
        """
        from memory.attributes import extract_attribute

        points: list[str] = []
        seen: set[str] = set()
        seen_attrs: set[str] = set()
        markers = ("i'm building", "im building", "i am building", "my ", "i like",
                   "i want", "i have", "favorite", "favourite", "estou construindo",
                   "meu ", "minha ", "eu gosto", "remember", "using", "it uses",
                   "actually")
        # Newest first, so when the user corrected a fact ("actually my
        # favorite game is Zelda now") the corrected value is seen first and
        # the earlier one is dropped — the summary must show the *current*
        # value, not both (spec: corrections in summary).
        for t in reversed(turns):
            if t.get("role") != "user":
                continue
            text = t.get("content", "").strip()
            low = text.lower()
            salient = any(m in low for m in markers) or any(h in low for h in _TOPIC_HINTS)
            if not salient:
                continue
            key = low
            if key in seen:
                continue
            attr = extract_attribute(text)
            if attr is not None:
                if attr[0] in seen_attrs:
                    continue  # a newer value for this attribute already kept
                seen_attrs.add(attr[0])
            seen.add(key)
            points.append(text.rstrip("."))
            if len(points) >= max_points:
                break
        points.reverse()
        if not points:
            return ""
        return "Earlier in this conversation: " + "; ".join(points) + "."

    # -- state + context ----------------------------------------------------

    def state(self, conversation_id: str) -> ConversationState:
        all_turns = self.history(conversation_id)
        recent = all_turns[-self.recent_turns:]
        summary = ""
        if len(all_turns) > self.summarize_after:
            older = all_turns[: -self.recent_turns]
            summary = self.summarize(older)
        return ConversationState(
            conversation_id=conversation_id,
            turn_count=len(all_turns),
            active_topics=self.extract_topics(all_turns),
            summary=summary,
            recent=recent,
        )

    def assemble(self, conversation_id: str, query: str, max_facts: int = 5) -> "ContextBundle":
        """The canonical context for a generation turn: a summary of older
        turns (when the conversation is long enough), the relevant long-term
        memories for this query, the recent turns, and the active topics.

        This is the single source the agent uses to build the model prompt,
        so summaries and corrected memories actually reach the model. Recent
        raw turns are returned too, but whether they are replayed into the
        prompt is the agent's decision (the 20M model is single-turn, so it
        currently does not — a trained 50M model can)."""
        st = self.state(conversation_id)
        facts = self.memory.get_relevant_memories(query, k=max_facts) if self.memory is not None else []
        return ContextBundle(
            summary=st.summary,
            relevant_facts=facts,
            recent=st.recent,
            active_topics=st.active_topics,
        )

    # -- entities & topics (rebuilt from history, so they survive restart) --

    def entity_tracker(self, conversation_id: str):
        """An EntityTracker replayed over this conversation's history."""
        from conversation.entities import EntityTracker

        et = EntityTracker()
        for i, turn in enumerate(self.history(conversation_id)):
            et.observe(turn.get("content", ""), i)
        return et

    def active_entities(self, conversation_id: str) -> list:
        return self.entity_tracker(conversation_id).active_entities()

    def topic_stack(self, conversation_id: str):
        """A TopicStack replayed over this conversation's user turns. Topic
        candidates come from the entities named in each message."""
        from conversation.entities import extract_entities
        from conversation.topics import TopicStack

        ts = TopicStack()
        for i, turn in enumerate(self.history(conversation_id)):
            if turn.get("role") != "user":
                continue
            content = turn.get("content", "")
            candidates = [surface for surface, _c, _t in extract_entities(content)]
            ts.note(content, i, candidates)
        return ts

    def resolve_reference(self, conversation_id: str, message: str):
        """Resolve a short contextual reference against this conversation.

        Tries, in order: an ordinal/list reference or affirmation ("the
        second one", "yes"), then a pronoun ("it", "isso") against the
        tracked entities. Returns a Resolution; kind == "none" (with
        `ambiguous`/`options` set when candidates exist) means it could not
        be resolved and the caller should ask rather than guess."""
        from conversation.reference import Resolution
        from conversation.reference import resolve_reference as _resolve

        recent = self.history(conversation_id, max_turns=self.recent_turns)
        res = _resolve(message, recent)
        if res.kind != "none":
            return res

        # Pronoun fallback via the entity tracker (finds a pronoun inside
        # the message, e.g. "it" in "why is it better?").
        er = self.entity_tracker(conversation_id).resolve_in_text(message)
        if er.entity is not None:
            conf = "high" if er.confidence >= 0.8 else "medium"
            return Resolution("entity", er.entity.text, conf, reason="pronoun")
        if er.ambiguous:
            return Resolution(
                "none", None, "low",
                options=[c.text for c in er.candidates],
                ambiguous=True, reason=er.reason,
            )
        return res

    def pending_question(self, conversation_id: str):
        """A PendingQuestion if the assistant's last turn asked the user to
        choose or confirm and is still unanswered, else None."""
        from conversation.pending import detect_pending_question

        history = self.history(conversation_id)
        for turn in reversed(history):
            if turn.get("role") == "assistant":
                return detect_pending_question(turn.get("content", ""))
            if turn.get("role") == "user":
                # A user turn after the assistant's question means it's no
                # longer pending (they already replied).
                break
        return None

    def build_conversation_context(self, conversation_id: str, message: str, max_facts: int = 5):
        """Assemble the full ConversationContext for the *incoming* message
        (not yet added to history): topic, entities, pending question,
        resolved reference, intent, summary, and relevant memories."""
        from conversation.context import ConversationContext, classify_intent
        from conversation.entities import extract_entities
        from conversation.pending import resolve_pending

        turn = len(self.history(conversation_id))

        ts = self.topic_stack(conversation_id)
        candidates = [surface for surface, _c, _t in extract_entities(message)]
        topic_event = ts.note(message, turn, candidates)
        current_topic = ts.current.name if ts.current else None
        topic_history = [t.name for t in ts.dormant]

        et = self.entity_tracker(conversation_id)
        et.observe(message, turn)
        active = [e.text for e in et.active_entities()]

        pending = self.pending_question(conversation_id)
        ref = self.resolve_reference(conversation_id, message)
        # A pending clarification answers against exactly the options the
        # assistant offered (clean entity extraction), so it takes priority
        # over the general reference resolver for this turn.
        resolved: str | None = None
        if pending is not None:
            resolved = resolve_pending(pending, message).resolved
        if resolved is None and ref.kind in {"list_item", "entity"}:
            resolved = ref.value

        intent = classify_intent(
            message,
            has_pending=pending is not None,
            topic_event=topic_event,
            reference_kind=ref.kind,
        )

        summary = self.state(conversation_id).summary
        facts = (
            [f["content"] for f in self.memory.get_relevant_memories(message, k=max_facts)]
            if self.memory is not None
            else []
        )

        return ConversationContext(
            conversation_id=conversation_id,
            current_message=message,
            intent=intent,
            current_topic=current_topic,
            topic_history=topic_history,
            active_entities=active,
            resolved_reference=resolved,
            pending_question=pending.text if pending is not None else None,
            summary=summary,
            relevant_memories=facts,
        )

    def build_context_block(self, conversation_id: str, query: str, max_facts: int = 5) -> str:
        """Assemble a compact, prioritised context string for a prompt:
        summary of older turns (if any) + relevant long-term memories. Recent
        turns are passed to the model separately as real chat turns, so they
        are not duplicated here."""
        st = self.state(conversation_id)
        parts: list[str] = []
        if st.summary:
            parts.append(st.summary)
        if self.memory is not None:
            facts = self.memory.get_relevant_memories(query, k=max_facts)
            if facts:
                lines = "\n".join(f"- {f['content']}" for f in facts)
                parts.append("Relevant memories:\n" + lines)
        return "\n\n".join(parts)
