"""ConversationContext: one structured object gathering everything the
response layer (or a future model) needs to answer the current message,
assembled in priority order.

It bundles the pieces the other conversation components produce — current
topic and topic history (TopicStack), active entities (EntityTracker), any
pending question (pending.py), the resolved reference for this message
(ReferenceResolver), the classified user intent, the conversation summary,
and the relevant long-term memories — so a caller has a single, compact,
prioritised view instead of calling six components by hand.

`render()` turns it into a small text block for a model prompt, keeping the
highest-priority information (current message, active reference, topic,
entities, pending question) and dropping anything empty. Deterministic and
CPU-only; nothing here calls the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# User-message intent categories. Deliberately coarse and deterministic.
_CORRECTION_RE = re.compile(
    r"^\s*(?:actually|no,?\s+i\s+meant|i\s+meant|correction:|wait,?\s+no"
    r"|na\s+verdade|n[aã]o,?\s+(?:eu\s+)?quis\s+dizer|corre[cç][aã]o:)\b",
    re.IGNORECASE,
)
_CONTINUE_RE = re.compile(
    r"^\s*(?:continue|go on|keep going|and then|what'?s next|next"
    r"|continua|continue|vai em frente|e depois|o que vem depois)\s*[.?!]*\s*$",
    re.IGNORECASE,
)


def classify_intent(
    message: str,
    *,
    has_pending: bool = False,
    topic_event: str | None = None,
    reference_kind: str | None = None,
) -> str:
    """Classify the user's message. Context matters: a bare "yes" answering
    a pending question is a clarification_response, not a lone affirmation."""
    from conversation.reference import _AFFIRM, _NEGATE
    from tools.smalltalk import match_smalltalk

    text = (message or "").strip()
    if not text:
        return "empty"
    low = text.lower().rstrip("?.! ")

    if _CORRECTION_RE.search(text):
        return "correction"
    if has_pending and (low in _AFFIRM or low in _NEGATE or reference_kind in {"list_item", "entity"}):
        return "clarification_response"
    if _CONTINUE_RE.match(text):
        return "continuation"
    if topic_event == "return":
        return "topic_return"
    if topic_event == "switch":
        return "topic_switch"
    # A bare "yes"/"no" is an affirmation/negation, not small talk — check it
    # before the smalltalk matcher (which also accepts acknowledgements).
    if low in _AFFIRM:
        return "affirmation"
    if low in _NEGATE:
        return "negation"
    sm = match_smalltalk(text)
    if sm is not None:
        return {"greeting": "greeting", "farewell": "farewell"}.get(sm[0], "small_talk")
    if reference_kind in {"list_item", "entity"}:
        return "reference"
    if text.rstrip().endswith("?"):
        return "question"
    return "statement"


@dataclass
class ConversationContext:
    conversation_id: str
    current_message: str
    intent: str = "statement"
    current_topic: str | None = None
    topic_history: list[str] = field(default_factory=list)  # dormant topic names
    active_entities: list[str] = field(default_factory=list)
    resolved_reference: str | None = None  # value the current reference points to
    pending_question: str | None = None
    summary: str = ""
    relevant_memories: list[str] = field(default_factory=list)

    def render(self) -> str:
        """A compact, prioritised context block for a model prompt. Empty
        sections are dropped so the prompt stays small (context priority)."""
        lines: list[str] = []
        if self.current_topic:
            lines.append(f"Current topic: {self.current_topic}")
        if self.resolved_reference:
            lines.append(f"The user is referring to: {self.resolved_reference}")
        if self.pending_question:
            lines.append(f"You are waiting on an answer to: {self.pending_question}")
        if self.active_entities:
            lines.append("Active things: " + ", ".join(self.active_entities[:5]))
        if self.summary:
            lines.append(self.summary)
        if self.relevant_memories:
            lines.append("Relevant memories:\n" + "\n".join(f"- {m}" for m in self.relevant_memories))
        if not lines:
            return ""
        return "[CONTEXT]\n" + "\n".join(lines) + "\n[/CONTEXT]"
