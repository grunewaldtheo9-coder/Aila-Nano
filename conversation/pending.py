"""PendingQuestion: track a question the assistant asked that the user
hasn't answered yet, and resolve the answer when it arrives.

When Aila asks "Which one do you mean, SQLite or PostgreSQL?", the user's
next "PostgreSQL" — or "the second one", or a bare "yes" to a one-option
proposition — answers that specific pending question. Without tracking it,
the short reply is meaningless. This detects an assistant clarification and
resolves the user's reply against exactly the options it offered.

Deterministic, CPU-only, English + Portuguese. Reuses the reference
resolver's option parsing and ordinal handling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from conversation.entities import extract_entities
from conversation.reference import _AFFIRM, _NEGATE, _ordinal_index, parse_options

# The assistant is asking the user to choose / confirm.
_CLARIFY_MARKERS = re.compile(
    r"\b(?:do you mean|which one|which of|did you mean|are you referring to"
    r"|você quer dizer|qual (?:deles|delas|dos dois)|você se refere a|quis dizer)\b",
    re.IGNORECASE,
)


@dataclass
class PendingQuestion:
    text: str                       # the assistant's question
    options: list[str] = field(default_factory=list)  # choices offered, if any
    kind: str = "clarification"     # "clarification" | "confirmation"
    asked_turn: int = -1


@dataclass
class PendingResolution:
    resolved: str | None = None
    confirmed: bool | None = None   # for a yes/no confirmation
    ambiguous: bool = False
    candidates: list[str] = field(default_factory=list)
    reason: str = ""


def detect_pending_question(assistant_text: str, turn: int = -1) -> PendingQuestion | None:
    """Return a PendingQuestion if the assistant's message asks the user to
    choose or confirm, else None."""
    text = (assistant_text or "").strip()
    # Needs a question. Usually it ends with "?", but a numbered-list
    # clarification often puts the "?" on the lead-in line ("Which would you
    # like?\n1. ...\n2. ..."), so accept a "?" anywhere.
    if "?" not in text:
        return None
    is_clarify = bool(_CLARIFY_MARKERS.search(text))
    numbered = parse_options(text)
    if is_clarify:
        # Prefer clean entity extraction for the offered choices — it avoids
        # dragging the question's lead-in ("Which one do you mean") in as an
        # option, and finds the single entity in "Do you mean PostgreSQL?".
        ents = [surface for surface, _c, _t in extract_entities(text)]
        options = ents if ents else numbered
        return PendingQuestion(text=text, options=options, kind="clarification", asked_turn=turn)
    if numbered:
        return PendingQuestion(text=text, options=numbered, kind="clarification", asked_turn=turn)
    return None


def resolve_pending(pending: PendingQuestion, user_message: str) -> PendingResolution:
    """Resolve the user's reply against a pending question's options."""
    msg = (user_message or "").strip()
    low = msg.lower().rstrip("?.! ")

    # 1) Naming an option directly ("PostgreSQL").
    for opt in pending.options:
        if re.search(rf"\b{re.escape(opt.lower())}\b", low):
            return PendingResolution(resolved=opt, confirmed=True)

    # 2) An ordinal against the offered options ("the second one").
    idx = _ordinal_index(msg)
    if idx is not None and pending.options:
        if idx == -1:
            return PendingResolution(resolved=pending.options[-1], confirmed=True)
        if 1 <= idx <= len(pending.options):
            return PendingResolution(resolved=pending.options[idx - 1], confirmed=True)
        return PendingResolution(ambiguous=True, candidates=pending.options, reason="out_of_range")

    # 3) Affirmation / negation of a single-option proposition
    #    ("Do you mean PostgreSQL?" -> "yes"/"no").
    if low in _AFFIRM:
        if len(pending.options) == 1:
            return PendingResolution(resolved=pending.options[0], confirmed=True)
        return PendingResolution(confirmed=True, reason="affirmed_without_single_option")
    if low in _NEGATE:
        return PendingResolution(confirmed=False, reason="rejected")

    # Couldn't tie the reply to the pending question.
    return PendingResolution(reason="unresolved", candidates=pending.options)
