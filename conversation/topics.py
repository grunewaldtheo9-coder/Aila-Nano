"""TopicStack: what the conversation is about now, and what it was about
before — so "back to Aila Nano" can restore an earlier thread.

A conversation drifts: the user talks about Aila Nano, detours into
Minecraft, then says "anyway, back to Aila Nano". A flat "current topic"
loses the earlier thread. This keeps an ordered stack — the last entry is
current, earlier entries are dormant — and:

- activates a topic (new or resurfaced) and moves it to current,
- detects an explicit return ("back to X", "voltando para X") and restores
  that topic,
- switches only on an explicit topic introduction or a transition marker
  ("by the way", "anyway"), so ordinary follow-ups ("and how fast is it?")
  stay on the current topic.

Deterministic, CPU-only, English + Portuguese. It tracks topic *names* the
caller supplies (from entities / topic extraction); it does not itself
decide what a "topic" is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# "back to X" / "voltando para X" — capture the target topic name.
_RETURN_RE = re.compile(
    r"\b(?:back to|going back to|return to|let'?s go back to|returning to|about)\s+(.+?)(?:\s+again)?[.?!]*$"
    r"|\b(?:voltando (?:para|ao|pro|no)|vamos voltar (?:para|ao|pro)|retornando ao|sobre)\s+(.+?)(?:\s+de novo)?[.?!]*$",
    re.IGNORECASE,
)
# A bare "let's go back" / "previous topic" with no explicit target.
_RETURN_BARE = re.compile(
    r"\b(?:go(?:ing)? back|previous topic|earlier topic|where we left off"
    r"|voltar ao assunto anterior|assunto anterior|voltando naquele assunto)\b",
    re.IGNORECASE,
)
# Transition markers that signal an intentional subject change.
_TRANSITION_RE = re.compile(
    r"\b(?:by the way|anyway|on another note|another thing|also|speaking of"
    r"|a prop[oó]sito|mudando de assunto|outra coisa|falando nisso)\b",
    re.IGNORECASE,
)
# Explicit "let's talk about X" / "what about X" topic introductions.
_INTRO_RE = re.compile(
    r"\b(?:let'?s talk about|talk about|what about|how about|tell me about"
    r"|vamos falar (?:de|sobre)|falar sobre|e sobre|me fala sobre)\b",
    re.IGNORECASE,
)


def _canonical(name: str) -> str:
    return name.strip().lower().rstrip("?.!")


@dataclass
class Topic:
    name: str
    canonical: str
    started_turn: int
    last_active_turn: int


class TopicStack:
    def __init__(self) -> None:
        # Order of activation; the last entry is the current topic.
        self.stack: list[Topic] = []

    @property
    def current(self) -> Topic | None:
        return self.stack[-1] if self.stack else None

    @property
    def previous(self) -> Topic | None:
        return self.stack[-2] if len(self.stack) >= 2 else None

    @property
    def dormant(self) -> list[Topic]:
        """Every topic except the current one, most-recently-active first."""
        return list(reversed(self.stack[:-1]))

    def _find(self, name: str) -> Topic | None:
        canon = _canonical(name)
        for t in self.stack:
            if t.canonical == canon or canon in t.canonical or t.canonical in canon:
                return t
        return None

    def activate(self, name: str, turn: int) -> Topic:
        """Make `name` the current topic — resurfacing it (moved to the top,
        keeping its history) if it already exists, else pushing it new."""
        existing = self._find(name)
        if existing is not None:
            existing.last_active_turn = turn
            self.stack.remove(existing)
            self.stack.append(existing)
            return existing
        topic = Topic(name=name.strip(), canonical=_canonical(name), started_turn=turn, last_active_turn=turn)
        self.stack.append(topic)
        return topic

    # -- message-driven updates --------------------------------------------

    # Phrases that mean "the earlier topic" rather than naming a topic.
    _META_TARGETS = frozenset({
        "the previous topic", "previous topic", "the earlier topic", "earlier topic",
        "the previous one", "previous one", "the last topic", "last topic",
        "o assunto anterior", "assunto anterior", "o anterior", "o tópico anterior",
    })

    def detect_return(self, message: str) -> str | None:
        """Return the topic name the user asked to go back to, "" for a bare
        "let's go back" / "back to the previous topic" with no explicit
        target, or None if this isn't a return request."""
        m = _RETURN_RE.search(message.strip())
        if m:
            target = (m.group(1) or m.group(2) or "").strip()
            if _canonical(target) in self._META_TARGETS:
                return ""
            return target
        if _RETURN_BARE.search(message):
            return ""
        return None

    def note(self, message: str, turn: int, candidate_topics: list[str]) -> str:
        """Update the stack for a user message. Returns the event:
        "return" (restored an earlier topic), "switch" (new/changed topic),
        or "continue" (stayed on the current topic — the default for
        follow-ups). Conservative: only an explicit return, topic
        introduction, or transition marker changes the current topic."""
        target = self.detect_return(message)
        if target is not None:
            if target and self._find(target) is not None:
                self.activate(target, turn)
                return "return"
            if target and candidate_topics:
                # named a target we don't have yet — treat as a switch to it
                self.activate(target, turn)
                return "switch"
            if not target and self.previous is not None:
                # bare "go back" -> the previous topic
                self.activate(self.previous.name, turn)
                return "return"
            return "continue"

        wants_switch = bool(_INTRO_RE.search(message) or _TRANSITION_RE.search(message))
        if wants_switch and candidate_topics:
            new = candidate_topics[0]
            if self.current is None or self._find(new) is None:
                self.activate(new, turn)
                return "switch"
            self.activate(new, turn)
            return "switch"

        # No current topic yet: adopt the most salient candidate.
        if self.current is None and candidate_topics:
            self.activate(candidate_topics[0], turn)
            return "switch"

        # Otherwise this is a follow-up: keep the current topic, refresh it.
        if self.current is not None:
            self.current.last_active_turn = turn
        return "continue"
