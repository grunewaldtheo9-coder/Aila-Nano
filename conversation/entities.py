"""EntityTracker: remember the things a conversation is about, and resolve
pronouns ("it", "isso") to them.

"PostgreSQL seems better." ... "I like it because it's simpler." — the "it"
means PostgreSQL only because PostgreSQL was the most recently mentioned
entity. This tracker records each entity's mentions (with a lightweight
semantic type) and resolves a pronoun to the single most-recently-mentioned
entity — or reports ambiguity when two entities are equally recent, so the
caller can ask instead of guessing.

Deterministic and CPU-only: entity recognition is a small lexicon plus
capitalised proper-noun capture, not an NLP model. English + Portuguese
pronouns. Nothing here calls the language model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# canonical -> lightweight semantic type. Extensible; unknown proper nouns
# are captured as "proper_noun".
_TYPE_LEXICON: dict[str, str] = {
    # databases / technology
    "postgresql": "technology", "postgres": "technology", "sqlite": "technology",
    "mongodb": "technology", "mysql": "technology", "redis": "technology",
    "python": "technology", "javascript": "technology", "rust": "technology",
    "faiss": "technology", "pytorch": "technology", "tensorflow": "technology",
    "arduino": "technology", "raspberry pi": "technology", "esp32": "technology",
    "docker": "technology", "linux": "technology", "windows": "technology",
    # games
    "minecraft": "game", "zelda": "game", "mario": "game", "roblox": "game",
    "terraria": "game", "fortnite": "game", "tetris": "game",
    # projects / products
    "aila nano": "project", "aila": "project",
}

# Known multi-word entities (matched before single tokens).
_MULTIWORD = sorted(
    (k for k in _TYPE_LEXICON if " " in k), key=lambda s: -len(s)
)

# Pronouns that refer back to a single recent entity.
_SINGULAR_PRONOUNS = frozenset({
    "it", "this", "that", "this one", "that one", "the previous one", "the same one",
    "isso", "isto", "aquilo", "esse", "essa", "esse aí", "aquele", "aquela",
    "ele", "ela", "o mesmo", "a mesma", "essa opção",
})
_PLURAL_PRONOUNS = frozenset({
    "they", "them", "those", "these",
    "eles", "elas", "esses", "essas", "aqueles", "aquelas",
})

# Capitalised tokens that are just sentence starts, not entities.
_COMMON_CAPS = frozenset({
    "I", "The", "A", "An", "It", "This", "That", "You", "We", "They", "He", "She",
    "What", "How", "Why", "When", "Where", "Which", "Who", "Do", "Does", "Is", "Are",
    "My", "Your", "So", "And", "But", "Or", "If", "For", "To", "Of", "On", "In",
    "Eu", "Você", "O", "A", "Os", "As", "Que", "Como", "Por", "Qual", "Meu", "Minha",
})

_CAP_TOKEN = re.compile(r"\b([A-Z][A-Za-z0-9+#.]*)\b")


@dataclass
class Entity:
    text: str
    canonical: str
    entity_type: str
    first_seen_turn: int
    last_seen_turn: int
    mentions: int = 1


@dataclass
class EntityResolution:
    entity: Entity | None = None
    confidence: float = 0.0
    ambiguous: bool = False
    candidates: list[Entity] = field(default_factory=list)
    reason: str = ""


def _canonical(text: str) -> str:
    return text.strip().lower()


def extract_entities(text: str) -> list[tuple[str, str, str]]:
    """Return (surface, canonical, type) for entities found in `text`.
    Lexicon terms are typed; other capitalised proper nouns are
    "proper_noun". Deterministic, order preserved, de-duplicated."""
    found: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    low = text.lower()

    for phrase in _MULTIWORD:
        if re.search(rf"\b{re.escape(phrase)}\b", low):
            canon = phrase
            if canon not in seen:
                seen.add(canon)
                found.append((phrase.title(), canon, _TYPE_LEXICON[phrase]))

    # Character indices where a new sentence starts (start of text, or just
    # after a .!? and whitespace). A capitalised word there is likely just
    # grammatical capitalisation ("Let's ...", "The ...") — only keep it as a
    # proper noun if it's a known entity.
    sentence_starts = {0}
    for m in re.finditer(r"[.!?]\s+", text):
        sentence_starts.add(m.end())

    for m in _CAP_TOKEN.finditer(text):
        surface = m.group(1)
        canon = _canonical(surface)
        if canon in seen:
            continue
        if canon in _TYPE_LEXICON:
            seen.add(canon)
            found.append((surface, canon, _TYPE_LEXICON[canon]))
        elif (
            surface not in _COMMON_CAPS
            and len(surface) >= 2
            and m.start() not in sentence_starts
        ):
            seen.add(canon)
            found.append((surface, canon, "proper_noun"))
    # single-token lexicon words that aren't capitalised (e.g. "python")
    for token in re.findall(r"[a-z0-9+#.]+", low):
        if token in _TYPE_LEXICON and token not in seen:
            seen.add(token)
            found.append((token, token, _TYPE_LEXICON[token]))
    return found


class EntityTracker:
    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}

    def observe(self, text: str, turn: int) -> list[Entity]:
        """Record the entities mentioned in `text` at `turn`. Returns the
        entities touched this turn (in mention order)."""
        touched: list[Entity] = []
        for surface, canon, etype in extract_entities(text):
            ent = self.entities.get(canon)
            if ent is None:
                ent = Entity(
                    text=surface, canonical=canon, entity_type=etype,
                    first_seen_turn=turn, last_seen_turn=turn, mentions=1,
                )
                self.entities[canon] = ent
            else:
                ent.last_seen_turn = turn
                ent.mentions += 1
            touched.append(ent)
        return touched

    def active_entities(self) -> list[Entity]:
        """All tracked entities, most recently mentioned first."""
        return sorted(
            self.entities.values(),
            key=lambda e: (e.last_seen_turn, e.mentions),
            reverse=True,
        )

    def resolve_in_text(self, text: str) -> EntityResolution:
        """Find a pronoun inside a (short) message and resolve it — so
        "why is it better?" resolves "it". Longer pronoun phrases ("this
        one") are matched before single tokens. Returns a no-pronoun result
        when the message contains none."""
        low = (text or "").lower()
        for pron in sorted(_SINGULAR_PRONOUNS | _PLURAL_PRONOUNS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(pron)}\b", low):
                return self.resolve_pronoun(pron)
        return EntityResolution(reason="no_pronoun")

    def resolve_pronoun(self, pronoun: str, entity_type: str | None = None) -> EntityResolution:
        """Resolve a pronoun to the single most-recently-mentioned entity.

        Ambiguous (no resolution) when two entities share the most recent
        turn — e.g. "it" right after "SQLite or PostgreSQL?". A plural
        pronoun ("they") over several recent entities returns them all as an
        (ambiguous) candidate set rather than a single entity."""
        p = pronoun.strip().lower().rstrip("?.! ")
        if p not in _SINGULAR_PRONOUNS and p not in _PLURAL_PRONOUNS:
            return EntityResolution(reason="not_a_pronoun")

        candidates = self.active_entities()
        if entity_type is not None:
            candidates = [e for e in candidates if e.entity_type == entity_type]
        if not candidates:
            return EntityResolution(reason="no_active_entity")

        if p in _PLURAL_PRONOUNS:
            recent_turn = candidates[0].last_seen_turn
            group = [e for e in candidates if e.last_seen_turn == recent_turn]
            if len(group) >= 2:
                return EntityResolution(
                    ambiguous=True, candidates=group, confidence=0.4,
                    reason="plural_refers_to_group",
                )
            return EntityResolution(entity=group[0], confidence=0.7)

        # Singular: the unique most-recent entity, or ambiguity on a tie.
        recent_turn = candidates[0].last_seen_turn
        most_recent = [e for e in candidates if e.last_seen_turn == recent_turn]
        if len(most_recent) == 1:
            return EntityResolution(entity=most_recent[0], confidence=0.9)
        return EntityResolution(
            ambiguous=True, candidates=most_recent, confidence=0.3,
            reason="multiple_active_entities",
        )
