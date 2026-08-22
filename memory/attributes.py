"""Extract a normalized *attribute key* from a remembered statement.

Correction ("Actually, my favorite game is Zelda now" replacing Minecraft)
needs a way to know two statements are about the *same* fact. A free-text
memory can't express that, so this maps statements to a stable key:

    "my favorite game is Minecraft"  -> ("favorite_game", "Minecraft")
    "my name is Theo"                -> ("name", "Theo")
    "I'm building an Arduino robot"  -> ("project", "an Arduino robot")
    "meu jogo favorito é Zelda"      -> ("favorite_game", "Zelda")

When a new memory shares an existing memory's key, the new one supersedes
the old (last-writer-wins) instead of both being kept as current. A
statement with no recognizable key (e.g. "I have a dog named Max") returns
None and is stored as an ordinary, non-versioned memory.

Deliberately conservative: only clear, declarative "this attribute is that
value" statements get a key. English and Portuguese.
"""

from __future__ import annotations

import re

# key patterns -> a function producing (attribute_key, value). Order matters;
# the first match wins.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "my favorite game is X" / "my favourite colour is X"
    (re.compile(r"\b(?:my\s+)?favou?rite\s+(\w+)\s+is\s+(.+)$", re.IGNORECASE), "favorite_{0}"),
    # "meu jogo favorito é X" / "minha cor favorita e X"
    (re.compile(r"\b(?:meu|minha)\s+(\w+)\s+favorit[oa]\s+(?:é|e)\s+(.+)$", re.IGNORECASE), "favorite_{0}"),
    # "my name is X" / "meu nome é X" / "call me X"
    (re.compile(r"\bmy\s+name\s+is\s+(.+)$", re.IGNORECASE), "name"),
    (re.compile(r"\bmeu\s+nome\s+(?:é|e)\s+(.+)$", re.IGNORECASE), "name"),
    (re.compile(r"\b(?:you\s+can\s+)?call\s+me\s+(.+)$", re.IGNORECASE), "name"),
    (re.compile(r"\bpode\s+me\s+chamar\s+de\s+(.+)$", re.IGNORECASE), "name"),
    # "I'm building X" / "my project is X"
    (re.compile(r"\bi(?:'m|\s+am)\s+building\s+(.+)$", re.IGNORECASE), "project"),
    (re.compile(r"\bmy\s+project\s+is\s+(.+)$", re.IGNORECASE), "project"),
    (re.compile(r"\bestou\s+construindo\s+(.+)$", re.IGNORECASE), "project"),
    (re.compile(r"\bmeu\s+projeto\s+(?:é|e)\s+(.+)$", re.IGNORECASE), "project"),
]

_PLURALS = {"games": "game", "movies": "movie", "colours": "colour", "colors": "color",
            "jogos": "jogo", "filmes": "filme", "cores": "cor"}


def _clean_value(value: str) -> str:
    return value.strip().rstrip(".!?").strip()


def extract_attribute(content: str) -> tuple[str, str] | None:
    """Return (attribute_key, value) for a declarative attribute statement,
    else None."""
    text = (content or "").strip()
    if not text:
        return None
    for pattern, key_template in _PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        if "{0}" in key_template:
            noun = m.group(1).lower()
            noun = _PLURALS.get(noun, noun)
            key = key_template.format(noun)
            value = _clean_value(m.group(2))
        else:
            key = key_template
            value = _clean_value(m.group(1))
        if not value:
            return None
        return key, value
    return None


# Key-only patterns for a *forget* request, where there is no value —
# "forget my favorite game" -> "favorite_game", "forget my name" -> "name".
_KEY_ONLY: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:my\s+)?favou?rite\s+(\w+)\b", re.IGNORECASE), "favorite_{0}"),
    (re.compile(r"\b(?:meu|minha)\s+(\w+)\s+favorit[oa]\b", re.IGNORECASE), "favorite_{0}"),
    (re.compile(r"\b(?:my\s+)?name\b", re.IGNORECASE), "name"),
    (re.compile(r"\b(?:meu\s+)?nome\b", re.IGNORECASE), "name"),
    (re.compile(r"\b(?:my\s+)?project\b", re.IGNORECASE), "project"),
    (re.compile(r"\b(?:meu\s+)?projeto\b", re.IGNORECASE), "project"),
]


def extract_attribute_key(content: str) -> str | None:
    """Return just the attribute key from a phrase that names an attribute
    without a value ("my favorite game", "my name") — for forget requests."""
    text = (content or "").strip()
    if not text:
        return None
    # A full "attr is value" statement also names a key.
    full = extract_attribute(text)
    if full:
        return full[0]
    for pattern, key_template in _KEY_ONLY:
        m = pattern.search(text)
        if not m:
            continue
        if "{0}" in key_template:
            noun = m.group(1).lower()
            noun = _PLURALS.get(noun, noun)
            return key_template.format(noun)
        return key_template
    return None


# Human-readable labels for the audit / memory listing.
_KEY_LABELS = {"name": "Name", "project": "Project"}


def label_for_key(key: str) -> str:
    if key in _KEY_LABELS:
        return _KEY_LABELS[key]
    if key.startswith("favorite_"):
        return "Favorite " + key[len("favorite_"):]
    return key.replace("_", " ").title()
