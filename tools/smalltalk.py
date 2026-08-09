"""Deterministic replies for very short conversational filler.

Why this is a rule table and not the model's job: greetings and
acknowledgements are the single most common short messages, and a
~20M-parameter model has no reliable behaviour for the ones it wasn't
explicitly fine-tuned on. In real use every unfamiliar filler word
collapsed onto the nearest trained example — "Bro", "Ok" and "nice!" all
came back as "Hi! How can I help you today?", which reads as the model
repeating itself and ignoring what was said (reported by the user:
"he repeats things some time").

Training more filler variants only moves the boundary; there is always
another one. A closed table of exact phrases removes the failure mode
entirely for the phrases it covers, and anything not in the table falls
through to normal handling untouched.

Design constraints that keep this safe:
- Exact match on a normalized phrase (lowercased, punctuation and
  repeated letters trimmed). No substring or prefix matching, so a real
  message that merely *starts* with "ok" ("ok so what is the capital of
  France?") is never intercepted.
- Only messages of at most MAX_SMALLTALK_WORDS words are considered.
- A trailing "?" disqualifies every intent except the ones that are
  *inherently* questions ("How are you?", "Tudo bem?"). So "Ok?" and
  "Nice?" keep going to the normal question path, while the most common
  greeting question of all still gets a real answer.
- Each intent has its own distinct reply, so different inputs visibly
  produce different outputs.
"""

from __future__ import annotations

import re

# Longest phrase in the table is 4 words once normalized ("How's it
# going" -> "how s it going", because stripping the apostrophe splits the
# contraction). Counted on the *normalized* phrase rather than the raw
# message so that asymmetry can't quietly make a table entry
# unreachable. Anything longer is real content, not filler.
MAX_SMALLTALK_WORDS = 4

_PUNCT = re.compile(r"[!.,~\-–—…\"'’?]+")
_REPEATED_CHARS = re.compile(r"(.)\1{2,}")
_FULL_COLLAPSE = re.compile(r"(.)\1+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/whitespace, and collapse stretched
    letters so "okkkk", "ok!!!" and "Ok" are one phrase. Portuguese
    laughter ("kkkkk") collapses to "kk", which the table lists."""
    lowered = (text or "").strip().lower()
    lowered = _PUNCT.sub(" ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return _REPEATED_CHARS.sub(r"\1\1", lowered)


# intent -> (phrases, english reply, portuguese reply).
#
# Replies are intentionally short and each ends by handing the turn back,
# so a filler exchange doesn't dead-end the conversation.
_INTENTS: list[tuple[str, tuple[str, ...], str, str]] = [
    (
        "greeting",
        (
            "hi", "hii", "hello", "helo", "hey", "heyy", "yo", "hiya", "sup",
            "hi there", "hello there", "hey there", "good morning",
            "good afternoon", "good evening", "howdy",
            "oi", "ola", "opa", "e ai", "eai", "bom dia", "boa tarde",
            "boa noite", "fala ai",
        ),
        "Hi! How can I help you today?",
        "Oi! Como posso ajudar você hoje?",
    ),
    (
        "how_are_you",
        (
            "how are you", "how r u", "how are u", "how you doing",
            "how s it going", "hows it going", "you good", "all good",
            "tudo bem", "tudo bom", "como vai", "como voce esta", "beleza",
        ),
        "I'm doing well, thanks! What can I help you with?",
        "Estou bem, obrigado! Com o que posso ajudar?",
    ),
    (
        "acknowledgement",
        (
            "ok", "okay", "k", "alright", "all right", "right",
            "got it", "gotcha", "understood", "i see", "sure", "fine",
            "noted", "makes sense", "fair enough",
            "ta", "ta bom", "tá", "certo", "entendi", "blz",
        ),
        "Got it. Anything else I can help with?",
        "Entendi. Posso ajudar em mais alguma coisa?",
    ),
    (
        "praise",
        (
            "nice", "cool", "great", "awesome", "amazing", "perfect",
            "excellent", "good", "good job", "well done", "very good",
            "impressive", "wow", "sweet", "brilliant",
            "legal", "otimo", "ótimo", "muito bom", "massa", "show",
            "excelente", "perfeito", "top",
        ),
        "Glad that helped! What would you like to do next?",
        "Que bom que ajudou! O que você quer fazer agora?",
    ),
    (
        "thanks",
        (
            "thanks", "thank you", "thanks a lot", "thank u", "thx", "ty",
            "many thanks", "cheers",
            "obrigado", "obrigada", "valeu", "vlw", "muito obrigado",
            "muito obrigada",
        ),
        "You're welcome! Let me know if you need anything else.",
        "De nada! É só me falar se precisar de mais alguma coisa.",
    ),
    (
        "address",
        (
            "bro", "bruh", "dude", "man", "buddy", "mate", "friend",
            "hey bro", "hey man", "yo bro",
            "mano", "cara", "velho", "brother", "irmao",
        ),
        "I'm here! What do you need?",
        "Estou aqui! Do que você precisa?",
    ),
    (
        "laughter",
        (
            "lol", "lmao", "haha", "hahaha", "hehe", "hah", "kk", "rs",
            "rsrs", "kkk",
        ),
        "Glad you're enjoying it! Anything I can help with?",
        "Que bom que você gostou! Posso ajudar em algo?",
    ),
    (
        "agreement",
        ("yes", "yeah", "yep", "yup", "sim", "isso", "exatamente", "exactly"),
        "Understood. What would you like next?",
        "Entendido. O que você quer agora?",
    ),
    (
        "disagreement",
        ("no", "nope", "nah", "not really", "nao", "não", "nada"),
        "No problem. Let me know whenever you need something.",
        "Sem problema. É só me chamar quando precisar de algo.",
    ),
    (
        "farewell",
        (
            "bye", "goodbye", "good bye", "see you", "see ya", "later",
            "good night", "that is all", "thats all",
            "tchau", "ate logo", "adeus", "ate mais", "falou",
        ),
        "Goodbye! Have a great day.",
        "Tchau! Tenha um ótimo dia.",
    ),
]

# Flattened for lookup: phrase -> (intent, en, pt). Built once at import.
_PHRASE_TABLE: dict[str, tuple[str, str, str]] = {}
for _intent, _phrases, _en, _pt in _INTENTS:
    for _phrase in _phrases:
        _PHRASE_TABLE.setdefault(_phrase, (_intent, _en, _pt))

# Intents whose phrases are questions in their own right, and so stay
# matchable with a trailing "?".
_QUESTION_INTENTS: frozenset[str] = frozenset({"how_are_you"})

# Phrases that are unambiguously Portuguese. The general language
# detector (webresearch.pipeline.detect_language) works on sentence-level
# evidence and can't call a single bare word like "beleza", so the table
# answers for itself: a Portuguese greeting always gets a Portuguese
# reply, whatever the detector said.
_PT_PHRASES: frozenset[str] = frozenset(
    {
        "oi", "ola", "opa", "e ai", "eai", "bom dia", "boa tarde", "boa noite",
        "fala ai", "tudo bem", "tudo bom", "como vai", "como voce esta",
        "beleza", "ta", "ta bom", "tá", "certo", "entendi", "blz",
        "legal", "otimo", "ótimo", "muito bom", "massa", "excelente",
        "perfeito", "obrigado", "obrigada", "valeu", "vlw", "muito obrigado",
        "muito obrigada", "mano", "cara", "velho", "irmao", "rs", "rsrs",
        "kk", "kkk", "sim", "isso", "exatamente", "nao", "não", "nada",
        "tchau", "ate logo", "adeus", "ate mais", "falou",
    }
)


def match_smalltalk(message: str, language: str = "en") -> tuple[str, str] | None:
    """Return (intent, reply) for a recognized filler phrase, else None.

    None means "not filler" — the caller must carry on with its normal
    routing. Never matches a question or a message longer than
    MAX_SMALLTALK_WORDS words.
    """
    raw = (message or "").strip()
    if not raw:
        return None

    phrase = normalize(raw)
    if not phrase or len(phrase.split()) > MAX_SMALLTALK_WORDS:
        return None
    entry = _PHRASE_TABLE.get(phrase)
    if entry is None:
        # "okkkk" normalizes to "okk" (runs are trimmed to two, not one,
        # so Portuguese laughter "kkkkk" -> "kk" survives as its own
        # phrase). Collapsing runs completely is the second attempt, and
        # only reached when the two-letter form matched nothing.
        entry = _PHRASE_TABLE.get(_FULL_COLLAPSE.sub(r"\1", phrase))
    if entry is None:
        return None
    intent, en, pt = entry

    # A question mark means the user is asking, not filling. "Ok?" and
    # "Nice?" must reach the normal question path — but "How are you?"
    # and "Tudo bem?" are questions *by nature*, and refusing those sent
    # the single most common opening line back to generation.
    if raw.rstrip().endswith("?") and intent not in _QUESTION_INTENTS:
        return None

    if phrase in _PT_PHRASES:
        language = "pt"
    return intent, (pt if language == "pt" else en)
