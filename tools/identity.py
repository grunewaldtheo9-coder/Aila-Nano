"""Deterministic answers to questions about Aila itself.

These questions are the ones a user asks first ("who made you?", "what
are you?"), they are the ones the project most needs to get right, and
they are the only questions the router deliberately keeps *away* from
the web (tools/router.py's _SELF_REFERENCE gate) — the web does not know
what Aila Nano is, and a search result about some other "Aila" would be
worse than no answer.

Until now that left generation as the only path, and at ~20M parameters
generation garbles them: "Who created Aila Company Solutions?" came back
as "Aila Company Solutions is the em Hameswald Benkendorf, to help answer
questions" in real use — the right facts, shredded. The facts are fixed
and known, so serving them from a table is strictly better: it cannot
garble, cannot hallucinate, and cannot drift from the identity training
data (tests/test_identity_facts_consistency.py checks the two agree).

Matching is intent-based and conservative: a message must look like a
question, mention Aila (or address "you"), and match one intent's
keywords. A first-person possessive ("my") disqualifies it outright so
"Do you remember my name?" stays a memory question, never an identity
one.
"""

from __future__ import annotations

import re

# Canonical facts. Single source of truth for the deterministic answers;
# datasets/scripts/generate_aila_identity_data.py teaches the same facts
# to the model itself.
COMPANY = "Aila Company Solutions"
FOUNDERS = "Theo Grunewald Hames and Guilherme Grunewald Benkendorf"
PARAMETER_COUNT = "about 20 million"

_ANSWERS_EN: dict[str, str] = {
    "creator": (
        f"I was created by {COMPANY}, founded by {FOUNDERS}. "
        "I'm Aila Nano, an original small language model — my architecture, "
        "tokenizer and weights were all built and trained from scratch, not "
        "wrapped around another company's model."
    ),
    "company_founders": (
        f"{COMPANY} was founded by {FOUNDERS}."
    ),
    "what_are_you": (
        f"I'm Aila Nano, a small language model with {PARAMETER_COUNT} parameters, "
        f"built and trained from scratch by {COMPANY}. I run on ordinary "
        "hardware — no GPU required."
    ),
    "name": (
        "My name is Aila Nano. The 'Aila' comes from "
        f"{COMPANY}, my creator, and 'Nano' reflects how small I am — "
        f"{PARAMETER_COUNT} parameters."
    ),
    "size": (
        f"I have {PARAMETER_COUNT} parameters, which makes me a small "
        "language model (an SLM) rather than a large one."
    ),
    "capabilities": (
        "I can chat, answer questions, do arithmetic exactly, remember "
        "things you tell me (say \"remember that ...\"), and look things up "
        "on the web when a question needs current facts. I'm a small model, "
        "so I'm best at short, direct questions."
    ),
}

_ANSWERS_PT: dict[str, str] = {
    "creator": (
        f"Fui criado pela {COMPANY}, fundada por {FOUNDERS}. "
        "Sou a Aila Nano, um modelo de linguagem pequeno e original — minha "
        "arquitetura, meu tokenizador e meus pesos foram construídos e "
        "treinados do zero."
    ),
    "company_founders": (
        f"A {COMPANY} foi fundada por {FOUNDERS}."
    ),
    "what_are_you": (
        f"Sou a Aila Nano, um modelo de linguagem pequeno com cerca de 20 milhões "
        f"de parâmetros, criado e treinado do zero pela {COMPANY}. Eu rodo em "
        "hardware comum — não preciso de placa de vídeo."
    ),
    "name": (
        f"Meu nome é Aila Nano. 'Aila' vem da {COMPANY}, minha criadora, e "
        "'Nano' reflete o meu tamanho — cerca de 20 milhões de parâmetros."
    ),
    "size": (
        "Tenho cerca de 20 milhões de parâmetros, o que me torna um modelo de "
        "linguagem pequeno (um SLM), não um grande."
    ),
    "capabilities": (
        "Posso conversar, responder perguntas, fazer contas com exatidão, "
        "lembrar coisas que você me contar (diga \"remember that ...\") e "
        "pesquisar na web quando a pergunta precisa de dados atuais. Sou um "
        "modelo pequeno, então me saio melhor com perguntas curtas e diretas."
    ),
}

# Intent detection. Ordered: the first pattern that matches wins, so the
# more specific intents come first (a question about the *company* must
# beat the general "who created you" pattern).
#
# Every pattern names its object explicitly ("...created you", "...your
# name") rather than leaving a wildcard gap. A loose gap looks harmless
# and isn't: "Who do you think made the pyramids?" mentions "you" and
# would match "who ... made", answering a pyramid question with Aila's
# creator. Anchored objects make a false positive essentially impossible.
_MAKE_VERBS_EN = r"created|made|built|developed|designed|trained|trains|owns|trained"
_MAKE_VERBS_PT = r"criou|fez|construiu|desenvolveu|treinou|projetou"
_AILA = r"(?:the\s+)?aila(?:\s+nano)?"

_INTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "company_founders",
        re.compile(
            rf"\bwho\s+(?:{_MAKE_VERBS_EN}|founded|started)\s+(?:the\s+)?aila\s+company\b"
            rf"|\bwho\s+(?:are\s+|is\s+)?the\s+(?:founders?|owners?)\s+of\s+(?:the\s+)?aila\b"
            rf"|\bquem\s+(?:fundou|{_MAKE_VERBS_PT})\s+a?\s*aila\s+company\b"
            rf"|\bfounders?\s+of\s+(?:the\s+)?aila\s+company\b",
            re.IGNORECASE,
        ),
    ),
    (
        "creator",
        re.compile(
            rf"\bwho\s+(?:{_MAKE_VERBS_EN})\s+(?:you|{_AILA})\b"
            rf"|\bwho\s+(?:is|are)\s+(?:behind|responsible\s+for)\s+(?:you|{_AILA})\b"
            rf"|\b(?:you\s+were|were\s+you)\s+(?:{_MAKE_VERBS_EN})\s+by\s+who(?:m)?\b"
            rf"|\bwho\s+(?:is|are)\s+your\s+(?:creators?|makers?|developers?|owners?)\b"
            rf"|\bquem\s+(?:{_MAKE_VERBS_PT})\s+(?:voc[eê]|a?\s*aila(?:\s+nano)?)\b"
            rf"|\bquem\s+[eé]\s+(?:o\s+)?seu\s+criador\b",
            re.IGNORECASE,
        ),
    ),
    (
        "size",
        re.compile(
            r"\bhow\s+many\s+(?:parameters|params)\b"
            r"|\bhow\s+(?:big|large)\s+(?:are|is)\s+(?:you|aila)\b"
            r"|\bquantos\s+par[aâ]metros\b"
            r"|\bqual\s+(?:é\s+)?o\s+seu\s+tamanho\b",
            re.IGNORECASE,
        ),
    ),
    (
        "capabilities",
        re.compile(
            r"\bwhat\s+can\s+you\s+(?:do|help)\b"
            r"|\bwhat\s+do\s+you\s+do\b"
            r"|\bwhat\s+are\s+you\s+able\s+to\s+do\b"
            r"|\bhow\s+can\s+you\s+help\b"
            r"|\bo\s+que\s+voc[eê]\s+(?:faz|pode\s+fazer|consegue\s+fazer|sabe\s+fazer)\b"
            r"|\bcomo\s+voc[eê]\s+pode\s+(?:me\s+)?ajudar\b",
            re.IGNORECASE,
        ),
    ),
    (
        "name",
        re.compile(
            r"\bwhat(?:'s|s| is|\s+is)\s+your\s+name\b"
            r"|\bwhat\s+are\s+you\s+called\b"
            r"|\bqual\s+(?:é\s+|e\s+)?(?:o\s+)?seu\s+nome\b"
            r"|\bcomo\s+voc[eê]\s+se\s+chama\b",
            re.IGNORECASE,
        ),
    ),
    (
        "what_are_you",
        re.compile(
            rf"\bwhat\s+(?:exactly\s+)?(?:is|are)\s+(?:you|{_AILA})\b"
            rf"|\bwho\s+(?:is|are)\s+(?:you|{_AILA})\b"
            rf"|\bwhat\s+{_AILA}\s+is\b"
            rf"|\b(?:tell\s+me\s+about|describe)\s+(?:you|yourself|{_AILA})\b"
            rf"|\bo\s+que\s+[eé]\s+a?\s*aila(?:\s+nano)?\b"
            rf"|\b(?:fale\s+sobre|descreva)\s+(?:voc[eê]|a?\s*aila(?:\s+nano)?)\b",
            re.IGNORECASE,
        ),
    ),
]

# The message must be about Aila: it names Aila, or addresses "you".
_ABOUT_AILA = re.compile(r"\b(aila|you|your|yours|yourself|voc[eê]|seu|sua|te)\b", re.IGNORECASE)

# ...but never when it's about the *user*. "Do you remember my name?"
# addresses Aila yet asks about the user — that belongs to memory.
_ABOUT_USER = re.compile(r"\b(my|mine|meu|minha|meus|minhas)\b", re.IGNORECASE)

_QUESTION_SHAPE = re.compile(
    r"^\s*(who|what|which|whose|how|why|are|is|do|does|did|can|tell|describe|"
    r"quem|qual|quais|o\s+que|como|por\s+que|fale|descreva|voc[eê])\b",
    re.IGNORECASE,
)


def match_identity_question(message: str, language: str = "en") -> tuple[str, str] | None:
    """Return (intent, answer) when `message` asks about Aila itself,
    else None (caller continues with its normal routing).

    Returns None for anything about the user, anything not shaped like a
    question, and anything that doesn't mention Aila or address it.
    """
    text = (message or "").strip()
    if not text:
        return None
    if _ABOUT_USER.search(text):
        return None
    if not _ABOUT_AILA.search(text):
        return None
    if not (_QUESTION_SHAPE.match(text) or text.rstrip().endswith("?")):
        return None

    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(text):
            answers = _ANSWERS_PT if language == "pt" else _ANSWERS_EN
            return intent, answers[intent]
    return None
