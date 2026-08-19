"""ToolRouter: the deterministic decision layer that runs before model
generation on every turn.

Why deterministic (rules), not model-driven function calling: an ~20M-
parameter model cannot reliably emit structured tool-call decisions, and
a wrong routing decision either leaks internal machinery to the user or
hallucinates. Rules are auditable, testable, and cheap. The Tool/
ToolRegistry abstraction (tools/base.py) stays the extension point for
future capabilities; the router is just the policy deciding which one
runs.

Routing policy per user message (first match wins):
1. Small talk  → fixed reply for short conversational filler ("ok",
                 "thanks", "bro"). See tools/smalltalk.py.
2. Arithmetic  → exact calculator answer (never let the model guess math).
3. Memory      → the user's own remembered facts, phrased deterministically.
4. Identity    → questions about Aila itself, answered from a fact table
                 (tools/identity.py) rather than generated.
5. Knowledge   → a relevant, confident, non-conflicted stored fact
                 answers directly. (Cheapest, fully offline.)
6. Web research→ only for factual-information questions the knowledge
                 base couldn't answer, when a Serper client is
                 configured. The result is always served as text; see
                 below for why it is never handed to the model.
7. Nothing     → plain model generation.

Why web results are never summarized by the model: at ~20M parameters,
generation given retrieved snippets does not summarize them, it
overwrites them. Real transcripts had "Who created Hames Eventos?" come
back as "I'm no other company to help a fun day at a time." with correct
snippets sitting in the prompt. So a research result is served verbatim
(hedged when confidence is low), and when research genuinely finds
nothing the router says so instead of letting the model invent an
answer. `RouteResult.context_snippets` remains part of the contract for
callers/future larger models, but the router no longer populates it from
web research.

The router never sends identity/self questions ("who created you") to
the web — the web doesn't know what Aila Nano is. Memory commands
("remember that...") are intercepted earlier, in agents/base.py, and
never reach the router.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from knowledge.base import KnowledgeBase
from memory.lexical import GENERIC_QUESTION_TERMS, tokenize
from memory.phrasing import memory_to_answer
from tools.calculator import try_calculate
from tools.identity import match_identity_question
from tools.smalltalk import match_smalltalk
from webresearch.pipeline import ResearchPipeline, detect_language

logger = logging.getLogger(__name__)

# Confidence at/above which a web research result is answered directly.
WEB_DIRECT_ANSWER_CONFIDENCE = 0.7

# Lexical relevance a stored memory needs before it is used as a *direct*
# answer rather than merely injected as context. Deliberately far above
# the injection threshold (0.2): injecting a loosely-related memory is
# harmless, but answering with the wrong one is not.
MEMORY_DIRECT_ANSWER_RELEVANCE = 0.5

# Interrogative openers that mark a message as an information question.
#
# The wh-words accept a bare "s" as well as "'s", because people drop the
# apostrophe constantly. Without it, "whats my name" was not recognised
# as a question at all, skipped the memory tier, and came back as
# generated nonsense — while "what's my name" worked perfectly (reported
# from real use). The optional "s" is deliberately NOT extended to
# is/are/do/does: "dos" is an ordinary Portuguese word, not a question.
_QUESTION_OPENERS = re.compile(
    r"^\s*(?:(?:who|what|when|where|why|how)(?:'s|s)?|which|"
    r"is|are|was|were|did|does|do|"
    r"quem|o\s+que|qual|quais|quando|onde|por\s+que|como|é|são|foi|eram)\b",
    re.IGNORECASE,
)

# Questions about Aila itself / the current conversation must never be
# routed to the web — they're answered by the model's own fine-tuned
# identity knowledge (or memory).
_SELF_REFERENCE = re.compile(
    r"\b(aila|you|your|yours|você|voce|seu|sua|teu|tua)\b", re.IGNORECASE
)

# First-person possessives: a question containing one ("What is my
# name?") asks about the *user*, which only stored memory can answer —
# no amount of pretraining knows it. When memory has nothing, saying so
# is the honest answer; generating is guaranteed to be either garbage or
# a fabrication.
_FIRST_PERSON_POSSESSIVE = re.compile(r"\b(my|mine|meu|minha|meus|minhas)\b", re.IGNORECASE)

# A question about the user's *own* attributes — preferences, identity,
# personal facts ("Do que eu gosto?", "Who am I?"). Only stored memory can
# answer these, so when memory has nothing, admitting it is the honest reply
# (and the translation fallback can then retry the same question across
# languages, finding a memory stored in the other language). Kept narrow on
# purpose: a bare "I"/"eu" also appears in general how-to questions ("How do
# I make bread?") that genuinely want the web, so the pronoun is required to
# sit with a stative / preference / identity verb, or in a fixed self-phrase.
_FIRST_PERSON_SELF = re.compile(
    r"\b(?:"
    # English
    r"i\s+(?:like|likes|love|hate|prefer|am|are|have|has|had|know|want|need|"
    r"enjoy|support|live|lived|study|studied|work|worked|was)\b"
    r"|i'?m\b"
    r"|who\s+am\s+i\b"
    r"|about\s+me\b"
    # Portuguese
    r"|eu\s+(?:me\s+)?(?:gosto|amo|odeio|prefiro|sou|tenho|tinha|sei|quero|"
    r"preciso|adoro|curto|torço|moro|morava|nasci|chamo|estudo|trabalho)\b"
    r"|quem\s+sou\s+eu\b"
    r"|sobre\s+mim\b"
    r"|comigo\b"
    r")",
    re.IGNORECASE,
)

# Words a bare follow-up may consist of and nothing else. English
# interrogatives are already stopwords (so they tokenize away to
# nothing); the Portuguese ones are not, and listing them here is
# deliberately narrower than adding them to STOPWORDS, which would
# change relevance scoring everywhere.
# Note the absence of "what": a bare "What?" or "What??" is confusion
# ("I didn't understand"), not a request for more on the same topic.
# Treating it as a follow-up replayed the previous answer word for word,
# which is precisely what the user was complaining about when they typed
# it. tools/smalltalk.py answers it as confusion instead.
_FOLLOW_UP_WORDS: frozenset[str] = frozenset(
    {
        "when", "where", "why", "who", "how", "which", "whose",
        "quando", "onde", "porque", "quem", "qual", "quais", "como",
        "quanto", "quanta", "quantos", "quantas",
    }
)

_NO_MEMORY_REPLY_EN = (
    "I don't have that in my memory yet. You can tell me with: "
    '"remember that ..."'
)
_NO_MEMORY_REPLY_PT = (
    "Ainda não tenho isso na minha memória. Você pode me dizer com: "
    '"remember that ..."'
)

# Prefix for a web result the pipeline was not confident about. The
# answer is still shown (it is far better than anything generation would
# produce) but it is labelled so the user can weigh it.
_HEDGE_EN = "I'm not fully certain, but here's what I found: {answer}"
_HEDGE_PT = "Não tenho certeza total, mas foi isto que encontrei: {answer}"

# Said when web research ran and genuinely came back with nothing. The
# alternative — generating — produced confident nonsense in real use, so
# admitting the miss is the honest and more useful reply.
_NO_ANSWER_EN = (
    "I looked that up but couldn't find a reliable answer. "
    "Try rephrasing it, or adding a bit more detail."
)
_NO_ANSWER_PT = (
    "Pesquisei, mas não encontrei uma resposta confiável. "
    "Tente reformular a pergunta ou dar mais detalhes."
)

# Said when the search itself could not be performed. Split by cause,
# because the user's next move differs: a rejected key needs replacing, a
# rate limit needs waiting, a network error needs a retry. Reported in
# plain language — no status codes, no key material.
_SEARCH_ERROR_REPLIES: dict[str, tuple[str, str]] = {
    "auth_failed": (
        "My web search key isn't being accepted right now, so I can't look "
        "that up. It usually means the key needs replacing or has run out of "
        "free searches — get a new one at serper.dev and put it in your .env "
        "file as SERPER_API_KEY.",
        "Minha chave de busca na web não está sendo aceita, então não consigo "
        "pesquisar isso. Normalmente isso quer dizer que a chave precisa ser "
        "trocada ou que as buscas gratuitas acabaram — pegue uma nova em "
        "serper.dev e coloque no arquivo .env como SERPER_API_KEY.",
    ),
    "rate_limited": (
        "I've hit the web search limit for now. Please try that question "
        "again in a little while.",
        "Atingi o limite de buscas na web por enquanto. Tente essa pergunta "
        "novamente daqui a pouco.",
    ),
    "search_failed": (
        "I couldn't reach the web to look that up just now. Please try again "
        "in a moment.",
        "Não consegui acessar a web para pesquisar isso agora. Tente "
        "novamente em instantes.",
    ),
}


@dataclass
class RouteResult:
    """What the router decided for one message.

    - direct_reply set → reply with this text; skip generation entirely.
    - neither field set → generate normally.

    `context_snippets` (rendered as a `[WEB]` block by agents/base.py) is
    part of the contract for a future larger model, but nothing populates
    it today: at ~20M parameters, generation given retrieved text
    overwrites it rather than summarizing it, so research results are
    served as text instead. See the module docstring.
    """

    direct_reply: str | None = None
    context_snippets: list[str] = field(default_factory=list)
    tool_used: str | None = None  # for logging/observability only


class ToolRouter:
    def __init__(
        self,
        knowledge: KnowledgeBase | None = None,
        research: ResearchPipeline | None = None,
        memory=None,  # memory.manager.MemoryManager | None
    ):
        self.knowledge = knowledge
        self.research = research
        self.memory = memory

    def route(self, message: str, previous_user_message: str | None = None) -> RouteResult:
        """Decide how to handle one user message. Never raises: any
        internal failure degrades to 'no routing' (plain generation).

        `previous_user_message` (the user's prior turn, if any) powers
        follow-up resolution: a bare "When?" after "Who founded Apple?"
        is expanded to carry the previous question's topic into knowledge
        lookup and web research. Purely lexical and deterministic — no
        model in the loop.
        """
        try:
            return self._route(message, previous_user_message)
        except Exception:  # noqa: BLE001 — the router must never kill a turn
            logger.exception("tool router failed; falling back to plain generation")
            return RouteResult()

    def _route(self, message: str, previous_user_message: str | None = None) -> RouteResult:
        message = (message or "").strip()
        if not message:
            return RouteResult()

        language = detect_language(message)

        # 1. Conversational filler ("ok", "thanks", "bro"). Exact-phrase
        #    only (tools/smalltalk.py), so anything with real content
        #    falls straight through.
        filler = match_smalltalk(message, language=language)
        if filler is not None:
            intent, reply = filler
            return RouteResult(direct_reply=reply, tool_used=f"smalltalk:{intent}")

        # 2. Exact arithmetic (always on the raw message — "When?" is
        #    never arithmetic, and expansion could only confuse this).
        calculated = try_calculate(message)
        if calculated is not None:
            template = "O resultado é {r}." if language == "pt" else "The answer is {r}."
            return RouteResult(direct_reply=template.format(r=calculated), tool_used="calculator")

        query = self._expand_follow_up(message, previous_user_message)

        # 3. The user's own remembered facts. Answered deterministically
        #    rather than injected-and-generated: the model garbles a
        #    correctly-retrieved memory at this scale (measured), and a
        #    remembered fact is something we know exactly. Runs before
        #    the information-question gate because personal questions
        #    ("Do you know my name?") legitimately address Aila directly,
        #    which that gate excludes.
        if self.memory is not None and self._looks_like_question(message):
            memories = self.memory.get_relevant_memories(
                query, k=1, threshold=MEMORY_DIRECT_ANSWER_RELEVANCE
            )
            if memories:
                answer = memory_to_answer(memories[0]["content"], language=language)
                if answer:
                    logger.info("memory hit id=%s", memories[0]["id"])
                    return RouteResult(direct_reply=answer, tool_used="memory")

            # A question about the user ("What is my name?") with nothing
            # remembered: memory is the only possible source, so admit it
            # rather than generate. Prevents both garbled output and any
            # chance of the model inventing a personal detail.
            if _FIRST_PERSON_POSSESSIVE.search(message) or _FIRST_PERSON_SELF.search(message):
                reply = _NO_MEMORY_REPLY_PT if language == "pt" else _NO_MEMORY_REPLY_EN
                return RouteResult(direct_reply=reply, tool_used="memory_miss")

        # 4. Questions about Aila itself. Answered from a fact table
        #    rather than generated: these are the questions the project
        #    most needs right, the web cannot answer them, and generation
        #    shredded them in real use. Runs after memory so anything the
        #    user explicitly told Aila still wins.
        identity = match_identity_question(message, language=language)
        if identity is not None:
            intent, answer = identity
            return RouteResult(direct_reply=answer, tool_used=f"identity:{intent}")

        if not self._is_information_question(query):
            return RouteResult()

        # 5. Stored global knowledge, in the language the question was
        #    asked in (a Portuguese question must not be answered with the
        #    English copy of a fact — the translation fallback handles the
        #    cross-language case properly).
        if self.knowledge is not None:
            item = self.knowledge.best_direct_answer(query, language=language)
            if item is not None:
                logger.info("knowledge hit id=%s relevance=%.2f", item.id, item.relevance)
                return RouteResult(direct_reply=item.answer, tool_used="knowledge")

        # 6. Web research. Whatever comes back is served as text — see
        #    the module docstring for why the model never gets to
        #    rewrite it.
        if self.research is not None:
            outcome = self.research.research(query)

            if outcome.ok and outcome.answer:
                if outcome.confidence >= WEB_DIRECT_ANSWER_CONFIDENCE:
                    return RouteResult(direct_reply=outcome.answer, tool_used="web_research")
                hedge = _HEDGE_PT if language == "pt" else _HEDGE_EN
                return RouteResult(
                    direct_reply=hedge.format(answer=outcome.answer),
                    tool_used="web_research_hedged",
                )

            # NOTE: deliberately no "serve the top snippet anyway"
            # fallback here. `reason="no_extractable_answer"` means the
            # pipeline already rejected every snippet for sharing too
            # little vocabulary with the question, so the snippets that
            # remain are precisely the off-topic ones — serving one
            # answers "Who is the president of Brazil?" with a cake
            # recipe. An admitted miss is the better answer.
            logger.info("web research unavailable/failed: %s", outcome.reason)
            error_reply = _SEARCH_ERROR_REPLIES.get(outcome.reason or "")
            if error_reply is not None:
                return RouteResult(
                    direct_reply=error_reply[1] if language == "pt" else error_reply[0],
                    tool_used=f"web_error:{outcome.reason}",
                )
            if outcome.reason != "web_search_disabled":
                # The web was reached and had nothing. Generating here is
                # how "Mrbest has how much subscribers on Youtube?" became
                # a paragraph about Aila Company Solutions.
                reply = _NO_ANSWER_PT if language == "pt" else _NO_ANSWER_EN
                return RouteResult(direct_reply=reply, tool_used="web_no_answer")
            # web_search_disabled: no API key configured. Fall through to
            # plain generation so an offline install still behaves the way
            # it did before web research existed.

        return RouteResult()

    def _expand_follow_up(self, message: str, previous_user_message: str | None) -> str:
        """Resolve short follow-up questions against the previous user
        turn: "When?" after "Who founded Apple?" becomes
        "Who founded Apple? When?" for retrieval purposes. Only fires for
        genuinely short, question-shaped messages whose own vocabulary is
        too thin to search — a full question is never rewritten."""
        if not previous_user_message:
            return message
        # Raw word count, NOT significant-token count: "What is
        # photosynthesis?" strips to a single significant token but is a
        # complete question, while a real follow-up ("When?", "E quando?")
        # is short in raw words too. Using tokenize() here misclassified
        # full questions as follow-ups (caught in testing).
        if len(message.split()) > 2:
            return message
        looks_like_question = message.rstrip().endswith("?") or bool(
            _QUESTION_OPENERS.match(message)
        )
        if not looks_like_question:
            return message
        # The message must carry no subject of its own. Without this,
        # *any* short question-shaped message inherited the previous
        # topic: after "Who founded Apple?", the replies to "Ok?", "Hi?"
        # and "Bro?" were all the Apple answer again — which is precisely
        # the "he repeats things" complaint this project started from.
        # "When?" and "E quando?" have nothing but an interrogative and
        # genuinely need the previous turn; "Ok?" does not.
        if tokenize(message) - _FOLLOW_UP_WORDS:
            return message
        if not tokenize(previous_user_message):
            return message
        return f"{previous_user_message.rstrip('?!. ')} {message}".strip()

    @staticmethod
    def _looks_like_question(message: str) -> bool:
        return bool(_QUESTION_OPENERS.match(message)) or message.rstrip().endswith("?")

    def _is_information_question(self, message: str) -> bool:
        """Only factual information questions leave the model: they must
        look like a question, name something searchable, and not be
        about Aila itself or the current conversation."""
        if _SELF_REFERENCE.search(message):
            return False
        if not self._looks_like_question(message):
            return False

        # Needs at least one word that actually names a subject. A bare
        # two-token count was the old rule and it rejected "Who is
        # MrBeast?" — one significant token, and exactly the kind of
        # question research exists for. Counting *distinctive* tokens
        # instead keeps "What is that?" (no subject at all) out while
        # letting single-subject questions through.
        significant = tokenize(message) - GENERIC_QUESTION_TERMS
        return len(significant) >= 1
