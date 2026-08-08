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
1. Arithmetic  → exact calculator answer (never let the model guess math).
2. Knowledge   → a relevant, confident, non-conflicted stored fact
                 answers directly. (Cheapest, fully offline.)
3. Web research→ only for factual-information questions the knowledge
                 base couldn't answer, when a Serper client is
                 configured. High-confidence result answers directly
                 (and was stored by the pipeline for next time);
                 medium-confidence results are returned as context
                 snippets for the model instead.
4. Nothing     → plain model generation (possibly with snippets).

The router never answers identity/self questions ("who created you") or
chit-chat from the web — those belong to the model's own fine-tuned
knowledge. Memory commands ("remember that...") are intercepted earlier,
in agents/base.py, and never reach the router.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from knowledge.base import KnowledgeBase
from memory.lexical import tokenize
from memory.phrasing import memory_to_answer
from tools.calculator import try_calculate
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
_QUESTION_OPENERS = re.compile(
    r"^\s*(who|what|when|where|which|why|how|is|are|was|were|did|does|do|"
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

_NO_MEMORY_REPLY_EN = (
    "I don't have that in my memory yet. You can tell me with: "
    '"remember that ..."'
)
_NO_MEMORY_REPLY_PT = (
    "Ainda não tenho isso na minha memória. Você pode me dizer com: "
    '"remember that ..."'
)


@dataclass
class RouteResult:
    """What the router decided for one message.

    Exactly one of:
    - direct_reply set  → reply with this text; skip generation entirely.
    - context_snippets  → generate normally, injecting these as [WEB] data.
    - neither           → generate normally with no additions.
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

        # 1. Exact arithmetic (always on the raw message — "When?" is
        #    never arithmetic, and expansion could only confuse this).
        calculated = try_calculate(message)
        if calculated is not None:
            language = detect_language(message)
            template = "O resultado é {r}." if language == "pt" else "The answer is {r}."
            return RouteResult(direct_reply=template.format(r=calculated), tool_used="calculator")

        query = self._expand_follow_up(message, previous_user_message)

        # 2. The user's own remembered facts. Answered deterministically
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
                answer = memory_to_answer(
                    memories[0]["content"], language=detect_language(message)
                )
                if answer:
                    logger.info("memory hit id=%s", memories[0]["id"])
                    return RouteResult(direct_reply=answer, tool_used="memory")

            # A question about the user ("What is my name?") with nothing
            # remembered: memory is the only possible source, so admit it
            # rather than generate. Prevents both garbled output and any
            # chance of the model inventing a personal detail.
            if _FIRST_PERSON_POSSESSIVE.search(message):
                language = detect_language(message)
                reply = _NO_MEMORY_REPLY_PT if language == "pt" else _NO_MEMORY_REPLY_EN
                return RouteResult(direct_reply=reply, tool_used="memory_miss")

        if not self._is_information_question(query):
            return RouteResult()

        # 3. Stored global knowledge.
        if self.knowledge is not None:
            item = self.knowledge.best_direct_answer(query)
            if item is not None:
                logger.info("knowledge hit id=%s relevance=%.2f", item.id, item.relevance)
                return RouteResult(direct_reply=item.answer, tool_used="knowledge")

        # 4. Web research.
        if self.research is not None:
            outcome = self.research.research(query)
            if outcome.ok and outcome.answer:
                if outcome.confidence >= WEB_DIRECT_ANSWER_CONFIDENCE:
                    return RouteResult(direct_reply=outcome.answer, tool_used="web_research")
                return RouteResult(context_snippets=outcome.snippets[:2], tool_used="web_research")
            logger.info("web research unavailable/failed: %s", outcome.reason)

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
        if not tokenize(previous_user_message):
            return message
        return f"{previous_user_message.rstrip('?!. ')} {message}".strip()

    @staticmethod
    def _looks_like_question(message: str) -> bool:
        return bool(_QUESTION_OPENERS.match(message)) or message.rstrip().endswith("?")

    def _is_information_question(self, message: str) -> bool:
        """Only factual information questions leave the model: they must
        look like a question, be substantial enough to search, and not be
        about Aila itself or the current conversation."""
        if _SELF_REFERENCE.search(message):
            return False
        if not self._looks_like_question(message):
            return False
        return len(tokenize(message)) >= 2
