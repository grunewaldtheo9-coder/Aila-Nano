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
from tools.calculator import try_calculate
from webresearch.pipeline import ResearchPipeline, detect_language

logger = logging.getLogger(__name__)

# Confidence at/above which a web research result is answered directly.
WEB_DIRECT_ANSWER_CONFIDENCE = 0.7

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
    ):
        self.knowledge = knowledge
        self.research = research

    def route(self, message: str) -> RouteResult:
        """Decide how to handle one user message. Never raises: any
        internal failure degrades to 'no routing' (plain generation)."""
        try:
            return self._route(message)
        except Exception:  # noqa: BLE001 — the router must never kill a turn
            logger.exception("tool router failed; falling back to plain generation")
            return RouteResult()

    def _route(self, message: str) -> RouteResult:
        message = (message or "").strip()
        if not message:
            return RouteResult()

        # 1. Exact arithmetic.
        calculated = try_calculate(message)
        if calculated is not None:
            language = detect_language(message)
            template = "O resultado é {r}." if language == "pt" else "The answer is {r}."
            return RouteResult(direct_reply=template.format(r=calculated), tool_used="calculator")

        if not self._is_information_question(message):
            return RouteResult()

        # 2. Stored global knowledge.
        if self.knowledge is not None:
            item = self.knowledge.best_direct_answer(message)
            if item is not None:
                logger.info("knowledge hit id=%s relevance=%.2f", item.id, item.relevance)
                return RouteResult(direct_reply=item.answer, tool_used="knowledge")

        # 3. Web research.
        if self.research is not None:
            outcome = self.research.research(message)
            if outcome.ok and outcome.answer:
                if outcome.confidence >= WEB_DIRECT_ANSWER_CONFIDENCE:
                    return RouteResult(direct_reply=outcome.answer, tool_used="web_research")
                return RouteResult(context_snippets=outcome.snippets[:2], tool_used="web_research")
            logger.info("web research unavailable/failed: %s", outcome.reason)

        return RouteResult()

    def _is_information_question(self, message: str) -> bool:
        """Only factual information questions leave the model: they must
        look like a question, be substantial enough to search, and not be
        about Aila itself or the current conversation."""
        if _SELF_REFERENCE.search(message):
            return False
        looks_like_question = bool(_QUESTION_OPENERS.match(message)) or message.rstrip().endswith("?")
        if not looks_like_question:
            return False
        return len(tokenize(message)) >= 2
