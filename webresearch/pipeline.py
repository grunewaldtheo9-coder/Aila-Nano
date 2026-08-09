"""ResearchPipeline: query → (cache | Serper) → rank → extract →
validate → dedup → store → outcome.

The pipeline never raises on external failure — every Serper/parse error
degrades to `ResearchOutcome(ok=False, reason=...)` so the caller (the
tool router) can fall back to the knowledge base or the model. It also
never returns unsanitized web text: every string that leaves this module
has passed webresearch/quality.py's sanitization.

Confidence model (simple, documented, testable):
- answer box present ................ base 0.8
- knowledge-graph description ....... base 0.7
- snippet-agreement extraction ...... base 0.5
- +0.1 if 2+ distinct domains corroborate the extracted answer
- +0.1 if the best source is tier 1
- capped to [0, 0.95] — web research never reaches 1.0 by design.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from knowledge.base import KnowledgeBase
from knowledge.store import KnowledgeStore
from memory.lexical import lexical_overlap_score, tokenize
from webresearch.quality import (
    complete_sentence,
    domain_tier,
    looks_truncated,
    rank_sources,
    sanitize_snippet,
)
from webresearch.serper import (
    SearchResponse,
    SerperAuthError,
    SerperClient,
    SerperError,
    SerperRateLimitError,
)

logger = logging.getLogger(__name__)

# Portuguese detection: cheap and transparent. Accented characters common
# in pt-BR; strong markers are words that are distinctively Portuguese
# (one occurrence is enough); weak markers are shared/ambiguous function
# words that only count in pairs.
_PT_CHARS = set("ãõáéíóúâêôàç")
_PT_STRONG_WORDS = {
    "quem", "qual", "quais", "quanto", "quanta", "onde", "quando", "porque",
    "você", "voce", "obrigado", "obrigada", "fundou", "criou",
    "significa", "chama", "mora", "gosta", "fala",
}
_PT_WEAK_WORDS = {"como", "por", "que", "foi", "sim", "para", "de", "da", "do", "uma", "um", "os", "as"}


def detect_language(text: str) -> str:
    lowered = (text or "").lower()
    if any(c in _PT_CHARS for c in lowered):
        return "pt"
    words = set(re.findall(r"[a-zà-ÿ]+", lowered))
    if words & _PT_STRONG_WORDS:
        return "pt"
    return "pt" if len(words & _PT_WEAK_WORDS) >= 2 else "en"


def normalize_query(query: str) -> str:
    """Cache key: lowercase, punctuation stripped, whitespace collapsed,
    significant words sorted so trivially-reordered questions share a
    cache entry."""
    tokens = sorted(tokenize(query))
    return " ".join(tokens) if tokens else (query or "").strip().lower()


@dataclass
class ResearchOutcome:
    ok: bool
    answer: str | None = None
    confidence: float = 0.0
    source_urls: list[str] = field(default_factory=list)
    source_titles: list[str] = field(default_factory=list)
    snippets: list[str] = field(default_factory=list)  # sanitized, for RAG context
    from_cache: bool = False
    stored: str | None = None  # 'created' | 'updated' | 'conflict' | None
    reason: str | None = None  # set when ok=False


class ResearchPipeline:
    def __init__(
        self,
        client: SerperClient | None,
        store: KnowledgeStore,
        knowledge: KnowledgeBase,
        cache_ttl_seconds: float = 168 * 3600,
    ):
        self.client = client  # None => web search unavailable/disabled
        self.store = store
        self.knowledge = knowledge
        self.cache_ttl_seconds = cache_ttl_seconds

    def research(self, query: str) -> ResearchOutcome:
        """Full pipeline for one question. Returns ok=False (never
        raises) when research is impossible or produced nothing usable."""
        query = (query or "").strip()
        if not query:
            return ResearchOutcome(ok=False, reason="empty_query")
        if self.client is None:
            return ResearchOutcome(ok=False, reason="web_search_disabled")

        language = detect_language(query)
        cache_key = normalize_query(query)

        response, from_cache, failure = self._get_results(query, cache_key, language)
        if response is None:
            return ResearchOutcome(ok=False, reason=failure or "search_failed")
        if not response.results and not response.answer_box_answer and not response.knowledge_graph_description:
            return ResearchOutcome(ok=False, reason="no_results")

        outcome = self._extract(query, response, language)
        outcome.from_cache = from_cache
        return outcome

    # -- steps -------------------------------------------------------------

    def _get_results(
        self, query: str, cache_key: str, language: str
    ) -> tuple[SearchResponse | None, bool, str | None]:
        """Returns (response, from_cache, failure_reason). Exactly one of
        response / failure_reason is set.

        The failure reason is specific rather than a single
        "search_failed" because the caller turns it into what the user
        reads, and the three cases need different advice: a rejected key
        needs a new key, a rate limit needs waiting, a network error
        needs a retry.
        """
        cached = self.store.get_cached_web_results(cache_key, self.cache_ttl_seconds)
        if cached is not None:
            logger.info("web cache hit for %r", cache_key)
            if cached:
                return SearchResponse.from_dict(cached[0]), True, None
            return None, True, "no_results"

        try:
            started = time.time()
            response = self.client.search(query, language=language)
            logger.info(
                "serper search ok: %d results in %.2fs", len(response.results), time.time() - started
            )
        except SerperAuthError as e:
            # The key is missing, revoked, or out of credits. Never logs
            # the key itself.
            logger.warning("serper rejected the API key: %s", e)
            return None, False, "auth_failed"
        except SerperRateLimitError as e:
            logger.warning("serper rate limited: %s", e)
            return None, False, "rate_limited"
        except SerperError as e:
            logger.warning("serper search failed: %s", e)
            return None, False, "search_failed"

        self.store.cache_web_results(cache_key, [response.to_dict()])
        return response, False, None

    def _extract(self, query: str, response: SearchResponse, language: str) -> ResearchOutcome:
        ranked = rank_sources(response.results)
        top = ranked[: 5]

        snippets: list[str] = []
        urls: list[str] = []
        titles: list[str] = []
        for r in top:
            clean = sanitize_snippet(r.snippet)
            if clean is None:
                continue
            snippets.append(clean)
            urls.append(r.link)
            titles.append(sanitize_snippet(r.title) or r.domain)

        # Candidate answers, best-sourced first. Collected rather than
        # short-circuited so a *complete* lower-tier candidate can be
        # preferred over a visibly cut-off higher-tier one — search
        # engines routinely return answer boxes and snippets that stop
        # mid-list ("...insurance, securities, ..."), and serving those
        # verbatim is what made Aila's answers read as unfinished
        # (reported from real use).
        candidates: list[tuple[str, float]] = []

        if response.answer_box_answer or response.answer_box_snippet:
            boxed = sanitize_snippet(response.answer_box_answer or response.answer_box_snippet)
            if boxed:
                candidates.append((boxed, 0.8))
        if response.knowledge_graph_description:
            desc = sanitize_snippet(response.knowledge_graph_description)
            if desc:
                title = sanitize_snippet(response.knowledge_graph_title or "") or ""
                candidates.append((f"{title}: {desc}" if title else desc, 0.7))
        # Snippets that actually share vocabulary with the question (an
        # off-topic snippet is worse than admitting failure).
        for s in snippets:
            if lexical_overlap_score(query, s) >= 0.2:
                candidates.append((s, 0.5))

        answer: str | None = None
        confidence = 0.0
        if candidates:
            complete = [c for c in candidates if not looks_truncated(c[0])]
            # Among complete candidates keep source order (answer box
            # first); if every candidate is cut off, take the best-sourced
            # one and repair it below.
            answer, confidence = (complete or candidates)[0]
            answer = complete_sentence(answer)
            if not answer:
                answer = None

        if answer is None:
            return ResearchOutcome(
                ok=False,
                snippets=[complete_sentence(s) for s in snippets],
                source_urls=urls,
                source_titles=titles,
                reason="no_extractable_answer",
            )

        # Corroboration: does a second, distinct domain's snippet agree?
        corroborating_domains = set()
        for r in top:
            clean = sanitize_snippet(r.snippet)
            if clean and lexical_overlap_score(answer, clean) >= 0.3:
                corroborating_domains.add(r.domain)
        if len(corroborating_domains) >= 2:
            confidence += 0.1
        if top and domain_tier(top[0].domain) == 1:
            confidence += 0.1
        confidence = max(0.0, min(0.95, confidence))

        verification = "corroborated" if len(corroborating_domains) >= 2 else "unverified"

        stored: str | None = None
        if confidence >= 0.5:
            stored, _ = self.knowledge.remember_answer(
                query,
                answer,
                confidence=confidence,
                source_urls=urls[:3],
                source_titles=titles[:3],
                language=language,
                verification=verification,
            )
        else:
            self.store.add_candidate(
                query, answer, confidence=confidence, reason="low_confidence",
                source_urls=urls[:3], source_titles=titles[:3],
            )

        return ResearchOutcome(
            ok=True,
            answer=answer,
            confidence=confidence,
            source_urls=urls[:3],
            source_titles=titles[:3],
            # Repaired too: the router may serve a snippet verbatim when
            # no answer could be extracted, so it must read as finished.
            snippets=[complete_sentence(s) for s in snippets[:3]],
            stored=stored,
        )
