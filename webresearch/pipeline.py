"""ResearchPipeline: query → (cache | Wikipedia | Serper) → rank →
extract → validate → dedup → store → outcome.

Sources are tried in order, first usable answer wins — see the
`ResearchPipeline` docstring for why Wikipedia leads.

The pipeline never raises on external failure — every provider/parse
error degrades to `ResearchOutcome(ok=False, reason=...)` so the caller
(the tool router) can fall back to the knowledge base or the model. It
also never returns unsanitized web text: every string that leaves this
module has passed webresearch/quality.py's sanitization.

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
    SerperTimeoutError,
)
from webresearch.wikipedia import (
    WikipediaError,
    WikipediaRateLimitError,
    WikipediaTimeoutError,
    WikipediaUnavailableError,
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
    provider: str | None = None  # 'wikipedia' | 'serper', set when ok=True


# Failure reasons ordered by how actionable they are for the user. When
# every provider failed, the one highest in this list is what gets
# reported: a rejected key is worth telling someone about, "nothing
# found" is not.
_REASON_PRIORITY = ("auth_failed", "rate_limited", "search_failed")

# How much vocabulary an extracted answer must share with the question
# before it can be served. Applies to every candidate — snippets and
# encyclopedia summaries alike — because an off-topic answer served
# confidently is worse than admitting the miss.
MIN_ANSWER_OVERLAP = 0.2


def _worst_reason(failures: list[str]) -> str:
    """Pick what to tell the user after every provider failed.

    If *any* provider actually reached the network and came back empty,
    that is the honest headline ("I looked and found nothing") — an
    auth error on a second, redundant provider is not the user's
    problem when the first one searched fine.
    """
    if not failures:
        return "no_results"
    searched = {"no_results", "no_extractable_answer"}
    if any(f in searched for f in failures):
        return "no_results"
    for reason in _REASON_PRIORITY:
        if reason in failures:
            return reason
    return failures[0]


class ResearchPipeline:
    """Research across one or more interchangeable providers.

    Providers are tried in order and the first *usable* answer wins.
    Wikipedia goes first when present: it needs no API key, has no quota,
    returns whole sentences rather than cut-off fragments, and is a
    tier-1 domain — so it is both the cheapest and the highest-quality
    source for the encyclopedic questions Aila is asked most. Serper
    backs it up for everything Wikipedia has no article about (current
    events, "how many subscribers", small local businesses).

    A provider "failing" is normal and never fatal: the next one is
    tried, and only when every provider has failed does the caller hear
    about it — with the most actionable reason of the ones collected.
    """

    def __init__(
        self,
        client: SerperClient | None,
        store: KnowledgeStore,
        knowledge: KnowledgeBase,
        cache_ttl_seconds: float = 168 * 3600,
        wikipedia=None,  # webresearch.wikipedia.WikipediaClient | None
        offline_cooldown_seconds: float = 60.0,
    ):
        self.client = client  # None => no Serper (no key configured)
        self.wikipedia = wikipedia  # None => Wikipedia disabled
        self.store = store
        self.knowledge = knowledge
        self.cache_ttl_seconds = cache_ttl_seconds

        # Offline circuit breaker. Without it, every question asked with
        # no internet pays a full timeout per provider before failing —
        # two providers at 8s each turns a chat into an 16-second wait per
        # message. After a connection failure we stop dialling out for a
        # cooldown and answer immediately from what we already know.
        self.offline_cooldown_seconds = offline_cooldown_seconds
        self._offline_until = 0.0

    # -- provider plumbing --------------------------------------------------

    def _providers(self) -> list[tuple[str, object, str]]:
        """(name, client, cache_prefix), best-first.

        Serper keeps the empty cache prefix it has always used, so web
        cache entries written by earlier versions stay valid.
        """
        providers: list[tuple[str, object, str]] = []
        if self.wikipedia is not None:
            providers.append(("wikipedia", self.wikipedia, "wiki:"))
        if self.client is not None:
            providers.append(("serper", self.client, ""))
        return providers

    @property
    def offline(self) -> bool:
        """True while the circuit breaker is open (a recent connection
        attempt failed outright)."""
        return time.time() < self._offline_until

    def _note_connection_failure(self) -> None:
        self._offline_until = time.time() + self.offline_cooldown_seconds
        logger.info("network looks down; skipping lookups for %.0fs", self.offline_cooldown_seconds)

    def _note_connection_success(self) -> None:
        self._offline_until = 0.0

    def research(self, query: str) -> ResearchOutcome:
        """Full pipeline for one question. Returns ok=False (never
        raises) when research is impossible or produced nothing usable."""
        query = (query or "").strip()
        if not query:
            return ResearchOutcome(ok=False, reason="empty_query")

        providers = self._providers()
        if not providers:
            return ResearchOutcome(ok=False, reason="web_search_disabled")

        language = detect_language(query)
        cache_key = normalize_query(query)

        # Cache is checked for every provider even while offline — a
        # previously fetched answer is still an answer.
        failures: list[str] = []
        for name, client, prefix in providers:
            response, from_cache, failure = self._get_results(
                name, client, query, prefix + cache_key, language
            )
            if response is None:
                failures.append(failure or "search_failed")
                continue
            if (
                not response.results
                and not response.answer_box_answer
                and not response.knowledge_graph_description
            ):
                failures.append("no_results")
                continue

            outcome = self._extract(query, response, language)
            if outcome.ok:
                outcome.from_cache = from_cache
                outcome.provider = name
                return outcome
            failures.append(outcome.reason or "no_extractable_answer")

        return ResearchOutcome(ok=False, reason=_worst_reason(failures))

    # -- steps -------------------------------------------------------------

    def _get_results(
        self, provider: str, client, query: str, cache_key: str, language: str
    ) -> tuple[SearchResponse | None, bool, str | None]:
        """Returns (response, from_cache, failure_reason). Exactly one of
        response / failure_reason is set.

        The failure reason is specific rather than a single
        "search_failed" because the caller turns it into what the user
        reads, and the cases need different advice: a rejected key needs
        a new key, a rate limit needs waiting, a network error needs a
        retry.
        """
        cached = self.store.get_cached_web_results(cache_key, self.cache_ttl_seconds)
        if cached is not None:
            logger.info("%s cache hit for %r", provider, cache_key)
            if cached:
                return SearchResponse.from_dict(cached[0]), True, None
            return None, True, "no_results"

        if self.offline:
            return None, False, "search_failed"

        try:
            started = time.time()
            response = client.search(query, language=language)
            logger.info(
                "%s search ok: %d results in %.2fs",
                provider,
                len(response.results),
                time.time() - started,
            )
            self._note_connection_success()
        except (SerperAuthError,) as e:
            # The key is missing, revoked, or out of credits. Never logs
            # the key itself. Reaching the server at all means we are
            # online, so this must not trip the offline breaker.
            logger.warning("%s rejected the API key: %s", provider, e)
            self._note_connection_success()
            return None, False, "auth_failed"
        except (SerperRateLimitError, WikipediaRateLimitError) as e:
            logger.warning("%s rate limited: %s", provider, e)
            self._note_connection_success()
            return None, False, "rate_limited"
        except (SerperTimeoutError, WikipediaTimeoutError, WikipediaUnavailableError) as e:
            logger.warning("%s unreachable: %s", provider, e)
            self._note_connection_failure()
            return None, False, "search_failed"
        except (SerperError, WikipediaError) as e:
            logger.warning("%s search failed: %s", provider, e)
            return None, False, "search_failed"

        self.store.cache_web_results(cache_key, [response.to_dict()])
        return response, False, None

    def _extract(self, query: str, response: SearchResponse, language: str) -> ResearchOutcome:
        ranked = rank_sources(response.results)
        top = ranked[: 5]

        snippets: list[str] = []
        raw_snippets: list[str] = []  # kept aligned with `snippets`
        urls: list[str] = []
        titles: list[str] = []
        for r in top:
            clean = sanitize_snippet(r.snippet)
            if clean is None:
                # Dropped (injection attempt / empty). Skipping it here
                # means `snippets` is NOT index-aligned with `top`, which
                # is why the raw text is collected in parallel rather
                # than zipped back together later.
                continue
            snippets.append(clean)
            raw_snippets.append(r.snippet)
            urls.append(r.link)
            titles.append(sanitize_snippet(r.title) or r.domain)

        # Candidate answers, best-sourced first, each tagged with whether
        # *the source* handed us something already cut off. Collected
        # rather than short-circuited so an intact lower-tier candidate
        # can beat a mid-list-truncated higher-tier one — search engines
        # routinely return answer boxes that stop at "...insurance,
        # securities, ...", and serving those is what made Aila's answers
        # read as unfinished (reported from real use).
        #
        # The flag is measured on the RAW text, before sanitizing. That
        # distinction matters: our own MAX_SNIPPET_CHARS cap also leaves
        # text mid-sentence, but a candidate is not *worse-sourced* for
        # being long. Conflating the two made "Who is MrBeast?" skip the
        # Wikipedia article about MrBeast (long, hit our cap) in favour of
        # a snippet about MrBeast Burger (short, intact).
        candidates: list[tuple[str, float, bool]] = []

        raw_box = response.answer_box_answer or response.answer_box_snippet
        if raw_box:
            boxed = sanitize_snippet(raw_box)
            if boxed:
                candidates.append((boxed, 0.8, looks_truncated(raw_box)))
        if response.knowledge_graph_description:
            desc = sanitize_snippet(response.knowledge_graph_description)
            if desc:
                title = sanitize_snippet(response.knowledge_graph_title or "") or ""
                combined = f"{title}: {desc}" if title else desc
                # Same on-topic gate the snippets get. Google's knowledge
                # graph is matched to the query by Google, but Wikipedia
                # summaries arrive from a *guessed* page title — a wrong
                # guess would otherwise be served at 0.7 confidence as a
                # confident, completely unrelated answer.
                if lexical_overlap_score(query, combined) >= MIN_ANSWER_OVERLAP:
                    candidates.append(
                        (combined, 0.7, looks_truncated(response.knowledge_graph_description))
                    )
        # Snippets that actually share vocabulary with the question (an
        # off-topic snippet is worse than admitting failure).
        for raw, s in zip(raw_snippets, snippets):
            if lexical_overlap_score(query, s) >= MIN_ANSWER_OVERLAP:
                candidates.append((s, 0.5, looks_truncated(raw)))

        answer: str | None = None
        confidence = 0.0
        if candidates:
            intact = [c for c in candidates if not c[2]]
            # Within each group, source order wins (answer box, then
            # encyclopedia summary, then snippets).
            for text, conf, _ in (intact or candidates):
                repaired = complete_sentence(text)
                if repaired:
                    answer, confidence = repaired, conf
                    break

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
