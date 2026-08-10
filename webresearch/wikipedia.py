"""Wikipedia client — a free, keyless knowledge source.

Why this exists alongside Serper: Serper needs a paid API key, and when
that key is missing, cancelled, or out of credits every factual question
falls through to a ~20M-parameter model, which answers confidently and
wrongly. Wikipedia's API needs no key, no account and no quota, so Aila
has a source that keeps working on its own.

It is also a *better* source for the questions Aila gets most:

- Summaries are whole paragraphs of finished sentences, not the
  cut-off fragments a search engine returns — which is what produced the
  "…insurance, securities, ..." complaint in the first place.
- wikipedia.org is a tier-1 domain in `webresearch/quality.py`, so an
  extracted answer earns the source-quality bonus honestly.

Deliberately reuses `SearchResponse`/`SearchResult` from the Serper
module rather than inventing a parallel shape: the pipeline, the
sanitizer, the ranker and the on-disk web cache then treat both sources
identically, and adding a third source later costs one adapter and no
downstream changes.

Two requests per lookup at worst:
1. REST summary for the subject, guessed straight from the question
   ("Who created Bambu Lab?" -> "Bambu_Lab"). Usually a direct hit.
2. If that 404s, the MediaWiki search API to find the real title, then
   the summary for the best match.

Everything returned is untrusted external DATA — the same rule as
Serper. This module only parses it into dataclasses; sanitization
happens in `webresearch/quality.py`.
"""

from __future__ import annotations

import json
import logging
import re
import socket
import urllib.error
import urllib.parse
import urllib.request

from memory.lexical import lexical_overlap_score
from webresearch.serper import SearchResponse, SearchResult

logger = logging.getLogger(__name__)

# Wikimedia's API etiquette asks for a descriptive User-Agent that
# identifies the tool. Requests without one are throttled harder.
USER_AGENT = "AilaNano/2.1 (https://github.com/grunewaldtheo9-coder/Aila-Nano)"

SUPPORTED_LANGUAGES = ("en", "pt")


class WikipediaError(Exception):
    """Base class for all Wikipedia client failures."""


class WikipediaTimeoutError(WikipediaError):
    """The request timed out."""


class WikipediaRateLimitError(WikipediaError):
    """Rate limited (HTTP 429)."""


class WikipediaUnavailableError(WikipediaError):
    """Wikipedia could not be reached (DNS, connection refused, offline)."""


# Words that open a question rather than name its subject. Stripped from
# the front of a query before guessing a page title, because Wikipedia's
# title lookup wants "Bambu Lab", not "who created bambu lab".
#
# The wh-words accept a bare "s" as well as "'s" — people drop the
# apostrophe constantly, and leaving "Whats" attached changed what
# Wikipedia's own search returned for "Whats the biggest youtuber in the
# world?" from MrBeast to an article about YouTuber *films*.
_LEADING_QUESTION_WORDS = re.compile(
    r"^\s*(?:"
    r"who(?:'s|s|\s+is|\s+are|\s+was|\s+were)?|what(?:'s|s|\s+is|\s+are|\s+was|\s+were)?|"
    r"when(?:'s|s|\s+was|\s+were|\s+did|\s+is)?|where(?:'s|s|\s+is|\s+are|\s+was)?|"
    r"which|why|how(?:\s+many|\s+much|\s+big|\s+old)?|"
    r"tell\s+me\s+about|explain|describe|define|"
    r"created|creates|make[sd]?|made|founded|found|built|build|invented|owns|owned|"
    r"quem(?:\s+[eé])?|o\s+que(?:\s+[eé])?|qual(?:\s+[eé])?|quais|quando|onde|"
    r"por\s*que|porque|como|quantos?|quantas?|"
    r"me\s+fale\s+sobre|fale\s+sobre|explique|descreva|criou|fundou|fez|"
    r"the|an?|o|os|as|de|da|do"
    r")\b\s*",
    re.IGNORECASE,
)

# Trailing filler that is never part of a page title.
_TRAILING_FILLER = re.compile(
    r"\s*\b(?:company|companies|corporation|inc|ltd|"
    r"empresa|empresas)\b\s*$",
    re.IGNORECASE,
)

_HTML_TAG = re.compile(r"<[^>]+>")


def extract_subject(query: str) -> str:
    """Best-effort page-title guess from a natural-language question.

    Strips leading question words repeatedly ("who created" -> "" leaves
    "Bambu Lab") and trailing punctuation. Returns "" when nothing
    identifiable is left, which the caller treats as "go straight to
    search" rather than requesting a nonsense title.
    """
    text = (query or "").strip().rstrip("?!.,;:").strip()
    if not text:
        return ""

    previous = None
    while text and text != previous:
        previous = text
        text = _LEADING_QUESTION_WORDS.sub("", text, count=1).strip()

    text = _TRAILING_FILLER.sub("", text).strip()
    # Collapse internal whitespace; Wikipedia titles use single spaces.
    return re.sub(r"\s+", " ", text)


def _strip_html(text: str) -> str:
    """MediaWiki search snippets embed <span class="searchmatch"> around
    hits. Store the words, never the markup."""
    cleaned = _HTML_TAG.sub("", text or "")
    return (
        cleaned.replace("&quot;", '"')
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#039;", "'")
        .replace("&nbsp;", " ")
    )


class WikipediaClient:
    """Keyless Wikipedia lookup. Same `search(query, language)` surface as
    `SerperClient`, so the pipeline can hold a list of interchangeable
    providers."""

    name = "wikipedia"

    def __init__(self, timeout_seconds: float = 8.0, max_results: int = 3):
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results

    # -- public API --------------------------------------------------------

    def search(self, query: str, language: str = "en") -> SearchResponse:
        """Look up `query` and return the best-matching article summary.

        Candidates come from two places — the title guessed straight from
        the question, and Wikipedia's own search — and the winner is the
        one whose summary shares the most vocabulary with the question,
        not simply the first one that exists.

        That ranking is the whole point. Taking the direct title hit
        answered "Who founded Apple?" with the article about *the fruit*:
        "Apple" is a real page, so the lookup succeeded and never looked
        further. Scoring candidates puts "Apple Inc." ahead, because its
        summary contains both "Apple" and "founded" while the fruit's
        contains only "Apple".
        """
        query = (query or "").strip()
        if not query:
            raise WikipediaError("Empty search query.")
        lang = language if language in SUPPORTED_LANGUAGES else "en"

        response = SearchResponse(query=query)

        subject = extract_subject(query)
        seen: set[str] = set()
        summaries: list[dict] = []

        direct = self._summary(subject, lang) if subject else None
        if direct is not None:
            summaries.append(direct)
            seen.add(direct["title"].lower())

        # Search as well when the direct hit is less than a perfect match
        # — it may be the wrong sense of an ambiguous name, as "Apple"
        # (the fruit) was for "Who founded Apple?".
        #
        # A perfect score means every significant word of the question
        # already appears in that article, so there is nothing better to
        # find; skipping the search there halves the requests for the
        # commonest shape of all ("What is photosynthesis?") and keeps
        # Aila well clear of Wikimedia's rate limits.
        if direct is None or self._score(query, direct) < 1.0:
            for title in self._search_titles(subject or query, lang):
                if len(summaries) >= self.max_results:
                    break
                if title.lower() in seen:
                    continue
                seen.add(title.lower())
                summary = self._summary(title, lang)
                if summary is not None:
                    summaries.append(summary)

        best = self._best_match(query, summaries)
        if best is not None:
            response.knowledge_graph_title = best["title"]
            response.knowledge_graph_description = best["extract"]
            for position, summary in enumerate(summaries, start=1):
                response.results.append(
                    SearchResult(
                        title=summary["title"],
                        link=summary["url"],
                        snippet=summary["extract"],
                        position=position,
                    )
                )
        return response

    @staticmethod
    def _score(query: str, summary: dict) -> float:
        return lexical_overlap_score(query, f"{summary['title']} {summary['extract']}")

    @classmethod
    def _best_match(cls, query: str, summaries: list[dict]) -> dict | None:
        """Highest word-overlap with the question; ties go to the earlier
        candidate (direct title hit first, then search order)."""
        best = None
        best_score = -1.0
        for summary in summaries:
            score = cls._score(query, summary)
            if score > best_score:
                best, best_score = summary, score
        return best

    # -- HTTP --------------------------------------------------------------

    def _summary(self, title: str, language: str) -> dict | None:
        """REST summary for an exact page title, or None when there is no
        such page / it isn't a usable article."""
        title = (title or "").strip()
        if not title:
            return None
        encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
        url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{encoded}"

        data = self._get_json(url, allow_missing=True)
        if data is None:
            return None

        # 'disambiguation' pages list unrelated subjects; 'no-extract'
        # pages have nothing to say. Neither answers a question.
        if data.get("type") in ("disambiguation", "no-extract"):
            return None
        extract = str(data.get("extract") or "").strip()
        page_title = str(data.get("title") or title).strip()
        if not extract or not page_title:
            return None

        page_url = ""
        content_urls = data.get("content_urls")
        if isinstance(content_urls, dict):
            desktop = content_urls.get("desktop")
            if isinstance(desktop, dict):
                page_url = str(desktop.get("page") or "")
        if not page_url:
            page_url = f"https://{language}.wikipedia.org/wiki/{encoded}"

        return {"title": page_title, "extract": extract, "url": page_url}

    def _search_titles(self, query: str, language: str) -> list[str]:
        """MediaWiki full-text search — used only when the direct title
        guess missed."""
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max(1, self.max_results),
                "format": "json",
                "srnamespace": 0,  # articles only, no Talk:/Category: pages
            }
        )
        url = f"https://{language}.wikipedia.org/w/api.php?{params}"

        data = self._get_json(url, allow_missing=True)
        if not isinstance(data, dict):
            return []
        search = (data.get("query") or {}).get("search")
        if not isinstance(search, list):
            return []
        titles = []
        for item in search:
            if isinstance(item, dict) and item.get("title"):
                titles.append(_strip_html(str(item["title"])))
        return titles

    def _get_json(self, url: str, allow_missing: bool = False):
        """One GET. Returns parsed JSON, or None for a 404 when
        `allow_missing` (a missing page is a normal outcome, not an
        error). Raises a typed WikipediaError otherwise."""
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 404 and allow_missing:
                return None
            if e.code == 429:
                raise WikipediaRateLimitError("Wikipedia rate limit reached (HTTP 429).") from e
            if e.code in (500, 502, 503, 504):
                raise WikipediaUnavailableError(f"Wikipedia is unavailable (HTTP {e.code}).") from e
            raise WikipediaError(f"Wikipedia HTTP error {e.code}.") from e
        except (TimeoutError, socket.timeout) as e:
            raise WikipediaTimeoutError(
                f"Wikipedia request timed out after {self.timeout_seconds}s."
            ) from e
        except urllib.error.URLError as e:
            if isinstance(getattr(e, "reason", None), (TimeoutError, socket.timeout)):
                raise WikipediaTimeoutError(
                    f"Wikipedia request timed out after {self.timeout_seconds}s."
                ) from e
            # DNS failure / connection refused — i.e. no internet.
            raise WikipediaUnavailableError(f"Could not reach Wikipedia: {e.reason}") from e

        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise WikipediaError("Wikipedia returned malformed JSON.") from e
