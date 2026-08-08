"""Serper (google.serper.dev) search client.

Security invariants:
- The API key is read from configuration/environment ONLY — never
  hardcoded, never logged, never included in error messages.
- Everything returned by the API is untrusted external DATA. This module
  parses it into plain dataclasses; nothing downstream may treat any of
  it as instructions (see webresearch/quality.py's sanitization and the
  prompt-side [WEB]/data framing in webresearch/pipeline.py).

Uses urllib from the standard library rather than adding an HTTP client
dependency — one POST endpoint doesn't justify one.
"""

from __future__ import annotations

import json
import logging
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

SERPER_ENDPOINT = "https://google.serper.dev/search"


class SerperError(Exception):
    """Base class for all Serper client failures."""


class SerperAuthError(SerperError):
    """Invalid or missing API key (HTTP 401/403)."""


class SerperRateLimitError(SerperError):
    """Rate limited (HTTP 429)."""


class SerperTimeoutError(SerperError):
    """The request timed out."""


@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str
    position: int
    domain: str = ""

    def __post_init__(self):
        if not self.domain:
            try:
                self.domain = urlparse(self.link).netloc.lower()
            except ValueError:
                self.domain = ""


@dataclass
class SearchResponse:
    query: str
    results: list[SearchResult] = field(default_factory=list)
    # Serper's direct-answer features, when present. Highest-value
    # extraction targets: Google has already done the synthesis.
    answer_box_answer: str | None = None
    answer_box_snippet: str | None = None
    knowledge_graph_title: str | None = None
    knowledge_graph_description: str | None = None
    knowledge_graph_attributes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "results": [vars(r) for r in self.results],
            "answer_box_answer": self.answer_box_answer,
            "answer_box_snippet": self.answer_box_snippet,
            "knowledge_graph_title": self.knowledge_graph_title,
            "knowledge_graph_description": self.knowledge_graph_description,
            "knowledge_graph_attributes": self.knowledge_graph_attributes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SearchResponse:
        results = [SearchResult(**r) for r in d.get("results", [])]
        return cls(
            query=d.get("query", ""),
            results=results,
            answer_box_answer=d.get("answer_box_answer"),
            answer_box_snippet=d.get("answer_box_snippet"),
            knowledge_graph_title=d.get("knowledge_graph_title"),
            knowledge_graph_description=d.get("knowledge_graph_description"),
            knowledge_graph_attributes=d.get("knowledge_graph_attributes", {}),
        )


class SerperClient:
    def __init__(self, api_key: str, timeout_seconds: float = 8.0, max_results: int = 5):
        if not api_key:
            raise SerperAuthError(
                "No Serper API key configured. Set SERPER_API_KEY in your environment "
                "or .env file (see .env.example)."
            )
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results

    def search(self, query: str, language: str = "en") -> SearchResponse:
        """One search request. Raises a typed SerperError subclass on any
        failure; never returns partial garbage."""
        query = (query or "").strip()
        if not query:
            raise SerperError("Empty search query.")

        payload = {"q": query, "num": self.max_results}
        if language == "pt":
            payload.update({"gl": "br", "hl": "pt-br"})

        request = urllib.request.Request(
            SERPER_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "X-API-KEY": self._api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            # Note: no API key and no response body in any of these messages.
            if e.code in (401, 403):
                raise SerperAuthError("Serper rejected the API key (HTTP %d)." % e.code) from e
            if e.code == 429:
                raise SerperRateLimitError("Serper rate limit reached (HTTP 429).") from e
            raise SerperError(f"Serper HTTP error {e.code}.") from e
        except (TimeoutError, socket.timeout) as e:
            raise SerperTimeoutError(
                f"Serper request timed out after {self.timeout_seconds}s."
            ) from e
        except urllib.error.URLError as e:
            if isinstance(getattr(e, "reason", None), (TimeoutError, socket.timeout)):
                raise SerperTimeoutError(
                    f"Serper request timed out after {self.timeout_seconds}s."
                ) from e
            raise SerperError(f"Serper request failed: {e.reason}") from e

        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            raise SerperError("Serper returned malformed JSON.") from e
        return self._parse(query, data)

    def _parse(self, query: str, data: dict) -> SearchResponse:
        if not isinstance(data, dict):
            raise SerperError("Serper returned an unexpected response shape.")

        response = SearchResponse(query=query)

        for i, item in enumerate(data.get("organic", []) or []):
            if not isinstance(item, dict):
                continue
            link = str(item.get("link", "") or "")
            title = str(item.get("title", "") or "")
            snippet = str(item.get("snippet", "") or "")
            if not link or not title:
                continue
            response.results.append(
                SearchResult(title=title, link=link, snippet=snippet, position=i + 1)
            )

        answer_box = data.get("answerBox")
        if isinstance(answer_box, dict):
            response.answer_box_answer = _clean_str(answer_box.get("answer"))
            response.answer_box_snippet = _clean_str(answer_box.get("snippet"))

        kg = data.get("knowledgeGraph")
        if isinstance(kg, dict):
            response.knowledge_graph_title = _clean_str(kg.get("title"))
            response.knowledge_graph_description = _clean_str(kg.get("description"))
            attributes = kg.get("attributes")
            if isinstance(attributes, dict):
                response.knowledge_graph_attributes = {
                    str(k): str(v) for k, v in attributes.items()
                }

        return response


def _clean_str(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None
