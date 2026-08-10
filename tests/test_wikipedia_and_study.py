"""Tests for the keyless Wikipedia source and self-directed study.

No test here touches the network: `WikipediaClient` is exercised by
monkeypatching its single HTTP method, so these stay meaningful offline
and in CI (and don't hammer Wikimedia). The one thing a fake cannot
verify — that the real API still returns what we parse — is covered by
the live check documented in docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import time

import pytest

from knowledge.base import KnowledgeBase
from knowledge.store import KnowledgeStore
from knowledge.study import SEED_TOPICS, StudySession, _topic_label
from tools.router import ToolRouter
from webresearch.pipeline import ResearchPipeline
from webresearch.serper import SearchResponse, SearchResult, SerperAuthError
from webresearch.wikipedia import (
    WikipediaClient,
    WikipediaRateLimitError,
    WikipediaUnavailableError,
    _strip_html,
    extract_subject,
)


@pytest.fixture
def store(tmp_path) -> KnowledgeStore:
    s = KnowledgeStore(str(tmp_path / "knowledge.db"))
    yield s
    s.close()


@pytest.fixture
def kb(store) -> KnowledgeBase:
    return KnowledgeBase(store)


# -- subject extraction -------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Who created Samsung?", "Samsung"),
        ("Who founded Bambu Lab?", "Bambu Lab"),
        ("What is Aila Nano?", "Aila Nano"),
        ("Tell me about the Roman Empire", "Roman Empire"),
        ("Quem criou a Samsung?", "Samsung"),
        ("O que é a Petrobras?", "Petrobras"),
        ("Explain gravity", "gravity"),
        ("   ", ""),
        ("", ""),
    ],
)
def test_extract_subject(query, expected):
    assert extract_subject(query) == expected


def test_extract_subject_leaves_unrecognized_questions_alone():
    """A question the stripper doesn't understand must stay intact rather
    than be mangled into a wrong page title — the full text still works
    as a search query, which is the safe fallback."""
    assert extract_subject("Mrbest has how much subscribers on Youtube?") == (
        "Mrbest has how much subscribers on Youtube"
    )


def test_strip_html_removes_search_markup():
    raw = 'Ragam <span class="searchmatch">bambu</span> &quot;dan&quot; kayu'
    assert _strip_html(raw) == 'Ragam bambu "dan" kayu'
    assert "<" not in _strip_html(raw)


# -- client behaviour (no network) --------------------------------------------


class FakeWikipedia(WikipediaClient):
    """Real client with only the HTTP call replaced, so URL building,
    parsing, candidate ranking and error mapping are all exercised."""

    def __init__(self, pages: dict, search_hits: dict | None = None, error=None, **kwargs):
        super().__init__(**kwargs)
        self.pages = pages  # title -> summary payload
        self.search_hits = search_hits or {}
        self.error = error
        self.requests: list[str] = []

    def _get_json(self, url: str, allow_missing: bool = False):
        self.requests.append(url)
        if self.error is not None:
            raise self.error
        if "/page/summary/" in url:
            title = url.rsplit("/", 1)[-1].replace("_", " ")
            import urllib.parse

            title = urllib.parse.unquote(title)
            # Real Wikipedia capitalizes the first letter of a title, so
            # "photosynthesis" and "Photosynthesis" are the same page.
            # The fake matches that, otherwise a lowercase subject guess
            # would "404" here but succeed against the real API.
            page = self.pages.get(title) or self.pages.get(title[:1].upper() + title[1:])
            if page is None:
                return None  # 404, same as the real client
            return page
        # search
        import urllib.parse

        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("srsearch", [""])[0]
        titles = self.search_hits.get(query, [])
        return {"query": {"search": [{"title": t} for t in titles]}}


def _page(title, extract, page_type="standard"):
    return {
        "type": page_type,
        "title": title,
        "extract": extract,
        "content_urls": {"desktop": {"page": f"https://en.wikipedia.org/wiki/{title}"}},
    }


def test_client_returns_a_summary_for_a_direct_title_hit():
    client = FakeWikipedia(
        pages={"Samsung": _page("Samsung", "Samsung Group is a South Korean conglomerate.")}
    )
    response = client.search("Who created Samsung?")
    assert response.knowledge_graph_title == "Samsung"
    assert "South Korean" in response.knowledge_graph_description
    assert response.results[0].domain == "en.wikipedia.org"


def test_client_prefers_the_candidate_that_matches_the_question():
    """The bug this ranking exists for: "Who founded Apple?" resolved to
    the article about the *fruit*, because "Apple" is a real page and the
    direct hit was taken without looking further."""
    client = FakeWikipedia(
        pages={
            "Apple": _page("Apple", "An apple is the round, edible fruit of an apple tree."),
            "Apple Inc.": _page(
                "Apple Inc.",
                "Apple Inc. is an American technology company founded in 1976 by Steve Jobs.",
            ),
        },
        search_hits={"Apple": ["Apple", "Apple Inc."]},
    )
    response = client.search("Who founded Apple?")
    assert response.knowledge_graph_title == "Apple Inc."
    assert "Steve Jobs" in response.knowledge_graph_description


def test_client_falls_back_to_search_when_the_title_guess_misses():
    client = FakeWikipedia(
        pages={"Rayleigh scattering": _page("Rayleigh scattering", "Rayleigh scattering makes the sky blue.")},
        search_hits={"is the sky blue": ["Rayleigh scattering"]},
    )
    response = client.search("why is the sky blue")
    assert response.knowledge_graph_title == "Rayleigh scattering"


def test_client_skips_disambiguation_pages():
    client = FakeWikipedia(
        pages={"Mercury": _page("Mercury", "Mercury may refer to:", page_type="disambiguation")},
        search_hits={"Mercury": []},
    )
    response = client.search("What is Mercury?")
    assert response.knowledge_graph_description is None
    assert response.results == []


def test_client_returns_empty_rather_than_raising_when_nothing_exists():
    client = FakeWikipedia(pages={}, search_hits={})
    response = client.search("Who created Hames Eventos?")
    assert response.results == []
    assert response.knowledge_graph_description is None


def test_client_never_requests_an_unsupported_language():
    client = FakeWikipedia(pages={"Samsung": _page("Samsung", "A conglomerate.")})
    client.search("Who created Samsung?", language="de")
    assert all("//en.wikipedia.org" in url for url in client.requests)

    client2 = FakeWikipedia(pages={"Petrobras": _page("Petrobras", "Uma empresa brasileira.")})
    client2.search("Quem criou a Petrobras?", language="pt")
    assert all("//pt.wikipedia.org" in url for url in client2.requests)


def test_client_rejects_an_empty_query():
    with pytest.raises(Exception):
        FakeWikipedia(pages={}).search("   ")


# -- pipeline: multiple providers ---------------------------------------------


class FakeSerper:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0

    def search(self, query, language="en"):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.response


def test_wikipedia_is_tried_before_serper(store, kb):
    wiki = FakeWikipedia(pages={"Samsung": _page("Samsung", "Samsung Group is a Korean conglomerate.")})
    serper = FakeSerper(response=SearchResponse(query="q", results=[]))
    pipe = ResearchPipeline(serper, store, kb, wikipedia=wiki)

    out = pipe.research("Who created Samsung?")
    assert out.ok
    assert out.provider == "wikipedia"
    assert serper.calls == 0, "Serper must not be paid for when Wikipedia answered"


def test_serper_covers_what_wikipedia_has_no_article_for(store, kb):
    wiki = FakeWikipedia(pages={}, search_hits={})
    serper = FakeSerper(
        response=SearchResponse(
            query="q",
            results=[
                SearchResult(
                    title="Hames Eventos",
                    link="https://hameseventos.example.com/",
                    snippet="Hames Eventos is an events company created by the Hames family.",
                    position=1,
                )
            ],
        )
    )
    pipe = ResearchPipeline(serper, store, kb, wikipedia=wiki)

    out = pipe.research("Who created Hames Eventos?")
    assert out.ok
    assert out.provider == "serper"
    assert serper.calls == 1


def test_a_dead_serper_key_does_not_stop_wikipedia(store, kb):
    """The exact situation that prompted this feature: the configured key
    was cancelled. Aila must keep working."""
    wiki = FakeWikipedia(pages={"Samsung": _page("Samsung", "Samsung Group is a Korean conglomerate.")})
    serper = FakeSerper(error=SerperAuthError("Serper rejected the API key (HTTP 403)."))
    router = ToolRouter(knowledge=kb, research=ResearchPipeline(serper, store, kb, wikipedia=wiki))

    result = router.route("Who created Samsung?")
    assert result.direct_reply is not None
    assert "Korean conglomerate" in result.direct_reply
    assert "key" not in result.direct_reply.lower()


def test_a_dead_key_is_reported_only_when_nothing_else_answered(store, kb):
    wiki = FakeWikipedia(pages={}, search_hits={})
    serper = FakeSerper(error=SerperAuthError("Serper rejected the API key (HTTP 403)."))
    router = ToolRouter(knowledge=kb, research=ResearchPipeline(serper, store, kb, wikipedia=wiki))

    result = router.route("Who created Hames Eventos?")
    # Wikipedia searched and found nothing, so "I found nothing" is the
    # honest headline — not an auth error on a redundant second source.
    assert result.tool_used == "web_no_answer"


def test_only_an_unreachable_network_reports_a_connection_error(store, kb):
    wiki = FakeWikipedia(pages={}, error=WikipediaUnavailableError("no route to host"))
    router = ToolRouter(knowledge=kb, research=ResearchPipeline(None, store, kb, wikipedia=wiki))

    result = router.route("Who created Samsung?")
    assert result.tool_used == "web_error:search_failed"
    assert "couldn't reach the web" in result.direct_reply


def test_wikipedia_rate_limit_is_reported_as_temporary(store, kb):
    wiki = FakeWikipedia(pages={}, error=WikipediaRateLimitError("429"))
    router = ToolRouter(knowledge=kb, research=ResearchPipeline(None, store, kb, wikipedia=wiki))
    result = router.route("Who created Samsung?")
    assert result.tool_used == "web_error:rate_limited"


def test_an_off_topic_article_is_never_served(store, kb):
    """A wrong page-title guess must not become a confident wrong answer.
    Wikipedia summaries arrive from a guess, unlike Google's knowledge
    graph, so they get the same on-topic gate the snippets get."""
    wiki = FakeWikipedia(
        pages={"Cake": _page("Cake", "Cake is a flour confection made from flour, sugar and eggs.")},
        search_hits={"president of Brazil": ["Cake"]},
    )
    pipe = ResearchPipeline(None, store, kb, wikipedia=wiki)
    out = pipe.research("Who is the president of Brazil?")
    assert out.ok is False
    assert store.all_knowledge() == [], "an off-topic answer must not be learned either"


def test_wikipedia_and_serper_use_separate_cache_entries(store, kb):
    wiki = FakeWikipedia(pages={}, search_hits={})
    serper = FakeSerper(
        response=SearchResponse(
            query="q",
            results=[
                SearchResult(
                    title="Hames Eventos",
                    link="https://hameseventos.example.com/",
                    snippet="Hames Eventos is an events company created by the Hames family.",
                    position=1,
                )
            ],
        )
    )
    pipe = ResearchPipeline(serper, store, kb, wikipedia=wiki)

    pipe.research("Who created Hames Eventos?")
    wiki_requests = len(wiki.requests)
    # Second time: both providers served from their own cache entries, so
    # neither makes a fresh request.
    pipe.research("Who created Hames Eventos?")
    assert serper.calls == 1
    assert len(wiki.requests) == wiki_requests


# -- offline circuit breaker --------------------------------------------------


def test_offline_breaker_stops_dialling_out_after_a_connection_failure(store, kb):
    """Without this, every question asked with no internet pays a full
    timeout per provider before failing."""
    wiki = FakeWikipedia(pages={}, error=WikipediaUnavailableError("no route to host"))
    pipe = ResearchPipeline(None, store, kb, wikipedia=wiki, offline_cooldown_seconds=60)

    assert pipe.offline is False
    pipe.research("Who created Samsung?")
    assert pipe.offline is True

    attempts = len(wiki.requests)
    pipe.research("Who founded Apple?")
    assert len(wiki.requests) == attempts, "no network call while the breaker is open"


def test_offline_breaker_expires(store, kb):
    wiki = FakeWikipedia(pages={}, error=WikipediaUnavailableError("no route to host"))
    pipe = ResearchPipeline(None, store, kb, wikipedia=wiki, offline_cooldown_seconds=0.01)
    pipe.research("Who created Samsung?")
    assert pipe.offline is True
    time.sleep(0.02)
    assert pipe.offline is False


def test_a_rejected_key_does_not_look_like_being_offline(store, kb):
    """Reaching the server and being turned away proves the network is
    up. Tripping the offline breaker there would wrongly suppress
    Wikipedia lookups for a minute."""
    serper = FakeSerper(error=SerperAuthError("Serper rejected the API key (HTTP 403)."))
    pipe = ResearchPipeline(serper, store, kb)
    pipe.research("Who created Samsung?")
    assert pipe.offline is False


def test_known_answers_still_work_while_offline(store, kb):
    """The point of learning: what Aila already knows needs no network."""
    kb.store.add_knowledge(
        "Who founded Apple?",
        "Apple was founded by Steve Jobs, Steve Wozniak and Ronald Wayne.",
        confidence=0.9,
    )
    wiki = FakeWikipedia(pages={}, error=WikipediaUnavailableError("offline"))
    router = ToolRouter(knowledge=kb, research=ResearchPipeline(None, store, kb, wikipedia=wiki))

    result = router.route("Who founded Apple?")
    assert result.tool_used == "knowledge"
    assert "Steve Jobs" in result.direct_reply
    assert wiki.requests == [], "a known answer must not touch the network at all"


# -- self-directed study ------------------------------------------------------


def _study_pipeline(store, kb, pages=None, **kwargs):
    wiki = FakeWikipedia(pages=pages or {}, **kwargs)
    return ResearchPipeline(None, store, kb, wikipedia=wiki), wiki


def test_study_learns_and_records_that_it_ran(store, kb):
    # A question the user asked and didn't get answered — the highest
    # value thing for Aila to go and learn.
    store.add_candidate("What is DNA?", "unclear", confidence=0.2, reason="low_confidence")
    pipe, _ = _study_pipeline(
        store, kb, pages={"DNA": _page("DNA", "DNA is a molecule carrying genetic instructions.")}
    )
    session = StudySession(store, pipe, max_topics=1)

    assert session.due() is True
    report = session.run()
    assert report.skipped is False
    assert report.studied == 1
    assert store.all_knowledge(), "what was studied must actually be stored"

    # ...and it does not run again on the next start.
    assert session.due() is False
    assert session.run().skipped is True


def test_study_prefers_questions_the_user_actually_asked(store, kb):
    store.add_candidate("Who created Hames Eventos?", "unclear", confidence=0.2, reason="low_confidence")
    pipe, _ = _study_pipeline(store, kb)
    session = StudySession(store, pipe, max_topics=2)

    topics = session.pick_topics()
    assert topics[0] == "Who created Hames Eventos?"
    assert topics[1] in SEED_TOPICS


def test_study_does_not_restudy_what_is_already_known(store, kb):
    store.add_knowledge(SEED_TOPICS[0], "An answer.", confidence=0.8)
    pipe, _ = _study_pipeline(store, kb)
    session = StudySession(store, pipe, max_topics=3)
    assert SEED_TOPICS[0] not in session.pick_topics()


def test_study_survives_a_provider_that_explodes(store, kb):
    class Exploding:
        offline = False

        def research(self, query):
            raise RuntimeError("boom")

    session = StudySession(store, Exploding(), max_topics=2)
    report = session.run()
    assert report.studied == 0
    assert report.failed == 2
    # It still records that it ran — otherwise a permanently broken
    # provider means retrying on every single startup.
    assert session.due() is False


def test_study_stops_early_when_the_network_drops(store, kb):
    wiki = FakeWikipedia(pages={}, error=WikipediaUnavailableError("offline"))
    pipe = ResearchPipeline(None, store, kb, wikipedia=wiki, offline_cooldown_seconds=60)
    session = StudySession(store, pipe, max_topics=5)

    report = session.run()
    assert report.studied == 0
    # One attempt trips the breaker; the remaining four are abandoned
    # rather than each waiting for its own timeout.
    assert report.failed == 1


def test_study_is_due_again_after_the_interval(store, kb):
    pipe, _ = _study_pipeline(store, kb)
    session = StudySession(store, pipe, max_topics=1, interval_seconds=0.01)
    session.mark_ran()
    assert session.due() is False
    time.sleep(0.02)
    assert session.due() is True


def test_study_recovers_from_a_corrupt_or_future_timestamp(store, kb):
    pipe, _ = _study_pipeline(store, kb)
    session = StudySession(store, pipe, max_topics=1)

    store.set_meta("last_study_at", "not-a-number")
    assert session.due() is True

    # A clock that jumped backwards must not block study forever.
    store.set_meta("last_study_at", str(time.time() + 10_000))
    assert session.due() is True


def test_forced_study_ignores_the_schedule(store, kb):
    pipe, _ = _study_pipeline(
        store, kb, pages={"DNA": _page("DNA", "DNA is a molecule carrying genetic instructions.")}
    )
    session = StudySession(store, pipe, max_topics=1)
    session.mark_ran()
    assert session.due() is False
    assert session.run(force=True).skipped is False


@pytest.mark.parametrize(
    "question,label",
    [
        ("What is DNA?", "DNA"),
        ("What is the Internet?", "Internet"),
        ("Who created Hames Eventos?", "Hames Eventos"),
        ("Photosynthesis", "Photosynthesis"),
    ],
)
def test_topic_labels_read_naturally(question, label):
    assert _topic_label(question) == label


def test_study_report_summary_is_empty_when_nothing_happened(store, kb):
    from knowledge.study import StudyReport

    assert StudyReport(skipped=True).summary() == ""
    assert StudyReport().summary() == ""
    assert "couldn't reach" in StudyReport(failed=2).summary()
    assert "Studied 1" in StudyReport(learned=["DNA"]).summary()


def test_a_perfect_direct_hit_skips_the_extra_search():
    """Request budget matters: Wikimedia rate-limits, and the commonest
    question shape ("What is X?") needs only one lookup."""
    client = FakeWikipedia(
        pages={"Photosynthesis": _page("Photosynthesis", "Photosynthesis converts light into chemical energy.")},
        search_hits={"Photosynthesis": ["Photosynthesis", "Chlorophyll"]},
    )
    response = client.search("What is photosynthesis?")
    assert response.knowledge_graph_title == "Photosynthesis"
    assert len(client.requests) == 1
    assert all("/page/summary/" in url for url in client.requests)


def test_an_imperfect_direct_hit_still_searches():
    client = FakeWikipedia(
        pages={
            "Apple": _page("Apple", "An apple is the round, edible fruit of an apple tree."),
            "Apple Inc.": _page("Apple Inc.", "Apple Inc. is a technology company founded by Steve Jobs."),
        },
        search_hits={"Apple": ["Apple Inc."]},
    )
    response = client.search("Who founded Apple?")
    assert response.knowledge_graph_title == "Apple Inc."
    assert any("srsearch" in url for url in client.requests)


# -- engine.study reporting ---------------------------------------------------


def test_as_question_does_not_wrap_something_already_a_question():
    """"/study who created Samsung" used to be researched as "What is who
    created Samsung?", which searches for nothing at all."""
    from engine.state import _as_question

    assert _as_question("photosynthesis") == "What is photosynthesis?"
    assert _as_question("who created Samsung") == "who created Samsung?"
    assert _as_question("What is DNA?") == "What is DNA?"
    assert _as_question("Quem criou a Petrobras") == "Quem criou a Petrobras?"
    assert _as_question("  ") == ""


class _StubResearch:
    """Minimal stand-in so study reporting can be tested per outcome."""

    offline = False

    def __init__(self, outcome):
        self.outcome = outcome

    def research(self, query):
        return self.outcome


def _engine_with(outcome, tmp_path, tokenizer, monkeypatch):
    from engine import AilaEngine, EngineSettings

    for var, value in {
        "AILA_CHECKPOINT": str(tmp_path / "missing.pt"),
        "AILA_FALLBACK_CHECKPOINT": str(tmp_path / "missing2.pt"),
        "AILA_TOKENIZER": tokenizer.model_path,
        "AILA_MEMORY_DB": str(tmp_path / "m.db"),
        "AILA_MEMORY_FAISS": str(tmp_path / "m.faiss"),
        "AILA_KNOWLEDGE_DB": str(tmp_path / "k.db"),
        "AILA_KNOWLEDGE_FAISS": str(tmp_path / "k.faiss"),
        "AILA_KNOWLEDGE_STORE_DB": str(tmp_path / "ks.db"),
        "AILA_DEVICE": "cpu",
    }.items():
        monkeypatch.setenv(var, value)

    engine = AilaEngine(EngineSettings())
    engine.router.research = _StubResearch(outcome)
    return engine


@pytest.mark.parametrize(
    "stored,learned,expected",
    [
        ("created", True, "Learned it."),
        ("updated", True, "Learned it."),
        ("conflict", False, "disagrees with something I already know"),
        ("rejected", False, "wasn't solid enough"),
        # The bug this covers: `stored=None` means the answer was found
        # but was too low-confidence to keep. Reporting that as "I
        # already knew this" claimed a success that never happened.
        (None, False, "not confident enough"),
    ],
)
def test_study_reports_each_storage_outcome_honestly(
    stored, learned, expected, tmp_path, tokenizer, monkeypatch
):
    from webresearch.pipeline import ResearchOutcome

    outcome = ResearchOutcome(ok=True, answer="Some answer.", confidence=0.6, stored=stored)
    with _engine_with(outcome, tmp_path, tokenizer, monkeypatch) as engine:
        got_learned, message = engine.study("photosynthesis")
        assert got_learned is learned
        assert expected in message


def test_study_reports_a_miss(tmp_path, tokenizer, monkeypatch):
    from webresearch.pipeline import ResearchOutcome

    outcome = ResearchOutcome(ok=False, reason="no_results")
    with _engine_with(outcome, tmp_path, tokenizer, monkeypatch) as engine:
        learned, message = engine.study("Hames Eventos")
        assert learned is False
        assert "couldn't find anything reliable" in message


def test_study_survives_a_research_layer_that_raises(tmp_path, tokenizer, monkeypatch):
    class Exploding:
        offline = False

        def research(self, query):
            raise RuntimeError("boom")

    from engine import AilaEngine, EngineSettings

    for var, value in {
        "AILA_CHECKPOINT": str(tmp_path / "missing.pt"),
        "AILA_FALLBACK_CHECKPOINT": str(tmp_path / "missing2.pt"),
        "AILA_TOKENIZER": tokenizer.model_path,
        "AILA_MEMORY_DB": str(tmp_path / "m.db"),
        "AILA_MEMORY_FAISS": str(tmp_path / "m.faiss"),
        "AILA_KNOWLEDGE_DB": str(tmp_path / "k.db"),
        "AILA_KNOWLEDGE_FAISS": str(tmp_path / "k.faiss"),
        "AILA_KNOWLEDGE_STORE_DB": str(tmp_path / "ks.db"),
        "AILA_DEVICE": "cpu",
    }.items():
        monkeypatch.setenv(var, value)

    with AilaEngine(EngineSettings()) as engine:
        engine.router.research = Exploding()
        learned, message = engine.study("photosynthesis")
        assert learned is False
        assert "couldn't study that" in message
