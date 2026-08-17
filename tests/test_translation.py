"""Translation: the deep-translator wrapper's defensive behaviour, and
the additive Portuguese->English->Portuguese fallback in the agent.

No test here touches the network. The wrapper is exercised with its
backend monkeypatched, and the agent fallback with a fake router and a
fake translator, so the routing *logic* — native first, translate only on
a miss, never degrade a confident native answer — is what's under test,
not Google Translate.
"""

from __future__ import annotations

import pytest

from tools.router import RouteResult
from translation.translator import Translator


# -- the wrapper --------------------------------------------------------------


class _Backend:
    """Stand-in for deep_translator.GoogleTranslator."""

    def __init__(self, source="auto", target="en", table=None, error=None):
        self.source = source
        self.target = target
        self._table = table or {}
        self._error = error

    def __call__(self, source, target):
        return _Backend(source, target, self._table, self._error)

    def translate(self, text):
        if self._error is not None:
            raise self._error
        return self._table.get(text, f"[{self.target}]{text}")


def _translator_with(table=None, error=None):
    t = Translator(enabled=True)
    t.available = True
    t._backend = _Backend(table=table, error=error)
    # A fresh cache per test, so one test's stubbed result can't leak.
    from translation import translator as mod

    mod._cached_translate.cache_clear()
    return t


def test_disabled_translator_is_a_passthrough():
    t = Translator(enabled=False)
    assert t.available is False
    assert t.to_english("Olá") == "Olá"
    assert t.to_portuguese("Hello") == "Hello"


def test_translation_round_trips():
    t = _translator_with(
        {"Quem criou a Samsung?": "Who created Samsung?", "The answer.": "A resposta."}
    )
    assert t.to_english("Quem criou a Samsung?") == "Who created Samsung?"
    assert t.to_portuguese("The answer.") == "A resposta."


def test_empty_text_is_never_translated():
    t = _translator_with()
    assert t.to_english("") == ""
    assert t.to_english("   ") == "   "


def test_a_backend_error_falls_back_to_the_original_text():
    t = _translator_with(error=RuntimeError("network down"))
    # Passthrough, not a crash — a failed translation must never break a turn.
    assert t.to_english("Quem criou a Samsung?") == "Quem criou a Samsung?"


def test_a_google_error_page_is_not_served_as_a_translation():
    """deep-translator can hand back Google's HTML/text error page as an
    ordinary string. Serving it as 'the translation' is worse than not
    translating, so it is treated as a failure."""
    t = _translator_with(
        {"Quem criou a Samsung?": "Error 500 (Server Error)!!1500. That’s an error."}
    )
    assert t.to_english("Quem criou a Samsung?") == "Quem criou a Samsung?"


def test_very_long_text_is_left_untranslated():
    from translation.translator import MAX_TRANSLATE_CHARS

    t = _translator_with()
    long_text = "a" * (MAX_TRANSLATE_CHARS + 1)
    assert t.to_english(long_text) == long_text


def test_unavailable_backend_makes_everything_passthrough(monkeypatch):
    """deep-translator not installed -> available False -> passthrough,
    with no branching required of callers."""
    import builtins

    real_import = builtins.__import__

    def no_deep_translator(name, *args, **kwargs):
        if name == "deep_translator":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_deep_translator)
    t = Translator(enabled=True)
    assert t.available is False
    assert t.to_english("Olá") == "Olá"


# -- the agent fallback -------------------------------------------------------


class _FakeRouter:
    """Returns a canned RouteResult per exact message."""

    def __init__(self, table):
        self.table = table
        self.seen: list[str] = []

    def route(self, message, previous_user_message=None):
        self.seen.append(message)
        return self.table.get(message, RouteResult())


class _FakeTranslator:
    available = True

    def __init__(self, to_en=None, to_pt=None):
        self._to_en = to_en or {}
        self._to_pt = to_pt or {}

    def to_english(self, text):
        return self._to_en.get(text, text)

    def to_portuguese(self, text):
        return self._to_pt.get(text, text)


def _agent(tiny_model, tokenizer, router, translator):
    from agents.registry import get_agent

    return get_agent("general", tiny_model, tokenizer, router=router, translator=translator)


def test_a_confident_native_answer_is_never_translated(tiny_model, tokenizer):
    router = _FakeRouter(
        {"Quem criou você?": RouteResult(direct_reply="Fui criado pela Aila.", tool_used="identity:creator")}
    )
    translator = _FakeTranslator()
    agent = _agent(tiny_model, tokenizer, router, translator)

    reply, snippets, tool = agent._resolve_route("c1", "Quem criou você?")
    assert reply == "Fui criado pela Aila."
    assert router.seen == ["Quem criou você?"]  # no English retry


def test_a_portuguese_miss_is_retried_in_english_and_translated_back(tiny_model, tokenizer):
    router = _FakeRouter(
        {
            # Native Portuguese: nothing.
            "Quem fundou a Creality?": RouteResult(),
            # Translated English: a confident web answer.
            "Who founded Creality?": RouteResult(
                direct_reply="Creality is a Chinese 3D printer company.", tool_used="web_research"
            ),
        }
    )
    translator = _FakeTranslator(
        to_en={"Quem fundou a Creality?": "Who founded Creality?"},
        to_pt={"Creality is a Chinese 3D printer company.": "A Creality é uma empresa chinesa."},
    )
    agent = _agent(tiny_model, tokenizer, router, translator)

    reply, _, tool = agent._resolve_route("c1", "Quem fundou a Creality?")
    assert reply == "A Creality é uma empresa chinesa."
    assert tool == "translated:web_research"


def test_a_soft_miss_still_gets_the_english_retry(tiny_model, tokenizer):
    """A Portuguese memory-miss is a `direct_reply`, but it means "I don't
    have this" — the English retry can still find an English-stored
    memory, and a confident answer beats the soft miss."""
    router = _FakeRouter(
        {
            "Qual é o meu nome?": RouteResult(
                direct_reply="Ainda não tenho isso na memória.", tool_used="memory_miss"
            ),
            "What is my name?": RouteResult(direct_reply="Your name is Theo.", tool_used="memory"),
        }
    )
    translator = _FakeTranslator(
        to_en={"Qual é o meu nome?": "What is my name?"},
        to_pt={"Your name is Theo.": "Seu nome é Theo."},
    )
    agent = _agent(tiny_model, tokenizer, router, translator)

    reply, _, tool = agent._resolve_route("c1", "Qual é o meu nome?")
    assert reply == "Seu nome é Theo."
    assert tool == "translated:memory"


def test_the_native_soft_miss_is_kept_when_english_also_misses(tiny_model, tokenizer):
    router = _FakeRouter(
        {
            "Qual é o meu nome?": RouteResult(
                direct_reply="Ainda não tenho isso na memória.", tool_used="memory_miss"
            ),
            "What is my name?": RouteResult(
                direct_reply="I don't have that yet.", tool_used="memory_miss"
            ),
        }
    )
    translator = _FakeTranslator(to_en={"Qual é o meu nome?": "What is my name?"})
    agent = _agent(tiny_model, tokenizer, router, translator)

    reply, _, tool = agent._resolve_route("c1", "Qual é o meu nome?")
    # The Portuguese soft-miss is the right thing to show, not a translated
    # English one.
    assert reply == "Ainda não tenho isso na memória."


def test_a_transient_web_error_is_not_retried_via_translation(tiny_model, tokenizer):
    """A rate limit hits the same backend in either language — translating
    the query cannot fix it, so the native error message is kept."""
    router = _FakeRouter(
        {
            "Quem fundou a Creality?": RouteResult(
                direct_reply="Atingi o limite de buscas.", tool_used="web_error:rate_limited"
            )
        }
    )
    translator = _FakeTranslator(to_en={"Quem fundou a Creality?": "Who founded Creality?"})
    agent = _agent(tiny_model, tokenizer, router, translator)

    reply, _, tool = agent._resolve_route("c1", "Quem fundou a Creality?")
    assert reply == "Atingi o limite de buscas."
    assert "Who founded Creality?" not in router.seen  # never retried


def test_english_messages_are_never_translated(tiny_model, tokenizer):
    router = _FakeRouter({"Who founded Creality?": RouteResult()})
    translator = _FakeTranslator()
    agent = _agent(tiny_model, tokenizer, router, translator)

    reply, _, _ = agent._resolve_route("c1", "Who founded Creality?")
    assert reply is None
    assert router.seen == ["Who founded Creality?"]  # routed once, in English


def test_no_translator_means_native_only(tiny_model, tokenizer):
    router = _FakeRouter(
        {"Quem fundou a Creality?": RouteResult(direct_reply="Não encontrei.", tool_used="web_no_answer")}
    )
    agent = _agent(tiny_model, tokenizer, router, translator=None)

    reply, _, tool = agent._resolve_route("c1", "Quem fundou a Creality?")
    assert reply == "Não encontrei."


def test_a_passthrough_translation_does_not_trigger_a_pointless_retry(tiny_model, tokenizer):
    """If translation returns the text unchanged (offline / error), the
    native route already covered it — there is nothing new to route."""
    router = _FakeRouter({"Quem fundou a Creality?": RouteResult()})
    # to_english returns the same string (nothing stubbed).
    translator = _FakeTranslator()
    agent = _agent(tiny_model, tokenizer, router, translator)

    reply, _, _ = agent._resolve_route("c1", "Quem fundou a Creality?")
    assert reply is None
    assert router.seen == ["Quem fundou a Creality?"]  # not routed a second time
