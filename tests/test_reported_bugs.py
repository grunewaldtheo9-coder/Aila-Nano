"""Regression tests for bugs found by the user in real conversation.

Each test names the bug and uses the *verbatim* message that triggered
it, so a future change that reintroduces the behaviour fails here with an
obvious explanation rather than a generic assertion.

Reported (transcript, three distinct problems):
  "Have some bugs; he repeats things some time he doesn't search when he
   has. One big problem is she doesn't say everything, making it
   incomplete, and saying ....saying"

1. Incomplete answers — every web-sourced answer trailed off in "...".
2. Doesn't search / doesn't use what it has — questions that should have
   been answered from the web came back as generated nonsense.
3. Repeats itself — unfamiliar short messages ("Bro", "Ok", "nice!") all
   produced the same canned greeting.
"""

from __future__ import annotations

import pytest

from knowledge.base import KnowledgeBase
from knowledge.store import KnowledgeStore
from tools.identity import match_identity_question
from tools.router import ToolRouter
from tools.smalltalk import match_smalltalk
from webresearch.pipeline import ResearchPipeline
from webresearch.quality import complete_sentence, looks_truncated
from webresearch.serper import (
    SearchResponse,
    SearchResult,
    SerperAuthError,
    SerperRateLimitError,
    SerperTimeoutError,
)


@pytest.fixture
def store(tmp_path) -> KnowledgeStore:
    s = KnowledgeStore(str(tmp_path / "knowledge.db"))
    yield s
    s.close()


@pytest.fixture
def kb(store) -> KnowledgeBase:
    return KnowledgeBase(store)


class FakeSerperClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0

    def search(self, query, language="en"):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.response


# -- bug 1: "she doesn't say everything ... and saying ....saying" ------------

# The exact strings Aila served, copied from the transcript.
SAMSUNG_CUT_OFF = (
    "Founded in 1938 by Lee Byung-chul as a trading company, Samsung "
    "diversified into various sectors, including food processing, textiles, "
    "insurance, securities, ..."
)
BAMBU_CUT_OFF = "Bambu Lab was founded in 2020 by Dr. Ye Tao, a former head of DJI ..."


@pytest.mark.parametrize("text", [SAMSUNG_CUT_OFF, BAMBU_CUT_OFF])
def test_cut_off_web_text_is_repaired_into_a_finished_sentence(text):
    assert looks_truncated(text)
    repaired = complete_sentence(text)
    assert "..." not in repaired
    assert "…" not in repaired
    assert repaired.endswith(".")
    assert not looks_truncated(repaired)
    # Repair only ever removes words — it must never invent any.
    assert repaired.rstrip(".") in text


def test_repair_prefers_dropping_the_cut_off_clause_when_a_sentence_is_complete():
    text = (
        "MrBeast is an American YouTuber and philanthropist. He is the most "
        "subscribed channel on YouTube. His videos typically feature ..."
    )
    repaired = complete_sentence(text)
    assert repaired.endswith("YouTube.")
    assert "typically feature" not in repaired


def test_complete_text_is_left_alone():
    text = "Apple was founded by Steve Jobs, Steve Wozniak and Ronald Wayne."
    assert looks_truncated(text) is False
    assert complete_sentence(text) == text
    # A short answer-box result has no final period but is not cut off —
    # it must not be mistaken for truncated and discarded in favour of a
    # worse candidate.
    assert looks_truncated("428 million subscribers") is False


def test_pipeline_prefers_a_complete_candidate_over_a_cut_off_answer_box(store, kb):
    response = SearchResponse(
        query="who created samsung",
        results=[
            SearchResult(
                title="Samsung - Wikipedia",
                link="https://en.wikipedia.org/wiki/Samsung",
                snippet="Samsung was founded by Lee Byung-chul in 1938 in Daegu, Korea.",
                position=1,
            )
        ],
        answer_box_snippet=SAMSUNG_CUT_OFF,
    )
    pipe = ResearchPipeline(FakeSerperClient(response=response), store, kb)
    out = pipe.research("Who created Samsung?")
    assert out.ok
    assert out.answer == "Samsung was founded by Lee Byung-chul in 1938 in Daegu, Korea."


def test_pipeline_repairs_the_answer_when_every_candidate_is_cut_off(store, kb):
    response = SearchResponse(
        query="who created samsung",
        results=[
            SearchResult(
                title="Samsung",
                link="https://example.org/samsung",
                snippet=SAMSUNG_CUT_OFF,
                position=1,
            )
        ],
        answer_box_snippet=SAMSUNG_CUT_OFF,
    )
    pipe = ResearchPipeline(FakeSerperClient(response=response), store, kb)
    out = pipe.research("Who created Samsung?")
    assert out.ok
    assert "..." not in out.answer
    assert out.answer.endswith("securities.")
    # Snippets are served verbatim by the router when no answer could be
    # extracted, so they must be repaired too.
    assert all("..." not in s for s in out.snippets)


def test_a_stored_answer_never_reaches_the_user_truncated(store, kb):
    """End-to-end: what the router hands back carries no ellipsis, and
    neither does the copy written into the knowledge base (which is what
    gets served on every later ask)."""
    response = SearchResponse(
        query="who created samsung",
        results=[
            SearchResult(
                title="Samsung - Wikipedia",
                link="https://en.wikipedia.org/wiki/Samsung",
                snippet=SAMSUNG_CUT_OFF,
                position=1,
            ),
            SearchResult(
                title="Samsung history",
                link="https://www.britannica.com/topic/Samsung",
                snippet=SAMSUNG_CUT_OFF,
                position=2,
            ),
        ],
        answer_box_snippet=SAMSUNG_CUT_OFF,
    )
    pipe = ResearchPipeline(FakeSerperClient(response=response), store, kb)
    router = ToolRouter(knowledge=kb, research=pipe)

    reply = router.route("Who created Samsung?").direct_reply
    assert reply is not None and "..." not in reply

    stored = store.all_knowledge()
    assert stored and "..." not in stored[0]["answer"]


# -- bug 2: "he doesn't search when he has" ----------------------------------


def _hames_response() -> SearchResponse:
    """A realistic result for a small company: no answer box, no
    knowledge graph, just snippets. Previously this produced
    ok=False/no direct reply, the model generated, and the user got
    'I'm no other company to help a fun day at a time.'"""
    return SearchResponse(
        query="who created hames eventos",
        results=[
            SearchResult(
                title="Hames Eventos",
                link="https://hameseventos.example.com/about",
                snippet=(
                    "Hames Eventos is an events company created by the Hames "
                    "family, organizing weddings and corporate events."
                ),
                position=1,
            )
        ],
    )


def test_small_company_question_is_answered_from_the_web_not_generated(store, kb):
    client = FakeSerperClient(response=_hames_response())
    router = ToolRouter(knowledge=kb, research=ResearchPipeline(client, store, kb))

    result = router.route("Who created Hames Eventos?")
    assert client.calls == 1, "the question must actually reach the web"
    # The key regression: this must never fall through to generation.
    assert result.direct_reply is not None
    assert "Hames" in result.direct_reply
    assert result.context_snippets == []


def test_youtube_subscriber_question_reaches_the_web(store, kb):
    """Verbatim from the transcript, misspelling included. It came back as
    a paragraph about Aila Company Solutions; it must go to the web."""
    response = SearchResponse(
        query="mrbeast subscribers",
        results=[
            SearchResult(
                title="MrBeast - YouTube",
                link="https://www.youtube.com/@MrBeast",
                snippet="MrBeast has over 400 million subscribers on YouTube.",
                position=1,
            )
        ],
        answer_box_answer="over 400 million subscribers",
    )
    client = FakeSerperClient(response=response)
    router = ToolRouter(knowledge=kb, research=ResearchPipeline(client, store, kb))

    result = router.route("Mrbest has how much subscribers on Youtube?")
    assert client.calls == 1
    assert result.direct_reply is not None
    assert "400 million" in result.direct_reply
    # "Youtube" contains the letters "you" — it must not be mistaken for
    # the user addressing Aila, which would block the search entirely.
    assert result.tool_used.startswith("web")


def test_router_never_serves_an_off_topic_snippet(store, kb):
    """Found while bug-hunting the fix itself. `no_extractable_answer`
    means the pipeline already rejected every snippet for sharing too
    little vocabulary with the question — so a "show the top snippet
    anyway" fallback answers "Who is the president of Brazil?" with a
    cake recipe. Admitting the miss is the better answer."""
    response = SearchResponse(
        query="q",
        results=[
            SearchResult(
                title="Cake recipes",
                link="https://example.com/cake",
                snippet="Preheat the oven to 180 degrees and beat the eggs with sugar.",
                position=1,
            )
        ],
    )
    client = FakeSerperClient(response=response)
    router = ToolRouter(knowledge=kb, research=ResearchPipeline(client, store, kb))

    result = router.route("Who is the president of Brazil?")
    assert result.tool_used == "web_no_answer"
    assert "oven" not in result.direct_reply
    assert "couldn't find" in result.direct_reply


def test_our_own_length_cap_does_not_create_an_incomplete_answer(store, kb):
    """Also found while bug-hunting: MAX_SNIPPET_CHARS cut a long snippet
    mid-sentence, and because the cut left no ellipsis the repair step
    couldn't tell and simply bolted a period on the end — reproducing the
    reported complaint from our own code rather than the search engine's."""
    from webresearch.quality import MAX_SNIPPET_CHARS

    filler = "Samsung was founded in 1938. "
    long_snippet = (
        filler * (MAX_SNIPPET_CHARS // len(filler) + 2)
    ) + "It later expanded into shipbuilding, insurance and electronics manufacturing"
    assert len(long_snippet) > MAX_SNIPPET_CHARS
    response = SearchResponse(
        query="who created samsung",
        results=[
            SearchResult(
                title="Samsung - Wikipedia",
                link="https://en.wikipedia.org/wiki/Samsung",
                snippet=long_snippet,
                position=1,
            )
        ],
    )
    pipe = ResearchPipeline(FakeSerperClient(response=response), store, kb)
    out = pipe.research("Who created Samsung?")

    assert out.ok
    assert "..." not in out.answer
    assert out.answer.endswith("1938.")


def test_router_admits_a_miss_instead_of_generating(store, kb):
    empty = SearchResponse(query="q", results=[])
    router = ToolRouter(knowledge=kb, research=ResearchPipeline(FakeSerperClient(empty), store, kb))
    result = router.route("Who created Hames Eventos?")
    assert result.tool_used == "web_no_answer"
    assert "couldn't find" in result.direct_reply


def test_router_reports_a_failed_search_as_a_failed_search(store, kb):
    client = FakeSerperClient(error=SerperTimeoutError("timed out"))
    router = ToolRouter(knowledge=kb, research=ResearchPipeline(client, store, kb))
    result = router.route("Who created Hames Eventos?")
    assert result.tool_used == "web_error:search_failed"
    assert "couldn't reach the web" in result.direct_reply


def test_router_explains_a_rejected_search_key_in_plain_language(store, kb):
    """Found while verifying this fix: the configured Serper key started
    returning HTTP 403. Reporting that as a generic network error sends
    the user off retrying something that will never succeed."""
    client = FakeSerperClient(error=SerperAuthError("Serper rejected the API key (HTTP 403)."))
    router = ToolRouter(knowledge=kb, research=ResearchPipeline(client, store, kb))
    result = router.route("Who created Hames Eventos?")
    assert result.tool_used == "web_error:auth_failed"
    assert "SERPER_API_KEY" in result.direct_reply
    assert "serper.dev" in result.direct_reply
    # The reply must never leak status codes or key material.
    assert "403" not in result.direct_reply


def test_router_reports_a_rate_limit_as_temporary(store, kb):
    client = FakeSerperClient(error=SerperRateLimitError("rate limited"))
    router = ToolRouter(knowledge=kb, research=ResearchPipeline(client, store, kb))
    result = router.route("Who created Hames Eventos?")
    assert result.tool_used == "web_error:rate_limited"
    assert "try that question again" in result.direct_reply


def test_offline_install_still_falls_through_to_the_model(store, kb):
    """With no Serper key configured, behaviour is unchanged from before
    web research existed — the model answers."""
    router = ToolRouter(knowledge=kb, research=ResearchPipeline(None, store, kb))
    assert router.route("Who created Hames Eventos?").direct_reply is None


def test_questions_about_aila_are_answered_from_facts_not_generated(store, kb):
    client = FakeSerperClient(response=_hames_response())
    router = ToolRouter(knowledge=kb, research=ResearchPipeline(client, store, kb))

    result = router.route("Who created Aila Company Solutions?")
    assert result.tool_used == "identity:company_founders"
    assert "Theo Grunewald Hames" in result.direct_reply
    assert "Guilherme Grunewald Benkendorf" in result.direct_reply
    assert client.calls == 0, "identity questions must never hit the web"

    assert router.route("Who created you?").tool_used == "identity:creator"
    assert router.route("What is Aila Nano?").tool_used == "identity:what_are_you"
    assert router.route("What can you do?").tool_used == "identity:capabilities"
    assert router.route("What is your name?").tool_used == "identity:name"
    assert router.route("How many parameters do you have?").tool_used == "identity:size"


@pytest.mark.parametrize(
    "message,intent",
    [
        # "Quem te criou?" puts the object pronoun *before* the verb, so the
        # "quem criou você" pattern (object after) missed it, while the
        # English "Who created you?" worked — a gap for a Portuguese user.
        ("Quem te criou?", "creator"),
        ("Quem te fez?", "creator"),
        ("Quem criou você?", "creator"),
        # "em beta" phrasing was uncovered, though "Are you in beta?" worked.
        ("Você está em beta?", "beta"),
        ("Você é beta?", "beta"),
    ],
)
def test_portuguese_identity_phrasings_match_like_their_english_twins(message, intent):
    match = match_identity_question(message, language="pt")
    assert match is not None and match[0] == intent


@pytest.mark.parametrize(
    "message",
    [
        "Quem te ligou ontem?",   # "te" + non-make verb: not about creation
        "Quem te ama?",           # ditto
        "Você está em casa?",     # "está em" but the object is "casa", not beta
        "O que é radiação beta?", # "beta" as a physics term, not Aila's status
    ],
)
def test_portuguese_identity_fixes_do_not_overreach(message):
    assert match_identity_question(message, language="pt") is None


@pytest.mark.parametrize(
    "message",
    [
        "Who created Samsung?",              # a different company
        "Who do you think made the pyramids?",  # mentions "you", isn't about Aila
        "What is my name?",                  # about the user -> memory
        "Do you remember my favorite color?",  # about the user -> memory
        "Tell me about the Roman Empire",     # unrelated
        "What is the capital of France?",     # unrelated
    ],
)
def test_identity_answers_do_not_hijack_other_questions(message):
    assert match_identity_question(message) is None


# -- bug 3: "he repeats things some time" ------------------------------------


def test_short_filler_messages_get_distinct_replies():
    """Verbatim from the transcript: "Bro", "Ok" and "nice!" every one of
    them answered "Hi! How can I help you today?"."""
    replies = {}
    for message in ("Bro", "Ok", "nice!", "thanks", "bye"):
        match = match_smalltalk(message)
        assert match is not None, f"{message!r} should be recognized as small talk"
        replies[message] = match[1]

    assert len(set(replies.values())) == len(replies), (
        f"filler messages must not all get the same reply: {replies}"
    )
    greeting = match_smalltalk("Hi")[1]
    assert greeting not in replies.values()


@pytest.mark.parametrize(
    "message,intent",
    [
        ("Hi", "greeting"),
        ("hello!", "greeting"),
        ("Bom dia", "greeting"),
        ("Ok", "acknowledgement"),
        ("okkkk", "acknowledgement"),
        ("Bro", "address"),
        ("nice!", "praise"),
        ("Thanks a lot", "thanks"),
        ("kkkkk", "laughter"),
        ("Tchau", "farewell"),
    ],
)
def test_filler_phrases_map_to_the_right_intent(message, intent):
    match = match_smalltalk(message)
    assert match is not None and match[0] == intent


def test_greeting_questions_are_still_small_talk():
    """A question mark once disqualified every filler phrase, which
    silently blocked the most common opening line of all and sent "How
    are you?" back to generation."""
    for message in ("How are you?", "How's it going?", "Tudo bem?", "Beleza?"):
        match = match_smalltalk(message)
        assert match is not None and match[0] == "how_are_you", message


def test_filler_with_a_question_mark_is_still_filler():
    """"Ok?" and "Nice?" used to be sent to the "normal question path" —
    which at this scale is model generation, i.e. nonsense. Matching is
    exact against a closed table of short phrases, so a real question
    cannot be swallowed by allowing the question mark."""
    assert match_smalltalk("Ok?")[0] == "acknowledgement"
    assert match_smalltalk("Nice?")[0] == "praise"
    assert match_smalltalk("Bro?")[0] == "address"

    # Real questions are still untouched. ("What?" on its own is
    # confusion, not a question — see the test below.)
    assert match_smalltalk("Who?") is None
    assert match_smalltalk("When?") is None
    assert match_smalltalk("ok so what is the capital of France?") is None


def test_every_phrase_in_the_table_is_reachable():
    """A phrase the normalizer can never produce is dead weight that
    silently stops working — "how s it going" was exactly that."""
    from tools.smalltalk import _PHRASE_TABLE

    unreachable = [p for p in _PHRASE_TABLE if match_smalltalk(p) is None]
    assert unreachable == []


def test_portuguese_filler_gets_a_portuguese_reply():
    assert match_smalltalk("Obrigado")[1].startswith("De nada")
    assert match_smalltalk("Oi")[1].startswith("Oi!")


@pytest.mark.parametrize(
    "message",
    [
        "ok so what is the capital of France?",  # starts with filler, has content
        "no idea what you mean by that",          # starts with "no"
        "Write me a nice poem",                   # contains "nice"
        "",
        "   ",
    ],
)
def test_real_messages_are_never_swallowed_as_filler(message):
    assert match_smalltalk(message) is None


def test_router_answers_filler_without_touching_any_tool(store, kb):
    client = FakeSerperClient(response=_hames_response())
    router = ToolRouter(knowledge=kb, research=ResearchPipeline(client, store, kb))

    assert router.route("Ok").tool_used == "smalltalk:acknowledgement"
    assert router.route("Bro").tool_used == "smalltalk:address"
    assert router.route("nice!").tool_used == "smalltalk:praise"
    assert client.calls == 0


def test_no_smalltalk_phrase_is_silently_shadowed():
    """The table is flattened with setdefault, so a phrase listed under
    two intents would quietly take the first and drop the second — a
    reply changing for no visible reason. Catch it at the source."""
    from tools.smalltalk import _INTENTS

    seen: dict[str, str] = {}
    duplicates = []
    for intent, phrases, _en, _pt in _INTENTS:
        for phrase in phrases:
            if phrase in seen:
                duplicates.append((phrase, seen[phrase], intent))
            seen[phrase] = intent
    assert duplicates == [], f"phrases listed under two intents: {duplicates}"


@pytest.mark.parametrize(
    "message",
    [
        "Are you finished?",
        "Are you in beta?",
        "Why are you called Beta?",
        "What does Beta mean?",
    ],
)
def test_beta_status_is_answered_deterministically(message):
    """Renaming the release to "Beta" invites the obvious follow-up. The
    model garbled it ("Going hardwa is their old one of her in two co-"),
    so it is answered from the fact table like the rest of identity."""
    match = match_identity_question(message)
    assert match is not None and match[0] == "beta"
    assert "beta" in match[1].lower()
    # It points at the reporting route, which is the useful part.
    assert "/support" in match[1] or "mailailacompanysolutions" in match[1]


def test_beta_intent_does_not_swallow_unrelated_beta_questions():
    assert match_identity_question("What is beta testing?") is None


# -- reported: install crashed with a torch pickling traceback ---------------


def test_an_lfs_pointer_is_reported_as_a_download_problem(tmp_path):
    """Reported verbatim: `_pickle.UnpicklingError: invalid load key, 'v'`.

    GitHub's "Download ZIP" replaces every large file with a ~130-byte
    text stub, so `best.pt` was a Git LFS pointer. The 'v' is the first
    byte of "version https://git-lfs.github.com/...". The raw traceback
    names neither the cause nor the fix.
    """
    from training.checkpoint import CheckpointNotDownloadedError, load_checkpoint

    pointer = tmp_path / "best.pt"
    pointer.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:0123456789abcdef\nsize 237717559\n"
    )

    with pytest.raises(CheckpointNotDownloadedError) as excinfo:
        load_checkpoint(str(pointer))

    message = str(excinfo.value)
    assert "Download raw file" in message
    assert "240 MB" in message
    assert "best.pt" in message


def test_a_real_checkpoint_is_not_mistaken_for_a_pointer(tmp_path):
    import torch

    from training.checkpoint import load_checkpoint

    path = tmp_path / "best.pt"
    torch.save({"config": {}, "step": 1}, path)
    assert load_checkpoint(str(path))["step"] == 1


def test_a_missing_file_still_reports_as_missing(tmp_path):
    from training.checkpoint import load_checkpoint

    with pytest.raises(FileNotFoundError):
        load_checkpoint(str(tmp_path / "nope.pt"))


def test_a_short_question_does_not_inherit_the_previous_topic(kb):
    """Found in a bug sweep, and it is the original "he repeats things"
    complaint in another guise: follow-up expansion fired for *any*
    short question-shaped message, so after "Who founded Apple?" the
    replies to "Ok?", "Hi?" and "Bro?" were all the Apple answer again."""
    kb.store.add_knowledge(
        "Who founded Apple?",
        "Apple was founded by Steve Jobs, Steve Wozniak and Ronald Wayne.",
        confidence=0.9,
    )
    router = ToolRouter(knowledge=kb)

    # A genuine follow-up carries no subject of its own and still works.
    for follow_up in ("When?", "Why?", "E quando?", "Onde?"):
        result = router.route(follow_up, previous_user_message="Who founded Apple?")
        assert result.tool_used == "knowledge", follow_up
        assert "Steve Jobs" in result.direct_reply

    # Anything that carries its own word must not inherit the topic.
    for filler in ("Ok?", "Hi?", "Sure?", "Bro?", "Nice?"):
        result = router.route(filler, previous_user_message="Who founded Apple?")
        assert result.tool_used != "knowledge", filler
        assert "Steve Jobs" not in (result.direct_reply or "")


def test_a_follow_up_skips_filler_and_resolves_against_the_real_question(
    tiny_model, tokenizer, tmp_path
):
    """Found in a bug sweep. Real transcript:

        You: Who founded Bambu Lab?   -> answered
        You: Ok?                      -> "Got it."
        You: When?                    -> the Wikipedia article on "OK"

    "Ok?" was the immediately preceding user turn, so the follow-up
    resolved against it. A follow-up means "more about what we were
    actually discussing"; an acknowledgement is not that.
    """
    from agents.registry import get_agent
    from memory.manager import MemoryManager
    from vectordb.embedder import AilaEmbedder

    embedder = AilaEmbedder(tiny_model, tokenizer)
    memory = MemoryManager(
        embedder, db_path=str(tmp_path / "m.db"), faiss_path=str(tmp_path / "m.faiss")
    )
    try:
        agent = get_agent("general", tiny_model, tokenizer, memory=memory)
        conversation = "c1"
        for role, content in [
            ("user", "Who founded Bambu Lab?"),
            ("assistant", "Bambu Lab was founded by engineers from DJI."),
            ("user", "Ok?"),
            ("assistant", "Got it. Anything else I can help with?"),
        ]:
            memory.add_turn(conversation, role, content)

        assert agent._previous_user_message(conversation) == "Who founded Bambu Lab?"
    finally:
        memory.close()


def test_previous_message_is_none_when_there_is_only_filler(
    tiny_model, tokenizer, tmp_path
):
    from agents.registry import get_agent
    from memory.manager import MemoryManager
    from vectordb.embedder import AilaEmbedder

    embedder = AilaEmbedder(tiny_model, tokenizer)
    memory = MemoryManager(
        embedder, db_path=str(tmp_path / "m2.db"), faiss_path=str(tmp_path / "m2.faiss")
    )
    try:
        agent = get_agent("general", tiny_model, tokenizer, memory=memory)
        memory.add_turn("c2", "user", "Hi")
        memory.add_turn("c2", "assistant", "Hi! How can I help you today?")
        memory.add_turn("c2", "user", "Ok")
        # Nothing substantive to inherit — better no antecedent than a
        # wrong one.
        assert agent._previous_user_message("c2") is None
    finally:
        memory.close()


def test_a_bare_what_is_confusion_not_a_follow_up(kb):
    """Reported transcript:

        You: Whats the biggest youtuber in the world?
        Aila: YouTuber film: A YouTuber film ...
        You: What??
        Aila: YouTuber film: A YouTuber film ...   <- repeated verbatim

    "What??" meant "I did not understand", but it was treated as a
    follow-up and replayed the answer the user was objecting to."""
    for message in ("What?", "What??", "what", "Huh?", "O que?", "Hein?"):
        match = match_smalltalk(message)
        assert match is not None and match[0] == "confusion", message

    kb.store.add_knowledge("Who founded Apple?", "Apple was founded by Steve Jobs.", confidence=0.9)
    router = ToolRouter(knowledge=kb)
    result = router.route("What??", previous_user_message="Who founded Apple?")
    assert result.tool_used == "smalltalk:confusion"
    assert "Steve Jobs" not in result.direct_reply

    # Real follow-ups are untouched.
    assert router.route("When?", previous_user_message="Who founded Apple?").tool_used == "knowledge"


def test_apostrophe_less_contractions_reach_the_right_wikipedia_article():
    """"Whats the biggest youtuber in the world?" searched Wikipedia with
    "Whats" still attached, which returned an article about YouTuber
    *films*. Stripping it is what makes the search find MrBeast."""
    from webresearch.wikipedia import extract_subject

    assert extract_subject("Whats the biggest youtuber in the world?") == (
        "biggest youtuber in the world"
    )
    assert extract_subject("Whos the president of Brazil?") == "president of Brazil"
    assert extract_subject("Wheres the Eiffel Tower?") == "Eiffel Tower"
    # The apostrophe form already worked and must keep working.
    assert extract_subject("What's photosynthesis?") == "photosynthesis"
