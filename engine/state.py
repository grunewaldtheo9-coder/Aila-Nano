"""AilaEngine: the interface-independent AI core.

Owns the tokenizer, model, memory, and knowledge index, and hands out
ready-to-use `Agent` instances. Nothing in this module knows or cares
what's driving it — `chat.py` (a terminal loop) is the first caller, but
a future desktop GUI, mobile app, or web service would construct and use
the exact same `AilaEngine` with no changes here. That boundary is
deliberate: interfaces are expected to come and go; the engine is not.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from pathlib import Path

from agents.base import Agent
from agents.registry import get_agent, list_agents
from engine.config import EngineSettings
from memory.manager import MemoryManager
from model.config import GPTConfig, nano_10m
from model.transformer import AilaNanoGPT
from tokenizer.tokenizer import AilaTokenizer
from training.checkpoint import load_checkpoint
from vectordb.embedder import AilaEmbedder
from vectordb.semantic_index import SemanticIndex

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]


class AilaEngine:
    """Load once, use everywhere. Construct one `AilaEngine` per process
    and reuse it — loading the model/tokenizer/memory is the expensive
    part; everything after that (`chat`, `chat_stream`, `get_agent`) is
    cheap.
    """

    def __init__(
        self,
        settings: EngineSettings | None = None,
        on_progress: ProgressCallback | None = None,
    ):
        self.settings = settings or EngineSettings()
        self._notify = on_progress or (lambda msg: None)
        self.device = self.settings.resolved_device()

        self._notify("Loading tokenizer...")
        self.tokenizer = self._load_tokenizer()
        self._notify("OK")

        self._notify("Loading model...")
        self.model, self.model_loaded_from = self._load_model()
        self._notify("OK")

        self._notify("Loading memory...")
        self.embedder = AilaEmbedder(self.model, self.tokenizer, device=self.device)
        self.memory = MemoryManager(
            self.embedder,
            db_path=self.settings.memory_db,
            faiss_path=self.settings.memory_faiss,
            max_facts=self.settings.retrieval_top_k,
            relevance_threshold=self.settings.relevance_threshold,
        )
        self._notify("OK")

        self._notify("Loading FAISS...")
        self.knowledge = SemanticIndex(
            self.embedder,
            db_path=self.settings.knowledge_db,
            faiss_path=self.settings.knowledge_faiss,
        )
        self._notify("OK")

        self._notify("Loading knowledge base...")
        # Set before the stack is built: the pipeline captures
        # `self._emit_status`, which reads this on every call.
        self._status_callback = None
        self.knowledge_store, self.knowledge_base, self.router = self._build_knowledge_stack()
        self.translator = self._build_translator()
        self._notify("OK")

        self._notify("Loading agents...")
        self._agent_cache: dict[str, Agent] = {}
        # Eagerly construct every registered agent so a misconfigured
        # persona fails fast at startup rather than on first use.
        for name in self.available_agents():
            self.get_agent(name)
        self._notify("OK")

    def _build_knowledge_stack(self):
        """Global knowledge base + web research + tool router.

        Web research is optional at every level: no SERPER_API_KEY (or
        AILA_WEB_SEARCH_ENABLED=false) simply means the router runs
        without a research pipeline — knowledge lookups and the
        calculator still work, and nothing errors. The 'firestore'
        storage backend is accepted but requires firebase-admin +
        GOOGLE_APPLICATION_CREDENTIALS; without them we fall back to
        SQLite with a logged warning rather than refusing to start.
        """
        from knowledge.base import KnowledgeBase
        from knowledge.store import KnowledgeStore
        from tools.router import ToolRouter
        from webresearch.pipeline import ResearchPipeline
        from webresearch.serper import SerperClient
        from webresearch.wikipedia import WikipediaClient

        store = None
        if self.settings.storage_backend == "firestore":
            from knowledge.firestore_backend import (
                FirestoreKnowledgeStore,
                FirestoreUnavailableError,
            )

            try:
                store = FirestoreKnowledgeStore(
                    project_id=os.environ.get("AILA_FIREBASE_PROJECT_ID")
                )
                logger.warning(
                    "Using the Firestore knowledge backend. Note: this adapter has "
                    "not been verified against a live Firestore project by the "
                    "Aila Nano test suite (no service-account credential is "
                    "shipped with the repo) — monitor it closely."
                )
            except FirestoreUnavailableError as e:
                logger.warning("Firestore backend unavailable (%s); falling back to SQLite.", e)

        if store is None:
            store = KnowledgeStore(self.settings.knowledge_store_db)
        base = KnowledgeBase(store)

        client = None
        if self.settings.web_search_enabled and self.settings.serper_api_key:
            client = SerperClient(
                self.settings.serper_api_key,
                timeout_seconds=self.settings.web_timeout_seconds,
                max_results=self.settings.web_max_results,
            )
        elif self.settings.web_search_enabled:
            logger.info("SERPER_API_KEY is not set — relying on Wikipedia alone.")

        # Wikipedia needs no key, so unless it is explicitly switched off
        # Aila always has *a* live source. This is what keeps her useful
        # when the Serper key is missing, cancelled, or out of credits —
        # the situation that previously dropped every factual question
        # onto a ~20M-parameter model's guesswork.
        wikipedia = (
            WikipediaClient(
                timeout_seconds=self.settings.web_timeout_seconds,
                max_results=self.settings.wikipedia_max_results,
            )
            if self.settings.wikipedia_enabled
            else None
        )

        research = (
            ResearchPipeline(
                client,
                store,
                base,
                cache_ttl_seconds=self.settings.web_cache_ttl_hours * 3600,
                wikipedia=wikipedia,
                on_status=self._emit_status,
            )
            if (client is not None or wikipedia is not None)
            else None
        )
        router = ToolRouter(knowledge=base, research=research, memory=self.memory)
        return store, base, router

    def _build_translator(self):
        """A Translator, or None when translation is switched off. Never
        raises: an unusable deep-translator just yields a translator whose
        `available` is False, which the agents treat as "no translation"."""
        if not self.settings.translation_enabled:
            return None
        from translation import Translator

        translator = Translator(enabled=True)
        if translator.available:
            logger.info("translation enabled (deep-translator)")
        else:
            logger.info("translation requested but deep-translator is unavailable")
        return translator

    # -- loading --------------------------------------------------------

    def _load_tokenizer(self) -> AilaTokenizer:
        return AilaTokenizer.load(self.settings.tokenizer_path)

    def _load_model(self) -> tuple[AilaNanoGPT, str | None]:
        for path in (self.settings.checkpoint_path, self.settings.fallback_checkpoint_path):
            if path and Path(path).exists():
                ckpt = load_checkpoint(path, map_location=self.device)
                model = AilaNanoGPT(GPTConfig.from_dict(ckpt["config"]))
                model.load_state_dict(ckpt["model_state_dict"])
                model.to(self.device)
                model.eval()
                logger.info("Loaded model from checkpoint: %s", path)
                return model, path

        logger.warning(
            "No checkpoint found at %s or %s — serving a freshly-initialized, UNTRAINED "
            "nano_10m model. Train Aila Nano first (see docs/TRAINING.md) for real responses.",
            self.settings.checkpoint_path,
            self.settings.fallback_checkpoint_path,
        )
        # The fallback config must still match the *actual* tokenizer's
        # vocab size — a mismatch here would let the model sample token
        # ids the tokenizer can't decode.
        cfg_dict = nano_10m().to_dict()
        cfg_dict["vocab_size"] = self.tokenizer.vocab_size
        model = AilaNanoGPT(GPTConfig.from_dict(cfg_dict))
        model.to(self.device)
        model.eval()
        return model, None

    # -- agents -----------------------------------------------------------

    def get_agent(self, agent_name: str) -> Agent:
        if agent_name not in self._agent_cache:
            self._agent_cache[agent_name] = get_agent(
                agent_name,
                self.model,
                self.tokenizer,
                memory=self.memory,
                knowledge=self.knowledge,
                device=self.device,
                router=self.router,
                allow_freeform=self.settings.allow_freeform,
                translator=self.translator,
                emit_status=self._emit_status,
            )
        return self._agent_cache[agent_name]

    def available_agents(self) -> list[str]:
        return list_agents()

    # -- convenience wrappers ----------------------------------------------

    def chat(self, conversation_id: str, message: str, agent_name: str = "general") -> str:
        """Non-streaming: get a full reply in one call."""
        return self.get_agent(agent_name).respond(conversation_id, message)

    def chat_stream(self, conversation_id: str, message: str, agent_name: str = "general"):
        """Streaming: yields text deltas as they're generated. Any
        interface can consume this — `chat.py` prints deltas as they
        arrive; a GUI could append them to a text widget the same way.
        """
        return self.get_agent(agent_name).respond_stream(conversation_id, message)

    @property
    def is_trained(self) -> bool:
        return self.model_loaded_from is not None

    def set_serper_api_key(self, api_key: str) -> bool:
        """Attach (or replace) the Serper source at runtime.

        Exists so the first-run setup can start using a pasted key
        immediately instead of telling the user to restart. Returns False
        if the key is empty or research is disabled entirely.

        Never logs the key.
        """
        from webresearch.serper import SerperClient

        api_key = (api_key or "").strip()
        research = getattr(self.router, "research", None)
        if not api_key or research is None:
            return False
        research.client = SerperClient(
            api_key,
            timeout_seconds=self.settings.web_timeout_seconds,
            max_results=self.settings.web_max_results,
        )
        self.settings.serper_api_key = api_key
        logger.info("Serper source attached at runtime")
        return True

    def check_serper_api_key(self, api_key: str) -> tuple[bool, str]:
        """One live search to see whether a key actually works.

        Worth the round trip: the previous key in this project was
        cancelled without anyone noticing, and every factual question
        quietly degraded for days. Checking at the moment it is pasted
        turns that into an immediate, fixable message.

        Returns (ok, human-readable reason). Never includes the key.
        """
        from webresearch.serper import (
            SerperAuthError,
            SerperClient,
            SerperError,
            SerperRateLimitError,
        )

        api_key = (api_key or "").strip()
        if not api_key:
            return False, "No key entered."
        try:
            SerperClient(api_key, timeout_seconds=self.settings.web_timeout_seconds).search(
                "test"
            )
        except SerperAuthError:
            return False, "That key was rejected. Check you copied all of it."
        except SerperRateLimitError:
            # The key is valid — it is just out of searches right now.
            return True, "The key works, but it has hit its search limit for now."
        except SerperError as e:
            return False, f"Couldn't reach the search service ({type(e).__name__})."
        return True, "The key works."

    def set_status_callback(self, callback) -> None:
        """Where to send short progress lines like "Searching Wikipedia...".

        A lookup can take a couple of seconds, and without this the chat
        simply stops responding with no explanation. Set to None to
        silence it.
        """
        self._status_callback = callback

    def _emit_status(self, message: str) -> None:
        callback = getattr(self, "_status_callback", None)
        if callback is None:
            return
        try:
            callback(message)
        except Exception:  # noqa: BLE001 — progress must never break a turn
            logger.exception("status callback failed")

    @property
    def web_search_active(self) -> bool:
        """True when at least one live source (Wikipedia or Serper) was
        constructed, i.e. research can run at all. Exposed (rather than
        left as a log line) so an interface can tell the user up front:
        without it, "why can't Aila look anything up?" has no visible
        answer, and factual questions quietly fall through to
        generation instead."""
        return self.router is not None and self.router.research is not None

    @property
    def research_sources(self) -> list[str]:
        """Names of the live sources available, best-first. Empty when
        research is switched off entirely."""
        research = getattr(self.router, "research", None)
        if research is None:
            return []
        try:
            return [name for name, _, _ in research._providers()]
        except Exception:  # noqa: BLE001 — a status readout must not crash
            logger.exception("could not list research sources")
            return []

    @property
    def known_fact_count(self) -> int:
        """How many researched facts Aila can answer from with no network
        at all — the number that goes up every time she learns."""
        try:
            return len(self.knowledge_store.all_knowledge())
        except Exception:  # noqa: BLE001 — a count must never break startup
            return 0

    # -- self-directed study -----------------------------------------------

    def study(self, topic: str) -> tuple[bool, str]:
        """Research one topic on demand and keep what is found.

        Returns (learned, message). Learning is exactly the normal
        research path, so nothing reaches the knowledge base that didn't
        clear the usual extraction and confidence gates.
        """
        topic = (topic or "").strip()
        if not topic:
            return False, "Tell me what to study, e.g. /study photosynthesis"
        research = getattr(self.router, "research", None)
        if research is None:
            return False, "I have no research sources enabled, so I can't study right now."

        question = _as_question(topic)
        try:
            outcome = research.research(question)
        except Exception as e:  # noqa: BLE001 — never break the session
            logger.exception("study failed for %r", topic)
            return False, f"I couldn't study that ({type(e).__name__})."

        if not (outcome.ok and outcome.answer):
            return False, "I looked that up but couldn't find anything reliable to learn."

        # `stored` says what the knowledge base did with it, and each
        # outcome means something different to the user. Reporting them
        # all as success would claim things that aren't true — in
        # particular `None` means "found, but too low-confidence to
        # keep", which is the opposite of having learned it.
        if outcome.stored in ("created", "updated"):
            return True, f"Learned it. {outcome.answer}"
        if outcome.stored == "conflict":
            return False, (
                "What I found disagrees with something I already know, so I've kept "
                "both aside rather than trusting either. Here's the new version: "
                f"{outcome.answer}"
            )
        if outcome.stored == "rejected":
            return False, "What I found wasn't solid enough to keep."
        return False, (
            "I found something, but I'm not confident enough in it to keep it: "
            f"{outcome.answer}"
        )

    def study_due(self) -> bool:
        """Whether a study round would actually do work right now.

        Exposed so an interface can say "Studying..." *before* the wait
        rather than after it — the study round blocks, and printing the
        message afterwards left the user staring at nothing.
        """
        from knowledge.study import StudySession

        research = getattr(self.router, "research", None)
        if research is None or not self.settings.daily_study_enabled:
            return False
        try:
            return StudySession(
                self.knowledge_store, research, max_topics=self.settings.study_topics_per_day
            ).due()
        except Exception:  # noqa: BLE001
            logger.exception("could not determine study schedule")
            return False

    def run_daily_study(self, force: bool = False):
        """One bounded round of self-directed study — see
        knowledge/study.py. Returns a StudyReport (possibly skipped)."""
        from knowledge.study import StudyReport, StudySession

        research = getattr(self.router, "research", None)
        if research is None or not self.settings.daily_study_enabled:
            return StudyReport(skipped=True)

        session = StudySession(
            self.knowledge_store,
            research,
            max_topics=self.settings.study_topics_per_day,
        )
        try:
            return session.run(force=force)
        except Exception:  # noqa: BLE001 — study must never break startup
            logger.exception("daily study failed")
            return StudyReport(skipped=True)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    # -- knowledge base ---------------------------------------------------

    def learn_file(self, path: str) -> int:
        """Chunk a local text file and index it into the shared knowledge
        base so agents can retrieve it as context (`agents/base.py`
        queries it alongside long-term memory when building each turn's
        system prompt). Returns the number of chunks indexed. Raises
        ValueError/OSError on bad input — callers (e.g. `chat.py`) should
        catch and report those.
        """
        from vectordb.chunking import ALLOWED_SUFFIXES, MAX_FILE_BYTES, chunk_text

        file_path = Path(path)
        if file_path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise ValueError(
                f"Unsupported file type '{file_path.suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}"
            )
        if not file_path.exists():
            raise FileNotFoundError(f"No such file: {path}")
        if file_path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"File too large (max {MAX_FILE_BYTES // (1024 * 1024)} MB).")

        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("File contained no text to index.")

        ids = self.knowledge.add_documents(
            chunks, metadatas=[{"source": str(file_path)} for _ in chunks]
        )
        self.knowledge.save()
        return len(ids)

    # -- lifecycle -----------------------------------------------------

    def save(self) -> None:
        self.memory.save()
        self.knowledge.save()

    def close(self) -> None:
        self.save()
        self.memory.close()
        self.knowledge.close()
        self.knowledge_store.close()

    def __enter__(self) -> AilaEngine:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


# Question openers, EN + PT. A topic that already reads as a question is
# researched as-is: wrapping it produced "What is who created Samsung?",
# which searches for nothing at all.
_QUESTION_SHAPED = re.compile(
    r"^\s*(who|what|when|where|which|why|how|is|are|was|were|do|does|did|can|"
    r"quem|o\s+que|qual|quais|quando|onde|por\s+que|porque|como|quantos?|quantas?)\b",
    re.IGNORECASE,
)


def _as_question(topic: str) -> str:
    """Turn a bare topic into something researchable.

    "photosynthesis"          -> "What is photosynthesis?"
    "who created Samsung"     -> "who created Samsung?"   (already a question)
    "What is DNA?"            -> unchanged
    """
    topic = (topic or "").strip()
    if not topic:
        return ""
    if topic.endswith("?"):
        return topic
    if _QUESTION_SHAPED.match(topic):
        return topic + "?"
    return f"What is {topic}?"
