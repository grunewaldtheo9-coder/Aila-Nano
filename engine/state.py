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
        self.knowledge_store, self.knowledge_base, self.router = self._build_knowledge_stack()
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
            logger.info("Web search enabled but SERPER_API_KEY is not set — running offline.")

        research = (
            ResearchPipeline(
                client,
                store,
                base,
                cache_ttl_seconds=self.settings.web_cache_ttl_hours * 3600,
            )
            if client is not None
            else None
        )
        router = ToolRouter(knowledge=base, research=research, memory=self.memory)
        return store, base, router

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
