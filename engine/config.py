"""Runtime configuration for the Aila Nano engine.

Centralizes every environment-variable-driven path/setting the engine
needs (checkpoint locations, tokenizer path, device, memory storage) so
any interface — the terminal chat today, or a future desktop/mobile/web
front end — configures itself identically. See docs/CONFIGURATION.md for
the full reference.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    """Numeric env var with a safe fallback. A typo in `.env`
    (AILA_WEB_MAX_RESULTS=five) must not kill startup with a raw
    ValueError traceback — warn and use the default instead."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using default %d.", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using default %s.", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class EngineSettings:
    # Each field reads its environment variable via default_factory, so
    # values are resolved fresh at *instantiation* time rather than baked
    # in at module-import time — matters for tests, and for any process
    # that constructs EngineSettings more than once with a changed
    # environment.
    # Prefers the 2.0 (~19.8M) checkpoint, falling back to the 1.0
    # (~10.9M) one. An existing 1.0-only install therefore keeps working
    # untouched, and adding the 2.0 file is all it takes to upgrade —
    # no config edit, no reinstall.
    checkpoint_path: str = field(
        default_factory=lambda: _env("AILA_CHECKPOINT", "checkpoints/finetune_20m/best.pt")
    )
    fallback_checkpoint_path: str = field(
        default_factory=lambda: _env("AILA_FALLBACK_CHECKPOINT", "checkpoints/finetune/best.pt")
    )
    tokenizer_path: str = field(
        default_factory=lambda: _env("AILA_TOKENIZER", "tokenizer/artifacts/aila_nano.model")
    )
    device: str = field(default_factory=lambda: _env("AILA_DEVICE", "auto"))
    memory_db: str = field(
        default_factory=lambda: _env("AILA_MEMORY_DB", "memory/data/aila_memory.db")
    )
    memory_faiss: str = field(
        default_factory=lambda: _env("AILA_MEMORY_FAISS", "memory/data/aila_memory.faiss")
    )
    knowledge_db: str = field(
        default_factory=lambda: _env("AILA_KNOWLEDGE_DB", "vectordb/index/knowledge.db")
    )
    knowledge_faiss: str = field(
        default_factory=lambda: _env("AILA_KNOWLEDGE_FAISS", "vectordb/index/knowledge.faiss")
    )
    default_agent: str = field(default_factory=lambda: _env("AILA_DEFAULT_AGENT", "general"))

    # -- Aila 2.0: global knowledge store + web research -------------------
    knowledge_store_db: str = field(
        default_factory=lambda: _env("AILA_KNOWLEDGE_STORE_DB", "knowledge/data/aila_knowledge.db")
    )
    storage_backend: str = field(
        default_factory=lambda: _env("AILA_STORAGE_BACKEND", "sqlite")
    )
    # Secret: only ever read from the environment / .env (gitignored).
    serper_api_key: str = field(default_factory=lambda: _env("SERPER_API_KEY", ""))
    web_search_enabled: bool = field(
        default_factory=lambda: _env_bool("AILA_WEB_SEARCH_ENABLED", True)
    )
    web_max_results: int = field(default_factory=lambda: _env_int("AILA_WEB_MAX_RESULTS", 5))
    web_timeout_seconds: float = field(
        default_factory=lambda: _env_float("AILA_WEB_TIMEOUT_SECONDS", 8.0)
    )
    web_cache_ttl_hours: float = field(
        default_factory=lambda: _env_float("AILA_WEB_CACHE_TTL_HOURS", 168.0)
    )
    retrieval_top_k: int = field(default_factory=lambda: _env_int("AILA_RETRIEVAL_TOP_K", 3))
    relevance_threshold: float = field(
        default_factory=lambda: _env_float("AILA_RELEVANCE_THRESHOLD", 0.2)
    )

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"
