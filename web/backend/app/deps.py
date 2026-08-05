"""Application state: the model, tokenizer, memory, and knowledge index
are loaded once at process startup and shared across requests via
FastAPI's dependency-injection system.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from fastapi import Request

from agents.registry import Agent, get_agent, list_agents
from memory.manager import MemoryManager
from model.config import GPTConfig, nano_10m
from model.transformer import AilaNanoGPT
from tokenizer.tokenizer import AilaTokenizer
from training.checkpoint import load_checkpoint
from vectordb.embedder import AilaEmbedder
from vectordb.semantic_index import SemanticIndex

logger = logging.getLogger(__name__)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class AilaSettings:
    # Each field reads its environment variable via default_factory, so
    # values are resolved fresh at *instantiation* time rather than baked
    # in at module-import time — important both for tests (which set env
    # vars per-case) and for any process that constructs AilaSettings more
    # than once with a changed environment.
    checkpoint_path: str = field(default_factory=lambda: _env("AILA_CHECKPOINT", "checkpoints/finetune/best.pt"))
    fallback_checkpoint_path: str = field(
        default_factory=lambda: _env("AILA_FALLBACK_CHECKPOINT", "checkpoints/pretrain/best.pt")
    )
    tokenizer_path: str = field(
        default_factory=lambda: _env("AILA_TOKENIZER", "tokenizer/artifacts/aila_nano.model")
    )
    device: str = field(default_factory=lambda: _env("AILA_DEVICE", "auto"))
    memory_db: str = field(default_factory=lambda: _env("AILA_MEMORY_DB", "memory/data/aila_memory.db"))
    memory_faiss: str = field(
        default_factory=lambda: _env("AILA_MEMORY_FAISS", "memory/data/aila_memory.faiss")
    )
    knowledge_db: str = field(
        default_factory=lambda: _env("AILA_KNOWLEDGE_DB", "vectordb/index/knowledge.db")
    )
    knowledge_faiss: str = field(
        default_factory=lambda: _env("AILA_KNOWLEDGE_FAISS", "vectordb/index/knowledge.faiss")
    )

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


class AilaState:
    """Holds everything a request handler needs. Constructed once in the
    FastAPI lifespan handler (see web/backend/app/main.py) and stashed on
    `app.state.aila`.
    """

    def __init__(self, settings: AilaSettings | None = None):
        self.settings = settings or AilaSettings()
        self.device = self.settings.resolved_device()

        self.tokenizer = self._load_tokenizer()
        self.model, self.model_loaded_from = self._load_model()
        self.embedder = AilaEmbedder(self.model, self.tokenizer, device=self.device)
        self.memory = MemoryManager(
            self.embedder,
            db_path=self.settings.memory_db,
            faiss_path=self.settings.memory_faiss,
        )
        self.knowledge = SemanticIndex(
            self.embedder,
            db_path=self.settings.knowledge_db,
            faiss_path=self.settings.knowledge_faiss,
        )
        self._agent_cache: dict[str, Agent] = {}

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

    def get_agent(self, agent_name: str) -> Agent:
        if agent_name not in self._agent_cache:
            self._agent_cache[agent_name] = get_agent(
                agent_name, self.model, self.tokenizer, memory=self.memory, device=self.device
            )
        return self._agent_cache[agent_name]

    def available_agents(self) -> list[str]:
        return list_agents()

    def close(self) -> None:
        self.memory.close()
        self.knowledge.close()


def get_state(request: Request) -> AilaState:
    return request.app.state.aila
