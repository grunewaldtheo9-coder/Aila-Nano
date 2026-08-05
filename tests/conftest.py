"""Shared pytest fixtures for the Aila Nano test suite.

Everything here is intentionally tiny (small vocab, few layers, low
dimensions) so the full suite runs in seconds on CPU while still
exercising every real code path (no mocking of the model/tokenizer).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from model.config import GPTConfig
from model.transformer import AilaNanoGPT
from tokenizer.tokenizer import AilaTokenizer
from tokenizer.trainer import train_tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CORPUS = REPO_ROOT / "datasets" / "sample" / "pretrain_sample.txt"
SAMPLE_FINETUNE = REPO_ROOT / "datasets" / "sample" / "finetune_sample.jsonl"
AILA_KNOWLEDGE = REPO_ROOT / "datasets" / "aila_knowledge" / "aila_company.jsonl"


@pytest.fixture(scope="session")
def tokenizer(tmp_path_factory) -> AilaTokenizer:
    out_dir = tmp_path_factory.mktemp("tokenizer_artifacts")
    prefix = str(out_dir / "aila_nano_test")
    train_tokenizer(input_files=str(SAMPLE_CORPUS), model_prefix=prefix, vocab_size=384)
    return AilaTokenizer.load(f"{prefix}.model")


@pytest.fixture
def tiny_config(tokenizer) -> GPTConfig:
    return GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=64,
        n_layers=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        mlp_hidden_mult=2.0,
        dropout=0.0,
        tie_embeddings=True,
        bias=False,
    )


@pytest.fixture
def tiny_model(tiny_config) -> AilaNanoGPT:
    return AilaNanoGPT(tiny_config)
