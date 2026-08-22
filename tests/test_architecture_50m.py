"""The nano_50m architecture preset: it builds, lands in the 45M-55M
target range, keeps the checkpoint-compatible layout of the smaller
presets, and can run a forward pass.
"""

from __future__ import annotations

import torch

from model.config import GPTConfig, nano_20m, nano_50m
from model.transformer import AilaNanoGPT
from model.utils import count_parameters


def test_nano_50m_is_in_the_target_range():
    model = AilaNanoGPT(nano_50m())
    n = count_parameters(model)
    assert 45_000_000 <= n <= 55_000_000, f"{n:,} is outside 45M-55M"


def test_nano_50m_exact_count_is_stable():
    # Locks the documented number so an accidental config edit is caught.
    assert count_parameters(AilaNanoGPT(nano_50m())) == 51_393_024


def test_nano_50m_is_larger_than_nano_20m():
    assert count_parameters(AilaNanoGPT(nano_50m())) > count_parameters(AilaNanoGPT(nano_20m()))


def test_nano_50m_keeps_the_compatible_layout():
    cfg = nano_50m()
    # Same tokenizer vocab and grouped-query/tied-embedding design as the
    # shipped presets, so training/inference/checkpoint code is unchanged.
    assert cfg.vocab_size == 8192
    assert cfg.tie_embeddings is True
    assert cfg.n_heads % cfg.n_kv_heads == 0
    assert cfg.d_model % cfg.n_heads == 0


def test_nano_50m_forward_pass_runs():
    cfg = nano_50m()
    model = AilaNanoGPT(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        logits, _ = model(ids)
    assert logits.shape == (1, 16, cfg.vocab_size)
