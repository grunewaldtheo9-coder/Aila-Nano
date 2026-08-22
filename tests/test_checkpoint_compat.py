"""Checkpoint self-description and compatibility validation — so a future
50M checkpoint is plug-and-play and a mismatched one fails with a clear
message instead of a cryptic shape error."""

from __future__ import annotations

import pytest
import torch

from model.config import nano_50m
from model.transformer import AilaNanoGPT
from training.checkpoint import (
    CheckpointIncompatibleError,
    checkpoint_metadata,
    describe_checkpoint,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_compatibility,
)


def test_metadata_describes_the_architecture():
    meta = checkpoint_metadata(AilaNanoGPT(nano_50m()))
    assert meta["model_name"] == "aila_nano"
    assert meta["parameters"] == 51_393_024
    assert meta["model_size"] == "51M"
    assert meta["d_model"] == 512
    assert meta["context_length"] == 1024


def test_saved_checkpoint_carries_metadata_and_config(tiny_model, tmp_path):
    path = str(tmp_path / "c.pt")
    save_checkpoint(path, tiny_model, optimizer=None, step=1, best_val_loss=0.0)
    ckpt = load_checkpoint(path)
    assert "metadata" in ckpt and ckpt["metadata"]["model_name"] == "aila_nano"
    assert ckpt["config"]["vocab_size"] == tiny_model.cfg.vocab_size
    assert "aila_nano" in describe_checkpoint(ckpt)


def test_compatibility_passes_for_matching_vocab(tiny_model, tmp_path):
    path = str(tmp_path / "c.pt")
    save_checkpoint(path, tiny_model, optimizer=None, step=1, best_val_loss=0.0)
    ckpt = load_checkpoint(path)
    # Same vocab as the model -> no error.
    validate_checkpoint_compatibility(ckpt, tiny_model.cfg.vocab_size)


def test_compatibility_fails_for_mismatched_vocab(tiny_model, tmp_path):
    path = str(tmp_path / "c.pt")
    save_checkpoint(path, tiny_model, optimizer=None, step=1, best_val_loss=0.0)
    ckpt = load_checkpoint(path)
    with pytest.raises(CheckpointIncompatibleError):
        validate_checkpoint_compatibility(ckpt, tiny_model.cfg.vocab_size + 1)


def test_compatibility_fails_without_config():
    with pytest.raises(CheckpointIncompatibleError):
        validate_checkpoint_compatibility({"model_state_dict": {}}, 8192)


def test_describe_handles_a_metadata_less_checkpoint():
    # Older checkpoints predate the metadata field; describe falls back to config.
    ckpt = {"config": {"d_model": 320, "n_layers": 15, "n_heads": 8, "max_seq_len": 512}}
    s = describe_checkpoint(ckpt)
    assert "d_model=320" in s
