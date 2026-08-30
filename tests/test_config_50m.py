"""The 50M model/training configs on disk agree with the nano_50m() preset,
the --preset selector, the parameter-count utilities, and the checkpoint
architecture-mismatch guard. Keeps config, code, and checkpoints internally
consistent so a future trained 50M checkpoint drops in cleanly.
"""

from __future__ import annotations

import pytest
import torch

from model.config import GPTConfig, nano_20m, nano_50m
from model.transformer import AilaNanoGPT
from model.utils import count_parameters
from training.checkpoint import (
    CheckpointIncompatibleError,
    save_checkpoint,
    load_checkpoint,
    validate_checkpoint_compatibility,
)


# -- config YAML matches the preset --------------------------------------

def test_model_yaml_matches_nano_50m_preset():
    cfg = GPTConfig.from_yaml("configs/model/nano_50m.yaml")
    assert cfg.to_dict() == nano_50m().to_dict()


def test_model_yaml_builds_in_range():
    cfg = GPTConfig.from_yaml("configs/model/nano_50m.yaml")
    n = count_parameters(AilaNanoGPT(cfg))
    assert 45_000_000 <= n <= 55_000_000


def test_training_configs_target_the_50m_checkpoint_dir():
    import yaml

    with open("configs/training/pretrain_50m.yaml") as f:
        pre = yaml.safe_load(f)
    assert pre["checkpoint_dir"].startswith("checkpoints/50m/")
    with open("configs/training/finetune_50m.yaml") as f:
        ft = yaml.safe_load(f)
    assert ft["out_dir"].startswith("checkpoints/50m/")


# -- --preset selector ----------------------------------------------------

def test_train_preset_selector_maps_to_nano_50m():
    from training.train import _PRESETS

    assert "nano_50m" in _PRESETS
    assert _PRESETS["nano_50m"]().to_dict() == nano_50m().to_dict()


# -- parameter accounting -------------------------------------------------

def test_total_and_trainable_counts_match_for_a_fully_trainable_model():
    model = AilaNanoGPT(nano_50m())
    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    assert total == trainable == 51_393_024


def test_trainable_count_excludes_frozen_params():
    model = AilaNanoGPT(nano_50m())
    for p in model.blocks.parameters():
        p.requires_grad = False
    assert count_parameters(model, trainable_only=True) < count_parameters(model)


# -- checkpoint architecture-mismatch guard (spec §6) ---------------------

def test_matching_architecture_passes(tmp_path):
    model = AilaNanoGPT(nano_50m())
    path = str(tmp_path / "c.pt")
    save_checkpoint(path, model, optimizer=None, step=1, best_val_loss=0.0)
    ckpt = load_checkpoint(path)
    # Same architecture expected -> no error.
    validate_checkpoint_compatibility(ckpt, 8192, expected_config=nano_50m())


def test_loading_20m_checkpoint_into_50m_expectation_is_rejected():
    # A 20M-shaped checkpoint config, validated against a 50M expectation,
    # must fail with a clear architecture-mismatch error — not a later
    # cryptic state_dict shape error.
    ckpt = {"config": nano_20m().to_dict(), "model_state_dict": {}}
    with pytest.raises(CheckpointIncompatibleError) as excinfo:
        validate_checkpoint_compatibility(ckpt, 8192, expected_config=nano_50m())
    msg = str(excinfo.value)
    assert "architecture mismatch" in msg.lower()
    assert "d_model" in msg


def test_expected_config_absent_keeps_old_behaviour():
    # Without an expected_config, only the vocab is checked (back-compat).
    ckpt = {"config": nano_20m().to_dict(), "model_state_dict": {}}
    validate_checkpoint_compatibility(ckpt, 8192)  # no raise
