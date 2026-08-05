"""Checkpoint save/load/resume for Aila Nano training runs.

A checkpoint bundles everything needed to exactly resume training (model
+ optimizer + scheduler step + RNG-affecting counters) as well as
everything needed to *use* the model standalone later (config + step +
val loss), so the same file works for both `--resume` and inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from model.config import GPTConfig
from model.transformer import AilaNanoGPT

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str,
    model: AilaNanoGPT,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    best_val_loss: float,
    extra: dict[str, Any] | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": model.cfg.to_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "extra": extra or {},
    }
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    Path(tmp_path).replace(path)  # atomic on POSIX — avoids truncated checkpoints on crash
    logger.info("Saved checkpoint to %s (step=%d)", path, step)


def load_checkpoint(
    path: str, map_location: str = "cpu"
) -> dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def load_model_from_checkpoint(path: str, map_location: str = "cpu") -> AilaNanoGPT:
    ckpt = load_checkpoint(path, map_location=map_location)
    cfg = GPTConfig.from_dict(ckpt["config"])
    model = AilaNanoGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def restore_training_state(
    path: str,
    model: AilaNanoGPT,
    optimizer: torch.optim.Optimizer,
    map_location: str = "cpu",
) -> tuple[int, float]:
    """Load model + optimizer state in-place; returns (step, best_val_loss)."""
    ckpt = load_checkpoint(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["step"], ckpt["best_val_loss"]
