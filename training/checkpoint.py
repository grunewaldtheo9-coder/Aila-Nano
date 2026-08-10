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


class CheckpointNotDownloadedError(RuntimeError):
    """The file is a Git LFS *pointer*, not the checkpoint itself.

    By far the most common real-world failure for anyone installing from
    a GitHub ZIP: "Download ZIP" silently substitutes a ~130-byte text
    stub for every LFS-tracked file. `torch.load` then dies on
    `_pickle.UnpicklingError: invalid load key, 'v'` — 'v' being the
    first byte of "version https://git-lfs.github.com/spec/v1" — which
    tells the user nothing about what to do.
    """


# First line of every Git LFS pointer file.
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"

# A real checkpoint is hundreds of MB; a pointer is a few hundred bytes.
# Used only to describe the file in the error message.
_POINTER_SIZE_CEILING = 4096


def _looks_like_lfs_pointer(path: Path) -> bool:
    try:
        if path.stat().st_size > _POINTER_SIZE_CEILING:
            return False
        with open(path, "rb") as f:
            return f.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX
    except OSError:
        return False


def load_checkpoint(
    path: str, map_location: str = "cpu"
) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if _looks_like_lfs_pointer(file_path):
        size = file_path.stat().st_size
        raise CheckpointNotDownloadedError(
            f"'{path}' is a {size}-byte placeholder, not the real model file "
            f"(a real one is ~240 MB).\n\n"
            "This happens when the project is downloaded with GitHub's "
            '"Download ZIP" button: it replaces large files with a small '
            "text stub.\n\n"
            "To fix it, download the model file on its own:\n"
            "  1. On GitHub, open the folder checkpoints, then finetune_20m\n"
            "  2. Click best.pt\n"
            '  3. Click the download arrow ("Download raw file")\n'
            f"  4. Replace '{path}' with the file you just downloaded\n"
            "     (it should be around 240 MB)"
        )

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
