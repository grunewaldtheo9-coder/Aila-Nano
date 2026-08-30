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


def checkpoint_metadata(model: AilaNanoGPT) -> dict[str, Any]:
    """Human-facing description of a model's architecture, stored alongside
    the weights so a checkpoint is self-describing (spec §7 checkpoint
    metadata). The authoritative architecture is still `config` — this is
    a convenience summary for reports and the CLI /model command."""
    cfg = model.cfg
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "model_name": "aila_nano",
        "model_size": f"{round(n_params / 1_000_000)}M",
        "parameters": n_params,
        "d_model": cfg.d_model,
        "layers": cfg.n_layers,
        "heads": cfg.n_heads,
        "kv_heads": cfg.n_kv_heads,
        "ffn": cfg.mlp_hidden_dim,
        "context_length": cfg.max_seq_len,
        "vocab_size": cfg.vocab_size,
    }


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
        "metadata": checkpoint_metadata(model),
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


class CheckpointIncompatibleError(RuntimeError):
    """The checkpoint's architecture doesn't match what it's being loaded
    into — most importantly a vocab size that disagrees with the tokenizer,
    which would let the model emit token ids the tokenizer can't decode.
    Raised with a clear message instead of a cryptic state_dict shape error
    (spec §7 checkpoint validation)."""


def describe_checkpoint(ckpt: dict[str, Any]) -> str:
    """One-line summary of a loaded checkpoint's architecture, from its
    stored metadata (falling back to `config` for older checkpoints that
    predate the metadata field)."""
    meta = ckpt.get("metadata")
    if not meta:
        cfg = ckpt.get("config", {})
        n = "?"
        return (
            f"aila_nano d_model={cfg.get('d_model')} layers={cfg.get('n_layers')} "
            f"heads={cfg.get('n_heads')} ctx={cfg.get('max_seq_len')} params={n}"
        )
    return (
        f"{meta['model_name']} {meta['model_size']} "
        f"({meta['parameters']:,} params) — d_model={meta['d_model']} "
        f"layers={meta['layers']} heads={meta['heads']} ffn={meta['ffn']} "
        f"ctx={meta['context_length']}"
    )


# Config fields that define the *architecture* (as opposed to training or
# behavioural settings). Two checkpoints agreeing on these can load each
# other's weights; disagreeing on any one produces a shape mismatch.
_ARCH_FIELDS = ("vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads",
                "mlp_hidden_mult", "tie_embeddings", "bias")


def validate_checkpoint_compatibility(
    ckpt: dict[str, Any],
    tokenizer_vocab_size: int,
    expected_config: "GPTConfig | dict | None" = None,
) -> None:
    """Raise CheckpointIncompatibleError if `ckpt` can't be used.

    Always checks the config is present and its vocab matches the tokenizer
    (a vocab mismatch would let the model emit undecodable token ids).

    When `expected_config` is given — a caller that wants a *specific*
    architecture (e.g. "load the 50M model") — the checkpoint's
    architecture-defining fields must match it, so a 20M checkpoint handed
    to a 50M expectation fails with a clear message instead of an obscure
    `load_state_dict` shape error. This runs *before* loading, so the
    failure is explained rather than a raw pickling/shape traceback."""
    cfg = ckpt.get("config")
    if not isinstance(cfg, dict):
        raise CheckpointIncompatibleError(
            "Checkpoint has no architecture config — it may be corrupted or from "
            "an incompatible version."
        )
    ckpt_vocab = cfg.get("vocab_size")
    if ckpt_vocab != tokenizer_vocab_size:
        raise CheckpointIncompatibleError(
            f"Checkpoint was trained with vocab_size={ckpt_vocab}, but the current "
            f"tokenizer has {tokenizer_vocab_size}. Use the tokenizer this "
            f"checkpoint was trained with, or the matching checkpoint for this "
            f"tokenizer — loading anyway would produce undecodable output."
        )

    if expected_config is not None:
        exp = expected_config.to_dict() if hasattr(expected_config, "to_dict") else dict(expected_config)
        mismatches = [
            (f, exp.get(f), cfg.get(f))
            for f in _ARCH_FIELDS
            if f in exp and exp.get(f) != cfg.get(f)
        ]
        if mismatches:
            exp_params = _approx_params(exp)
            got_params = _approx_params(cfg)
            detail = ", ".join(f"{f}: expected {e}, found {g}" for f, e, g in mismatches)
            raise CheckpointIncompatibleError(
                f"Checkpoint architecture mismatch: expected a "
                f"~{exp_params // 1_000_000}M configuration, found an incompatible "
                f"~{got_params // 1_000_000}M one ({detail}). Point at the matching "
                f"checkpoint for this architecture, or load without forcing a "
                f"specific architecture (the architecture is normally read from "
                f"the checkpoint itself)."
            )


def _approx_params(cfg: dict[str, Any]) -> int:
    """A rough parameter estimate from an architecture config dict, for
    human-readable size labels in error messages (embeddings + blocks).
    Not exact — only used to say "~20M" vs "~50M"."""
    try:
        d = int(cfg["d_model"])
        layers = int(cfg["n_layers"])
        vocab = int(cfg["vocab_size"])
        n_heads = int(cfg["n_heads"])
        n_kv = int(cfg["n_kv_heads"])
        mult = float(cfg["mlp_hidden_mult"])
        head_dim = d // n_heads
        ff = max(8, 8 * round(mult * d / 8))
        attn = d * (d + 2 * n_kv * head_dim + d)  # q,k,v,o (approx)
        mlp = d * ff * 3  # SwiGLU: gate, up, down
        per_block = attn + mlp + 2 * d  # + norms
        emb = vocab * d  # tied
        return emb + layers * per_block + d
    except Exception:
        return 0


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
