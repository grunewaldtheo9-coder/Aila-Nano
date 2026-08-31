#!/usr/bin/env python3
"""Data-scaling experiment runner for Aila Nano.

Trains the *same* model on the *same* corpus at a series of token budgets,
so we can measure — not guess — how much Aila Nano 50M improves as it sees
more high-quality data. Each budget's run writes a machine-readable JSON
result (parameters, tokens, epochs, losses, perplexity, time, tokens/sec,
dataset version + hash, language mixture) into a results directory, and
`scripts/scaling_report.py` turns those into a comparison table + plot.

Everything reported is measured from the actual run. Budgets whose result
JSON already exists are skipped (`--skip-existing`), so a long sweep is
resumable across sessions. Runs never train past the requested token
budget (see training.trainer.resolve_max_steps).

Example (small, CPU-real):
    python scripts/scaling_experiment.py \
        --model-config configs/model/nano_50m_cpu256.yaml \
        --base-train-config configs/training/pretrain_50m.yaml \
        --token-budgets 2000000,5000000 \
        --out-root checkpoints/50m_data_scaling \
        --results-dir experiments/50m_data_scaling \
        --dataset-version aila_pretrain_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from model.config import GPTConfig, nano_10m, nano_20m, nano_50m
from model.transformer import AilaNanoGPT
from training.trainer import Trainer, TrainingConfig

logger = logging.getLogger(__name__)

_PRESETS = {"nano_10m": nano_10m, "nano_20m": nano_20m, "nano_50m": nano_50m}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _language_mixture(dataset_version: str | None) -> dict | None:
    """Read the language distribution from a dataset manifest if one exists
    at datasets/pretrain/<version>/manifest.json (produced by the data
    pipeline). Returns None when there is no manifest — never fabricated."""
    if not dataset_version:
        return None
    manifest = Path("datasets/pretrain") / dataset_version / "manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text())
            return data.get("language_distribution")
        except Exception:
            return None
    return None


def _build_model(args) -> GPTConfig:
    if args.model_config:
        return GPTConfig.from_yaml(args.model_config)
    if args.preset:
        return _PRESETS[args.preset]()
    return nano_50m()


@torch.no_grad()
def _mean_loss(trainer: Trainer, loader, iters: int) -> float:
    trainer.model.eval()
    losses, it = [], iter(loader)
    for _ in range(iters):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x, y = x.to(trainer.device), y.to(trainer.device)
        losses.append(trainer._forward_loss(x, y).item())
    trainer.model.train()
    return sum(losses) / max(1, len(losses))


def run_budget(args, budget: int, model_cfg: GPTConfig, base_train: dict) -> dict:
    """Train one budget and return its measured result dict."""
    import math

    ckpt_dir = str(Path(args.out_root) / f"tokens_{budget}")
    tb_dir = str(Path(args.out_root) / "tb" / f"tokens_{budget}")
    train_kwargs = dict(base_train)
    train_kwargs.update(
        max_tokens=budget,
        checkpoint_dir=ckpt_dir,
        tensorboard_dir=tb_dir,
        dataset_version=args.dataset_version,
    )
    cfg = TrainingConfig(**train_kwargs)

    model = AilaNanoGPT(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    trainer = Trainer(model, cfg)
    logger.info(
        "=== budget %s tokens -> %d steps (%d tokens/step), %d params ===",
        f"{budget:,}", trainer.max_steps, trainer.tokens_per_step, n_params,
    )

    t0 = time.time()
    state = trainer.train()
    elapsed = time.time() - t0

    tokens_seen = state.step * trainer.tokens_per_step
    unique_tokens = trainer.train_ds.num_tokens
    final_val = trainer.evaluate()
    final_train = _mean_loss(trainer, trainer.train_loader, cfg.eval_iters)

    result = {
        "model_parameters": n_params,
        "training_tokens_budget": budget,
        "tokens_seen": tokens_seen,
        "unique_tokens": unique_tokens,
        "epochs": round(tokens_seen / max(1, unique_tokens), 4),
        "tokens_per_parameter": round(tokens_seen / max(1, n_params), 4),
        "final_train_loss": round(final_train, 4),
        "best_val_loss": round(state.best_val_loss, 4),
        "final_val_loss": round(final_val, 4),
        "best_val_perplexity": round(math.exp(min(state.best_val_loss, 20)), 4),
        "training_time_sec": round(elapsed, 1),
        "tokens_per_sec": round(tokens_seen / max(elapsed, 1e-9), 1),
        "effective_max_steps": trainer.max_steps,
        "dataset_version": args.dataset_version,
        "dataset_sha256": trainer.dataset_meta["train_sha256"],
        "language_mixture": _language_mixture(args.dataset_version),
        "seed": cfg.seed,
        "context_length": model_cfg.max_seq_len,
        "checkpoint_dir": ckpt_dir,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Aila Nano data-scaling experiment runner.")
    p.add_argument("--model-config", default=None, help="Model YAML (default: nano_50m preset).")
    p.add_argument("--preset", default=None, choices=sorted(_PRESETS))
    p.add_argument("--base-train-config", required=True, help="TrainingConfig YAML (base hyperparams).")
    p.add_argument("--token-budgets", required=True,
                   help="Comma-separated token budgets, e.g. 2000000,5000000,10000000")
    p.add_argument("--out-root", default="checkpoints/50m_data_scaling")
    p.add_argument("--results-dir", default="experiments/50m_data_scaling")
    p.add_argument("--dataset-version", default=None)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip a budget whose result JSON already exists (resumable sweep).")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    args = parse_args()

    budgets = [int(b) for b in args.token_budgets.split(",") if b.strip()]
    model_cfg = _build_model(args)
    with open(args.base_train_config) as f:
        base_train = yaml.safe_load(f)
    # These are set per-budget; drop from the base so they don't double up.
    for k in ("max_tokens", "max_epochs", "checkpoint_dir", "tensorboard_dir", "dataset_version"):
        base_train.pop(k, None)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for budget in budgets:
        out_json = results_dir / f"tokens_{budget}.json"
        if args.skip_existing and out_json.exists():
            logger.info("skip %s (result exists: %s)", f"{budget:,}", out_json)
            continue
        result = run_budget(args, budget, model_cfg, base_train)
        out_json.write_text(json.dumps(result, indent=2))
        logger.info(
            "budget %s -> best_val %.4f (ppl %.2f), %d tok/s, %.0fs  [%s]",
            f"{budget:,}", result["best_val_loss"], result["best_val_perplexity"],
            result["tokens_per_sec"], result["training_time_sec"], out_json,
        )


if __name__ == "__main__":
    main()
