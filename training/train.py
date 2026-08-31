#!/usr/bin/env python3
"""Pretraining entrypoint.

Usage:
    python -m training.train --model-config configs/model/nano_10m.yaml \
        --train-config configs/training/pretrain.yaml

    # resume the latest checkpoint in the run's checkpoint_dir:
    python -m training.train --train-config configs/training/pretrain.yaml --resume
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from model.config import GPTConfig, nano_10m, nano_20m, nano_50m
from model.transformer import AilaNanoGPT
from training.trainer import Trainer, TrainingConfig

# Built-in architecture presets, selectable with --preset without needing a
# separate YAML file. --model-config still takes precedence when both given.
_PRESETS = {"nano_10m": nano_10m, "nano_20m": nano_20m, "nano_50m": nano_50m}


def parse_args():
    p = argparse.ArgumentParser(description="Train Aila Nano from scratch.")
    p.add_argument("--model-config", type=str, default=None, help="YAML path; overrides --preset")
    p.add_argument(
        "--preset", type=str, default=None, choices=sorted(_PRESETS),
        help="Built-in architecture preset (default: nano_10m when no --model-config).",
    )
    p.add_argument("--train-config", type=str, required=True, help="YAML path for TrainingConfig")
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in checkpoint_dir")
    p.add_argument("--resume-from", type=str, default=None, help="Resume from a specific checkpoint file")
    # Optional budget overrides (also settable in the train-config YAML).
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Token budget: stop after ~this many tokens (never trains past it).")
    p.add_argument("--max-epochs", type=float, default=None,
                   help="Epoch budget: stop after this many passes over the corpus.")
    p.add_argument("--max-steps", type=int, default=None, help="Override max optimizer steps.")
    p.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint_dir.")
    p.add_argument("--dataset-version", type=str, default=None,
                   help="Record this dataset version string in the checkpoint metadata.")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    args = parse_args()

    if args.model_config:
        model_cfg = GPTConfig.from_yaml(args.model_config)
    elif args.preset:
        model_cfg = _PRESETS[args.preset]()
    else:
        model_cfg = nano_10m()
    with open(args.train_config) as f:
        train_cfg = TrainingConfig(**yaml.safe_load(f))

    # CLI overrides win over the YAML (only when provided).
    for attr, val in (
        ("max_tokens", args.max_tokens),
        ("max_epochs", args.max_epochs),
        ("max_steps", args.max_steps),
        ("checkpoint_dir", args.checkpoint_dir),
        ("dataset_version", args.dataset_version),
    ):
        if val is not None:
            setattr(train_cfg, attr, val)

    model = AilaNanoGPT(model_cfg)
    n_params = model.num_parameters()
    logging.info("Model initialized: %d parameters (%.2fM)", n_params, n_params / 1e6)

    trainer = Trainer(model, train_cfg)
    if args.resume_from:
        trainer.resume(args.resume_from)
    elif args.resume:
        trainer.resume()

    trainer.train()
    logging.info("Training complete. Best val loss: %.4f", trainer.state.best_val_loss)


if __name__ == "__main__":
    main()
