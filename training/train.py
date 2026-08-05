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

from model.config import GPTConfig, nano_10m
from model.transformer import AilaNanoGPT
from training.trainer import Trainer, TrainingConfig


def parse_args():
    p = argparse.ArgumentParser(description="Train Aila Nano from scratch.")
    p.add_argument("--model-config", type=str, default=None, help="YAML path; default: nano_10m preset")
    p.add_argument("--train-config", type=str, required=True, help="YAML path for TrainingConfig")
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in checkpoint_dir")
    p.add_argument("--resume-from", type=str, default=None, help="Resume from a specific checkpoint file")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    args = parse_args()

    model_cfg = GPTConfig.from_yaml(args.model_config) if args.model_config else nano_10m()
    with open(args.train_config) as f:
        train_cfg = TrainingConfig(**yaml.safe_load(f))

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
