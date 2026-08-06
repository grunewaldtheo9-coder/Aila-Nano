#!/usr/bin/env python3
"""Sanity-check tool: run a fixed set of test prompts through every
checkpoint in a fine-tuning output directory, so a real generation
regression (e.g. a repetition-loop collapse from overfitting) can be
caught by eye across a whole run instead of only checking the final or
"best" (lowest val-loss) checkpoint — teacher-forced val_loss can look
great while autoregressive generation has already collapsed.

Usage:
    python scripts/compare_finetune_checkpoints.py \
        --dir checkpoints/finetune_real --tokenizer tokenizer/artifacts/aila_nano.model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.registry import get_agent
from model.config import GPTConfig
from model.transformer import AilaNanoGPT
from tokenizer import AilaTokenizer
from training.checkpoint import load_checkpoint

TEST_PROMPTS = [
    ("general", "Who created you?"),
    ("general", "What is Aila Nano?"),
    ("general", "Hello, how are you?"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="Fine-tune output dir containing checkpoints")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--pattern", default="epoch_*.pt")
    return p.parse_args()


def main():
    args = parse_args()
    tok = AilaTokenizer.load(args.tokenizer)
    checkpoints = sorted(Path(args.dir).glob(args.pattern))
    if not checkpoints:
        print(f"No checkpoints matching {args.pattern} in {args.dir}")
        return

    for ckpt_path in checkpoints:
        ckpt = load_checkpoint(str(ckpt_path))
        model = AilaNanoGPT(GPTConfig.from_dict(ckpt["config"]))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        print(f"\n{'=' * 60}\n{ckpt_path.name}  (step={ckpt['step']})\n{'=' * 60}")
        for persona, question in TEST_PROMPTS:
            agent = get_agent(persona, model, tok, device="cpu")
            reply = agent.respond("cmp", question, remember_turn=False)
            # Truncate very long/degenerate replies for a scannable report.
            shown = reply if len(reply) < 200 else reply[:200] + "... [truncated]"
            print(f"  [{persona}] Q: {question}\n    A: {shown}")


if __name__ == "__main__":
    main()
