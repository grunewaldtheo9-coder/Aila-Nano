#!/usr/bin/env python3
"""Evaluate a checkpoint's language capability: English vs Portuguese
perplexity on held-out sets, plus a generation-degradation (repetition)
check. Reports the two languages separately so a Portuguese improvement
that damages English is visible.

Uses the tokenizer that MATCHES the checkpoint (the checkpoint records its
vocab; a mismatch is rejected up front with a clear error), so it works for
both current (8192) and future v2 (16384) checkpoints.

Example:
    python -m evaluation.evaluate \
        --checkpoint checkpoints/50m/finetune/best.pt \
        --tokenizer tokenizer/artifacts/aila_nano.model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import GPTConfig
from model.generate import generate
from model.transformer import AilaNanoGPT
from tokenizer import AilaTokenizer
from training.checkpoint import load_checkpoint, validate_checkpoint_compatibility

from evaluation.perplexity import evaluate_language_file, repetition_score

import torch

_DEFAULT_PROMPTS = ["Once upon a time", "The weather today", "Era uma vez", "A cidade de"]


def _load_prompts() -> list[tuple[str, str]]:
    """Fixed bilingual generation prompts (committed under evaluation/data/),
    kept constant across checkpoints so comparisons are meaningful. Falls
    back to a small default set if the files are absent."""
    out: list[tuple[str, str]] = []
    for lang, path in (("en", "evaluation/data/prompts_en.txt"),
                       ("pt", "evaluation/data/prompts_pt.txt")):
        p = Path(path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.append((lang, line.strip()))
    if not out:
        out = [("en" if i < 2 else "pt", pr) for i, pr in enumerate(_DEFAULT_PROMPTS)]
    return out


def load_model(checkpoint: str, tokenizer: AilaTokenizer, device="cpu"):
    ckpt = load_checkpoint(checkpoint, map_location=device)
    validate_checkpoint_compatibility(ckpt, tokenizer.vocab_size)
    model = AilaNanoGPT(GPTConfig.from_dict(ckpt["config"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default="tokenizer/artifacts/aila_nano.model")
    p.add_argument("--eval-en", default="evaluation/data/eval_en.txt")
    p.add_argument("--eval-pt", default="evaluation/data/eval_pt.txt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    tok = AilaTokenizer.load(args.tokenizer)
    model, ckpt = load_model(args.checkpoint, tok, device=args.device)

    en = evaluate_language_file(model, tok, args.eval_en, device=args.device)
    pt = evaluate_language_file(model, tok, args.eval_pt, device=args.device)

    gens = []
    for lang, prompt in _load_prompts():
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=args.device)
        out = generate(model, ids, max_new_tokens=40, temperature=0.7, top_k=40, eos_id=tok.eos_id)
        text = tok.decode(out[0, ids.shape[1]:].tolist())
        gens.append({"lang": lang, "prompt": prompt, "generation": text.strip(), **repetition_score(text)})

    result = {
        "checkpoint": args.checkpoint,
        "tokenizer": args.tokenizer,
        "tokenizer_vocab_size": tok.vocab_size,
        "checkpoint_step": ckpt.get("step"),
        "checkpoint_dataset": ckpt.get("extra", {}).get("dataset_version"),
        "english_perplexity": en,
        "portuguese_perplexity": pt,
        "generations": gens,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\n[written {args.out}]")


if __name__ == "__main__":
    main()
