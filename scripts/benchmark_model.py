#!/usr/bin/env python3
"""Benchmark a fine-tuned Aila Nano checkpoint — used to compare 1.0 vs
2.0 with numbers instead of claims.

Metrics:
- teacher-forced validation loss / perplexity over the instruction
  datasets (deterministic, comparable across checkpoints that share the
  tokenizer);
- identity accuracy: fraction of identity questions whose sampled reply
  contains the expected keyword(s);
- robustness: malformed/misspelled/short inputs must produce a
  non-empty, non-crashing reply;
- Portuguese basic: PT greetings/identity get a reply containing an
  expected PT keyword;
- generation latency (tokens/s over the benchmark's sampled replies).

Sampling uses each agent's default settings with a fixed seed, so runs
are repeatable.

Usage:
    python scripts/benchmark_model.py --checkpoint checkpoints/finetune/best.pt
    python scripts/benchmark_model.py --checkpoint checkpoints/finetune_20m/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from agents.registry import get_agent
from finetuning.dataset import InstructionDataset, collate_instruction_batch
from model.config import GPTConfig
from model.transformer import AilaNanoGPT
from tokenizer import AilaTokenizer
from training.checkpoint import load_checkpoint

IDENTITY_CASES = [
    ("Who created you?", ["aila"]),
    ("What is Aila Nano?", ["model"]),
    ("Who founded Aila Company Solutions?", ["theo", "guilherme", "grunewald"]),
    ("Introduce yourself.", ["aila"]),
]

ROBUSTNESS_CASES = [
    "who crated apple",
    "Who created apple???",
    "what is appl",
    "Open AI~",
    "APPLE founders",
    "hello!!",
    "ok",
    "??",
]

PT_CASES = [
    ("Olá!", ["olá", "ajudar", "oi"]),
    ("Quem criou você?", ["aila"]),
    ("Obrigado!", ["nada", "ajudar"]),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default="tokenizer/artifacts/aila_nano.model")
    p.add_argument(
        "--data",
        nargs="+",
        default=[
            "datasets/aila_knowledge/aila_company.jsonl",
            "datasets/sample/finetune_sample.jsonl",
            "datasets/aila_knowledge/portuguese_basic.jsonl",
        ],
    )
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


@torch.no_grad()
def eval_loss(model, tokenizer, data_paths, max_seq_len) -> float:
    ds = InstructionDataset(data_paths, tokenizer, max_seq_len=max_seq_len)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=8, shuffle=False,
        collate_fn=lambda b: collate_instruction_batch(b, pad_id=tokenizer.pad_id),
    )
    total, batches = 0.0, 0
    for batch in loader:
        _, loss = model(batch["input_ids"], targets=batch["labels"])
        total += loss.item()
        batches += 1
    return total / max(1, batches)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    tok = AilaTokenizer.load(args.tokenizer)
    ckpt = load_checkpoint(args.checkpoint)
    model = AilaNanoGPT(GPTConfig.from_dict(ckpt["config"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    loss = eval_loss(model, tok, args.data, model.cfg.max_seq_len)

    def reply(question: str) -> str:
        agent = get_agent("general", model, tok, device="cpu")
        return agent.respond("bench", question, remember_turn=False)

    started = time.time()
    generated_chars = 0

    identity_hits = 0
    for question, keywords in IDENTITY_CASES:
        r = reply(question).lower()
        generated_chars += len(r)
        if any(k in r for k in keywords):
            identity_hits += 1

    robust_ok = 0
    for question in ROBUSTNESS_CASES:
        try:
            r = reply(question)
            generated_chars += len(r)
            if isinstance(r, str) and r.strip():
                robust_ok += 1
        except Exception:  # noqa: BLE001 — a crash is exactly what we're measuring
            pass

    pt_hits = 0
    for question, keywords in PT_CASES:
        r = reply(question).lower()
        generated_chars += len(r)
        if any(k in r for k in keywords):
            pt_hits += 1

    elapsed = time.time() - started
    report = {
        "checkpoint": args.checkpoint,
        "parameters": n_params,
        "val_loss": round(loss, 4),
        "val_perplexity": round(float(torch.exp(torch.tensor(loss))), 2),
        "identity_accuracy": f"{identity_hits}/{len(IDENTITY_CASES)}",
        "robustness_no_crash_nonempty": f"{robust_ok}/{len(ROBUSTNESS_CASES)}",
        "portuguese_basic": f"{pt_hits}/{len(PT_CASES)}",
        "generation_seconds_total": round(elapsed, 1),
        "generated_chars": generated_chars,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
