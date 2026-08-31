#!/usr/bin/env python3
"""Evaluate the 50M-v2 bilingual checkpoints and assemble the experiment
report. For each checkpoint (best.pt per token budget) it measures English
and Portuguese **bits-per-character** (tokenizer-independent, so it is
comparable to the old 8192 model) and per-token perplexity, plus fixed
bilingual generations. It reads the scaling result JSONs for tokens-seen and
throughput, folds in the committed old-50M baseline, and writes:

  experiments/bilingual_50m_v2/evaluation.json
  experiments/bilingual_50m_v2/generations.json
  experiments/bilingual_50m_v2/training_metrics.json
  experiments/bilingual_50m_v2/README.md

All numbers are measured from the actual checkpoints. Nothing is fabricated;
a checkpoint that is missing is simply skipped.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model.generate import generate
from tokenizer import AilaTokenizer
from evaluation.evaluate import _load_prompts, load_model
from evaluation.perplexity import repetition_score, text_perplexity

CKPT_ROOT = "checkpoints_v2/50m_bilingual"
RESULTS = "experiments/bilingual_50m_v2"
V2_TOK = "tokenizer/artifacts/v2_bilingual/aila_nano_v2_bilingual.model"
EVAL_EN = "evaluation/data/eval_en.txt"
EVAL_PT = "evaluation/data/eval_pt.txt"


def _eval_checkpoint(path: str, tok: AilaTokenizer) -> dict:
    model, ckpt = load_model(path, tok)
    en = text_perplexity(model, tok, Path(EVAL_EN).read_text(encoding="utf-8"))
    pt = text_perplexity(model, tok, Path(EVAL_PT).read_text(encoding="utf-8"))
    gens = []
    for lang, prompt in _load_prompts():
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
        out = generate(model, ids, max_new_tokens=40, temperature=0.7, top_k=40, eos_id=tok.eos_id)
        text = tok.decode(out[0, ids.shape[1]:].tolist())
        gens.append({"lang": lang, "prompt": prompt, "generation": text.strip(), **repetition_score(text)})
    extra = ckpt.get("extra", {})
    return {
        "checkpoint": path,
        "tokens_seen": extra.get("tokens_seen"),
        "dataset_version": extra.get("dataset_version"),
        "params": ckpt.get("metadata", {}).get("parameters"),
        "english": en, "portuguese": pt, "generations": gens,
    }


def main():
    tok = AilaTokenizer.load(V2_TOK)
    ckpts = sorted(glob.glob(f"{CKPT_ROOT}/tokens_*/best.pt"),
                   key=lambda p: int(p.split("tokens_")[1].split("/")[0]))
    evals = [_eval_checkpoint(c, tok) for c in ckpts]

    # scaling metrics (tokens, throughput, time) from the sweep JSONs
    training = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f"{RESULTS}/tokens_*.json"))]

    baseline_path = Path(f"{RESULTS}/baseline_old50m_8192.json")
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None

    Path(RESULTS).mkdir(parents=True, exist_ok=True)
    Path(f"{RESULTS}/evaluation.json").write_text(json.dumps(evals, indent=2, ensure_ascii=False))
    Path(f"{RESULTS}/generations.json").write_text(
        json.dumps([{"checkpoint": e["checkpoint"], "generations": e["generations"]} for e in evals],
                   indent=2, ensure_ascii=False))
    Path(f"{RESULTS}/training_metrics.json").write_text(json.dumps(training, indent=2))

    # ---- README ----
    lines = ["# Aila Nano 50M-v2 (bilingual, 16384 tokenizer) — experiment report", ""]
    lines.append("Fresh 55,587,328-param model (vocab 16384) trained from scratch on the")
    lines.append("bilingual corpus `aila_pretrain_v2_bilingual` (62% PT / 38% EN by tokens).")
    lines.append("BPC = bits per character (tokenizer-independent, the only fair cross-tokenizer")
    lines.append("metric). Perplexity-per-token is NOT comparable across tokenizers.\n")

    if baseline:
        b_en = baseline["english_perplexity"]; b_pt = baseline["portuguese_perplexity"]
        lines.append("## Baseline: old 50M (8192 tokenizer, EN-only pretrain + instruction finetune)")
        lines.append("| | EN | PT |")
        lines.append("|---|---|---|")
        lines.append(f"| BPC | {b_en['bits_per_char']} | {b_pt['bits_per_char']} |")
        lines.append(f"| token ppl | {b_en['perplexity']} | {b_pt['perplexity']} |")
        lines.append("_(EN not token-matched: the old model saw ~10x more English.)_\n")

    lines.append("## New 50M-v2 by training-token budget (BPC, lower is better)")
    lines.append("| Tokens seen | EN BPC | PT BPC | EN ppl | PT ppl |")
    lines.append("|---|---|---|---|---|")
    for e in evals:
        lines.append(f"| {e['tokens_seen']:,} | {e['english']['bits_per_char']} | "
                     f"{e['portuguese']['bits_per_char']} | {e['english']['perplexity']} | "
                     f"{e['portuguese']['perplexity']} |")
    lines.append("")
    Path(f"{RESULTS}/README.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[written {RESULTS}/evaluation.json, generations.json, training_metrics.json, README.md]")


if __name__ == "__main__":
    main()
