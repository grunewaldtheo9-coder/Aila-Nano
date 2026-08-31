#!/usr/bin/env python3
"""Measure how efficiently the current SentencePiece tokenizer encodes
English vs Portuguese, so a decision to (not) grow the vocabulary is made
on evidence, not vibes (spec: "Do not change the production tokenizer
without measured evidence").

Reports, per language sample:
  * tokens / character   (lower = more efficient)
  * tokens / word        (≈1.0 is ideal; >1 means words are being split)
  * characters / token
  * fraction of common Portuguese words that get split into >1 piece
  * a few example fragmentations

Run on real text (defaults to the local TinyStories sample for EN and the
bundled Portuguese examples for PT), or pass your own files.

Usage:
    python scripts/tokenizer_stats.py \
        --tokenizer tokenizer/artifacts/aila_nano.model
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer import AilaTokenizer

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Common Brazilian-Portuguese words a good PT tokenizer should keep whole-ish.
_COMMON_PT_WORDS = [
    "obrigado", "obrigada", "você", "não", "coração", "informação",
    "cidade", "trabalho", "português", "criança", "então", "também",
    "porque", "melhor", "empresa", "linguagem", "aprender", "manhã",
]


def _sample_text() -> tuple[str, str]:
    """Default EN / PT samples from files in the repo (real text)."""
    en = ""
    ts = Path("datasets/raw/tinystories.txt")
    if ts.exists():
        with open(ts, encoding="utf-8", errors="ignore") as f:
            en = f.read(200_000)
    pt_parts = []
    pt_jsonl = Path("datasets/aila_knowledge/portuguese_basic.jsonl")
    if pt_jsonl.exists():
        for line in pt_jsonl.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            pt_parts.append(" ".join(str(rec.get(k, "")) for k in ("instruction", "output", "input")))
    return en, "\n".join(pt_parts)


def efficiency(tok: AilaTokenizer, text: str) -> dict:
    text = text.strip()
    if not text:
        return {"chars": 0, "words": 0, "tokens": 0}
    ids = tok.encode(text, add_bos=False, add_eos=False)
    chars = len(text)
    words = len(_WORD_RE.findall(text))
    return {
        "chars": chars,
        "words": words,
        "tokens": len(ids),
        "tokens_per_char": round(len(ids) / chars, 4) if chars else None,
        "tokens_per_word": round(len(ids) / words, 4) if words else None,
        "chars_per_token": round(chars / len(ids), 4) if ids else None,
    }


def word_fragmentation(tok: AilaTokenizer, words: list[str]) -> dict:
    split, examples = 0, []
    for w in words:
        ids = tok.encode(w, add_bos=False, add_eos=False)
        if len(ids) > 1:
            split += 1
            if len(examples) < 8:
                pieces = [tok.decode([i], skip_special_tokens=True) for i in ids]
                examples.append({"word": w, "pieces": pieces, "n": len(ids)})
    return {
        "n_words": len(words),
        "split_words": split,
        "split_fraction": round(split / len(words), 4) if words else None,
        "examples": examples,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="tokenizer/artifacts/aila_nano.model")
    p.add_argument("--en-file", default=None)
    p.add_argument("--pt-file", default=None)
    p.add_argument("--out", default=None, help="Optional JSON output path.")
    return p.parse_args()


def main():
    args = parse_args()
    tok = AilaTokenizer.load(args.tokenizer)

    en_default, pt_default = _sample_text()
    en = Path(args.en_file).read_text(encoding="utf-8", errors="ignore") if args.en_file else en_default
    pt = Path(args.pt_file).read_text(encoding="utf-8", errors="ignore") if args.pt_file else pt_default

    result = {
        "vocab_size": tok.vocab_size,
        "english": efficiency(tok, en),
        "portuguese": efficiency(tok, pt),
        "portuguese_word_fragmentation": word_fragmentation(tok, _COMMON_PT_WORDS),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\n[written {args.out}]")


if __name__ == "__main__":
    main()
