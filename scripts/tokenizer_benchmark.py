#!/usr/bin/env python3
"""Benchmark tokenizer candidates against the production tokenizer on
held-out English and Portuguese text, and select by measured evidence.

Metrics per tokenizer (on eval_en.txt / eval_pt.txt, which never entered
training): tokens/word, tokens/char, tokens/sentence(line), chars/token,
byte-fallback rate, accented-character handling (are ã/õ/ç/ê… real vocab
pieces or byte-fallback?), common-PT-word fragmentation, and vocabulary
utilization on the eval sets.

Emits a JSON with everything and a Markdown comparison table (the Phase-11
table). Does not change any tokenizer; measurement only.

Example:
    python scripts/tokenizer_benchmark.py \
        --production tokenizer/artifacts/aila_nano.model \
        --candidates tokenizer/artifacts/candidates/*/*.model \
        --eval-en datasets/tokenizer_corpus/eval_en.txt \
        --eval-pt datasets/tokenizer_corpus/eval_pt.txt \
        --out experiments/tokenizer_benchmark.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer import AilaTokenizer

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

_PT_WORDS = [
    "informação", "educação", "coração", "português", "programação",
    "inteligência", "computador", "desenvolvimento", "conversação",
    "aprendizagem", "obrigado", "você", "não", "manhã", "então",
]
_PT_ACCENTS = list("ãõáéíóúâêôç")


def _text_efficiency(tok: AilaTokenizer, text: str) -> dict:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    chars = sum(len(ln) for ln in lines)
    words = sum(len(_WORD_RE.findall(ln)) for ln in lines)
    fallback = set(tok.byte_fallback_ids)
    total_tokens = 0
    fallback_tokens = 0
    used_ids: set[int] = set()
    for ln in lines:
        ids = tok.encode(ln, add_bos=False, add_eos=False)
        total_tokens += len(ids)
        used_ids.update(ids)
        fallback_tokens += sum(1 for i in ids if i in fallback)
    return {
        "lines": len(lines), "chars": chars, "words": words, "tokens": total_tokens,
        "tokens_per_word": round(total_tokens / words, 4) if words else None,
        "tokens_per_char": round(total_tokens / chars, 4) if chars else None,
        "tokens_per_sentence": round(total_tokens / len(lines), 4) if lines else None,
        "chars_per_token": round(chars / total_tokens, 4) if total_tokens else None,
        "byte_fallback_rate": round(fallback_tokens / total_tokens, 5) if total_tokens else None,
        "distinct_ids_used": len(used_ids),
    }


def _pt_word_fragmentation(tok: AilaTokenizer) -> dict:
    split, pieces_total, examples = 0, 0, []
    for w in _PT_WORDS:
        ids = tok.encode(w, add_bos=False, add_eos=False)
        pieces_total += len(ids)
        if len(ids) > 1:
            split += 1
        if len(examples) < 12:
            examples.append({"word": w, "n_pieces": len(ids),
                             "pieces": [tok.id_to_piece(i) for i in ids]})
    return {
        "n_words": len(_PT_WORDS), "split_words": split,
        "split_fraction": round(split / len(_PT_WORDS), 4),
        "avg_pieces_per_word": round(pieces_total / len(_PT_WORDS), 4),
        "examples": examples,
    }


def _accent_handling(tok: AilaTokenizer) -> dict:
    fallback = set(tok.byte_fallback_ids)
    in_vocab, via_fallback = [], []
    for ch in _PT_ACCENTS:
        ids = tok.encode(ch, add_bos=False, add_eos=False)
        # A single-char that encodes with any byte-fallback piece is not
        # represented natively in the vocab.
        if any(i in fallback for i in ids):
            via_fallback.append(ch)
        else:
            in_vocab.append(ch)
    return {
        "accented_chars": _PT_ACCENTS,
        "in_vocab": in_vocab,
        "via_byte_fallback": via_fallback,
        "in_vocab_fraction": round(len(in_vocab) / len(_PT_ACCENTS), 4),
    }


def benchmark_one(model_path: str, en_text: str, pt_text: str) -> dict:
    tok = AilaTokenizer.load(model_path)
    en = _text_efficiency(tok, en_text)
    pt = _text_efficiency(tok, pt_text)
    return {
        "model_path": model_path,
        "vocab_size": tok.vocab_size,
        "english": en,
        "portuguese": pt,
        "pt_word_fragmentation": _pt_word_fragmentation(tok),
        "accent_handling": _accent_handling(tok),
        "vocab_utilization_pt": round((pt["distinct_ids_used"]) / tok.vocab_size, 4),
        "vocab_utilization_en": round((en["distinct_ids_used"]) / tok.vocab_size, 4),
    }


def comparison_table(results: list[dict]) -> str:
    header = (
        "| Tokenizer | Vocab | EN tok/word | PT tok/word | PT byte-fallback | "
        "PT word-split | Accents in-vocab |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in results:
        name = Path(r["model_path"]).stem
        rows.append(
            f"| {name} | {r['vocab_size']} | {r['english']['tokens_per_word']} | "
            f"{r['portuguese']['tokens_per_word']} | "
            f"{r['portuguese']['byte_fallback_rate']} | "
            f"{r['pt_word_fragmentation']['split_fraction']} | "
            f"{r['accent_handling']['in_vocab_fraction']} |"
        )
    return header + "\n".join(rows)


def recommend(results: list[dict]) -> dict:
    """Pick the best tokenizer by measured evidence: strongly reward PT
    efficiency (lower tok/word, lower fallback) without badly hurting EN.
    A simple, transparent score — lower is better."""
    best = None
    for r in results:
        # Explicit None checks — a real 0.0 byte-fallback rate is the *best*
        # case and must not be coerced to a penalty by `0.0 or 1.0`.
        en_tpw = r["english"]["tokens_per_word"]
        pt_tpw = r["portuguese"]["tokens_per_word"]
        fb = r["portuguese"]["byte_fallback_rate"]
        en_tpw = 99.0 if en_tpw is None else en_tpw
        pt_tpw = 99.0 if pt_tpw is None else pt_tpw
        fb = 1.0 if fb is None else fb
        # PT weighted 2x (the bottleneck), EN 1x, fallback penalty.
        score = 2.0 * pt_tpw + 1.0 * en_tpw + 5.0 * fb
        r["_score"] = round(score, 4)
        if best is None or score < best["_score"]:
            best = r
    return best


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--production", default="tokenizer/artifacts/aila_nano.model")
    p.add_argument("--candidates", nargs="+", required=True, help="Candidate .model files/globs.")
    p.add_argument("--eval-en", required=True)
    p.add_argument("--eval-pt", required=True)
    p.add_argument("--out", default="experiments/tokenizer_benchmark.json")
    p.add_argument("--report", default="experiments/TOKENIZER_REPORT.md")
    return p.parse_args()


def main():
    args = parse_args()
    en_text = Path(args.eval_en).read_text(encoding="utf-8", errors="ignore")
    pt_text = Path(args.eval_pt).read_text(encoding="utf-8", errors="ignore")

    model_paths = [args.production]
    for pat in args.candidates:
        model_paths.extend(sorted(glob.glob(pat)))
    # de-dup while preserving order
    seen, ordered = set(), []
    for m in model_paths:
        if m not in seen and Path(m).exists():
            seen.add(m); ordered.append(m)

    results = [benchmark_one(m, en_text, pt_text) for m in ordered]
    best = recommend(results)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"results": results, "recommended": best["model_path"]},
                                         indent=2, ensure_ascii=False))
    table = comparison_table(results)
    report = (
        "# Tokenizer benchmark (EN + PT, held-out)\n\n"
        f"Evaluated on held-out text that never entered training.\n\n"
        f"{table}\n\n"
        f"**Recommended (measured): `{Path(best['model_path']).stem}`** "
        f"(score {best['_score']}, lower is better; PT weighted 2×, EN 1×, "
        f"byte-fallback penalized).\n"
    )
    Path(args.report).write_text(report)
    print(table)
    print(f"\nRecommended: {best['model_path']} (score {best['_score']})")
    print(f"[written {args.out}, {args.report}]")


if __name__ == "__main__":
    main()
