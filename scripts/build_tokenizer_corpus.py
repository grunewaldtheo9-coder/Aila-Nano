#!/usr/bin/env python3
"""Build a balanced English+Portuguese *text* corpus for training tokenizer
candidates, plus disjoint held-out EN/PT evaluation files that never enter
tokenizer training (or model training).

The English half is sampled from TinyStories; the Portuguese half from the
provided PT text files (Project Gutenberg public-domain literature). Each
paragraph is Gutenberg-stripped, cleaned, and language-verified, so English
paragraphs cannot sneak into the PT half (and vice-versa). The EN/PT byte
ratio is configurable so English does not dominate the vocabulary
(`--pt-fraction`).

Outputs (under --out-dir):
  tokenizer_train.txt   one document per line, shuffled EN+PT
  eval_en.txt           held-out English lines (never trained on)
  eval_pt.txt           held-out Portuguese lines (never trained on)
  corpus_stats.json     byte/char/line counts per language

Deterministic (seeded). Streaming/​line-based; never loads a whole corpus
structure into memory beyond the document list.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "datasets" / "scripts"))

import langid  # noqa: E402
from build_pretrain_corpus import split_documents, strip_gutenberg_boilerplate  # noqa: E402
from clean_text import clean_document, is_low_quality  # noqa: E402


def _load_docs(paths: list[str], expect_lang: str, min_chars: int) -> list[str]:
    docs: list[str] = []
    for fp in paths:
        raw = Path(fp).read_text(encoding="utf-8", errors="ignore")
        raw = strip_gutenberg_boilerplate(raw)
        for doc in split_documents(raw, min_chars=min_chars):
            doc = clean_document(doc)
            if len(doc) < min_chars or is_low_quality(doc):
                continue
            if langid.identify(doc) != expect_lang:
                continue
            # SentencePiece trains on lines; collapse internal newlines.
            docs.append(" ".join(doc.split()))
    return docs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pt-input", nargs="+", required=True, help="Portuguese text files/globs.")
    p.add_argument("--en-input", nargs="+", required=True, help="English text file(s) (e.g. TinyStories).")
    p.add_argument("--out-dir", default="datasets/tokenizer_corpus")
    p.add_argument("--pt-fraction", type=float, default=0.5, help="Target PT share of the training bytes.")
    p.add_argument("--eval-fraction", type=float, default=0.1)
    p.add_argument("--min-doc-chars", type=int, default=120)
    p.add_argument("--en-max-bytes", type=int, default=4_000_000, help="Cap EN input read (TinyStories is large).")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    pt_files: list[str] = []
    for pat in args.pt_input:
        pt_files.extend(sorted(glob.glob(pat)))
    pt_docs = _load_docs(pt_files, "pt", args.min_doc_chars)

    # English: read a bounded slice so TinyStories doesn't blow up memory.
    en_tmp: list[str] = []
    read = 0
    for fp in args.en_input:
        with open(fp, encoding="utf-8", errors="ignore") as f:
            for chunk in f:
                en_tmp.append(chunk)
                read += len(chunk)
                if read >= args.en_max_bytes:
                    break
        if read >= args.en_max_bytes:
            break
    en_text = "".join(en_tmp)
    en_parts = en_text.split("<|endoftext|>") if "<|endoftext|>" in en_text else en_text.split("\n\n")
    en_docs: list[str] = []
    for doc in en_parts:
        doc = clean_document(doc)
        if len(doc) < args.min_doc_chars or is_low_quality(doc):
            continue
        if langid.identify(doc) != "en":
            continue
        en_docs.append(" ".join(doc.split()))

    # Balance by bytes toward the target PT fraction (down-sample the larger
    # side; never repeat documents).
    def _bytes(docs):
        return sum(len(d) for d in docs)

    rng.shuffle(pt_docs); rng.shuffle(en_docs)
    pt_bytes, en_bytes = _bytes(pt_docs), _bytes(en_docs)
    # Want pt_bytes / (pt_bytes+en_bytes) ~= pt_fraction. Cap EN accordingly.
    if pt_bytes > 0 and args.pt_fraction < 1.0:
        target_en = int(pt_bytes * (1 - args.pt_fraction) / args.pt_fraction)
        if en_bytes > target_en:
            kept, acc = [], 0
            for d in en_docs:
                kept.append(d); acc += len(d)
                if acc >= target_en:
                    break
            en_docs = kept

    def _split_eval(docs):
        n_eval = max(1, int(args.eval_fraction * len(docs)))
        return docs[n_eval:], docs[:n_eval]  # train, eval (disjoint)

    pt_train, pt_eval = _split_eval(pt_docs)
    en_train, en_eval = _split_eval(en_docs)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_docs = pt_train + en_train
    rng.shuffle(train_docs)
    (out / "tokenizer_train.txt").write_text("\n".join(train_docs) + "\n", encoding="utf-8")
    (out / "eval_en.txt").write_text("\n".join(en_eval) + "\n", encoding="utf-8")
    (out / "eval_pt.txt").write_text("\n".join(pt_eval) + "\n", encoding="utf-8")

    stats = {
        "pt_documents_train": len(pt_train), "en_documents_train": len(en_train),
        "pt_documents_eval": len(pt_eval), "en_documents_eval": len(en_eval),
        "pt_bytes_train": _bytes(pt_train), "en_bytes_train": _bytes(en_train),
        "pt_fraction_actual": round(_bytes(pt_train) / max(1, _bytes(pt_train) + _bytes(en_train)), 4),
        "pt_chars_train": sum(len(d) for d in pt_train),
        "en_chars_train": sum(len(d) for d in en_train),
        "seed": args.seed,
        "pt_sources": pt_files,
    }
    (out / "corpus_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\n[written {out}/tokenizer_train.txt, eval_en.txt, eval_pt.txt]")


if __name__ == "__main__":
    main()
