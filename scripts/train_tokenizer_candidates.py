#!/usr/bin/env python3
"""Train bilingual (EN+PT) SentencePiece tokenizer candidates at several
vocabulary sizes, WITHOUT touching the production tokenizer.

Each candidate is written to `tokenizer/artifacts/candidates/<name>/` with a
small manifest (vocab size, character coverage, training corpus + its hash,
git commit, timestamp) so it is reproducible and clearly versioned. The
existing `tokenizer/artifacts/aila_nano.model` — and every checkpoint that
depends on it — is left completely unchanged.

Key change vs the production tokenizer for Portuguese: the training corpus
is ~50% Portuguese (so PT subwords/characters earn vocabulary) and
`character_coverage` defaults to 1.0 (so accented characters ã/õ/ç/ê get
real vocab entries instead of byte-fallback).

Example:
    python scripts/train_tokenizer_candidates.py \
        --corpus datasets/tokenizer_corpus/tokenizer_train.txt \
        --vocab-sizes 8192,12288,16384
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer.trainer import train_tokenizer


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, help="Plain-text training corpus (one doc per line).")
    p.add_argument("--vocab-sizes", default="8192,12288,16384")
    p.add_argument("--character-coverage", type=float, default=1.0)
    p.add_argument("--out-root", default="tokenizer/artifacts/candidates")
    p.add_argument("--name-prefix", default="bilingual")
    return p.parse_args()


def main():
    args = parse_args()
    vocab_sizes = [int(v) for v in args.vocab_sizes.split(",") if v.strip()]
    corpus_hash = _file_sha256(args.corpus)
    commit = _git_commit()

    for vocab in vocab_sizes:
        name = f"{args.name_prefix}_{vocab}"
        out_dir = Path(args.out_root) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = str(out_dir / name)
        print(f"=== training {name} (vocab={vocab}, coverage={args.character_coverage}) ===")
        model_path = train_tokenizer(
            input_files=[args.corpus],
            model_prefix=prefix,
            vocab_size=vocab,
            model_type="bpe",
            character_coverage=args.character_coverage,
        )
        manifest = {
            "name": name,
            "vocab_size": vocab,
            "character_coverage": args.character_coverage,
            "model_type": "bpe",
            "training_corpus": args.corpus,
            "training_corpus_sha256": corpus_hash,
            "model_path": model_path,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "note": "Bilingual EN+PT candidate. Does NOT replace tokenizer/artifacts/aila_nano.model.",
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"  -> {model_path}")

    print("\nAll candidates trained. Production tokenizer unchanged.")


if __name__ == "__main__":
    main()
