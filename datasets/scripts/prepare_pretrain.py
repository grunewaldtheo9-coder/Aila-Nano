#!/usr/bin/env python3
"""Tokenize one or more plain-text corpora into the flat uint16 `.bin`
shards `training/dataset.py` reads, with a train/val split.

Usage:
    python datasets/scripts/prepare_pretrain.py \
        --input datasets/sample/pretrain_sample.txt \
        --tokenizer tokenizer/artifacts/aila_nano.model \
        --out-dir datasets/processed \
        --val-fraction 0.05
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from tokenizer import AilaTokenizer
from training.dataset import write_token_bin

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True, help="One or more UTF-8 text files")
    p.add_argument("--tokenizer", required=True, help="Path to trained .model file")
    p.add_argument("--out-dir", default="datasets/processed")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    args = parse_args()

    tok = AilaTokenizer.load(args.tokenizer)

    all_ids: list[int] = []
    for path in args.input:
        text = Path(path).read_text(encoding="utf-8")
        # Split on blank lines into documents and add BOS/EOS per document
        # so the model learns clean document boundaries rather than
        # treating the whole corpus as one endless stream.
        docs = [d.strip() for d in text.split("\n\n") if d.strip()] or [text]
        for doc in docs:
            all_ids.extend(tok.encode(doc, add_bos=True, add_eos=True))
        logger.info("Tokenized %s -> running total %d tokens", path, len(all_ids))

    ids = np.array(all_ids, dtype=np.int64)

    n_val = max(1, int(len(ids) * args.val_fraction))
    # Contiguous split (not shuffled) so validation still measures
    # coherent-document perplexity rather than scrambled tokens.
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_token_bin(train_ids, str(out_dir / "pretrain_train.bin"))
    write_token_bin(val_ids, str(out_dir / "pretrain_val.bin"))

    logger.info(
        "Wrote %d train tokens and %d val tokens to %s", len(train_ids), len(val_ids), out_dir
    )


if __name__ == "__main__":
    main()
