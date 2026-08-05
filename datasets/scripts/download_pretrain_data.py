#!/usr/bin/env python3
"""Download, clean, and deduplicate public-domain / permissively-licensed
pretraining corpora for Aila Nano.

Requires the extra dependencies in requirements-datasets.txt:
    pip install -r requirements-datasets.txt

Sources (see datasets/README.md for full license text and rationale):

  - roneneldan/TinyStories (CDLA-Sharing-1.0) — short, simple, coherent
    English stories, purpose-built for training very small language
    models. This is the primary corpus for Aila Nano: at ~10.9M
    parameters, the model has far more capacity to spare learning clean,
    simple grammar and narrative structure than it does trying to absorb
    web-scale, high-entropy text.
  - wikitext-103-raw-v1 (CC BY-SA 3.0) — encyclopedic text, added in a
    smaller proportion for topical/factual breadth beyond stories.

Usage:
    python datasets/scripts/download_pretrain_data.py \
        --out-dir datasets/raw \
        --tinystories-docs 200000 \
        --wikitext-docs 20000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Our own datasets/ directory has no __init__.py (it's a data directory,
# not a Python package) and this script needs the *installed*
# `datasets` PyPI package (Hugging Face) for `load_dataset` below. Those
# two facts combined mean `import datasets.scripts...` can never resolve
# to our own datasets/scripts/ — Python's import system always prefers a
# regular installed package over a same-named namespace package, no
# matter what's prepended to sys.path. So we import clean_text/dedupe as
# bare modules from this script's own directory instead, sidestepping
# the `datasets` package name entirely.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from clean_text import clean_corpus  # noqa: E402
from dedupe import dedupe_corpus  # noqa: E402

logger = logging.getLogger(__name__)


def _require_hf_datasets():
    try:
        import datasets as hf_datasets  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "This script needs the Hugging Face `datasets` library. Install it with:\n"
            "    pip install -r requirements-datasets.txt"
        ) from e


def download_tinystories(max_docs: int) -> list[str]:
    from datasets import load_dataset

    logger.info("Downloading roneneldan/TinyStories (up to %d docs)...", max_docs)
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    docs = []
    for i, row in enumerate(ds):
        if i >= max_docs:
            break
        docs.append(row["text"])
    logger.info("Fetched %d TinyStories documents.", len(docs))
    return docs


def download_wikitext(max_docs: int) -> list[str]:
    from datasets import load_dataset

    logger.info("Downloading wikitext-103-raw-v1 (up to %d docs)...", max_docs)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    # wikitext ships line-by-line; join consecutive non-empty lines into
    # documents split at its own article-heading markers ("= Title =").
    docs, current = [], []
    for row in ds:
        line = row["text"]
        if line.strip().startswith("= ") and current:
            docs.append("\n".join(current))
            current = []
            if len(docs) >= max_docs:
                break
        if line.strip():
            current.append(line.strip())
    if current and len(docs) < max_docs:
        docs.append("\n".join(current))
    logger.info("Fetched %d Wikitext documents.", len(docs))
    return docs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="datasets/raw")
    p.add_argument("--tinystories-docs", type=int, default=200_000)
    p.add_argument("--wikitext-docs", type=int, default=20_000)
    p.add_argument("--near-dedupe-threshold", type=float, default=0.85)
    p.add_argument(
        "--max-near-dedupe-docs",
        type=int,
        default=20_000,
        help=(
            "Safety cap: near-duplicate detection (dedupe.py::near_dedupe) is O(n^2) "
            "and will effectively hang on large corpora. If a corpus has more than "
            "this many documents after cleaning, near-dedup is skipped automatically "
            "(exact-duplicate removal still runs) unless --force-near-dedupe is set."
        ),
    )
    p.add_argument(
        "--force-near-dedupe",
        action="store_true",
        help="Run near-duplicate detection even above --max-near-dedupe-docs. Can take a very long time.",
    )
    p.add_argument("--skip-tinystories", action="store_true")
    p.add_argument("--skip-wikitext", action="store_true")
    return p.parse_args()


def _dedupe_with_safety_cap(docs: list[str], args) -> list[str]:
    near_threshold = args.near_dedupe_threshold
    if len(docs) > args.max_near_dedupe_docs and not args.force_near_dedupe:
        logger.warning(
            "%d documents exceeds --max-near-dedupe-docs=%d; skipping the O(n^2) "
            "near-duplicate pass (exact-duplicate removal still applies). Pass "
            "--force-near-dedupe to run it anyway (can take a very long time at this size).",
            len(docs),
            args.max_near_dedupe_docs,
        )
        near_threshold = None
    return dedupe_corpus(docs, near_threshold=near_threshold)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    args = parse_args()
    _require_hf_datasets()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_tinystories:
        docs = download_tinystories(args.tinystories_docs)
        docs = clean_corpus(docs)
        docs = _dedupe_with_safety_cap(docs, args)
        path = out_dir / "tinystories.txt"
        path.write_text("\n\n".join(docs), encoding="utf-8")
        logger.info("Wrote %d cleaned TinyStories docs to %s", len(docs), path)

    if not args.skip_wikitext:
        docs = download_wikitext(args.wikitext_docs)
        docs = clean_corpus(docs, min_chars=200)
        docs = _dedupe_with_safety_cap(docs, args)
        path = out_dir / "wikitext.txt"
        path.write_text("\n\n".join(docs), encoding="utf-8")
        logger.info("Wrote %d cleaned Wikitext docs to %s", len(docs), path)

    logger.info(
        "Done. Next: train a tokenizer on these files, then run "
        "datasets/scripts/prepare_pretrain.py to produce token .bin shards."
    )


if __name__ == "__main__":
    main()
