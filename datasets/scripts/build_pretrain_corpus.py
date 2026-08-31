#!/usr/bin/env python3
"""Reproducible pretraining-corpus builder.

Takes raw text files, runs them through the cleaning / language-ID /
filtering / deduplication pipeline, tokenizes the survivors into the flat
uint16 `.bin` shards the trainer reads, makes a deterministic train/val
split, and writes a versioned manifest (sources, hashes, token counts,
language distribution, filter config). Every stage reports measurable
statistics — nothing is silently discarded.

The language mixture is configurable (`--target-ratios pt=0.7,en=0.3`):
documents are language-identified and, when ratios are given, down-sampled
per language to approximate the target proportion (never up-sampled /
repeated — a language short of its target simply stays under it, reported
honestly).

Processing is document-at-a-time so it does not require loading a whole
web-scale corpus into RAM; at nano scale the dedup index is in-memory,
which is documented as the current limit.

Example (Portuguese public-domain literature):
    python datasets/scripts/build_pretrain_corpus.py \
        --input datasets/raw/pt/*.txt \
        --tokenizer tokenizer/artifacts/aila_nano.model \
        --version aila_pretrain_pt_v1 \
        --val-fraction 0.05 \
        --source-name "Project Gutenberg (Machado de Assis)" \
        --source-license "Public domain (Project Gutenberg License)"
"""

from __future__ import annotations

import argparse
import glob
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import langid
from clean_text import clean_document, is_low_quality
from dataset_manifest import Source, build_manifest, write_manifest
from dedupe import exact_dedupe, near_dedupe

from tokenizer import AilaTokenizer
from training.dataset import write_token_bin

logger = logging.getLogger(__name__)

_GUT_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE | re.DOTALL)
_GUT_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE | re.DOTALL)


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove the Project Gutenberg license header/footer, keeping only the
    work itself (boilerplate removal). No-op for non-Gutenberg text."""
    m1 = _GUT_START.search(text)
    if m1:
        text = text[m1.end():]
    m2 = _GUT_END.search(text)
    if m2:
        text = text[: m2.start()]
    return text


def split_documents(text: str, min_chars: int = 200) -> list[str]:
    """Split a long text into paragraph-ish documents on blank lines."""
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if len(p.strip()) >= min_chars]


def _parse_ratios(spec: str | None) -> dict[str, float]:
    if not spec:
        return {}
    out = {}
    for part in spec.split(","):
        k, _, v = part.partition("=")
        out[k.strip()] = float(v)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True, help="Text files / globs.")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--version", required=True, help="Dataset version name.")
    p.add_argument("--out-root", default="datasets/pretrain")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--min-doc-chars", type=int, default=200)
    p.add_argument("--target-ratios", default=None,
                   help="Optional per-language target proportions, e.g. pt=0.7,en=0.3")
    p.add_argument("--source-name", default="")
    p.add_argument("--source-license", default="")
    p.add_argument("--source-url", default="")
    p.add_argument("--no-near-dedup", action="store_true",
                   help="Skip the O(n^2) near-duplicate pass (exact dedup still runs). "
                        "Use for large corpora of distinct works where near-dup adds "
                        "little; keeps the build tractable on CPU.")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    args = parse_args()

    files: list[str] = []
    for pat in args.input:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        raise SystemExit(f"No input files matched: {args.input}")

    stats = {"raw_documents": 0, "after_length_filter": 0, "after_quality_filter": 0,
             "after_lang_filter": 0, "after_dedup": 0, "docs_by_language": {}}
    docs: list[str] = []
    for fp in files:
        raw = Path(fp).read_text(encoding="utf-8", errors="ignore")
        raw = strip_gutenberg_boilerplate(raw)
        for doc in split_documents(raw, min_chars=args.min_doc_chars):
            stats["raw_documents"] += 1
            doc = clean_document(doc)
            if len(doc) < args.min_doc_chars:
                continue
            stats["after_length_filter"] += 1
            if is_low_quality(doc):
                continue
            stats["after_quality_filter"] += 1
            lang = langid.identify(doc)
            stats["docs_by_language"][lang] = stats["docs_by_language"].get(lang, 0) + 1
            if lang == "unknown":
                continue
            stats["after_lang_filter"] += 1
            docs.append(doc)

    # Deduplicate (exact then optionally near-duplicate).
    before = len(docs)
    docs = exact_dedupe(docs)
    after_exact = len(docs)
    if not args.no_near_dedup:
        docs = near_dedupe(docs, threshold=0.8)
    stats["after_dedup"] = len(docs)
    stats["exact_duplicates_removed"] = before - after_exact
    stats["near_duplicates_removed"] = after_exact - stats["after_dedup"]

    # Optional per-language down-sampling toward target ratios (never repeat).
    ratios = _parse_ratios(args.target_ratios)
    lang_of = {id(d): langid.identify(d) for d in docs}
    if ratios:
        rng = np.random.default_rng(args.seed)
        by_lang: dict[str, list[str]] = {}
        for d in docs:
            by_lang.setdefault(lang_of[id(d)], []).append(d)
        total = len(docs)
        kept: list[str] = []
        for lang, group in by_lang.items():
            target = ratios.get(lang)
            if target is None:
                kept.extend(group)
                continue
            cap = int(target * total)
            if len(group) > cap:
                idx = rng.choice(len(group), size=cap, replace=False)
                group = [group[i] for i in sorted(idx)]
            kept.extend(group)
        docs = kept

    if not docs:
        raise SystemExit("No documents survived filtering.")

    # Tokenize (survivors) and measure tokens by language.
    tok = AilaTokenizer.load(args.tokenizer)
    all_ids: list[int] = []
    tokens_by_lang: dict[str, int] = {}
    doc_langs: list[str] = []
    for d in docs:
        ids = tok.encode(d, add_bos=True, add_eos=True)
        all_ids.extend(ids)
        lang = lang_of.get(id(d), langid.identify(d))
        tokens_by_lang[lang] = tokens_by_lang.get(lang, 0) + len(ids)
        doc_langs.append(lang)

    # Deterministic train/val split at the document level (no leakage: a
    # document is entirely in train or entirely in val).
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(docs))
    n_val = max(1, int(args.val_fraction * len(docs)))
    val_idx = set(order[:n_val].tolist())

    train_ids: list[int] = []
    val_ids: list[int] = []
    for i, d in enumerate(docs):
        ids = tok.encode(d, add_bos=True, add_eos=True)
        (val_ids if i in val_idx else train_ids).extend(ids)

    out_dir = Path(args.out_root) / args.version
    out_dir.mkdir(parents=True, exist_ok=True)
    train_bin = str(out_dir / "train.bin")
    val_bin = str(out_dir / "val.bin")
    write_token_bin(train_ids, train_bin)
    write_token_bin(val_ids, val_bin)

    doc_lang_dist: dict[str, int] = {}
    for lang in doc_langs:
        doc_lang_dist[lang] = doc_lang_dist.get(lang, 0) + 1

    manifest = build_manifest(
        version=args.version,
        train_bin=train_bin, val_bin=val_bin,
        tokenizer_path=args.tokenizer, tokenizer_vocab_size=tok.vocab_size,
        sources=[Source(name=args.source_name or "unspecified",
                        license=args.source_license or "UNVERIFIED",
                        url=args.source_url,
                        provenance="Fetched via build_pretrain_corpus.py; verify license before redistribution.",
                        documents=len(docs), tokens=len(all_ids))],
        language_distribution=doc_lang_dist,
        token_distribution=tokens_by_lang,
        filter_config={
            "min_doc_chars": args.min_doc_chars,
            "clean_document": True, "is_low_quality": True,
            "langid_filter": True, "exact_dedupe": True,
            "near_dedupe_threshold": 0.8, "target_ratios": ratios or None,
        },
        document_count=len(docs),
    )
    write_manifest(manifest, root=args.out_root)

    total_tokens = len(all_ids)
    logger.info("=== corpus %s ===", args.version)
    logger.info("raw documents:            %d", stats["raw_documents"])
    logger.info("after length filter:      %d", stats["after_length_filter"])
    logger.info("after quality filter:     %d", stats["after_quality_filter"])
    logger.info("after language filter:    %d", stats["after_lang_filter"])
    logger.info("after dedup:              %d (exact -%d, near -%d)",
                stats["after_dedup"], stats["exact_duplicates_removed"], stats["near_duplicates_removed"])
    logger.info("final documents:          %d", len(docs))
    logger.info("docs by language:         %s", doc_lang_dist)
    logger.info("tokens total:             %d (train %d, val %d)",
                total_tokens, len(train_ids), len(val_ids))
    logger.info("tokens by language:       %s", tokens_by_lang)
    removed_pct = 100.0 * (stats["raw_documents"] - len(docs)) / max(1, stats["raw_documents"])
    logger.info("documents removed:        %.1f%%", removed_pct)
    logger.info("manifest:                 %s", out_dir / "manifest.json")


if __name__ == "__main__":
    main()
