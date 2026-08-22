#!/usr/bin/env python3
"""Split conversational data into train/validation sets.

Reads one or more JSONL files, keeps only valid, de-duplicated
conversations, shuffles deterministically, and writes `train.jsonl` and
`validation.jsonl`. Validation must be *unseen* conversations (spec §53),
so the split is on whole conversations and duplicates are removed first.

Usage:
    python datasets/scripts/split_conversations.py \
        datasets/conversational/generated/generated.jsonl \
        --out-dir datasets/conversational/split --val-fraction 0.1
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from finetuning.chat_format import validate_conversation


def _load(paths: list[str]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if validate_conversation(record) is not None:
                continue
            key = json.dumps(record.get("messages", []), sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            out.append(record)
    return out


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="JSONL files or globs")
    p.add_argument("--out-dir", default="datasets/conversational/split")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    paths: list[str] = []
    for pattern in args.inputs:
        paths.extend(sorted(glob.glob(pattern, recursive=True)))
    records = _load(paths)
    if not records:
        print("No valid conversations found.")
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * args.val_fraction))
    val, train = records[:n_val], records[n_val:]

    out_dir = Path(args.out_dir)
    _write(out_dir / "train.jsonl", train)
    _write(out_dir / "validation.jsonl", val)
    print(f"Wrote {len(train)} train + {len(val)} validation conversations to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
