#!/usr/bin/env python3
"""Validate conversational datasets (the `messages` JSONL schema).

Checks every record against finetuning.chat_format.validate_conversation
(valid alternating roles, non-empty content, at least one assistant turn),
reports malformed lines, and flags exact-duplicate conversations. Exits
non-zero if any file has an error, so it can gate CI / a training run
(spec §27, §29).

Usage:
    python datasets/scripts/validate_conversations.py datasets/conversational/*.jsonl
    python datasets/scripts/validate_conversations.py            # defaults to datasets/conversational/
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from finetuning.chat_format import validate_conversation


def _canonical(conv: dict) -> str:
    return json.dumps(conv.get("messages", []), sort_keys=True, ensure_ascii=False)


def validate_file(path: str) -> tuple[int, int, int, list[str]]:
    """Return (ok, bad, duplicates, error_messages) for one file."""
    ok = bad = duplicates = 0
    errors: list[str] = []
    seen: set[str] = set()
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            bad += 1
            errors.append(f"{path}:{line_no}: malformed JSON ({e})")
            continue
        reason = validate_conversation(record)
        if reason is not None:
            bad += 1
            errors.append(f"{path}:{line_no}: {reason}")
            continue
        key = _canonical(record)
        if key in seen:
            duplicates += 1
            errors.append(f"{path}:{line_no}: duplicate conversation")
            continue
        seen.add(key)
        ok += 1
    return ok, bad, duplicates, errors


def main(argv: list[str]) -> int:
    patterns = argv or ["datasets/conversational/*.jsonl"]
    paths: list[str] = []
    for p in patterns:
        paths.extend(sorted(glob.glob(p)))
    if not paths:
        print("No files matched.")
        return 1

    total_ok = total_bad = total_dupes = 0
    all_errors: list[str] = []
    for path in paths:
        ok, bad, dupes, errors = validate_file(path)
        total_ok += ok
        total_bad += bad
        total_dupes += dupes
        all_errors.extend(errors)
        print(f"{path}: {ok} valid, {bad} invalid, {dupes} duplicate")

    print(f"\nTotal: {total_ok} valid, {total_bad} invalid, {total_dupes} duplicate")
    if all_errors:
        print("\nProblems:")
        for e in all_errors[:50]:
            print(f"  {e}")
    return 0 if (total_bad == 0 and total_dupes == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
