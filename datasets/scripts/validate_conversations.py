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


def _assistant_turns(conv: dict) -> int:
    return sum(1 for m in conv.get("messages", []) if m.get("role") == "assistant")


def _user_turns(conv: dict) -> int:
    return sum(1 for m in conv.get("messages", []) if m.get("role") == "user")


def compute_stats(records: list[dict]) -> str:
    """Human-readable statistics over valid records — real counts only."""
    if not records:
        return "No conversations."
    langs: dict[str, int] = {}
    cats: dict[str, int] = {}
    turn_buckets = {"2-turn": 0, "3-5 turn": 0, "6-10 turn": 0, "10+ turn": 0}
    req_mem = req_web = 0
    total_msgs = 0
    total_turns = 0
    turn_counts: list[int] = []
    for r in records:
        lang = r.get("language", "?")
        langs[lang] = langs.get(lang, 0) + 1
        cat = r.get("category", "?")
        cats[cat] = cats.get(cat, 0) + 1
        req_mem += 1 if r.get("requires_memory") else 0
        req_web += 1 if r.get("requires_web") else 0
        total_msgs += len(r.get("messages", []))
        # A "turn" = a user+assistant exchange; count user turns as turns.
        t = _user_turns(r) + _assistant_turns(r)
        turn_counts.append(t)
        total_turns += t
        if t <= 2:
            turn_buckets["2-turn"] += 1
        elif t <= 5:
            turn_buckets["3-5 turn"] += 1
        elif t <= 10:
            turn_buckets["6-10 turn"] += 1
        else:
            turn_buckets["10+ turn"] += 1
    turn_counts.sort()
    median = turn_counts[len(turn_counts) // 2]
    lines = ["Dataset Statistics", "------------------", f"Total conversations: {len(records)}", ""]
    lines += [f"Total messages: {total_msgs}", f"Average turns: {total_turns / len(records):.1f}", f"Median turns: {median}", ""]
    lines.append("Languages:")
    for k, v in sorted(langs.items()):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Turn depth:")
    for k, v in turn_buckets.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Categories:")
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(f"Requires memory: {req_mem}")
    lines.append(f"Requires web:    {req_web}")
    return "\n".join(lines)


def _load_valid_records(path: str) -> list[dict]:
    out: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if validate_conversation(record) is None:
            out.append(record)
    return out


def main(argv: list[str]) -> int:
    show_stats = "--stats" in argv
    argv = [a for a in argv if a != "--stats"]
    patterns = argv or ["datasets/conversational/**/*.jsonl"]
    paths: list[str] = []
    for p in patterns:
        paths.extend(sorted(glob.glob(p, recursive=True)))
    if not paths:
        print("No files matched.")
        return 1

    total_ok = total_bad = total_dupes = 0
    all_errors: list[str] = []
    all_records: list[dict] = []
    for path in paths:
        ok, bad, dupes, errors = validate_file(path)
        total_ok += ok
        total_bad += bad
        total_dupes += dupes
        all_errors.extend(errors)
        all_records.extend(_load_valid_records(path))
        print(f"{path}: {ok} valid, {bad} invalid, {dupes} duplicate")

    print(f"\nTotal: {total_ok} valid, {total_bad} invalid, {total_dupes} duplicate")
    if show_stats:
        print()
        print(compute_stats(all_records))
    if all_errors:
        print("\nProblems:")
        for e in all_errors[:50]:
            print(f"  {e}")
    return 0 if (total_bad == 0 and total_dupes == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
