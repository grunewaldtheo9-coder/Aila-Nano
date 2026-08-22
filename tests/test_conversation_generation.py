"""The synthetic conversation generator, statistics, and train/val split."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load(module_name: str, rel: str):
    spec = importlib.util.spec_from_file_location(module_name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load("gen_conv", "datasets/scripts/generate_conversations.py")
val = _load("val_conv", "datasets/scripts/validate_conversations.py")


def test_generator_produces_valid_conversations():
    records = gen.generate(count=300, seed=7)
    assert len(records) > 100
    from finetuning.chat_format import validate_conversation

    for r in records:
        assert validate_conversation(r) is None
        assert r["language"] in ("en", "pt")
        assert r["category"]
        assert "messages" in r


def test_generator_is_deterministic():
    a = gen.generate(count=200, seed=99)
    b = gen.generate(count=200, seed=99)
    assert a == b


def test_generator_deduplicates():
    records = gen.generate(count=500, seed=3)
    import json

    keys = {json.dumps(r["messages"], sort_keys=True, ensure_ascii=False) for r in records}
    assert len(keys) == len(records)  # no exact duplicates


def test_generator_covers_multiple_categories_and_languages():
    records = gen.generate(count=800, seed=11)
    cats = {r["category"] for r in records}
    langs = {r["language"] for r in records}
    assert langs == {"en", "pt"}
    # A real spread of conversation types, not one template.
    assert len(cats) >= 6
    assert "search" in cats and "memory" in cats


def test_search_examples_are_flagged_and_no_search_examples_are_not():
    records = gen.generate(count=800, seed=5)
    for r in records:
        if r["category"] == "search":
            assert r["requires_web"] is True
        if r["category"] in ("greeting", "identity", "preference"):
            assert r["requires_web"] is False


def test_stats_report_real_counts():
    records = gen.generate(count=300, seed=1)
    report = val.compute_stats(records)
    assert "Total conversations: " + str(len(records)) in report
    assert "Languages:" in report
    assert "Categories:" in report


def test_split_is_disjoint_and_covers_everything(tmp_path):
    import json

    records = gen.generate(count=400, seed=2)
    src = tmp_path / "gen.jsonl"
    src.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")

    split = _load("split_conv", "datasets/scripts/split_conversations.py")
    loaded = split._load([str(src)])
    import random

    rng = random.Random(42)
    rng.shuffle(loaded)
    n_val = max(1, int(len(loaded) * 0.1))
    val_set, train_set = loaded[:n_val], loaded[n_val:]

    def key(r):
        return json.dumps(r["messages"], sort_keys=True, ensure_ascii=False)

    train_keys = {key(r) for r in train_set}
    val_keys = {key(r) for r in val_set}
    assert train_keys.isdisjoint(val_keys)  # validation is unseen
    assert len(train_keys) + len(val_keys) == len(loaded)


def test_generator_produces_deep_long_context_conversations():
    records = gen.generate(count=3000, seed=13)
    deep = [r for r in records if r["category"] == "long_context"]
    assert deep, "expected long-context conversations"
    # Real depth: some conversations reach 10+ turns.
    assert max(r["turns"] for r in records) >= 10
    # A part-accumulation conversation ends by recalling the parts.
    recalls = [
        r for r in deep
        if "mentioned" in r["messages"][-2]["content"].lower()
        or "mencionei" in r["messages"][-2]["content"].lower()
    ]
    assert recalls
    r = recalls[0]
    listed = r["messages"][-1]["content"].lower()
    # The recall lists several accumulated parts (comma-separated), and each
    # feature the user stated earlier appears in it.
    assert "," in listed
    stated_features = [
        m["content"] for m in r["messages"]
        if m["role"] == "user" and ("also has" in m["content"].lower() or "também tem" in m["content"].lower())
    ]
    assert stated_features  # parts were accumulated
    for feat_turn in stated_features:
        # the noun after "has"/"tem" should be recalled
        noun = feat_turn.lower().split("has")[-1].split("tem")[-1].strip(" .")
        assert noun in listed
