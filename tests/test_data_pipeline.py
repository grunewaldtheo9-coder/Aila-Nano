"""Language identification, dataset manifest/versioning, and train/val
leakage checks for the pretraining data pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPTS = str(Path(__file__).resolve().parent.parent / "datasets" / "scripts")
sys.path.insert(0, _SCRIPTS)

import langid  # noqa: E402
from dataset_manifest import Source, build_manifest, load_manifest, write_manifest  # noqa: E402

from training.dataset import corpus_fingerprint, write_token_bin  # noqa: E402


# -- language identification ----------------------------------------------

def test_identifies_english():
    assert langid.identify("The cat sat on the mat and they were very happy about it.") == "en"


def test_identifies_portuguese_by_words_and_diacritics():
    assert langid.identify("Você não sabe o que aconteceu porque não estava com a gente.") == "pt"


def test_short_or_foreign_text_is_unknown():
    assert langid.identify("") == "unknown"
    assert langid.identify("123 456 789") == "unknown"


def test_portuguese_diacritics_signal():
    # Accented characters push the PT score up.
    assert langid.language_scores("coração informação você não")["pt"] > 0


def test_corpus_distribution_counts_documents():
    docs = [
        "The dog ran and the children laughed with them.",
        "Bom dia, você não vai acreditar no que aconteceu porque foi incrível.",
        "xyzzy 42",
    ]
    dist = langid.corpus_language_distribution(docs)
    assert dist["en"] == 1 and dist["pt"] == 1 and dist["unknown"] == 1


# -- manifest / versioning ------------------------------------------------

def test_manifest_records_hash_tokens_and_sources(tmp_path):
    tb = str(tmp_path / "t.bin"); write_token_bin(list(range(3000)), tb)
    vb = str(tmp_path / "v.bin"); write_token_bin(list(range(300)), vb)
    m = build_manifest(
        version="unit_v1", train_bin=tb, val_bin=vb,
        tokenizer_path="tok.model", tokenizer_vocab_size=8192,
        sources=[Source(name="TinyStories", license="CDLA-Sharing-1.0")],
        language_distribution={"en": 10},
    )
    assert m.train_sha256 == corpus_fingerprint(tb)["sha256"]
    assert m.train_tokens == 3000
    assert m.val_tokens == 300
    assert m.git_commit is None or isinstance(m.git_commit, str)
    assert m.sources[0]["license"] == "CDLA-Sharing-1.0"


def test_manifest_roundtrip_to_disk(tmp_path):
    tb = str(tmp_path / "t.bin"); write_token_bin(list(range(1000)), tb)
    m = build_manifest(version="unit_v2", train_bin=tb,
                       tokenizer_path="tok.model", tokenizer_vocab_size=8192)
    write_manifest(m, root=str(tmp_path / "pre"))
    loaded = load_manifest("unit_v2", root=str(tmp_path / "pre"))
    assert loaded["version"] == "unit_v2"
    assert loaded["train_tokens"] == 1000


# -- train/val leakage (verifiable via hashes) ----------------------------

def _window_hashes(bin_path, seq_len=16):
    import hashlib
    mm = np.memmap(bin_path, dtype=np.uint16, mode="r")
    out = set()
    for i in range(0, len(mm) - seq_len, seq_len):
        out.add(hashlib.sha1(mm[i : i + seq_len].tobytes()).hexdigest())
    return out


def test_no_train_val_leakage_when_split_is_disjoint(tmp_path):
    ids = list(range(4000))
    tb = str(tmp_path / "t.bin"); write_token_bin(ids[:3600], tb)
    vb = str(tmp_path / "v.bin"); write_token_bin(ids[3600:], vb)
    # A clean split shares no identical windows.
    assert _window_hashes(tb).isdisjoint(_window_hashes(vb))


def test_leakage_is_detected_when_val_overlaps_train(tmp_path):
    ids = list(range(4000))
    tb = str(tmp_path / "t.bin"); write_token_bin(ids, tb)
    vb = str(tmp_path / "v.bin"); write_token_bin(ids[:1000], vb)  # subset of train
    # Overlapping content -> shared window hashes -> leakage caught.
    assert not _window_hashes(tb).isdisjoint(_window_hashes(vb))
