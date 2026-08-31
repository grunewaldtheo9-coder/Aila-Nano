"""Bilingual tokenizer: candidate training, benchmark metrics, the promoted
v2 artifact, and backward compatibility with the production tokenizer.

These tests train a tiny tokenizer from an in-memory EN+PT corpus (fast, no
network) to exercise the pipeline, and additionally check the committed v2
artifact when present."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tokenizer import AilaTokenizer
from tokenizer.trainer import train_tokenizer_from_iterator

_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load_script("tokenizer_benchmark")

_PT_LINES = [
    "O coração da informação é a educação e a programação.",
    "Você não imagina a inteligência do computador e a conversação.",
    "A manhã trouxe desenvolvimento, aprendizagem e português.",
    # a line carrying every target accent so full-coverage puts them in-vocab
    "ã õ á é í ó ú â ê ô ç: acentuação, coração, avó, pôr, você, três.",
] * 40
_EN_LINES = [
    "The heart of information is education and programming.",
    "You cannot imagine the intelligence of the computer.",
    "The morning brought development, learning and progress.",
] * 40


def _train_tiny(tmp_path, vocab_size, coverage):
    prefix = str(tmp_path / f"tk_{vocab_size}")
    train_tokenizer_from_iterator(
        iter(_PT_LINES + _EN_LINES), prefix, vocab_size=vocab_size,
        character_coverage=coverage, hard_vocab_limit=False,
    )
    return AilaTokenizer.load(prefix + ".model")


def test_candidate_trains_and_has_requested_vocab(tmp_path):
    tok = _train_tiny(tmp_path, 512, 1.0)
    assert tok.vocab_size == 512


def test_full_coverage_puts_accents_in_vocab(tmp_path):
    tok = _train_tiny(tmp_path, 512, 1.0)
    acc = bench._accent_handling(tok)
    # With PT in the corpus and coverage 1.0, accented chars are real pieces.
    assert acc["in_vocab_fraction"] >= 0.8, acc["via_byte_fallback"]


def test_bilingual_reduces_pt_fragmentation_vs_english_only(tmp_path):
    pt_only_en_corpus = AilaTokenizer  # placeholder to keep names clear
    bilingual = _train_tiny(tmp_path, 512, 1.0)
    # An English-only tokenizer of the same size fragments PT much more.
    en_prefix = str(tmp_path / "en_only")
    train_tokenizer_from_iterator(iter(_EN_LINES), en_prefix, vocab_size=512,
                                  character_coverage=0.9995, hard_vocab_limit=False)
    en_only = AilaTokenizer.load(en_prefix + ".model")
    pt_bi = bench._pt_word_fragmentation(bilingual)["avg_pieces_per_word"]
    pt_en = bench._pt_word_fragmentation(en_only)["avg_pieces_per_word"]
    assert pt_bi < pt_en, (pt_bi, pt_en)


def test_benchmark_text_efficiency_shape(tmp_path):
    tok = _train_tiny(tmp_path, 512, 1.0)
    eff = bench._text_efficiency(tok, "\n".join(_PT_LINES[:5]))
    for k in ("tokens_per_word", "tokens_per_char", "tokens_per_sentence",
              "byte_fallback_rate", "distinct_ids_used"):
        assert k in eff


def test_recommend_prefers_lower_pt_tokens_per_word():
    results = [
        {"model_path": "a", "english": {"tokens_per_word": 1.2},
         "portuguese": {"tokens_per_word": 2.8, "byte_fallback_rate": 0.1}},
        {"model_path": "b", "english": {"tokens_per_word": 1.2},
         "portuguese": {"tokens_per_word": 1.4, "byte_fallback_rate": 0.0}},
    ]
    assert bench.recommend(results)["model_path"] == "b"


# -- production tokenizer unchanged + v2 artifact -------------------------

def test_production_tokenizer_still_8192():
    tok = AilaTokenizer.load("tokenizer/artifacts/aila_nano.model")
    assert tok.vocab_size == 8192


V2 = Path("tokenizer/artifacts/v2_bilingual/aila_nano_v2_bilingual.model")


@pytest.mark.skipif(not V2.exists(), reason="v2 tokenizer artifact not present")
def test_v2_tokenizer_is_16384_and_handles_accents():
    tok = AilaTokenizer.load(str(V2))
    assert tok.vocab_size == 16384
    acc = bench._accent_handling(tok)
    assert acc["in_vocab_fraction"] == 1.0
    # PT word fragmentation far below the production tokenizer. On the hard
    # morphological word-list production averages ~4 pieces/word; v2 ~2.0.
    prod = AilaTokenizer.load("tokenizer/artifacts/aila_nano.model")
    v2_pieces = bench._pt_word_fragmentation(tok)["avg_pieces_per_word"]
    prod_pieces = bench._pt_word_fragmentation(prod)["avg_pieces_per_word"]
    assert v2_pieces < prod_pieces
    assert v2_pieces <= 2.5


@pytest.mark.skipif(not V2.exists(), reason="v2 tokenizer artifact not present")
def test_v2_roundtrips_portuguese_text():
    tok = AilaTokenizer.load(str(V2))
    s = "Coração, informação e educação em português."
    assert "ç" in tok.decode(tok.encode(s, add_bos=False, add_eos=False))
