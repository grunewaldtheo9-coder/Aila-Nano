"""Evaluation harness: per-language perplexity, repetition/degradation
metric, deterministic scoring, and checkpoint/tokenizer compatibility."""

from __future__ import annotations

import pytest

from model.config import tiny_debug
from model.transformer import AilaNanoGPT
from tokenizer.trainer import train_tokenizer_from_iterator
from tokenizer import AilaTokenizer

from evaluation.perplexity import (
    evaluate_language_file,
    repetition_score,
    text_perplexity,
)


def _tiny_tok(tmp_path):
    prefix = str(tmp_path / "tk")
    corpus = [
        "the cat sat on the mat", "a dog ran in the park", "hello world",
        "the quick brown fox jumps", "she sells sea shells", "a bird in the hand",
    ] * 40
    # hard_vocab_limit=False: a tiny corpus can't fill a large vocab, and we
    # only need a working tokenizer for the test.
    train_tokenizer_from_iterator(iter(corpus), prefix, vocab_size=320,
                                  character_coverage=1.0, hard_vocab_limit=False)
    return AilaTokenizer.load(prefix + ".model")


def test_text_perplexity_is_finite_and_positive(tmp_path):
    tok = _tiny_tok(tmp_path)
    cfg = tiny_debug()
    cfg.vocab_size = tok.vocab_size
    model = AilaNanoGPT(cfg)
    r = text_perplexity(model, tok, "the cat sat on the mat\na dog ran in the park")
    assert r["tokens"] > 0
    assert r["loss"] is not None and r["loss"] > 0
    assert r["perplexity"] > 1.0


def test_perplexity_is_deterministic(tmp_path):
    tok = _tiny_tok(tmp_path)
    cfg = tiny_debug(); cfg.vocab_size = tok.vocab_size
    model = AilaNanoGPT(cfg).eval()
    a = text_perplexity(model, tok, "hello world\nthe dog ran")
    b = text_perplexity(model, tok, "hello world\nthe dog ran")
    assert a == b


def test_long_line_is_fully_scored_not_truncated(tmp_path):
    # Regression: a line longer than the context window must have EVERY
    # transition scored (chunked), not truncated to seq_cap while still
    # counting the whole line's characters — otherwise BPC is understated.
    tok = _tiny_tok(tmp_path)
    cfg = tiny_debug(); cfg.vocab_size = tok.vocab_size; cfg.max_seq_len = 16
    model = AilaNanoGPT(cfg)
    long_line = " ".join(["the cat sat"] * 30)  # >> 16 tokens
    ids = tok.encode(long_line, add_bos=True, add_eos=True)
    assert len(ids) > cfg.max_seq_len  # precondition: would truncate
    r = text_perplexity(model, tok, long_line)
    assert r["tokens"] == len(ids) - 1  # all transitions scored, none dropped
    assert r["chars"] == len(long_line)


def test_empty_text_reports_zero_tokens(tmp_path):
    tok = _tiny_tok(tmp_path)
    cfg = tiny_debug(); cfg.vocab_size = tok.vocab_size
    model = AilaNanoGPT(cfg)
    r = text_perplexity(model, tok, "\n\n   \n")
    assert r["tokens"] == 0 and r["perplexity"] is None


def test_evaluate_language_file(tmp_path):
    tok = _tiny_tok(tmp_path)
    cfg = tiny_debug(); cfg.vocab_size = tok.vocab_size
    model = AilaNanoGPT(cfg)
    p = tmp_path / "en.txt"; p.write_text("the cat sat\na dog ran\nhello world\n")
    r = evaluate_language_file(model, tok, str(p))
    assert r["tokens"] > 0 and r["path"] == str(p)


# -- repetition / degradation --------------------------------------------

def test_repetition_score_detects_loops():
    looped = repetition_score("aila aila aila aila aila aila aila aila")
    varied = repetition_score("the quick brown fox jumps over the lazy dog today")
    assert looped["repeat_ngram_fraction"] > varied["repeat_ngram_fraction"]
    assert looped["top_token_share"] > 0.5


def test_repetition_short_text_is_safe():
    r = repetition_score("hi")
    assert r["repeat_ngram_fraction"] == 0.0


# -- held-out eval sets exist and are non-trivial ------------------------

def test_committed_eval_sets_present():
    import pathlib
    en = pathlib.Path("evaluation/data/eval_en.txt").read_text(encoding="utf-8")
    pt = pathlib.Path("evaluation/data/eval_pt.txt").read_text(encoding="utf-8")
    assert len([l for l in en.splitlines() if l.strip()]) >= 5
    assert len([l for l in pt.splitlines() if l.strip()]) >= 5
    # Portuguese eval genuinely contains accented characters.
    assert any(c in pt for c in "ãõáéíóúâêôç")
