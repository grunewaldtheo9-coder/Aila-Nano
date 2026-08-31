"""Bilingual 50M-v2 pieces: the 16384-vocab model config, bits-per-character
evaluation (the fair cross-tokenizer metric), device selection, and
bilingual corpus mixing."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch

from model.config import GPTConfig
from model.transformer import AilaNanoGPT
from tokenizer import AilaTokenizer
from tokenizer.trainer import train_tokenizer_from_iterator
from training.trainer import TrainingConfig
from evaluation.perplexity import text_perplexity

_ROOT = Path(__file__).resolve().parent.parent


def _load_script(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -- v2 model config ------------------------------------------------------

def test_v2_model_config_is_16384_and_55m():
    cfg = GPTConfig.from_yaml("configs/model/nano_50m_v2_bilingual.yaml")
    assert cfg.vocab_size == 16384
    n = sum(p.numel() for p in AilaNanoGPT(cfg).parameters())
    assert n == 55_587_328  # +4.2M vs the 8192 model, from the tied embedding


def test_v2_model_accepts_full_vocab_ids():
    cfg = GPTConfig.from_yaml("configs/model/nano_50m_v2_bilingual.yaml")
    cfg.max_seq_len = 32
    model = AilaNanoGPT(cfg)
    ids = torch.tensor([[16383, 0, 100, 16000, 42]])  # includes the top id
    with torch.no_grad():
        logits, _ = model(ids)
    assert logits.shape[-1] == 16384


# -- bits per character (fair cross-tokenizer metric) --------------------

def _tiny_model_and_tok(tmp_path):
    prefix = str(tmp_path / "tk")
    corpus = ["the cat sat on the mat", "a dog ran fast", "hello world today"] * 40
    train_tokenizer_from_iterator(iter(corpus), prefix, vocab_size=320,
                                  character_coverage=1.0, hard_vocab_limit=False)
    tok = AilaTokenizer.load(prefix + ".model")
    from model.config import tiny_debug
    cfg = tiny_debug(); cfg.vocab_size = tok.vocab_size
    return AilaNanoGPT(cfg).eval(), tok


def test_bits_per_char_present_and_consistent(tmp_path):
    model, tok = _tiny_model_and_tok(tmp_path)
    r = text_perplexity(model, tok, "the cat sat on the mat\na dog ran fast")
    assert r["bits_per_char"] is not None and r["bits_per_char"] > 0
    # BPC = (total NLL nats / ln2) / chars = per-token loss * tokens/char / ln2
    approx = r["loss"] * r["tokens_per_char"] / math.log(2)
    assert abs(approx - r["bits_per_char"]) < 0.05


def test_bpc_is_deterministic(tmp_path):
    model, tok = _tiny_model_and_tok(tmp_path)
    a = text_perplexity(model, tok, "hello world today\nthe cat sat")
    b = text_perplexity(model, tok, "hello world today\nthe cat sat")
    assert a["bits_per_char"] == b["bits_per_char"]


# -- device selection -----------------------------------------------------

def test_device_auto_resolves_to_available_device():
    resolved = TrainingConfig(device="auto").resolved_device()
    assert resolved in ("cpu", "cuda")
    if not torch.cuda.is_available():
        assert resolved == "cpu"


def test_device_explicit_cpu_is_respected():
    assert TrainingConfig(device="cpu").resolved_device() == "cpu"


# -- bilingual corpus mixing ---------------------------------------------

def test_language_ratio_parsing():
    build = _load_script("datasets/scripts/build_pretrain_corpus.py", "build_pretrain_corpus")
    ratios = build._parse_ratios("pt=0.7,en=0.3")
    assert ratios == {"pt": 0.7, "en": 0.3}
    assert build._parse_ratios(None) == {}


def test_gutenberg_boilerplate_is_stripped():
    build = _load_script("datasets/scripts/build_pretrain_corpus.py", "build_pretrain_corpus")
    text = ("header junk\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n"
            "real content here\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nfooter junk")
    stripped = build.strip_gutenberg_boilerplate(text)
    assert "real content here" in stripped
    assert "header junk" not in stripped and "footer junk" not in stripped


# -- v2 bilingual corpus manifest (when built) ---------------------------

_MANIFEST = Path("datasets/pretrain/aila_pretrain_v2_bilingual/manifest.json")


def test_v2_corpus_manifest_records_both_languages():
    import pytest
    if not _MANIFEST.exists():
        pytest.skip("v2 bilingual corpus not built in this environment")
    import json
    m = json.loads(_MANIFEST.read_text())
    assert m["tokenizer_vocab_size"] == 16384
    assert "pt" in m["language_distribution"] and "en" in m["language_distribution"]
    assert m["token_distribution"].get("pt", 0) > 0
