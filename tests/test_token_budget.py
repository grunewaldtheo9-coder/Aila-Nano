"""Token/epoch budget enforcement and reproducible dataset identity in
checkpoints — the infrastructure the data-scaling experiments depend on."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from model.transformer import AilaNanoGPT
from training.checkpoint import load_checkpoint
from training.dataset import corpus_fingerprint, write_token_bin
from training.trainer import Trainer, TrainingConfig, resolve_max_steps


# -- resolve_max_steps (pure) --------------------------------------------

def test_no_budget_uses_max_steps():
    assert resolve_max_steps(500, tokens_per_step=1000, corpus_tokens=100_000) == 500


def test_token_budget_caps_below_max_steps():
    # 10,000 tokens / 1,000 tokens-per-step = 10 steps, below the 500 cap.
    assert resolve_max_steps(500, 1000, 100_000, max_tokens=10_000) == 10


def test_token_budget_rounds_up_a_partial_step():
    assert resolve_max_steps(500, 1000, 100_000, max_tokens=10_500) == 11


def test_epoch_budget_caps():
    # 2 epochs * 5,000 corpus tokens = 10,000 tokens / 1,000 = 10 steps.
    assert resolve_max_steps(500, 1000, 5_000, max_epochs=2.0) == 10


def test_smallest_budget_wins():
    # max_steps=500, tokens->10 steps, epochs->4 steps: the epoch budget wins.
    assert resolve_max_steps(500, 1000, 2_000, max_tokens=10_000, max_epochs=2.0) == 4


def test_budget_never_below_one_step():
    assert resolve_max_steps(500, 1000, 100, max_tokens=1) == 1


# -- corpus fingerprint ---------------------------------------------------

def test_corpus_fingerprint_is_deterministic_and_counts_tokens(tmp_path):
    ids = list(range(5000))
    p = str(tmp_path / "a.bin")
    write_token_bin(ids, p)
    fp1 = corpus_fingerprint(p)
    fp2 = corpus_fingerprint(p)
    assert fp1["sha256"] == fp2["sha256"]
    assert fp1["num_tokens"] == 5000


def test_corpus_fingerprint_changes_with_content(tmp_path):
    a = str(tmp_path / "a.bin"); write_token_bin(list(range(5000)), a)
    b = str(tmp_path / "b.bin"); write_token_bin(list(range(1, 5001)), b)
    assert corpus_fingerprint(a)["sha256"] != corpus_fingerprint(b)["sha256"]


# -- trainer honours the budget + records provenance ----------------------

def _tiny_bins(tmp_path, cfg_vocab):
    ids = np.random.randint(0, cfg_vocab, size=4000).tolist()
    tb, vb = tmp_path / "t.bin", tmp_path / "v.bin"
    write_token_bin(ids[:3600], str(tb))
    write_token_bin(ids[3600:], str(vb))
    return str(tb), str(vb)


def test_trainer_stops_at_token_budget(tmp_path, tiny_config):
    tb, vb = _tiny_bins(tmp_path, tiny_config.vocab_size)
    seq = tiny_config.max_seq_len
    tokens_per_step = 4 * 1 * seq  # batch_size * grad_accum * seq_len
    budget = 6 * tokens_per_step + 1  # -> ceil = 7 steps, well below max_steps
    cfg = TrainingConfig(
        train_bin=tb, val_bin=vb, max_lr=1e-3, min_lr=1e-4, warmup_steps=1,
        max_steps=100, max_tokens=budget, batch_size=4, eval_interval=50,
        eval_iters=1, log_interval=50, checkpoint_dir=str(tmp_path / "c"),
        early_stopping_patience=None, device="cpu", amp=False,
        tensorboard_dir=str(tmp_path / "tb"), num_workers=0,
        dataset_version="test_v1",
    )
    trainer = Trainer(AilaNanoGPT(tiny_config), cfg)
    assert trainer.max_steps == math.ceil(budget / tokens_per_step)  # 7, not 100
    state = trainer.train()
    assert state.step == trainer.max_steps


def test_checkpoint_records_dataset_metadata(tmp_path, tiny_config):
    tb, vb = _tiny_bins(tmp_path, tiny_config.vocab_size)
    cfg = TrainingConfig(
        train_bin=tb, val_bin=vb, max_lr=1e-3, min_lr=1e-4, warmup_steps=1,
        max_steps=3, batch_size=4, eval_interval=3, eval_iters=1, log_interval=3,
        checkpoint_dir=str(tmp_path / "c"), early_stopping_patience=None,
        device="cpu", amp=False, tensorboard_dir=str(tmp_path / "tb"),
        num_workers=0, dataset_version="aila_pretrain_test",
    )
    trainer = Trainer(AilaNanoGPT(tiny_config), cfg)
    trainer.train()
    ckpt = load_checkpoint(str(Path(cfg.checkpoint_dir) / "best.pt"))
    extra = ckpt["extra"]
    assert extra["dataset_version"] == "aila_pretrain_test"
    assert extra["train_sha256"] == corpus_fingerprint(tb)["sha256"]
    assert extra["tokens_seen"] == ckpt["step"] * extra["tokens_per_step"]
    assert extra["train_tokens"] == corpus_fingerprint(tb)["num_tokens"]
