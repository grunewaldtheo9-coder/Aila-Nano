from pathlib import Path

import numpy as np
import torch

from model.transformer import AilaNanoGPT
from training.checkpoint import load_checkpoint, restore_training_state, save_checkpoint
from training.dataset import TokenBinDataset, write_token_bin
from training.scheduler import CosineWarmupScheduler
from training.trainer import Trainer, TrainingConfig


def test_cosine_scheduler_warmup_then_decay():
    sched = CosineWarmupScheduler(max_lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=100)
    assert sched.get_lr(0) < sched.get_lr(9)  # ramps up during warmup
    assert abs(sched.get_lr(10) - 1e-3) < 1e-9  # peak right after warmup
    assert sched.get_lr(100) == 1e-4  # floors at min_lr past max_steps
    assert sched.get_lr(55) < sched.get_lr(10)  # decays in between


def test_token_bin_roundtrip(tmp_path):
    ids = list(range(1000))
    path = str(tmp_path / "tokens.bin")
    write_token_bin(ids, path)

    ds = TokenBinDataset(path, seq_len=16, samples_per_epoch=5)
    assert len(ds) == 5
    x, y = ds[0]
    assert x.shape == (16,)
    assert y.shape == (16,)
    # y is x shifted by one position, by construction
    assert torch.equal(y[:-1], x[1:])


def test_checkpoint_save_and_resume_restores_state(tmp_path, tiny_config):
    model = AilaNanoGPT(tiny_config)
    optimizer = model.configure_optimizer(weight_decay=0.01, learning_rate=1e-3, betas=(0.9, 0.95))
    path = str(tmp_path / "ckpt.pt")

    save_checkpoint(path, model, optimizer, step=42, best_val_loss=1.234)
    ckpt = load_checkpoint(path)
    assert ckpt["step"] == 42
    assert ckpt["best_val_loss"] == 1.234
    assert ckpt["config"]["vocab_size"] == tiny_config.vocab_size

    model2 = AilaNanoGPT(tiny_config)
    optimizer2 = model2.configure_optimizer(weight_decay=0.01, learning_rate=1e-3, betas=(0.9, 0.95))
    step, best_val_loss = restore_training_state(path, model2, optimizer2)
    assert step == 42
    assert best_val_loss == 1.234
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


def test_trainer_runs_and_checkpoints(tmp_path, tiny_config):
    ids = np.random.randint(0, tiny_config.vocab_size, size=2000).tolist()
    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    write_token_bin(ids[:1800], str(train_bin))
    write_token_bin(ids[1800:], str(val_bin))

    model = AilaNanoGPT(tiny_config)
    cfg = TrainingConfig(
        train_bin=str(train_bin),
        val_bin=str(val_bin),
        max_lr=1e-3,
        min_lr=1e-4,
        warmup_steps=2,
        max_steps=10,
        batch_size=4,
        eval_interval=5,
        eval_iters=2,
        log_interval=5,
        checkpoint_dir=str(tmp_path / "ckpts"),
        early_stopping_patience=None,
        device="cpu",
        amp=False,
        tensorboard_dir=str(tmp_path / "tb"),
        num_workers=0,
    )
    trainer = Trainer(model, cfg)
    state = trainer.train()

    assert state.step == 10
    assert (Path(cfg.checkpoint_dir) / "step_0000010.pt").exists()
    assert (Path(cfg.checkpoint_dir) / "best.pt").exists()


def test_trainer_prunes_to_exactly_keep_last_n_checkpoints(tmp_path, tiny_config):
    # Regression test: `excess = len(ckpts) - keep_last_n_checkpoints` must
    # be clamped to >= 0 before slicing `ckpts[:excess]` — a negative
    # excess is not a no-op slice in Python (it counts back from the end),
    # so this used to silently cap every run at 1 retained rolling
    # checkpoint no matter what keep_last_n_checkpoints was set to.
    ids = np.random.randint(0, tiny_config.vocab_size, size=2000).tolist()
    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    write_token_bin(ids[:1800], str(train_bin))
    write_token_bin(ids[1800:], str(val_bin))

    model = AilaNanoGPT(tiny_config)
    cfg = TrainingConfig(
        train_bin=str(train_bin),
        val_bin=str(val_bin),
        max_lr=1e-3,
        min_lr=1e-4,
        warmup_steps=2,
        max_steps=50,
        batch_size=4,
        eval_interval=5,
        eval_iters=2,
        log_interval=5,
        checkpoint_dir=str(tmp_path / "ckpts"),
        keep_last_n_checkpoints=3,
        early_stopping_patience=None,
        device="cpu",
        amp=False,
        tensorboard_dir=str(tmp_path / "tb"),
        num_workers=0,
    )
    trainer = Trainer(model, cfg)
    trainer.train()

    rolling_checkpoints = sorted(Path(cfg.checkpoint_dir).glob("step_*.pt"))
    assert len(rolling_checkpoints) == 3
    assert [p.name for p in rolling_checkpoints] == [
        "step_0000040.pt",
        "step_0000045.pt",
        "step_0000050.pt",
    ]


def test_trainer_resume_continues_from_saved_step(tmp_path, tiny_config):
    ids = np.random.randint(0, tiny_config.vocab_size, size=2000).tolist()
    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    write_token_bin(ids[:1800], str(train_bin))
    write_token_bin(ids[1800:], str(val_bin))

    base_cfg = dict(
        train_bin=str(train_bin),
        val_bin=str(val_bin),
        max_lr=1e-3,
        min_lr=1e-4,
        warmup_steps=2,
        batch_size=4,
        eval_interval=5,
        eval_iters=2,
        log_interval=5,
        checkpoint_dir=str(tmp_path / "ckpts"),
        early_stopping_patience=None,
        device="cpu",
        amp=False,
        num_workers=0,
    )

    model1 = AilaNanoGPT(tiny_config)
    trainer1 = Trainer(model1, TrainingConfig(max_steps=5, tensorboard_dir=str(tmp_path / "tb1"), **base_cfg))
    trainer1.train()

    model2 = AilaNanoGPT(tiny_config)
    trainer2 = Trainer(model2, TrainingConfig(max_steps=10, tensorboard_dir=str(tmp_path / "tb2"), **base_cfg))
    trainer2.resume()
    assert trainer2.state.step == 5
    state = trainer2.train()
    assert state.step == 10


def test_trainer_stops_safely_on_non_finite_loss(tmp_path, tiny_config):
    # Spec §11: a NaN/Inf loss must stop training before it poisons the
    # checkpoint — the loop breaks, the step counter does not advance past
    # the bad step, and no "best.pt" is written for the non-finite state.
    ids = np.random.randint(0, tiny_config.vocab_size, size=2000).tolist()
    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    write_token_bin(ids[:1800], str(train_bin))
    write_token_bin(ids[1800:], str(val_bin))

    model = AilaNanoGPT(tiny_config)
    cfg = TrainingConfig(
        train_bin=str(train_bin), val_bin=str(val_bin),
        max_lr=1e-3, min_lr=1e-4, warmup_steps=2, max_steps=10, batch_size=4,
        eval_interval=5, eval_iters=2, log_interval=5,
        checkpoint_dir=str(tmp_path / "ckpts"), early_stopping_patience=None,
        device="cpu", amp=False, tensorboard_dir=str(tmp_path / "tb"), num_workers=0,
    )
    trainer = Trainer(model, cfg)

    # Force a non-finite loss that is still connected to the graph so
    # backward() runs and produces non-finite grads.
    def nan_loss(x, y):
        return next(trainer.model.parameters()).sum() * float("nan")

    trainer._forward_loss = nan_loss
    before = [p.detach().clone() for p in trainer.model.parameters()]
    state = trainer.train()

    assert state.step == 0  # broke on the first (bad) step, never advanced
    assert not (Path(cfg.checkpoint_dir) / "best.pt").exists()
    # The non-finite gradient must NOT have been applied to the weights: the
    # guard runs before the optimizer step, so params are byte-for-byte the
    # same (no NaNs poisoning the model in memory either).
    for p_before, p_after in zip(before, trainer.model.parameters()):
        assert torch.equal(p_before, p_after)
