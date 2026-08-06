"""The Aila Nano pretraining loop.

Supports CPU, single CUDA GPU today, and is structured so multi-GPU
(DistributedDataParallel) can be added later by wrapping `self.model` and
guarding the logging/checkpointing calls with a rank==0 check — the loop
itself doesn't need to change.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.transformer import AilaNanoGPT
from training.checkpoint import restore_training_state, save_checkpoint
from training.dataset import TokenBinDataset
from training.scheduler import CosineWarmupScheduler

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    # data
    train_bin: str = "datasets/processed/pretrain_train.bin"
    val_bin: str = "datasets/processed/pretrain_val.bin"

    # optimization
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    max_steps: int = 5000
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    batch_size: int = 32

    # evaluation / checkpointing
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 20
    checkpoint_dir: str = "checkpoints/pretrain"
    keep_last_n_checkpoints: int = 3
    early_stopping_patience: int | None = 10  # in "eval_interval" units; None disables

    # misc
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    amp: bool = True
    seed: int = 1337
    tensorboard_dir: str = "runs/pretrain"
    num_workers: int = 2

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainState:
    step: int = 0
    best_val_loss: float = field(default=float("inf"))
    steps_since_improvement: int = 0


class Trainer:
    def __init__(self, model: AilaNanoGPT, cfg: TrainingConfig):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.resolved_device())
        self.model.to(self.device)

        torch.manual_seed(cfg.seed)

        self.optimizer = model.configure_optimizer(
            weight_decay=cfg.weight_decay, learning_rate=cfg.max_lr, betas=cfg.betas
        )
        self.scheduler = CosineWarmupScheduler(
            max_lr=cfg.max_lr,
            min_lr=cfg.min_lr,
            warmup_steps=cfg.warmup_steps,
            max_steps=cfg.max_steps,
        )

        self.train_ds = TokenBinDataset(cfg.train_bin, seq_len=model.cfg.max_seq_len)
        self.val_ds = TokenBinDataset(cfg.val_bin, seq_len=model.cfg.max_seq_len)
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
        )

        self.amp_enabled = cfg.amp and self.device.type in ("cuda", "cpu")
        self.amp_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16
        self.scaler = torch.amp.GradScaler(
            enabled=self.amp_enabled and self.device.type == "cuda"
        )

        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=cfg.tensorboard_dir)
        self.state = TrainState()

    # -- checkpoint / resume ------------------------------------------------

    def resume(self, path: str | None = None) -> None:
        path = path or self._latest_checkpoint_path()
        if path is None:
            logger.info("No checkpoint found to resume from; starting fresh.")
            return
        step, best_val_loss = restore_training_state(
            path, self.model, self.optimizer, map_location=str(self.device)
        )
        self.state.step = step
        self.state.best_val_loss = best_val_loss
        logger.info("Resumed from %s at step %d (best_val_loss=%.4f)", path, step, best_val_loss)

    def _latest_checkpoint_path(self) -> str | None:
        ckpts = sorted(Path(self.cfg.checkpoint_dir).glob("step_*.pt"))
        return str(ckpts[-1]) if ckpts else None

    def _save(self, tag: str | None = None) -> None:
        name = tag or f"step_{self.state.step:07d}"
        path = str(Path(self.cfg.checkpoint_dir) / f"{name}.pt")
        save_checkpoint(
            path, self.model, self.optimizer, self.state.step, self.state.best_val_loss
        )
        self._prune_checkpoints()

    def _prune_checkpoints(self) -> None:
        ckpts = sorted(Path(self.cfg.checkpoint_dir).glob("step_*.pt"))
        # `excess` must be clamped to >= 0: a negative excess passed to
        # `ckpts[:excess]` is not a no-op slice — Python treats a negative
        # stop as "count back from the end", so e.g. 2 checkpoints with
        # keep_last_n_checkpoints=3 gives excess=-1 and `ckpts[:-1]`
        # deletes the oldest one anyway, even though we're under the
        # limit. Unclamped, this silently capped every run at 1 retained
        # rolling checkpoint regardless of keep_last_n_checkpoints.
        excess = max(0, len(ckpts) - self.cfg.keep_last_n_checkpoints)
        for p in ckpts[:excess]:
            p.unlink(missing_ok=True)

    # -- core loop ------------------------------------------------------

    def _forward_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled
        ):
            _, loss = self.model(x, targets=y)
        return loss

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        losses = []
        it = iter(self.val_loader)
        for _ in range(min(self.cfg.eval_iters, len(self.val_loader))):
            try:
                x, y = next(it)
            except StopIteration:
                break
            x, y = x.to(self.device), y.to(self.device)
            loss = self._forward_loss(x, y)
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / max(1, len(losses))

    def train(self) -> TrainState:
        self.model.train()
        train_iter = iter(self.train_loader)
        t0 = time.time()
        running_loss = 0.0

        while self.state.step < self.cfg.max_steps:
            lr = self.scheduler.set_lr(self.optimizer, self.state.step)
            self.optimizer.zero_grad(set_to_none=True)

            step_loss = 0.0
            for _ in range(self.cfg.grad_accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)
                x, y = x.to(self.device), y.to(self.device)

                loss = self._forward_loss(x, y) / self.cfg.grad_accum_steps
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                step_loss += loss.item()

            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            running_loss += step_loss
            self.state.step += 1

            if self.state.step % self.cfg.log_interval == 0:
                elapsed = time.time() - t0
                avg_loss = running_loss / self.cfg.log_interval
                tok_per_sec = (
                    self.cfg.batch_size
                    * self.cfg.grad_accum_steps
                    * self.model.cfg.max_seq_len
                    * self.cfg.log_interval
                    / max(elapsed, 1e-9)
                )
                logger.info(
                    "step %d/%d | loss %.4f | lr %.2e | grad_norm %.3f | %.0f tok/s",
                    self.state.step,
                    self.cfg.max_steps,
                    avg_loss,
                    lr,
                    grad_norm,
                    tok_per_sec,
                )
                self.writer.add_scalar("train/loss", avg_loss, self.state.step)
                self.writer.add_scalar("train/lr", lr, self.state.step)
                self.writer.add_scalar("train/grad_norm", grad_norm, self.state.step)
                self.writer.add_scalar("train/tokens_per_sec", tok_per_sec, self.state.step)
                running_loss = 0.0
                t0 = time.time()

            if self.state.step % self.cfg.eval_interval == 0 or self.state.step == self.cfg.max_steps:
                val_loss = self.evaluate()
                perplexity = math.exp(min(val_loss, 20))
                logger.info(
                    "  [eval] step %d | val_loss %.4f | perplexity %.2f",
                    self.state.step,
                    val_loss,
                    perplexity,
                )
                self.writer.add_scalar("val/loss", val_loss, self.state.step)
                self.writer.add_scalar("val/perplexity", perplexity, self.state.step)

                improved = val_loss < self.state.best_val_loss
                if improved:
                    self.state.best_val_loss = val_loss
                    self.state.steps_since_improvement = 0
                    self._save(tag="best")
                else:
                    self.state.steps_since_improvement += 1

                self._save()  # rolling step_XXXXXXX.pt for resume

                if (
                    self.cfg.early_stopping_patience is not None
                    and self.state.steps_since_improvement >= self.cfg.early_stopping_patience
                ):
                    logger.info(
                        "Early stopping: no val improvement for %d eval intervals.",
                        self.cfg.early_stopping_patience,
                    )
                    break

        self.writer.close()
        return self.state
