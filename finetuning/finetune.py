#!/usr/bin/env python3
"""Instruction fine-tuning loop and CLI entrypoint.

Full fine-tuning (not LoRA/adapters) is used deliberately: at ~10.9M
parameters the whole model already fits comfortably in memory and trains
in seconds-to-minutes per epoch even on CPU, so parameter-efficient
fine-tuning would only add complexity without a real efficiency win here.

Supports continual fine-tuning: point `--init-checkpoint` at any previous
checkpoint (a pretrained base model *or* an already fine-tuned one) to
keep adapting the model as new instruction data becomes available.

Usage:
    python -m finetuning.finetune \
        --init-checkpoint checkpoints/pretrain/best.pt \
        --tokenizer tokenizer/artifacts/aila_nano.model \
        --data datasets/aila_knowledge/aila_company.jsonl datasets/finetune/instructions.jsonl \
        --out-dir checkpoints/finetune
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from finetuning.dataset import InstructionDataset, collate_instruction_batch
from model.transformer import AilaNanoGPT
from tokenizer.tokenizer import AilaTokenizer
from training.checkpoint import load_checkpoint, save_checkpoint
from training.metrics import AverageMeter, perplexity
from training.scheduler import CosineWarmupScheduler

logger = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    epochs: int = 3
    batch_size: int = 8
    max_lr: float = 5e-5
    min_lr: float = 5e-6
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    val_fraction: float = 0.1
    log_interval: int = 10
    out_dir: str = "checkpoints/finetune"
    device: str = "auto"
    seed: int = 1337
    # A rolling `epoch_NNN.pt` is written every `checkpoint_every_n_epochs`
    # epochs (in addition to `best.pt`, always kept up to date), and only
    # the last `keep_last_n_checkpoints` of those are retained. Without
    # this, a many-epoch run on a small dataset — a completely reasonable
    # thing to do here, since fine-tuning is cheap — silently fills the
    # disk: one checkpoint per epoch adds up fast (each is a full copy of
    # the model + optimizer state).
    keep_last_n_checkpoints: int = 3
    checkpoint_every_n_epochs: int = 1

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


def run_finetune(
    model: AilaNanoGPT,
    tokenizer: AilaTokenizer,
    data_paths: list[str],
    cfg: FinetuneConfig,
) -> AilaNanoGPT:
    device = torch.device(cfg.resolved_device())
    model.to(device)
    torch.manual_seed(cfg.seed)

    full_ds = InstructionDataset(data_paths, tokenizer, max_seq_len=model.cfg.max_seq_len)
    n_val = max(1, int(len(full_ds) * cfg.val_fraction)) if len(full_ds) > 1 else 0
    n_train = len(full_ds) - n_val
    if n_val > 0:
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed)
        )
    else:
        train_ds, val_ds = full_ds, None

    collate = partial(collate_instruction_batch, pad_id=tokenizer.pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, drop_last=False
    )
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
        if val_ds is not None
        else None
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))

    optimizer = model.configure_optimizer(
        weight_decay=cfg.weight_decay, learning_rate=cfg.max_lr, betas=cfg.betas
    )
    scheduler = CosineWarmupScheduler(
        max_lr=cfg.max_lr, min_lr=cfg.min_lr, warmup_steps=warmup_steps, max_steps=total_steps
    )

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(cfg.out_dir) / "tb"))

    step = 0
    best_val_loss = float("inf")
    model.train()
    for epoch in range(cfg.epochs):
        meter = AverageMeter()
        for batch in train_loader:
            lr = scheduler.set_lr(optimizer, step)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(input_ids, targets=labels)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            meter.update(loss.item())
            step += 1
            if step % cfg.log_interval == 0:
                logger.info(
                    "epoch %d | step %d/%d | loss %.4f | lr %.2e | grad_norm %.3f",
                    epoch,
                    step,
                    total_steps,
                    meter.avg,
                    lr,
                    grad_norm,
                )
                writer.add_scalar("finetune/loss", meter.avg, step)
                writer.add_scalar("finetune/lr", lr, step)
                meter.reset()

        if val_loader is not None:
            val_loss = _evaluate(model, val_loader, device)
            logger.info(
                "  [epoch %d done] val_loss %.4f | perplexity %.2f",
                epoch,
                val_loss,
                perplexity(val_loss),
            )
            writer.add_scalar("finetune/val_loss", val_loss, step)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    str(Path(cfg.out_dir) / "best.pt"), model, optimizer, step, best_val_loss
                )

        if epoch % cfg.checkpoint_every_n_epochs == 0 or epoch == cfg.epochs - 1:
            save_checkpoint(
                str(Path(cfg.out_dir) / f"epoch_{epoch:03d}.pt"),
                model,
                optimizer,
                step,
                best_val_loss,
            )
            _prune_epoch_checkpoints(cfg.out_dir, cfg.keep_last_n_checkpoints)

    writer.close()
    return model


def _prune_epoch_checkpoints(out_dir: str, keep_last_n: int) -> None:
    checkpoints = sorted(Path(out_dir).glob("epoch_*.pt"))
    # Regression note: `excess` must be clamped to >= 0 before slicing.
    # `checkpoints[:excess]` with a *negative* excess is not a no-op —
    # Python slicing treats a negative stop as "count back from the end",
    # so e.g. 2 checkpoints with keep_last_n=3 gives excess=-1 and
    # `checkpoints[:-1]` deletes the oldest one anyway, even though we're
    # under the limit. This silently capped every run at 1 retained
    # checkpoint regardless of keep_last_n — caught by
    # tests/test_finetuning.py::test_run_finetune_prunes_old_epoch_checkpoints.
    excess = max(0, len(checkpoints) - keep_last_n)
    for p in checkpoints[:excess]:
        p.unlink(missing_ok=True)


@torch.no_grad()
def _evaluate(model: AilaNanoGPT, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    meter = AverageMeter()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        _, loss = model(input_ids, targets=labels)
        meter.update(loss.item())
    model.train()
    return meter.avg


def parse_args():
    p = argparse.ArgumentParser(description="Instruction fine-tune Aila Nano.")
    p.add_argument("--init-checkpoint", required=True, help="Pretrained or previous-finetune checkpoint")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--data", nargs="+", required=True, help="One or more JSONL instruction files")
    p.add_argument("--config", default=None, help="YAML file with FinetuneConfig fields (optional)")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-lr", type=float, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    args = parse_args()

    from model.config import GPTConfig

    ckpt = load_checkpoint(args.init_checkpoint)
    model = AilaNanoGPT(GPTConfig.from_dict(ckpt["config"]))
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded init checkpoint from %s (step %d)", args.init_checkpoint, ckpt["step"])

    tokenizer = AilaTokenizer.load(args.tokenizer)

    cfg_kwargs: dict = {}
    if args.config:
        import yaml

        with open(args.config) as f:
            cfg_kwargs.update(yaml.safe_load(f))
    for key in ("out_dir", "epochs", "batch_size", "max_lr", "device"):
        val = getattr(args, key)
        if val is not None:
            cfg_kwargs[key] = val

    cfg = FinetuneConfig(**cfg_kwargs)
    run_finetune(model, tokenizer, args.data, cfg)


if __name__ == "__main__":
    main()
