"""Cosine learning-rate schedule with linear warmup (GPT-3 / Chinchilla
style) — the de-facto standard for transformer pretraining.
"""

from __future__ import annotations

import math


class CosineWarmupScheduler:
    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
    ):
        if min_lr > max_lr:
            raise ValueError("min_lr must be <= max_lr")
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def get_lr(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        if step >= self.max_steps:
            return self.min_lr
        decay_ratio = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        decay_ratio = min(max(decay_ratio, 0.0), 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def set_lr(self, optimizer, step: int) -> float:
        lr = self.get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr
