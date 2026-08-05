"""Small reusable training metrics."""

from __future__ import annotations

import math


class AverageMeter:
    """Tracks a running mean (e.g. loss) without keeping full history."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def perplexity(loss: float, cap: float = 20.0) -> float:
    """exp(loss), capped to avoid overflow when loss is still very high
    early in training (e.g. before the model has learned anything)."""
    return math.exp(min(loss, cap))
