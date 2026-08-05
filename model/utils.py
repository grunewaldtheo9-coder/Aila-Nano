"""Small helpers for inspecting model size."""

from __future__ import annotations

import torch.nn as nn


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def format_param_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def parameter_breakdown(module: nn.Module) -> dict[str, int]:
    """Group parameter counts by top-level submodule name — useful for
    sanity-checking where a model's budget is going.
    """
    breakdown: dict[str, int] = {}
    for name, p in module.named_parameters():
        top = name.split(".")[0]
        breakdown[top] = breakdown.get(top, 0) + p.numel()
    return breakdown
