"""Small helpers for inspecting model size."""

from __future__ import annotations

import torch.nn as nn


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    """Total parameter count of `module`. With `trainable_only=True`, count
    only parameters that require gradients (what an optimizer would update)."""
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


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
