#!/usr/bin/env python3
"""Print the exact parameter count and per-component breakdown for a
GPTConfig preset. Used to tune model/config.py's `nano_10m()` (~10.9M)
and `nano_20m()` (~19.8M) to land on their targets.

Usage:
    python scripts/count_params.py                  # 1.0 preset
    python scripts/count_params.py --preset nano_20m
    python scripts/count_params.py --n-layers 10 --d-model 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import GPTConfig, nano_10m, nano_20m
from model.transformer import AilaNanoGPT
from model.utils import count_parameters, format_param_count, parameter_breakdown

PRESETS = {"nano_10m": (nano_10m, 10_900_000), "nano_20m": (nano_20m, 20_000_000)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="nano_10m")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-kv-heads", type=int, default=None)
    parser.add_argument("--mlp-hidden-mult", type=float, default=None)
    args = parser.parse_args()

    preset_fn, target = PRESETS[args.preset]
    cfg = preset_fn()
    overrides = {
        k: v for k, v in vars(args).items() if v is not None and k != "preset"
    }
    field_map = {
        "vocab_size": "vocab_size",
        "max_seq_len": "max_seq_len",
        "n_layers": "n_layers",
        "d_model": "d_model",
        "n_heads": "n_heads",
        "n_kv_heads": "n_kv_heads",
        "mlp_hidden_mult": "mlp_hidden_mult",
    }
    cfg_dict = cfg.to_dict()
    for arg_name, field in field_map.items():
        if arg_name in overrides:
            cfg_dict[field] = overrides[arg_name]
    cfg = GPTConfig.from_dict(cfg_dict)

    model = AilaNanoGPT(cfg)
    total = count_parameters(model)
    non_embed = model.num_parameters(exclude_embeddings=True)

    print("Config:")
    for k, v in cfg.to_dict().items():
        print(f"  {k}: {v}")
    print(f"  head_dim: {cfg.head_dim}")
    print(f"  mlp_hidden_dim: {cfg.mlp_hidden_dim}")
    print()
    print("Parameter breakdown (top-level):")
    for name, n in parameter_breakdown(model).items():
        print(f"  {name:12s} {n:>12,}  ({format_param_count(n)})")
    print()
    print(f"Total parameters:              {total:>12,}  ({format_param_count(total)})")
    print(f"Total (excluding embeddings):  {non_embed:>12,}  ({format_param_count(non_embed)})")
    print(f"Target: {format_param_count(target)}  |  Delta: {(total - target) / target:+.2%}")


if __name__ == "__main__":
    main()
