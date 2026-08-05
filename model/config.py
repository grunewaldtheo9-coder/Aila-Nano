"""Architecture configuration for the Aila Nano transformer.

The default `nano_10m()` preset is tuned (see scripts/count_params.py) to
land at ~10.9M total parameters with tied input/output embeddings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import yaml


@dataclass
class GPTConfig:
    # -- vocabulary / sequence -----------------------------------------
    vocab_size: int = 8192
    max_seq_len: int = 512

    # -- depth / width ---------------------------------------------------
    n_layers: int = 8
    d_model: int = 208
    n_heads: int = 8
    n_kv_heads: int = 4  # grouped-query attention; == n_heads gives plain MHA
    mlp_hidden_mult: float = 2.75  # SwiGLU hidden dim = round(mlp_hidden_mult * d_model)

    # -- regularization / numerics ---------------------------------------
    dropout: float = 0.1
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    initializer_range: float = 0.02

    # -- behavior ---------------------------------------------------------
    tie_embeddings: bool = True
    bias: bool = False  # no bias terms in Linear layers (GPT-3 / Llama style)

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def mlp_hidden_dim(self) -> int:
        # Round to the nearest multiple of 8 for friendlier matmul shapes.
        raw = int(self.mlp_hidden_mult * self.d_model)
        return max(8, 8 * round(raw / 8))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> GPTConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict) -> GPTConfig:
        return cls(**data)


def nano_10m() -> GPTConfig:
    """The Aila Nano ~10.9M-parameter preset used for pretraining.

    Sized by scripts/count_params.py — see docs/MODEL_CARD.md for the exact
    parameter breakdown by component.
    """
    return GPTConfig(
        vocab_size=8192,
        max_seq_len=512,
        n_layers=12,
        d_model=256,
        n_heads=8,
        n_kv_heads=4,
        mlp_hidden_mult=2.72,
        dropout=0.1,
        tie_embeddings=True,
        bias=False,
    )


def tiny_debug() -> GPTConfig:
    """A tiny config for fast unit tests / CI (not for real training)."""
    return GPTConfig(
        vocab_size=256,
        max_seq_len=64,
        n_layers=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        mlp_hidden_mult=2.0,
        dropout=0.0,
        tie_embeddings=True,
        bias=False,
    )
