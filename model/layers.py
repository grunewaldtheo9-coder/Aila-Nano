"""Building-block layers for the Aila Nano transformer.

Design decisions (all chosen for the best quality-per-parameter at ~10.9M
scale, following what current open-weight LLMs converged on):

- RMSNorm instead of LayerNorm: one fewer learned vector (no bias/mean
  term), slightly cheaper, and empirically matches LayerNorm's stability
  in pre-norm transformers (used by Llama, Mistral, Gemma, ...).
- Rotary position embeddings (RoPE) instead of learned absolute position
  embeddings: zero extra parameters (vs. max_seq_len * d_model for a
  learned table), and encodes *relative* position, which generalizes
  better to sequences shorter/longer than seen at training time.
- SwiGLU MLP instead of GELU-MLP: consistently improves loss-per-parameter
  in the literature (Shazaeer, 2020; used by PaLM, Llama). Its hidden size
  is scaled down from the usual 4x so the total parameter count stays
  comparable to a plain 4x-GELU MLP despite using a third weight matrix.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for numerical stability under mixed precision,
        # then cast back to the input dtype before scaling.
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (norm.to(dtype)) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block: down(silu(gate(x)) * up(x))."""

    def __init__(self, d_model: int, hidden_dim: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


def precompute_rope_cache(
    head_dim: int, max_seq_len: int, theta: float = 10000.0, device=None, dtype=torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for rotary embeddings.

    Returns tensors of shape (max_seq_len, head_dim) ready to be sliced per
    forward pass and applied to query/key tensors.
    """
    assert head_dim % 2 == 0, "RoPE requires an even head_dim"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)  # (max_seq_len, head_dim / 2)
    freqs = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, head_dim)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query/key tensors.

    q, k: (batch, n_heads, seq_len, head_dim)
    cos, sin: (seq_len, head_dim) — already sliced to the current positions.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot.to(q.dtype), k_rot.to(k.dtype)
