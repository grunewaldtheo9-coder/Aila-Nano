"""Aila Nano: a decoder-only GPT transformer, ~10.9M parameters at the
`nano_10m()` config (see model/config.py).

Architecture summary (Llama-style pre-norm decoder):
    token embedding (tied with output head)
      -> [ RMSNorm -> GQA causal self-attention (+RoPE) -> residual
           RMSNorm -> SwiGLU MLP                         -> residual ] x n_layers
      -> RMSNorm
      -> linear head (tied weights) -> logits
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import CausalSelfAttention, KVCache
from model.config import GPTConfig
from model.layers import RMSNorm, SwiGLU, precompute_rope_cache


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = CausalSelfAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            dropout=cfg.dropout,
            bias=cfg.bias,
        )
        self.mlp_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mlp = SwiGLU(
            cfg.d_model, cfg.mlp_hidden_dim, bias=cfg.bias, dropout=cfg.dropout
        )

    def forward(self, x, cos, sin, attn_mask=None, kv_cache: KVCache | None = None):
        x = x + self.attn(self.attn_norm(x), cos, sin, attn_mask=attn_mask, kv_cache=kv_cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class AilaNanoGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

        cos, sin = precompute_rope_cache(
            cfg.head_dim, cfg.max_seq_len, theta=cfg.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2-style scaled init on residual-output projections, so that
        # the variance of the residual stream doesn't grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(
                    p, mean=0.0, std=cfg.initializer_range / math.sqrt(2 * cfg.n_layers)
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.token_emb.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_caches: list[KVCache] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        input_ids: (batch, seq_len)
        targets:   (batch, seq_len) next-token targets, -100 to ignore a position.
        kv_caches: one KVCache per layer, for incremental decoding.

        Returns (logits, loss). loss is None if targets is None.
        """
        bsz, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}"
            )

        start_pos = kv_caches[0].seq_len if kv_caches is not None else 0
        cos = self.rope_cos[start_pos : start_pos + seq_len].to(input_ids.device)
        sin = self.rope_sin[start_pos : start_pos + seq_len].to(input_ids.device)

        x = self.dropout(self.token_emb(input_ids))
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, cos, sin, kv_cache=cache)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def new_kv_caches(self) -> list[KVCache]:
        return [KVCache() for _ in range(self.cfg.n_layers)]

    @torch.no_grad()
    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the transformer trunk and return the final-norm hidden
        states (batch, seq_len, d_model) instead of vocabulary logits.

        This is what powers Aila Nano's own text embeddings (see
        vectordb/embedder.py) — the model produces its own semantic
        representations rather than depending on an external embedding
        API, consistent with the project's "no external AI APIs" rule.
        """
        bsz, seq_len = input_ids.shape
        cos = self.rope_cos[:seq_len].to(input_ids.device)
        sin = self.rope_sin[:seq_len].to(input_ids.device)
        x = self.token_emb(input_ids)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.final_norm(x)

    def configure_optimizer(
        self, weight_decay: float, learning_rate: float, betas: tuple[float, float]
    ) -> torch.optim.Optimizer:
        """Split parameters into decayed (weight matrices) and non-decayed
        (norms, biases, embeddings-as-1D... actually embeddings are 2D but
        conventionally excluded) groups, AdamW-style (Loshchilov & Hutter).
        """
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() < 2:
                no_decay.append(p)
            else:
                decay.append(p)
        optim_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
