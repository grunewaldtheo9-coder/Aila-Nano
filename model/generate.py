"""Autoregressive text generation for AilaNanoGPT.

Uses an incremental KV cache so each new token costs a single forward pass
over one position instead of the whole growing sequence — the standard
approach for efficient decoder-only inference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model.transformer import AilaNanoGPT


@torch.no_grad()
def generate(
    model: AilaNanoGPT,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = 0.95,
    repetition_penalty: float = 1.15,
    eos_id: int | None = None,
) -> torch.Tensor:
    """Sample a continuation for `input_ids` (shape: (batch, prompt_len)).

    Returns the full sequence (prompt + generated), shape (batch, prompt_len + n_generated).
    Generation for a batch element stops early (padding with eos_id) once it
    emits `eos_id`; sequences shorter than the batch max are marked with
    `eos_id` past the true end.
    """
    model.eval()
    device = input_ids.device
    bsz, prompt_len = input_ids.shape
    max_seq_len = model.cfg.max_seq_len
    if prompt_len > max_seq_len:
        raise ValueError(
            f"Prompt length {prompt_len} exceeds model max_seq_len {max_seq_len}; "
            f"truncate the prompt before calling generate()."
        )
    # Hard cap so a caller-supplied max_new_tokens can never push the total
    # sequence (prompt + generated) past the model's RoPE cache — beyond
    # that, position embeddings simply don't exist.
    max_new_tokens = min(max_new_tokens, max_seq_len - prompt_len)

    kv_caches = model.new_kv_caches()

    generated = input_ids
    finished = torch.zeros(bsz, dtype=torch.bool, device=device)

    # Prime the KV cache with the full prompt in one forward pass.
    logits, _ = model(input_ids, kv_caches=kv_caches)
    next_logits = logits[:, -1, :]

    for _ in range(max_new_tokens):
        next_logits = _apply_repetition_penalty(next_logits, generated, repetition_penalty)
        next_token = _sample(next_logits, temperature=temperature, top_k=top_k, top_p=top_p)

        if eos_id is not None:
            next_token = torch.where(
                finished, torch.full_like(next_token, eos_id), next_token
            )
            finished = finished | (next_token.squeeze(-1) == eos_id)

        generated = torch.cat([generated, next_token], dim=1)
        if eos_id is not None and bool(finished.all()):
            break

        logits, _ = model(next_token, kv_caches=kv_caches)
        next_logits = logits[:, -1, :]

    return generated


@torch.no_grad()
def generate_stream(
    model: AilaNanoGPT,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = 0.95,
    repetition_penalty: float = 1.15,
    eos_id: int | None = None,
):
    """Generator version of `generate` for a single prompt (batch size 1),
    yielding one new token id at a time as it's produced — what the web
    backend's SSE `/chat/stream` endpoint consumes.
    """
    if input_ids.shape[0] != 1:
        raise ValueError("generate_stream only supports batch size 1")

    model.eval()
    prompt_len = input_ids.shape[1]
    max_new_tokens = min(max_new_tokens, model.cfg.max_seq_len - prompt_len)
    kv_caches = model.new_kv_caches()

    generated = input_ids
    logits, _ = model(input_ids, kv_caches=kv_caches)
    next_logits = logits[:, -1, :]

    for _ in range(max_new_tokens):
        next_logits = _apply_repetition_penalty(next_logits, generated, repetition_penalty)
        next_token = _sample(next_logits, temperature=temperature, top_k=top_k, top_p=top_p)
        token_id = int(next_token.item())

        generated = torch.cat([generated, next_token], dim=1)
        yield token_id

        if eos_id is not None and token_id == eos_id:
            return

        logits, _ = model(next_token, kv_caches=kv_caches)
        next_logits = logits[:, -1, :]


def _apply_repetition_penalty(
    logits: torch.Tensor, generated: torch.Tensor, penalty: float
) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    for b in range(generated.shape[0]):
        seen = torch.unique(generated[b])
        seen_logits = logits[b, seen]
        logits[b, seen] = torch.where(
            seen_logits > 0, seen_logits / penalty, seen_logits * penalty
        )
    return logits


def _sample(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = 0.95,
) -> torch.Tensor:
    """logits: (batch, vocab_size) -> next token ids (batch, 1)."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_val = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
        logits = torch.where(logits < kth_val, torch.full_like(logits, float("-inf")), logits)

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs, dim=-1)
        # Remove tokens once cumulative probability exceeds top_p, but always
        # keep at least the single most-likely token.
        remove = cum_probs - probs > top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
