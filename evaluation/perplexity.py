"""Per-language perplexity and generation-degradation metrics for a
checkpoint. Deterministic and CPU-only.

Perplexity is the standard, honest way to compare *language-model* quality
across checkpoints and languages: lower = the model finds the held-out text
less surprising. We report English and Portuguese separately so a change
that helps one language and hurts the other is visible, not averaged away.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch


@torch.no_grad()
def text_perplexity(model, tokenizer, text: str, device="cpu", max_seq_len: int | None = None) -> dict:
    """Mean cross-entropy and perplexity of `model` over `text`, evaluated
    line by line (teacher forcing). Lines longer than the model context are
    truncated to it.

    Returns per-token loss + perplexity AND **bits-per-character (BPC)**.
    BPC is the total predicted negative log-likelihood (in bits) divided by
    the number of *characters* — it is tokenizer-independent, so it is the
    only fair way to compare models that use different tokenizers/vocabs
    (e.g. the 8192 vs the 16384 tokenizer). Per-token perplexity is NOT
    comparable across tokenizers; BPC is."""
    model.eval()
    seq_cap = max_seq_len or model.cfg.max_seq_len
    total_loss, total_tokens, total_chars = 0.0, 0, 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        ids = tokenizer.encode(line, add_bos=True, add_eos=True)
        if len(ids) < 2:
            continue
        ids = ids[:seq_cap]
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        _, loss = model(x, targets=y)
        n = y.numel()
        total_loss += loss.item() * n   # sum of NLL (nats) over predicted tokens
        total_tokens += n
        total_chars += len(line)
    if total_tokens == 0:
        return {"loss": None, "perplexity": None, "tokens": 0, "chars": 0, "bits_per_char": None}
    mean_loss = total_loss / total_tokens
    # BPC = total NLL in bits / characters. Tokenizer-independent.
    bpc = (total_loss / math.log(2)) / total_chars if total_chars else None
    return {
        "loss": round(mean_loss, 4),
        "perplexity": round(math.exp(min(mean_loss, 20)), 4),
        "tokens": total_tokens,
        "chars": total_chars,
        "tokens_per_char": round(total_tokens / total_chars, 4) if total_chars else None,
        "bits_per_char": round(bpc, 4) if bpc is not None else None,
    }


def evaluate_language_file(model, tokenizer, path: str, device="cpu") -> dict:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    result = text_perplexity(model, tokenizer, text, device=device)
    result["path"] = path
    return result


def repetition_score(text: str, n: int = 3) -> dict:
    """Degradation signal for generated text: the fraction of n-grams that
    are repeats (1 - distinct_ngrams/total_ngrams). High = the model is
    looping. Also reports the single most-repeated token's share."""
    tokens = text.split()
    if len(tokens) < n:
        return {"repeat_ngram_fraction": 0.0, "top_token_share": 0.0, "n_tokens": len(tokens)}
    ngrams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    distinct = len(set(ngrams))
    from collections import Counter
    counts = Counter(tokens)
    top_share = counts.most_common(1)[0][1] / len(tokens)
    return {
        "repeat_ngram_fraction": round(1 - distinct / len(ngrams), 4),
        "top_token_share": round(top_share, 4),
        "n_tokens": len(tokens),
    }
