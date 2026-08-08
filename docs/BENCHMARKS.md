# Benchmarks: Aila Nano 1.0 vs 2.0

All numbers here were produced by `scripts/benchmark_model.py` on the
machine that trained both models (4 vCPU, 15 GB RAM, CPU only), with no
training job competing for CPU. Reproduce with:

```bash
python scripts/benchmark_model.py --checkpoint checkpoints/finetune/best.pt
python scripts/benchmark_model.py --checkpoint checkpoints/finetune_20m/best.pt
```

## Fair comparison

Both models evaluated on **only the two instruction datasets both were
fine-tuned on** (`aila_company.jsonl` + `finetune_sample.jsonl`). 2.0 was
additionally trained on `portuguese_basic.jsonl`; including it would
penalize 1.0 for data it never saw.

| Metric | 1.0 (10,877,184 params) | 2.0 (19,796,160 params) | Change |
|---|---|---|---|
| Validation loss | 1.7065 | **1.0081** | −41% |
| Validation perplexity | 5.51 | **2.74** | −50% |
| Robustness (non-empty, no crash on 8 malformed inputs) | 7/8 | **8/8** | +1 |
| Identity keyword accuracy | 1/4 | 1/4 | — |
| Portuguese keyword accuracy | 1/3 | 1/3 | — |
| Generation time (15 prompts) | 5.6 s | 8.2 s | ~1.5× slower |

Pretraining (before instruction tuning):

| | 1.0 | 2.0 |
|---|---|---|
| Best pretrain val loss | — | 2.6304 |
| Best pretrain perplexity | 16.43 | **13.88** |
| Steps × tokens | 650 × 16,384 | 700 × 16,384 |

## What the numbers do and don't mean

**Real:** the language-modeling improvement. Perplexity halved on the
same data, and 2.0 no longer produces an empty reply for any malformed
input.

**Not real:** a dramatically better chat experience. Side-by-side on
identical prompts, **both models still garble text**. Examples, verbatim:

> **"What can you help me with?"**
> 1.0 — `The gift  InPI œLetâ€œI, €TMt have about  locers.`
> 2.0 — `As Aila N...able, I can help answer questions, generate and take it one day at a time, hold convers...`

> **"Who created you?"**
> 1.0 — `I was created byano, built from scratch by Aila N`
> 2.0 — `I was created by`

2.0 recovers trained content more often (note it reaches "founders",
"approximately", "Guil…ald Grune…" on the identity questions, which 1.0
mostly doesn't), but neither model reliably produces clean factual
sentences. The identity and Portuguese keyword scores are unchanged
precisely because of this: the right *content* often appears, mangled
enough to miss the keyword check.

## Why 2.0 is still the better system

The chat quality users actually experience comes mostly from the
deterministic layer added in 2.0 — arithmetic, user memory, the global
knowledge base, and web research — which returns exact answers and does
not depend on the model's generation quality at all:

```
You: What is 12 * 8?              -> The answer is 96.
You: remember that my name is Theo -> Got it — I'll remember that my name is Theo.
You: What is my name?              -> Your name is Theo.
You: Who founded Apple?            -> Apple was founded by Steve Jobs, Steve Wozniak
                                      and Ronald Wayne in 1976.
```

Those are verbatim outputs, and they are identical on either checkpoint.

## The real bottleneck

Not parameter count. Both models were pretrained on ~11.5M tokens of
TinyStories (simple English children's stories) — orders of magnitude
below what a ~20M-parameter model needs, and the reason its prose sounds
childlike and drifts into story fragments. Doubling parameters against
the same tiny corpus is why the metric gains didn't translate into
visibly better answers.

The highest-value next step is **more and better pretraining data on a
GPU**, not a bigger model — see [GPU_TRAINING.md](GPU_TRAINING.md).
