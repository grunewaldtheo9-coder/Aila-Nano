# Benchmarks: `nano_10m` vs `nano_20m`

A comparison of the two *model sizes* in this repo, not of product
releases. Aila Nano Beta ships the `nano_20m` weights.

All numbers here were produced by `scripts/benchmark_model.py` on the
machine that trained both models (4 vCPU, 15 GB RAM, CPU only), with no
training job competing for CPU. Reproduce with:

```bash
python scripts/benchmark_model.py --checkpoint checkpoints/finetune/best.pt
python scripts/benchmark_model.py --checkpoint checkpoints/finetune_20m/best.pt
```

## Fair comparison

Both models evaluated on **only the two instruction datasets both were
fine-tuned on** (`aila_company.jsonl` + `finetune_sample.jsonl`). `nano_20m` was
additionally trained on `portuguese_basic.jsonl`; including it would
penalize 1.0 for data it never saw.

| Metric | `nano_10m` (10,877,184 params) | `nano_20m` (19,796,160 params) | Change |
|---|---|---|---|
| Validation loss | 1.7065 | **1.0081** | −41% |
| Validation perplexity | 5.51 | **2.74** | −50% |
| Robustness (non-empty, no crash on 8 malformed inputs) | 7/8 | **8/8** | +1 |
| Identity keyword accuracy | 1/4 | 1/4 | — |
| Portuguese keyword accuracy | 1/3 | 1/3 | — |
| Generation time (15 prompts) | 5.6 s | 8.2 s | ~1.5× slower |

Pretraining (before instruction tuning):

| | `nano_10m` | `nano_20m` |
|---|---|---|
| Best pretrain val loss | — | 2.6304 |
| Best pretrain perplexity | 16.43 | **13.88** |
| Steps × tokens | 650 × 16,384 | 700 × 16,384 |

## What the numbers do and don't mean

**Real:** the language-modeling improvement. Perplexity halved on the
same data, and `nano_20m` no longer produces an empty reply for any malformed
input.

**Not real:** a dramatically better chat experience. Side-by-side on
identical prompts, **both models still garble text**. Examples, verbatim:

> **"What can you help me with?"**
> 1.0 — `The gift  InPI œLetâ€œI, €TMt have about  locers.`
> `nano_20m` — `As Aila N...able, I can help answer questions, generate and take it one day at a time, hold convers...`

> **"Who created you?"**
> 1.0 — `I was created byano, built from scratch by Aila N`
> `nano_20m` — `I was created by`

`nano_20m` recovers trained content more often (note it reaches "founders",
"approximately", "Guil…ald Grune…" on the identity questions, which 1.0
mostly doesn't), but neither model reliably produces clean factual
sentences. The identity and Portuguese keyword scores are unchanged
precisely because of this: the right *content* often appears, mangled
enough to miss the keyword check.

## Why the bigger model is still the better system

The chat quality users actually experience comes mostly from the
deterministic layer around the model — arithmetic, user memory, the global
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

---

## Beta retrain (`checkpoints/finetune_beta` → shipped)

The shipped weights were trained on identity data that stated a wrong
parameter count (10.9M, from before the model grew to 19.8M) and did not
know the release name. That data was corrected and the model re-trained
on the same recipe (40 epochs, `configs/training/finetune_beta.yaml`,
initialised from `checkpoints/pretrain_20m/best.pt`).

| Metric | Previous weights | Beta retrain |
|---|---|---|
| Loss on the two datasets **unchanged** between the runs | 1.9525 (ppl 7.05) | **1.5554 (ppl 4.74)** |
| Robustness (no crash, non-empty on 8 malformed inputs) | 8/8 | 8/8 |
| Identity keyword accuracy (4 cases) | 1/4 | 1/4 |
| Portuguese keyword accuracy (3 cases) | 1/3 | 0/3 |

**How to read this honestly.**

The headline number in `scripts/benchmark_model.py` (val_loss 1.3037 →
0.8811) is *not* a fair comparison and is not quoted above: it evaluates
on `aila_company.jsonl`, which changed between the two runs, so the new
model is partly being scored on data it memorised and the old one on
data it never saw. The table therefore measures only the two datasets
that were byte-identical for both runs — where the improvement is real
(perplexity 7.05 → 4.74) and apples-to-apples.

The identity and Portuguese rows are 4- and 3-case keyword checks over
generations that are incoherent in both models; a one-case difference
there is noise, not signal. Side-by-side generation on ten prompts
showed no quality difference worth claiming in either direction — both
still garble open-ended text, exactly as
[MODEL_CARD.md](MODEL_CARD.md#out-of-scope--limitations) says they will.

**So why ship it?** Not for a quality jump — for correctness. The
previous weights were trained to state a false parameter count. These
are not. The deterministic layer (identity, small talk, memory,
arithmetic, research) is what users actually experience for those
questions, and it is unchanged by the retrain.

---

## Why freeform generation is off by default

Measured on the shipped weights: twelve prompts the model was
*fine-tuned on*, sampled at its default settings, produced usable text
three times.

| Prompt (all present in the fine-tuning data) | Reply |
|---|---|
| "Write a Python function that reverses a string." | "Have a great day." |
| "What is a variable in programming?" | "A capital do Ficoect ajudar?" |
| "Tell me a joke." | "A cat pulls the cat over to their room. Anna is very happy..." |
| "Write a short poem about rain." | "The rain pulls the rain over the town. It is not a on top of a a few d..." |

An out-of-vocabulary check was tried as a garbage *detector* — score each
reply by the fraction of words appearing in neither the training data nor
the tokenizer's whole-word vocabulary — and **it does not work**:
fine-tuned prompts averaged 0.24 out-of-vocabulary, untrained ones 0.16.
There is no separation to threshold on, because generation is unreliable
everywhere rather than only on unfamiliar input. It was not shipped.

So `AILA_ALLOW_FREEFORM` defaults to `false`: a message the deterministic
layer cannot handle gets an honest list of what Aila *can* do. That is
not a capability being removed — the capability produced invented words.
Set `AILA_ALLOW_FREEFORM=true` to let the model try anyway.

The fix for this is more pretraining data, not a bigger safety net; see
[MODEL_CARD.md](MODEL_CARD.md#out-of-scope--limitations).
