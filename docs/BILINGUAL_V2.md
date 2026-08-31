# Aila Nano 50M-v2 (bilingual) — training experiment & analysis

This documents a **real, measured** experiment: training a fresh 50M model
with the 16,384-vocab bilingual tokenizer on a bilingual EN+PT corpus, on
CPU only. Every number below is measured from actual checkpoints. What was
*not* run is stated plainly.

## Setup

- **Hardware:** 4-core Intel Xeon @ 2.10 GHz, ~15 GiB RAM, **no GPU**
  (`CUDA available: False`). Measured throughput ~650–770 tok/s.
- **Model:** `configs/model/nano_50m_v2_bilingual.yaml` — the nano_50m
  architecture with `vocab_size=16384`. **55,587,328 parameters** (+4.2M vs
  the 8192 model, entirely from the larger tied embedding). Trained **from
  scratch**; the old 8192 checkpoints are not loaded (incompatible vocab).
- **Tokenizer:** `tokenizer/artifacts/v2_bilingual/aila_nano_v2_bilingual.model`
  (16,384; see `docs/TOKENIZER.md`).
- **Corpus:** `aila_pretrain_v2_bilingual` — 1,270,784 tokens, **62% PT /
  38% EN** by tokens (PT 789,207 / EN 481,577), TinyStories (EN, CDLA) +
  Project Gutenberg PT literature (public domain). All training budgets stay
  **under the ~1.2M unique-token ceiling — no data repetition.**
- **Metric:** **bits-per-character (BPC)** — tokenizer-independent, so it is
  the *only* fair way to compare the 16,384-vocab model against the old
  8,192-vocab model. Per-token perplexity is **not** comparable across
  tokenizers and is shown only for within-model trends.

## Results (measured)

**Baseline — old 50M (8192 tokenizer, EN-only pretrain + instruction finetune):**

| | EN | PT |
|---|---|---|
| BPC | 1.5843 | **5.8512** |
| token ppl | 129.3 | 2599.7 |

**New 50M-v2 (bilingual) by training-token budget:**

| Tokens seen | Tokens/param | EN BPC | PT BPC | EN ppl* | PT ppl* | tok/s | time |
|---|---|---|---|---|---|---|---|
| 307,200 | 0.0055 | 2.4418 | 3.0045 | 1352.0 | 2016.7 | 655 | 469 s |
| 602,112 | 0.0108 | 2.2631 | 2.9181 | 797.8 | 1620.5 | 705 | 854 s |
| 1,105,920 | 0.0199 | **2.0673** | **2.7813** | 447.5 | 1145.9 | 720 | 1535 s |

*within-model only; not comparable to the 8192 baseline.

## Analysis (answered from measurements, not intuition)

**Q1 — Did tokenizer v2 improve Portuguese representation?** **Yes**, measured
in two ways: at the tokenizer level (PT tok/word 2.85 → 1.41, byte-fallback
9.6% → 0.005%, accents 1/11 → 11/11 in-vocab — `docs/TOKENIZER.md`), and at
the model level (PT BPC 5.85 → 2.78).

**Q2 — Did Portuguese perplexity improve?** **Yes, dramatically**, on the fair
BPC metric: **PT BPC 5.85 → 2.78, ≈52% lower**, despite the v2 model seeing
only ~1.1M total tokens vs the old model's ~11M EN tokens + finetune. The old
model had essentially never seen Portuguese; the bilingual v2 has.

**Q3 — How does PT BPC change with more tokens?** It falls monotonically —
3.00 → 2.92 → 2.78 across 300k → 600k → 1.1M — and is **still descending with
no plateau** at tokens/param ≈ 0.02.

**Q4 — Does English degrade?** **Not a fair comparison, and not evidence of
degradation.** The v2's EN BPC (2.07) is higher than the old model's (1.58),
but the old model saw ~10–25× more English (≈11M EN tokens + finetune vs the
v2's ≈0.42M EN tokens). The v2's EN BPC is itself dropping steeply with
tokens (2.44 → 2.26 → 2.07), i.e. it is simply undertrained on English, not
harmed by bilingual data. A **token-matched control** (an 8192 and/or a v2
model trained on the same budget) is required to answer this cleanly and was
**not run** (CPU time).

**Q5 — Is 50M capacity sufficient to keep scaling data?** **Yes.** Both EN and
PT BPC drop steeply with more tokens and show no plateau at tokens/param ≈
0.02 — the model is nowhere near a capacity limit; it is deeply
token/data-limited. There is ample headroom to keep feeding it data.

**Q6 — What is the remaining bottleneck?** Primarily **Portuguese data
quantity + training-token budget**. The **tokenizer bottleneck is resolved**
(v2). Portuguese **data quality/diversity** (currently literary-only) is a
secondary concern. **Model capacity is not** the bottleneck.

**Q7 — Is there evidence to justify 100M/120M now?** **No.** The curves show
large, unexhausted data-driven gains at tokens/param ≈ 0.02 (compute-optimal
is ~20). Scaling parameters now would be premature. **Data first, tokens
first, parameters later.**

## Qualitative (honest, not cherry-picked)

At 1.1M tokens the v2 model produces **recognizable Portuguese** vocabulary
and morphology in the corpus's literary register — e.g. *"o seu coração, que
lhe…"*, *"a vida, por uma"*, *"o amor; e o que não"* — a clear qualitative
jump from the old model, which returned `.` or English text for PT prompts.
It is **not yet coherent or fluent** (repetition, archaic Gutenberg spelling
like *elle/espirito*), which is expected at ~0.02 tokens/param. English
generations are weak for the same reason (little EN training). Full outputs:
`experiments/bilingual_50m_v2/generations.json`.

## What remains incomplete (honest)

- **No long run.** The largest budget is 1.1M tokens (~26 min CPU). A
  competent bilingual model needs orders of magnitude more — GPU territory.
  Everything is prepared and resumable to run it there.
- **No token-matched old-vs-new control** for the EN question (Q4).
- **Portuguese corpus is literary-only** (Gutenberg). Broad Brazilian-web PT
  (Wikipedia/OSCAR, filtered) is the next data addition; the pipeline
  supports it (HF `datasets` is not installed here, so it was not fetched).
- **No v2 instruction/Aila fine-tune** (the deterministic router and
  conversation infra are unchanged and remain authoritative).

## Reproduce

```bash
# corpus (v2 tokenizer)
python datasets/scripts/build_pretrain_corpus.py \
  --input "datasets/raw/pt_all/*.txt" datasets/raw/tinystories_slice.txt \
  --tokenizer tokenizer/artifacts/v2_bilingual/aila_nano_v2_bilingual.model \
  --version aila_pretrain_v2_bilingual --target-ratios "pt=0.5,en=0.5" --no-near-dedup
# scaling sweep
python scripts/scaling_experiment.py \
  --model-config configs/model/nano_50m_v2_bilingual.yaml \
  --base-train-config configs/training/scaling_50m_v2_base.yaml \
  --token-budgets 300000,600000,1100000 \
  --out-root checkpoints_v2/50m_bilingual --results-dir experiments/bilingual_50m_v2 \
  --dataset-version aila_pretrain_v2_bilingual --skip-existing
# evaluate + report
python scripts/eval_v2_checkpoints.py
```
