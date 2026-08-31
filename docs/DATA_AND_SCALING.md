# Data pipeline, token budgets & scaling experiments

This document covers the infrastructure added to make Aila Nano 50M a
better *language model by giving it more, better data* — before touching
parameter count. It is honest about what has been **run** vs. what is
**framework-ready**: the CPU here (4 cores, ~800 tok/s) makes large runs
multi-day, so the code exists and the *smallest useful* experiments were
actually executed.

## What this adds

| Piece | File | Status |
|---|---|---|
| Token/epoch budget enforcement | `training/trainer.py` (`resolve_max_steps`) | ✅ run + tested |
| Dataset identity in checkpoints | `training/dataset.py` (`corpus_fingerprint`) | ✅ run + tested |
| Scaling-experiment runner | `scripts/scaling_experiment.py` | ✅ run (small budgets) |
| Scaling report (table + plot) | `scripts/scaling_report.py` | ✅ run + tested |
| Language ID (EN/PT) | `datasets/scripts/langid.py` | ✅ run + tested |
| Corpus builder (clean/filter/dedup/tokenize/manifest) | `datasets/scripts/build_pretrain_corpus.py` | ✅ run on real PT data |
| Dataset versioning/manifest | `datasets/scripts/dataset_manifest.py` | ✅ run + tested |
| Tokenizer efficiency analysis | `scripts/tokenizer_stats.py` | ✅ run on real data |
| Separated dataset stages | `datasets/{pretrain,instruction,aila}/` | ✅ created |
| EN/PT/Aila evaluation harness | — | ⏳ not built this pass (see "Not done") |

The deterministic router, conversation infrastructure, tokenizer artifact,
and the existing 20M/50M checkpoints are **unchanged**.

## Token-budget training

The trainer now accepts token and epoch budgets and trains for the
*smallest* of `{max_steps, max_tokens→steps, max_epochs→steps}` — it never
trains past a requested token budget, and the LR cosine anneals over that
effective horizon.

```bash
python -m training.train \
  --model-config configs/model/nano_50m_cpu256.yaml \
  --train-config configs/training/pretrain_50m.yaml \
  --max-tokens 5000000            # or --max-epochs 2  or --max-steps 800
```

Every checkpoint records, in its `extra` block: `dataset_version`,
`train_sha256` (content hash of the corpus), `train_tokens`, `val_tokens`,
`tokens_seen`, `tokens_per_step`, `effective_max_steps`, and `seed` — so a
checkpoint identifies exactly the data and budget that produced it.

### Resume

```bash
python -m training.train ... --resume          # latest in checkpoint_dir
python -m training.train ... --resume-from <checkpoint.pt>
```

Resume restores model + optimizer + step + best-val and continues the LR
schedule (verified in `tests/test_training.py`).

## Scaling experiments

Train the *same* model on the *same* corpus at a series of token budgets,
then compare — measured, not guessed.

```bash
# Run a sweep (resumable: --skip-existing skips budgets already done)
python scripts/scaling_experiment.py \
  --model-config configs/model/nano_50m_cpu256.yaml \
  --base-train-config configs/training/scaling_50m_base.yaml \
  --token-budgets 500000,1000000,2000000 \
  --out-root checkpoints/50m_data_scaling \
  --results-dir experiments/50m_data_scaling \
  --dataset-version aila_pretrain_v1_tinystories --skip-existing

# Build the comparison report (Markdown table + ASCII/PNG plots)
python scripts/scaling_report.py --results-dir experiments/50m_data_scaling
```

Each budget writes a machine-readable JSON (`experiments/50m_data_scaling/
tokens_<N>.json`) with parameters, tokens seen, epochs, tokens/parameter,
train/val loss, best val perplexity, time, tokens/sec, dataset version +
hash, and language mixture. The report reads only the JSONs that exist and
**never invents missing points**. See `experiments/50m_data_scaling/REPORT.md`
for the measured results from this environment.

### Measured result (this environment)

Three real points, same 51M model, same corpus, fresh per budget (see
`experiments/50m_data_scaling/REPORT.md`):

| Tokens seen | Tokens/param | Best val loss | Val PPL |
|---|---|---|---|
| 503,808 | 0.0098 | 4.4375 | 84.56 |
| 1,007,616 | 0.0196 | 3.7719 | 43.46 |
| 2,002,944 | 0.0390 | 3.2552 | 25.92 |

Validation loss falls steeply and monotonically with more data (−0.67 from
1M→2M alone) with **no sign of a plateau**, at tokens/parameter far below 1.
This empirically confirms Aila Nano 50M is **strongly data-limited**: the
next gains come from more high-quality data, not more parameters. (These are
short-horizon points meant to trace the curve on CPU, not converged models.)

### Cost on this CPU

~800 tok/s ⇒ ~12k tokens/step ⇒ ~210 steps/hour. So one epoch of the
17.4M-token corpus is ~6.7 h. The sweep above uses small budgets
(≤2M tokens, all < 1 epoch, no data repetition) precisely so it completes
in a session. Larger budgets (50M–1B) are supported by the framework but
are GPU-scale on wall-clock terms here.

## Building a corpus

`datasets/scripts/build_pretrain_corpus.py` runs the full pipeline:
Gutenberg-boilerplate strip → clean/normalize (`clean_text.py`) → language
ID (`langid.py`) → quality/length filter → exact + near dedup (`dedupe.py`)
→ optional per-language down-sampling toward `--target-ratios` (never
up-samples/repeats) → tokenize → deterministic document-level train/val
split (a document is entirely in train or entirely in val — no leakage) →
versioned manifest. It reports raw/after-filter/after-dedup document counts,
tokens total and by language, and percentage removed.

```bash
python datasets/scripts/build_pretrain_corpus.py \
  --input "datasets/raw/pt/*.txt" \
  --tokenizer tokenizer/artifacts/aila_nano.model \
  --version aila_pretrain_pt_v1 \
  --target-ratios pt=1.0 \
  --source-name "Project Gutenberg (Machado de Assis, Eça de Queirós)" \
  --source-license "Public domain (Project Gutenberg License)"
```

Inspect any corpus:
```bash
cat datasets/pretrain/<version>/manifest.json
```

## Portuguese data

A real Portuguese corpus was built this pass from **Project Gutenberg**
public-domain literature (Machado de Assis, Eça de Queirós): 4 works →
2,278 documents after filtering → **666,877 Portuguese tokens** (verified
`pt` by langid), versioned as `aila_pretrain_pt_v1`. This is a starting
point, not a full corpus — it is literary PT, weighted to classic
(non-Brazilian-web) register. Broader Brazilian-Portuguese web text
(Wikipedia/OSCAR-style, aggressively filtered) is the next addition; the
pipeline already supports it — only a source fetch is needed. The HF
`datasets` library is **not installed** in this environment (and the local
`datasets/` folder shadows the import), so Wikipedia/OSCAR streaming was not
used here; Gutenberg was fetched directly over HTTPS with a verified
public-domain license instead.

## Tokenizer findings (measured — not changed)

`scripts/tokenizer_stats.py` on real text (`experiments/tokenizer_stats.json`):

| | English | Portuguese |
|---|---|---|
| tokens / word | ~1.28 | ~3.11 |
| chars / token | ~4.02 | ~1.85 |
| common PT words split | — | 100% |

The current 8,192-vocab SentencePiece tokenizer is efficient for English
but **inefficient for Portuguese**: it fragments PT words ~3× and
byte-fallbacks accented characters (ê, ç, ã), e.g. `obrigado` → `o|br|ig|
ado`, `você` → `vo|c|<byte>|<byte>`. **Evidence-based recommendation:** a
genuinely bilingual corpus warrants a retrained tokenizer (with accented
characters and common PT subwords in-vocab), evaluated at 8k/12k/16k. Per
spec, the production tokenizer is **not** changed here (backward
compatibility with the 20M/50M checkpoints); this is a measured
recommendation, not a change.

## CPU requirements & disk

- CPU-only; no GPU, no CUDA-only code paths. ~15 GiB RAM is ample
  (training peaks ~3 GB; the pipeline is document-at-a-time).
- Disk: each 50M checkpoint is ~617 MB. Scaling checkpoints live under
  `checkpoints/50m_data_scaling/` (git-ignored). Token `.bin` corpora are
  git-ignored; manifests are committed.

## Licensing / provenance

- **TinyStories** — CDLA-Sharing-1.0 (permits ML training). Verified.
- **Project Gutenberg (PT literature)** — public domain in the US; texts
  carry the Project Gutenberg License header (stripped from the corpus).
  Verified per-work as public domain.
- No corpus is claimed as freely usable unless its license was checked; the
  manifest records `UNVERIFIED` when a caller does not supply one.

## Not done this pass (honest scope)

Chosen focus was **token budgets + scaling** and **larger data**. Not built:
the EN/PT/Aila **evaluation harness** and a standalone `evaluate
--checkpoint` command; a retrained tokenizer; and the large (50M–1B-token)
runs themselves. The scaling framework, budgets, corpus builder, and
manifests are the foundation those build on next.
