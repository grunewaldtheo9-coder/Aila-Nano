# Bilingual tokenizer (v2) — why, how, and the evidence

## Why the tokenizer needed to change

The production tokenizer (`tokenizer/artifacts/aila_nano.model`, 8,192-vocab
SentencePiece BPE) was trained on an English-only corpus with
`character_coverage=0.9995`. Measured on held-out text, it is efficient for
English but badly inefficient for Portuguese:

| | English | Portuguese |
|---|---|---|
| tokens / word | 1.18 | **2.85** |
| byte-fallback rate | ~0 | **9.6%** |
| accented chars in-vocab (ã õ á é í ó ú â ê ô ç) | — | **1 of 11** |
| common PT words split | — | **100%** |

`obrigado` → `o|br|ig|ado`, `você` → `vo|c|<byte>|<byte>`. The accented
characters were not in the vocabulary at all, so they fell to byte-fallback.
This wastes context and makes Portuguese ~2–3× more expensive to model —
a real bottleneck, not a cosmetic one.

## What we built (measurement, not vibes)

A candidate-training + benchmark pipeline, evaluated on **held-out** EN/PT
text that never entered training:

- `scripts/build_tokenizer_corpus.py` — a balanced EN+PT text corpus
  (**~50% PT / 50% EN** by bytes; configurable `--pt-fraction` so English
  can't dominate the vocabulary), with disjoint held-out EN/PT eval files.
- `scripts/train_tokenizer_candidates.py` — trains 8K/12K/16K candidates
  with `character_coverage=1.0` into `tokenizer/artifacts/candidates/`.
  **The production tokenizer is never touched.**
- `scripts/tokenizer_benchmark.py` — measures tokens/word, tokens/char,
  tokens/sentence, byte-fallback rate, accented-char handling, PT word
  fragmentation, and vocab utilization; writes
  `experiments/tokenizer_benchmark.json` + `experiments/TOKENIZER_REPORT.md`
  and selects by a transparent score (PT weighted 2×, EN 1×, byte-fallback
  penalized).

### Measured comparison (Phase-11 table)

| Tokenizer | Vocab | EN tok/word | PT tok/word | PT byte-fallback | PT word-split | Accents in-vocab |
|---|---|---|---|---|---|---|
| aila_nano (production) | 8192 | 1.177 | 2.850 | 9.58% | 100% | 9% (1/11) |
| bilingual_8192 | 8192 | 1.238 | 1.566 | 0.005% | 53% | 100% |
| bilingual_12288 | 12288 | 1.205 | 1.469 | 0.005% | 47% | 100% |
| **bilingual_16384 (chosen)** | 16384 | 1.190 | **1.414** | 0.005% | 47% | **100%** |

**Portuguese fragmentation roughly halved (2.85 → 1.41 tok/word), byte-fallback
effectively eliminated, all 11 target accents in-vocabulary, and English
essentially unchanged (+1.1%).** The 16K candidate wins on PT efficiency at
negligible EN cost.

## The v2 tokenizer and backward compatibility

The chosen candidate is promoted (not swapped) to a **new versioned
artifact**: `tokenizer/artifacts/v2_bilingual/aila_nano_v2_bilingual.model`
(vocab 16,384) with a manifest recording sources, license, corpus, model
hash, and git commit.

**Nothing existing is changed or broken:**
- `tokenizer/artifacts/aila_nano.model` (8,192) remains the production
  tokenizer for **all existing 20M and 50M checkpoints**.
- v2 has `vocab_size=16384`, so it is for **future v2 checkpoints only** —
  a model must be (pre)trained with it, since its embedding table is a
  different size. `validate_checkpoint_compatibility` already rejects a
  tokenizer/checkpoint vocab mismatch with a clear message, so an old
  checkpoint can never silently load the new tokenizer (or vice-versa).

## Evaluation harness

`evaluation/` measures a checkpoint's **per-language** capability, kept
separate from Aila-specific behaviour (which the deterministic router and
its own tests own):

```bash
python -m evaluation.evaluate --checkpoint checkpoints/50m/finetune/best.pt \
    --tokenizer tokenizer/artifacts/aila_nano.model
```

It reports English vs Portuguese perplexity on held-out sets
(`evaluation/data/eval_{en,pt}.txt`, committed, never trained on) plus a
repetition/degradation check on short generations. Measured on the current
50M finetune checkpoint:

| | English | Portuguese |
|---|---|---|
| perplexity | 129.3 | **2599.7** |
| tokens (10 held-out sentences) | 152 | 343 |

The current model is ~20× worse on Portuguese than English (and PT needs
2.3× the tokens for the same sentences — the fragmentation showing through).
This is the honest baseline a future bilingual v2 model will be measured
against.

## Reproduce

```bash
# 1. Portuguese public-domain text (Project Gutenberg), verified pt by langid
#    (already staged under datasets/raw/pt_tok/, git-ignored)
# 2. Balanced EN+PT tokenizer corpus + held-out eval
python scripts/build_tokenizer_corpus.py \
  --pt-input "datasets/raw/pt_tok/*.txt" --en-input datasets/raw/tinystories.txt \
  --out-dir datasets/tokenizer_corpus --pt-fraction 0.5
# 3. Train candidates (production tokenizer untouched)
python scripts/train_tokenizer_candidates.py \
  --corpus datasets/tokenizer_corpus/tokenizer_train.txt --vocab-sizes 8192,12288,16384
# 4. Benchmark + select
python scripts/tokenizer_benchmark.py \
  --candidates "tokenizer/artifacts/candidates/*/*.model" \
  --eval-en datasets/tokenizer_corpus/eval_en.txt --eval-pt datasets/tokenizer_corpus/eval_pt.txt
```

## Sources & provenance

- **English:** TinyStories — CDLA-Sharing-1.0 (verified).
- **Portuguese:** 8 Project Gutenberg public-domain works (Machado de Assis,
  José de Alencar, Aluísio Azevedo, Camões…), langid-verified `pt`. Classic
  **literary** register (mostly PT-BR plus *Os Lusíadas* PT-PT). This is
  enough to fix the *tokenizer* (accents + morphology), but it is **not** a
  broad Brazilian-Portuguese web corpus — that (Wikipedia/OSCAR, aggressively
  filtered) is the next data addition and the pipeline already supports it.

## What is NOT done (honest scope)

- No v2 **model** was trained (a new 16K-vocab model must be pretrained from
  scratch — multi-day on this 4-core CPU). The tokenizer, corpus, eval
  harness, and metadata are the foundation for that run; it is **not** run
  here, and no results are claimed for it.
- The production tokenizer and default model are unchanged. No 120M.

## CPU / disk

Tokenizer training is seconds on CPU; the benchmark is trivial. Downloads
were bounded Gutenberg texts (~4 MB), processed then kept under
`datasets/raw/` (git-ignored). Candidate `.model` binaries are git-ignored
(regenerable); the chosen v2 model (~513 KB) is committed.
