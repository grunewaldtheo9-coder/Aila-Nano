# Training Guide

## Overview

Training Aila Nano is four steps: **tokenizer → pretraining data →
pretraining → fine-tuning**. Each step's output is a plain artifact
(a `.model` file, `.bin` token shards, a `.pt` checkpoint) that the next
step reads, so any step can be re-run independently.

## 1. Train the tokenizer

```python
from tokenizer import train_tokenizer

train_tokenizer(
    input_files=["datasets/raw/tinystories.txt", "datasets/raw/wikitext.txt"],
    model_prefix="tokenizer/artifacts/aila_nano",
    vocab_size=8192,
)
```

Train it on the same corpus (or a representative sample of it) you're
about to pretrain on — this is a one-time step per corpus; retraining the
tokenizer later invalidates any checkpoint trained against the old one
(the vocabulary, and therefore the embedding matrix, changes).

## 2. Get pretraining data

```bash
pip install -r requirements-datasets.txt
python datasets/scripts/download_pretrain_data.py --out-dir datasets/raw
```

Downloads, cleans, and deduplicates TinyStories + Wikitext-103 (see
[`datasets/README.md`](../datasets/README.md) for why these two).
Near-duplicate detection (`datasets/scripts/dedupe.py::near_dedupe`) is
O(n²) and will effectively hang on a large corpus, so it's automatically
skipped (exact-duplicate removal still runs) once a source has more than
`--max-near-dedupe-docs` (default 20,000) documents — pass
`--force-near-dedupe` to run it anyway if you're willing to wait. Swap in
your own corpus by pointing `prepare_pretrain.py` (next step) at any
UTF-8 text file(s) instead — one document per paragraph (blank-line
separated) works best, since the script inserts `<s>`/`</s>` at document
boundaries.

```bash
python datasets/scripts/prepare_pretrain.py \
    --input datasets/raw/tinystories.txt datasets/raw/wikitext.txt \
    --tokenizer tokenizer/artifacts/aila_nano.model \
    --out-dir datasets/processed \
    --val-fraction 0.02
```

Writes `datasets/processed/pretrain_{train,val}.bin` — flat `uint16`
token-id arrays, memory-mapped by the trainer so corpus size is never
bounded by RAM.

## 3. Pretrain

```bash
python -m training.train \
    --model-config configs/model/nano_10m.yaml \
    --train-config configs/training/pretrain.yaml
```

Key knobs in `configs/training/pretrain.yaml` (see
[docs/CONFIGURATION.md](CONFIGURATION.md) for the full field reference):

- `batch_size` / `grad_accum_steps` — effective batch size is their
  product; raise `grad_accum_steps` instead of `batch_size` if you hit an
  out-of-memory error. **On CPU this matters more than it sounds**: without
  a flash-attention kernel, PyTorch's CPU attention keeps the full
  `(batch, heads, seq_len, seq_len)` score tensor around for backward, so
  memory scales roughly linearly with `batch_size`. Measured on
  `nano_10m` (`max_seq_len=512`): batch 4 → ~2.4 GB peak RSS, batch 8 →
  ~4.6 GB, batch 16 → ~8.9 GB, batch 32 → OOM on a 15 GB machine. The
  shipped default (`batch_size=4`, `grad_accum_steps=16`, effective batch
  64) targets ~2.5 GB peak, safe on modest hardware; raise `batch_size`
  and lower `grad_accum_steps` proportionally if you have RAM (or a GPU)
  to spare.
- `max_steps` / `warmup_steps` — the cosine schedule decays from `max_lr`
  to `min_lr` over `[warmup_steps, max_steps]`.
- `eval_interval` / `early_stopping_patience` — training stops early if
  validation loss hasn't improved for `early_stopping_patience`
  consecutive evaluations; set to `null` to disable.

Monitor with TensorBoard:

```bash
tensorboard --logdir runs/pretrain
```

**Resume** an interrupted run:

```bash
python -m training.train --train-config configs/training/pretrain.yaml --resume
# or from a specific checkpoint:
python -m training.train --train-config configs/training/pretrain.yaml \
    --resume-from checkpoints/pretrain/step_0012000.pt
```

Checkpoints land in `checkpoint_dir` (default `checkpoints/pretrain/`):
a rolling `step_XXXXXXX.pt` (for resume; only the last
`keep_last_n_checkpoints` are kept) and `best.pt` (lowest validation
loss seen so far — this is what you fine-tune from).

For a fast correctness check of the whole pipeline without a real corpus,
use `configs/training/pretrain_smoketest.yaml` against
`datasets/sample/pretrain_sample.txt` — runs to completion in seconds on
CPU.

## 4. Fine-tune

```bash
python -m finetuning.finetune \
    --init-checkpoint checkpoints/pretrain/best.pt \
    --tokenizer tokenizer/artifacts/aila_nano.model \
    --data datasets/aila_knowledge/aila_company.jsonl datasets/sample/finetune_sample.jsonl \
    --config configs/training/finetune.yaml
```

Always include `datasets/aila_knowledge/aila_company.jsonl` so the model
retains knowledge of who built it. Add your own JSONL files (same
`{instruction, input, output, system}` schema — see
[`finetuning/format.py`](../finetuning/format.py)) for any additional
behavior you want to teach it.

**Continual fine-tuning**: point `--init-checkpoint` at a previous
fine-tune checkpoint (not just a pretrained one) to keep adapting the
model as new instruction data becomes available, without starting over:

```bash
python -m finetuning.finetune \
    --init-checkpoint checkpoints/finetune/best.pt \
    --tokenizer tokenizer/artifacts/aila_nano.model \
    --data datasets/new_instructions.jsonl \
    --out-dir checkpoints/finetune_v2
```

## 5. Chat with it

```bash
AILA_CHECKPOINT=checkpoints/finetune/best.pt \
AILA_TOKENIZER=tokenizer/artifacts/aila_nano.model \
python chat.py
```

See [docs/CONFIGURATION.md](CONFIGURATION.md) for all `AILA_*` env vars.
