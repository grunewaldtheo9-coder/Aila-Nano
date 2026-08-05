# Datasets

Aila Nano's data is organized into three categories, each with its own
licensing story. **Nothing here is scraped ad hoc or used without a clear,
AI-training-compatible license.**

## 1. Pretraining corpora (`datasets/raw/`, `datasets/processed/`)

Not committed to the repository (see `.gitignore`) — fetched on demand by
`datasets/scripts/download_pretrain_data.py`. Two sources:

| Dataset | License | Why it's used |
|---|---|---|
| [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories) | CDLA-Sharing-1.0 | Short, simple, coherent English stories written specifically to be learnable by very small language models (Eldan & Li, 2023, *"TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"*). At Aila Nano's ~10.9M parameter scale, this is a far better fit than raw web text — it's the primary pretraining corpus. |
| [`wikitext-103-raw-v1`](https://huggingface.co/datasets/wikitext) | CC BY-SA 3.0 | Verified, encyclopedic Wikipedia text, added in a smaller proportion to broaden topical/factual coverage beyond narrative stories. |

Both licenses explicitly permit reuse (including for model training) with
attribution; CC BY-SA / CDLA-Sharing additionally require share-alike
redistribution of the *dataset itself* if redistributed — this project
does not redistribute the raw corpora, only the scripts to fetch them.

If either dataset's license or availability ever changes in a way that's
incompatible with training, `download_pretrain_data.py` is the single
place to swap in a replacement — see the module docstring for the
selection rationale so a substitute can be judged on the same criteria
(small-model suitability, license clarity, quality).

Processing pipeline (`datasets/scripts/`):

1. `download_pretrain_data.py` — streams the source datasets, avoiding a
   full local copy of upstream data dumps.
2. `clean_text.py` — Unicode normalization, control-character stripping,
   whitespace collapsing, low-quality-document filtering.
3. `dedupe.py` — exact (hash-based) and near-duplicate (shingle overlap)
   deduplication.
4. `prepare_pretrain.py` — tokenizes the cleaned corpus with a trained
   Aila Nano tokenizer and writes flat `uint16` token-id `.bin` shards
   (train/val split) for `training/dataset.py`.

## 2. Aila Company knowledge (`datasets/aila_knowledge/`)

`aila_company.jsonl` — a small, hand-written, original instruction-tuning
dataset teaching Aila Nano who it is and who built it (Aila Company
Solutions; founders Theo Grunewald Hames and Guilherme Grunewald
Benkendorf). Authored entirely for this project — no external license
concerns. Kept separate from the public pretraining/instruction data so it
can always be included in fine-tuning regardless of which public datasets
are swapped in or out.

## 3. Samples (`datasets/sample/`)

Tiny, hand-written, original text and instruction examples
(`pretrain_sample.txt`, `finetune_sample.jsonl`) used by the test suite
and for smoke-testing the training/fine-tuning pipelines without
downloading anything. Not meant to train a useful model on their own.

## Fine-tuning instruction data

Beyond `aila_knowledge/`, general instruction-tuning data should follow
the same JSONL schema (see `finetuning/format.py`):

```json
{"instruction": "...", "input": "", "output": "...", "system": "..."}
```

Good publicly-licensed sources to extend with (not bundled in this repo —
fetch and license-check independently before use):
[`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
(CC BY-SA 3.0, human-written instructions) and
[`HuggingFaceH4/no_robots`](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
(CC BY-NC 4.0 — non-commercial only, verify this fits your use case before
using it).
