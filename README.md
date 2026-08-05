# Aila Nano

**An original, from-scratch small language model — built, trained, and owned by [Aila Company Solutions](docs/MODEL_CARD.md#aila-company-solutions).**

Aila Nano is a decoder-only transformer with **~10.9 million parameters**
(10,877,184 exactly at the default config — see
[`scripts/count_params.py`](scripts/count_params.py)), designed to learn
language, answer questions, generate and continue text, be fine-tuned on
new data, and run entirely locally — CPU or GPU, no external AI API
required.

Created by **Theo Grunewald Hames** and **Guilherme Grunewald Benkendorf**.

> Aila Nano is not a wrapper around OpenAI, Anthropic, Google, or any other
> company's model. Every component below — tokenizer, architecture,
> training loop, embeddings — is implemented in this repository.

## Why "Nano"

At ~10.9M parameters, Aila Nano is roughly 1,000x smaller than GPT-2 Small
class models were once considered "small." That's a deliberate design
choice, not a limitation to apologize for: it trains in minutes on a
laptop CPU, fine-tunes in seconds, runs inference without a GPU, and is
small enough that every architectural decision below can be reasoned
about and inspected line-by-line rather than taken on faith. See
[docs/MODEL_CARD.md](docs/MODEL_CARD.md) for what that scale does and
doesn't make Aila Nano good at.

## What's in this repository

| Module | Purpose |
|---|---|
| [`tokenizer/`](tokenizer/) | SentencePiece BPE tokenizer: train, encode, decode, save/load |
| [`model/`](model/) | The decoder-only GPT architecture (RoPE, GQA, SwiGLU, RMSNorm) |
| [`training/`](training/) | Pretraining loop: data loading, AMP, checkpointing, resume, TensorBoard |
| [`finetuning/`](finetuning/) | Instruction-tuning on JSONL data, with prompt/loss masking |
| [`datasets/`](datasets/) | Dataset download/clean/dedupe scripts + the Aila Company knowledge set |
| [`vectordb/`](vectordb/) | FAISS-backed semantic search, powered by Aila Nano's own embeddings |
| [`memory/`](memory/) | Conversation, long-term, and semantic memory with relevance ranking |
| [`agents/`](agents/) | Four assistant personas sharing one model (General/Programming/Research/Writing) |
| [`web/backend/`](web/backend/) | FastAPI server: chat (incl. streaming), agents, memory, uploads |
| [`web/frontend/`](web/frontend/) | Next.js chat UI: dark mode, history, streaming, file upload |
| [`configs/`](configs/) | YAML configs for the model and training/fine-tuning runs |
| [`scripts/`](scripts/) | Utility scripts (parameter counting, etc.) |
| [`tests/`](tests/) | Pytest suite covering every module above |

## Quickstart

```bash
# 1. Install
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the cuXXX index for GPU
pip install -r requirements.txt

# 2. Train a tokenizer. The bundled sample corpus is tiny (a few dozen
#    lines, just enough to smoke-test the pipeline), so it can only
#    support a small vocab — train on a real corpus (docs/TRAINING.md)
#    with vocab_size=8192 for an actual model.
python -c "
from tokenizer import train_tokenizer
train_tokenizer('datasets/sample/pretrain_sample.txt', 'tokenizer/artifacts/aila_nano_sample', vocab_size=512)
"

# 3. Check the (real, production-sized) model hits the ~10.9M parameter target
python scripts/count_params.py

# 4. Prepare data and pretrain a tiny demo-sized model on the sample
#    corpus, just to prove the pipeline end-to-end (see docs/TRAINING.md
#    for real-corpus + real nano_10m-config instructions — this step uses
#    configs/model/quickstart_demo.yaml, not the production architecture)
python datasets/scripts/prepare_pretrain.py \
    --input datasets/sample/pretrain_sample.txt \
    --tokenizer tokenizer/artifacts/aila_nano_sample.model \
    --out-dir datasets/processed \
    --val-fraction 0.15
python -m training.train \
    --model-config configs/model/quickstart_demo.yaml \
    --train-config configs/training/pretrain_smoketest.yaml

# 5. Instruction-tune, including Aila's own knowledge about itself
#    (checkpoints/smoketest/ is where pretrain_smoketest.yaml writes to —
#    a real run with configs/training/pretrain.yaml writes to
#    checkpoints/pretrain/ instead, see docs/TRAINING.md)
python -m finetuning.finetune \
    --init-checkpoint checkpoints/smoketest/best.pt \
    --tokenizer tokenizer/artifacts/aila_nano_sample.model \
    --data datasets/aila_knowledge/aila_company.jsonl datasets/sample/finetune_sample.jsonl

# 6. Serve it
uvicorn web.backend.app.main:app --reload --port 8000

# 7. Run the web UI
cd web/frontend && npm install && npm run dev
```

Full walkthroughs: [docs/INSTALL.md](docs/INSTALL.md) ·
[docs/TRAINING.md](docs/TRAINING.md).

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the model, training, memory, and serving layers fit together
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md) — parameters, intended use, limitations, Aila Company Solutions
- [docs/TRAINING.md](docs/TRAINING.md) — pretraining and fine-tuning guide
- [docs/INSTALL.md](docs/INSTALL.md) — installation (CPU/GPU/Docker)
- [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) — repo layout, testing, contributing
- [docs/API.md](docs/API.md) — FastAPI backend endpoint reference
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) — every config file and env var
- [datasets/README.md](datasets/README.md) — dataset sources and licenses

## License

Apache License 2.0 — see [LICENSE](LICENSE). Copyright Aila Company
Solutions.
