# Aila Nano Beta

**An original, from-scratch small language model — built, trained, and owned by [Aila Company Solutions](docs/MODEL_CARD.md#aila-company-solutions).**

*Beta: a working early release, not a finished product. It is genuinely
usable and genuinely rough in places — please report problems to
mailailacompanysolutions@gmail.com, or type `/support` in the chat.*

Aila Nano is a decoder-only transformer. Two architecture sizes exist —
`nano_10m` at ~10.9M parameters (10,877,184) and the shipped `nano_20m`
at ~19.8M parameters (19,796,160), both measured programmatically via
[`scripts/count_params.py`](scripts/count_params.py) — designed to learn
language, answer questions, generate and continue text, be fine-tuned on
new data, and run entirely locally — CPU or GPU, no external AI API
required for the model itself.

Around the model sits an external intelligence layer:
long-term user memory, a global knowledge base with validation and
deduplication, optional web research through Serper (with caching,
source ranking, and prompt-injection defense), an exact-arithmetic
calculator, and a deterministic tool router that decides per question
whether to answer from memory, knowledge, the web, arithmetic — or
generate.

The architectural bet: at this scale the model is a decent text
generator but an unreliable fact repeater, so **everything the system
knows exactly is answered by code, and generation is reserved for
open-ended language.** Ask it `12 * 8` and you get exactly `96`; tell it
your name and ask for it back and you get your name, verbatim, every
time. User memory and global knowledge are kept strictly separate, so
one user's facts can never surface for another.

Created by **Theo Grunewald Hames** and **Guilherme Grunewald Benkendorf**.

> Aila Nano is not a wrapper around OpenAI, Anthropic, Google, or any other
> company's model. Every component below — tokenizer, architecture,
> training loop, embeddings — is implemented in this repository.

## Terminal-first

Aila Nano runs entirely in one terminal window. No server, no browser, no
frontend build step:

```bash
python chat.py
```

That's it. It loads the tokenizer, model, memory, and agents, and drops
you into a chat prompt. See [Quickstart](#quickstart) below for the setup
steps that come before it.

The AI itself (`engine.AilaEngine`) is kept completely independent of
`chat.py` — the terminal is simply today's interface to it. A future
desktop app, mobile app, or web service would sit on top of the exact
same engine with no changes to the model, memory, or agents. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for that boundary.

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
| [`engine/`](engine/) | The interface-independent AI core — loads everything above, exposes `chat()`/`chat_stream()` |
| [`knowledge/`](knowledge/) | Global knowledge base: validated facts, candidates, web cache, dedup/conflict handling |
| [`webresearch/`](webresearch/) | Serper search client, source ranking, sanitization, research pipeline |
| [`tools/`](tools/) | Tool router + calculator; extension point for future capabilities |
| [`chat.py`](chat.py) | The terminal chat interface — the only thing you run |
| [`configs/`](configs/) | YAML configs for the model and training/fine-tuning runs |
| [`scripts/`](scripts/) | Utilities: parameter counting, benchmarking, checkpoint comparison, training supervisor |
| [`tests/`](tests/) | Pytest suite covering every module above |

## Quickstart

```bash
# 1. Install
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
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

# 6. Talk to it
AILA_CHECKPOINT=checkpoints/finetune/best.pt \
AILA_TOKENIZER=tokenizer/artifacts/aila_nano_sample.model \
python chat.py
```

```
=====================================
Aila Nano Beta
Small Language Model
Python 3.13
CPU Mode

Loading tokenizer...
OK
Loading model...
OK
Loading memory...
OK
Loading FAISS...
OK
Loading knowledge base...
OK
Loading agents...
OK
Ready.

You: What is 12 * 8?
Aila: The answer is 96.

You: remember that my name is Theo
Aila: Got it — I'll remember that my name is Theo.

You: What is my name?
Aila: Your name is Theo.

You: Who founded Apple?
Aila: Apple was founded by Steve Jobs, Steve Wozniak and Ronald Wayne in 1976.
```

(That last one comes from the knowledge base once researched — the same
question, however rephrased, is answered offline afterwards.)

Type `exit` or `quit` to leave, `/help` to see the rest of the commands
(switching agents, `/remember` `/forget` `/memories`, indexing a file
into the knowledge base, `/study <topic>` to learn something on the
spot, `/knows` to see how much she has learned, and `/support` /
`/feedback` to report a problem to Aila Company Solutions).

Aila looks things up on **Wikipedia** — free, no API key, no account —
and on Google via serper.dev if you set `SERPER_API_KEY`. Everything she
researches is stored, so it answers instantly and works with no
internet afterwards. Once a day at startup she also goes back and
studies questions she previously failed to answer. Plain-language commands work too:
"remember that ...", "forget that ...", "what do you remember about me?"

Full walkthroughs: [docs/INSTALL.md](docs/INSTALL.md) ·
[docs/TRAINING.md](docs/TRAINING.md).

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the model, training, memory, and engine layers fit together
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md) — parameters, intended use, limitations, Aila Company Solutions
- [docs/TRAINING.md](docs/TRAINING.md) — pretraining and fine-tuning guide
- [docs/INSTALL.md](docs/INSTALL.md) — installation (CPU/GPU/Docker)
- [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) — repo layout, testing, contributing
- [docs/API.md](docs/API.md) — `AilaEngine` Python API reference (for building new interfaces)
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) — every config file and env var
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — measured 1.0 vs 2.0 comparison, honestly interpreted
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — real failure modes and how to fix them
- [docs/GPU_TRAINING.md](docs/GPU_TRAINING.md) — training on a rented cloud GPU (costs, setup, configs)
- [datasets/README.md](datasets/README.md) — dataset sources and licenses

## License

Apache License 2.0 — see [LICENSE](LICENSE). Copyright Aila Company
Solutions.
