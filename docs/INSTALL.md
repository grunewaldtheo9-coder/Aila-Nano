# Installation

## Requirements

- Python 3.10–3.13 (developed and tested against 3.11 and 3.13)
- ~2 GB disk for Python dependencies (PyTorch is the bulk of it)
- A CUDA GPU is optional — everything in this repo runs on CPU, just slower

That's it. There is no separate frontend build, no Node.js, no browser —
Aila Nano is a single Python program.

## 1. Set up the environment

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell or cmd):**

```bat
python -m venv .venv
.venv\Scripts\activate
```

## 2. Install dependencies

```bash
# Install PyTorch first, matching your hardware:
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU-only
# pip install torch --index-url https://download.pytorch.org/whl/cu121 # CUDA 12.1

pip install -r requirements.txt          # core runtime
pip install -r requirements-dev.txt      # + testing/linting (optional)
pip install -r requirements-datasets.txt # + Hugging Face `datasets`, for downloading real pretraining corpora (optional)
```

Every package above is verified to install cleanly (no source builds, no
manual fixes) on a fresh Python 3.10–3.13 environment. If you hit a
build error on an unusual platform, it's almost certainly a missing
prebuilt wheel for that specific platform/Python combination for one of
`torch`, `faiss-cpu`, or `sentencepiece` — check that package's PyPI page
for supported platforms before filing an issue here.

Verify the install:

```bash
python -c "import torch, sentencepiece, faiss; print('OK, CUDA:', torch.cuda.is_available())"
python scripts/count_params.py   # should print ~10.88M total parameters
```

## 3. Run it

```bash
python chat.py
```

If no trained checkpoint exists yet, `chat.py` still starts and lets you
chat — with an untrained (gibberish-output) model, and a note telling you
so. See [docs/TRAINING.md](TRAINING.md) to actually train one.

## 4. Docker

```bash
docker build -t aila-nano .
docker run -it --rm \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/tokenizer/artifacts:/app/tokenizer/artifacts" \
    -v "$(pwd)/memory/data:/app/memory/data" \
    -v "$(pwd)/vectordb/index:/app/vectordb/index" \
    aila-nano
```

`-it` is required — `chat.py` is an interactive terminal program. The
volume mounts mean a model trained on the host is immediately usable
inside the container, and memory/knowledge-base data persist across
container restarts. For a CUDA image, see the comment at the top of
`Dockerfile`.

## GPU notes

Nothing in the codebase branches on CPU vs. GPU except device selection
(`device: "auto"` in the training/engine configs picks CUDA if
available). Multi-GPU (`DistributedDataParallel`) is not wired up yet —
`training/trainer.py`'s docstring notes where to add it — but the single
biggest lever for training speed is simply having *a* GPU: `torch.amp`
mixed precision is already enabled by default.
