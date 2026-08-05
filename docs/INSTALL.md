# Installation

## Requirements

- Python 3.10+
- Node.js 18+ (for the web frontend)
- ~2 GB disk for Python dependencies (PyTorch is the bulk of it)
- A CUDA GPU is optional — everything in this repo runs on CPU, just slower

## 1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install PyTorch first, matching your hardware:
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU-only
# pip install torch --index-url https://download.pytorch.org/whl/cu121 # CUDA 12.1

pip install -r requirements.txt          # core runtime
pip install -r requirements-dev.txt      # + testing/linting (optional)
pip install -r requirements-datasets.txt # + Hugging Face `datasets`, for downloading real pretraining corpora (optional)
```

Verify the install:

```bash
python -c "import torch, sentencepiece, faiss, fastapi; print('OK, CUDA:', torch.cuda.is_available())"
python scripts/count_params.py   # should print ~10.88M total parameters
```

## 2. Web frontend

```bash
cd web/frontend
npm install
cp .env.local.example .env.local   # NEXT_PUBLIC_API_URL, defaults to http://localhost:8000
```

## 3. Running everything

```bash
# Terminal 1 — backend (loads a checkpoint if one exists at
# checkpoints/finetune/best.pt or checkpoints/pretrain/best.pt, otherwise
# serves an untrained model so the API/UI are still explorable)
source .venv/bin/activate
uvicorn web.backend.app.main:app --reload --port 8000

# Terminal 2 — frontend
cd web/frontend && npm run dev
```

Open http://localhost:3000. See [docs/TRAINING.md](TRAINING.md) to
actually train a model before expecting coherent output.

## 4. Docker

```bash
docker compose up --build
```

Starts the backend on :8000 and frontend on :3000. `checkpoints/`,
`tokenizer/artifacts/`, `memory/data/`, and `vectordb/index/` are
bind-mounted so a model trained on the host is immediately usable inside
the container, and memory/vector data persist across container restarts.
For a CUDA backend image, see the comment at the top of `Dockerfile`.

## GPU notes

Nothing in the codebase branches on CPU vs. GPU except device selection
(`device: "auto"` in the training/serving configs picks CUDA if
available). Multi-GPU (`DistributedDataParallel`) is not wired up yet —
`training/trainer.py`'s docstring notes where to add it — but the single
biggest lever for training speed is simply having *a* GPU: `torch.amp`
mixed precision is already enabled by default.
