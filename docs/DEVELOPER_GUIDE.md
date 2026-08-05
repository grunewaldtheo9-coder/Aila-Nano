# Developer Guide

## Repository layout

```
tokenizer/     SentencePiece BPE: train, encode, decode, special tokens
model/         AilaNanoGPT architecture, config, generation
training/      Pretraining loop, dataset, scheduler, checkpointing
finetuning/    Instruction-tuning format, dataset, loop
datasets/      Download/clean/dedupe scripts, Aila knowledge, samples
vectordb/      FAISS index, Aila-Nano-powered embedder, document store
memory/        Conversation/long-term/semantic memory, ranking
agents/        Persona classes + registry, all sharing one model
engine/        Interface-independent AI core (AilaEngine, EngineSettings)
tools/         Extension point for future capabilities (empty registry today)
chat.py        The terminal interface — the only thing you run
configs/       YAML configs for model/training/finetuning
scripts/       Standalone utility scripts
tests/         pytest suite (one file per module)
docs/          You are here
checkpoints/   Trained model checkpoints (gitignored; runtime output)
```

Each top-level Python directory (`tokenizer/`, `model/`, `training/`, …)
is an independently importable package — there's no umbrella `aila_nano/`
namespace package wrapping them. This mirrors the module boundaries
described in the project's design brief directly onto the filesystem.
`vectordb/` is named that (not `faiss/`, as literally specified) to avoid
shadowing the real `faiss` PyPI package on `sys.path`, since the repo
root is importable.

There is no web/HTTP layer. `engine/` is the interface-independent AI
core; `chat.py` is the terminal interface built on it. A future
GUI/mobile/web interface would be a new, equally thin file importing
`engine.AilaEngine` — see `docs/ARCHITECTURE.md`'s roadmap section and
`docs/API.md` for that boundary.

## Setting up for development

**Linux/macOS:**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
```

**Windows:**

```bat
python -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
```

Supported/tested on Python 3.10 through 3.13.

## Running tests

```bash
pytest tests/ -v
pytest tests/ --cov=. --cov-report=term-missing   # with coverage
pytest tests/test_model.py -v                     # a single module
```

The suite trains a real (tiny) tokenizer and instantiates a real (tiny)
model in `tests/conftest.py` — nothing is mocked. It runs in a few
seconds on CPU and exercises every module: tokenizer round-tripping,
model forward/KV-cache-correctness/generation, a full training loop
(including checkpoint save/resume), fine-tuning with loss masking,
vector search, memory ranking, all four agents (including knowledge-base
retrieval), the `AilaEngine` API end to end, and `chat.py`'s command
handling.

## Linting

```bash
ruff check .          # see [tool.ruff] in pyproject.toml for the rule set
ruff check . --fix    # auto-fix what's fixable
```

## Verifying the parameter budget

```bash
python scripts/count_params.py
```

Prints the exact parameter count and a per-component breakdown for
`model.config.nano_10m()`. If you change any architecture field in
`configs/model/nano_10m.yaml`, re-run this to confirm the total is still
~10.9M — pass overrides directly, e.g.:

```bash
python scripts/count_params.py --n-layers 10 --d-model 256
```

## Adding a new agent persona

1. Create `agents/<name>_assistant.py` subclassing `agents.base.Agent`
   with a distinct `system_prompt` and (optionally) `default_settings`.
2. Register it in `agents/registry.py::AGENT_REGISTRY`.
3. Add a test in `tests/test_agents.py` following the existing pattern.

No model or training changes are needed — every agent shares the same
underlying weights. `chat.py`'s `/agents` and `/agent <name>` commands
pick up new personas automatically via `engine.available_agents()`.

## Adding a new dataset source

See [`datasets/README.md`](../datasets/README.md) and the docstring at
the top of `datasets/scripts/download_pretrain_data.py` for the
selection criteria (small-model suitability, license clarity, quality)
a replacement source should be judged against.

## Adding a future capability (search, file reading, tool calling, ...)

See the roadmap section of `docs/ARCHITECTURE.md`. The intended shape is
a `tools.base.Tool` subclass registered in a `tools.registry.ToolRegistry`
— nothing is wired into generation yet, so building one today means
implementing the tool and deciding how/when an interface (or eventually
the model itself, once function-calling training exists) invokes it.

## Commit / contribution conventions

- One logical change per commit; imperative-mood subject lines.
- Run `pytest tests/ -v` and `ruff check .` before committing.
- Update the relevant doc under `docs/` alongside any behavioral change —
  docs that drift from the code are worse than no docs.
