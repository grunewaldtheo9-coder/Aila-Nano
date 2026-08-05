# Configuration Guide

## Model config (`configs/model/*.yaml` → `model.config.GPTConfig`)

| Field | Default (`nano_10m`) | Meaning |
|---|---|---|
| `vocab_size` | 8192 | Must match the trained tokenizer's vocab size exactly |
| `max_seq_len` | 512 | Context length (also the size of the precomputed RoPE table) |
| `n_layers` | 12 | Transformer blocks |
| `d_model` | 256 | Residual stream / embedding dimension |
| `n_heads` | 8 | Query attention heads (`d_model` must be divisible by this) |
| `n_kv_heads` | 4 | Key/value heads for GQA (`n_heads` must be divisible by this; equal to `n_heads` = plain MHA) |
| `mlp_hidden_mult` | 2.72 | SwiGLU hidden dim = `round(mlp_hidden_mult × d_model / 8) × 8` |
| `dropout` | 0.1 | Applied in attention, MLP, and embeddings |
| `norm_eps` | 1e-5 | RMSNorm epsilon |
| `rope_theta` | 10000.0 | RoPE base frequency |
| `initializer_range` | 0.02 | Std-dev for weight init |
| `tie_embeddings` | true | Share input embedding and output head weights |
| `bias` | false | Linear layers have no bias terms |

Load with `GPTConfig.from_yaml(path)`, or use a preset directly:
`model.config.nano_10m()` (production) / `model.config.tiny_debug()`
(fast tests, not for real training).

## Pretraining config (`configs/training/*.yaml` → `training.trainer.TrainingConfig`)

| Field | Meaning |
|---|---|
| `train_bin` / `val_bin` | Paths to the `.bin` token shards from `prepare_pretrain.py` |
| `max_lr` / `min_lr` | Cosine schedule endpoints |
| `warmup_steps` / `max_steps` | Schedule shape; `max_steps` is also the training length |
| `weight_decay` / `betas` | AdamW hyperparameters (decay applied only to ≥2D params) |
| `grad_clip` | Global gradient-norm clip |
| `grad_accum_steps` | Micro-batches accumulated per optimizer step (effective batch = `batch_size × grad_accum_steps`) |
| `batch_size` | Per-micro-batch size |
| `eval_interval` / `eval_iters` | How often, and over how many val batches, to evaluate |
| `log_interval` | Steps between TensorBoard/log-line writes |
| `checkpoint_dir` / `keep_last_n_checkpoints` | Where checkpoints go; old rolling checkpoints are pruned |
| `early_stopping_patience` | Eval intervals without improvement before stopping; `null` disables |
| `device` | `"auto"` (CUDA if available, else CPU), `"cpu"`, or `"cuda"` |
| `amp` | Mixed precision (fp16+GradScaler on CUDA, bf16 on CPU) |
| `tensorboard_dir` | `tensorboard --logdir <this>` |

`configs/training/pretrain.yaml` — real-run defaults.
`configs/training/pretrain_smoketest.yaml` — tiny/fast, for CI and
pipeline sanity checks.

## Fine-tuning config (`configs/training/finetune.yaml` → `finetuning.finetune.FinetuneConfig`)

| Field | Meaning |
|---|---|
| `epochs` | Full passes over the instruction data |
| `batch_size` | Examples per batch (dynamically padded, not windowed) |
| `max_lr` / `min_lr` / `warmup_ratio` | Cosine schedule; warmup is a *fraction* of total steps |
| `val_fraction` | Held-out fraction of the instruction data for validation |
| `out_dir` | Where `epoch_NNN.pt` and `best.pt` are written |

## Backend environment variables (`web/backend/app/deps.py::AilaSettings`)

| Variable | Default | Meaning |
|---|---|---|
| `AILA_CHECKPOINT` | `checkpoints/finetune/best.pt` | Preferred checkpoint to serve |
| `AILA_FALLBACK_CHECKPOINT` | `checkpoints/pretrain/best.pt` | Used if the above doesn't exist |
| `AILA_TOKENIZER` | `tokenizer/artifacts/aila_nano.model` | Must match the checkpoint's tokenizer |
| `AILA_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `AILA_MEMORY_DB` / `AILA_MEMORY_FAISS` | `memory/data/aila_memory.{db,faiss}` | Conversation + long-term memory storage |
| `AILA_KNOWLEDGE_DB` / `AILA_KNOWLEDGE_FAISS` | `vectordb/index/knowledge.{db,faiss}` | Uploaded-document knowledge base |
| `AILA_CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |

If neither checkpoint path exists, the server logs a warning and serves a
freshly-initialized, untrained model (vocab size taken from the loaded
tokenizer) so the API/UI remain explorable before training completes.

## Frontend environment variables (`web/frontend/.env.local`)

| Variable | Default | Meaning |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend base URL |
