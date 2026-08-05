# Architecture

## System overview

```
                         ┌─────────────────────────┐
                         │   web/frontend (Next.js) │
                         │   chat UI, dark mode,    │
                         │   history, uploads       │
                         └────────────┬─────────────┘
                                      │ HTTP / SSE
                         ┌────────────▼─────────────┐
                         │  web/backend (FastAPI)    │
                         │  routers: chat, agents,   │
                         │  memory, upload, health   │
                         └──┬───────────┬────────────┘
                             │           │
              ┌──────────────▼──┐   ┌────▼─────────────┐
              │  agents/         │   │  memory/          │
              │  (persona +      │◄──┤  conversation +   │
              │   prompt build)  │   │  long-term +      │
              └───────┬──────────┘   │  semantic memory  │
                      │              └────────┬──────────┘
                      │                       │
              ┌───────▼───────────────────────▼──────────┐
              │        model/ (AilaNanoGPT) +             │
              │        tokenizer/ (AilaTokenizer)         │
              └───────────────────┬────────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │  vectordb/ (FAISS +        │
                    │  Aila Nano's own embedder)  │
                    └─────────────────────────────┘

training/ and finetuning/ produce the checkpoints that model/ + tokenizer/
load; datasets/ produces the data training/ and finetuning/ consume.
```

Every layer above the model is built to share **one** `AilaNanoGPT`
instance: the four agent personas differ only in system prompt and
sampling defaults (`agents/base.py`), and even the vector embeddings used
for semantic search and memory retrieval come from the same model's own
hidden states (`vectordb/embedder.py`) rather than a separate embedding
model or external API.

## The model (`model/`)

Decoder-only, pre-norm transformer, Llama-family design choices scaled
down to ~10.9M parameters:

- **Tokenization → embedding**: SentencePiece BPE, vocab size 8192,
  tied input/output embeddings (halves the single largest parameter
  block: one `vocab_size × d_model` matrix instead of two).
- **Position encoding**: Rotary Position Embeddings (RoPE), not a
  learned absolute position table — zero extra parameters, encodes
  *relative* position, and generalizes better to sequence lengths not
  seen during training.
- **Attention**: grouped-query attention (GQA) — 8 query heads, 4 KV
  heads, `head_dim=32`. KV heads shared across query-head groups trims
  attention parameters vs. plain multi-head attention with negligible
  quality cost at this scale, freeing budget for more layers.
- **Normalization**: RMSNorm (pre-norm placement around attention and
  MLP), not LayerNorm — one fewer learned vector, same stability
  properties in a pre-norm transformer.
- **Feed-forward**: SwiGLU (`down(silu(gate(x)) * up(x))`), hidden dim
  ≈2.72× `d_model` (vs. the usual 4× for a 2-matrix GELU-MLP) — SwiGLU's
  third weight matrix is offset by a smaller hidden size, so it costs
  about the same parameter budget as a plain GELU-MLP while consistently
  improving loss-per-parameter (Shazeer, 2020).
- **Initialization**: `N(0, 0.02²)` for all linear/embedding weights,
  with GPT-2-style scaled init (`std / sqrt(2 × n_layers)`) on residual
  *output* projections (`o_proj`, `down_proj`) so residual-stream
  variance doesn't grow with depth.

Default (`model/config.py::nano_10m()`):

| | |
|---|---|
| vocab_size | 8192 |
| max_seq_len | 512 |
| n_layers | 12 |
| d_model | 256 |
| n_heads / n_kv_heads | 8 / 4 |
| head_dim | 32 |
| mlp_hidden_dim | 696 (SwiGLU) |
| **Total parameters** | **10,877,184 (~10.88M)** |

Run `python scripts/count_params.py` for the live breakdown by component.

### Generation (`model/generate.py`)

Autoregressive sampling with an incremental KV cache (numerically
verified identical to a full non-cached forward pass — see
`tests/test_model.py::test_kv_cache_matches_full_forward`), temperature,
top-k, top-p (nucleus), and repetition penalty. `generate_stream()`
yields one token at a time for the backend's SSE endpoint.

## Training (`training/`)

- **Data**: pretraining corpora are packed into a flat `uint16` token-id
  binary file and memory-mapped (`training/dataset.py`) — random
  contiguous windows are sampled as training examples, so nothing has to
  fit in RAM regardless of corpus size.
- **Optimizer**: AdamW with weight decay applied only to ≥2D parameters
  (weight matrices), not to norms/biases/1D params
  (`AilaNanoGPT.configure_optimizer`).
- **Schedule**: linear warmup → cosine decay (`training/scheduler.py`).
- **Mixed precision**: `torch.amp.autocast` — fp16 + `GradScaler` on
  CUDA, bf16 (no scaler needed) on CPU.
- **Robustness**: gradient clipping, gradient accumulation, periodic
  validation with early stopping, checkpoint save/prune/resume
  (`training/checkpoint.py`, atomic writes via tmp-file + rename),
  TensorBoard logging of loss/LR/grad-norm/perplexity/throughput.

## Fine-tuning (`finetuning/`)

Full fine-tuning (every parameter trainable), not LoRA/adapters — at
10.9M parameters the whole model already trains fast enough on CPU that
parameter-efficient methods would only add complexity. Data is JSONL
`{instruction, input, output, system}` (`finetuning/format.py`), formatted
into the chat token layout:

```
<s> [<|system|> ... <|end|>] <|user|> ... <|end|> <|assistant|> ... <|end|> </s>
```

The loss is masked (`label = -100`) everywhere except the assistant's
response span, so the model is never trained to predict its own prompt.
`finetuning/finetune.py` supports **continual fine-tuning**: point
`--init-checkpoint` at any prior checkpoint — pretrained or already
fine-tuned — to keep adapting the model as new instruction data arrives.

## Vector database & embeddings (`vectordb/`)

`AilaEmbedder` mean-pools Aila Nano's own final-norm hidden states
(`AilaNanoGPT.forward_hidden`) over non-padding tokens and L2-normalizes
the result — the model produces its own semantic embeddings, so cosine
similarity is just inner product. `FaissIndex` wraps a FAISS
`IndexFlatIP` inside an `IndexIDMap2` (exact search, caller-controlled
integer ids, supports deletion) — the right trade-off at the scale a
nano-model deployment realistically indexes; swap in `IndexHNSWFlat` if
an index grows past roughly a million vectors. `DocumentStore` persists
text/metadata in SQLite, keyed by the same ids as the FAISS index.
`SemanticIndex` ties the three together into add/search/delete.

## Memory (`memory/`)

Three memory types, one manager:

- **Conversation memory** — the running back-and-forth of one
  `conversation_id`, stored in SQLite, rendered back into chat turns.
- **Long-term memory** — durable facts that outlive any single
  conversation (`remember()` / `forget()`), each with an importance
  score.
- **Semantic memory** — retrieves long-term facts *by meaning*: a FAISS
  index keyed by the same ids as the long-term-facts table (no duplicated
  text storage between SQL and FAISS).

Retrieved facts are re-ranked (`memory/ranking.py`) by a weighted
combination of semantic relevance, exponential recency decay, and
stored importance — not relevance alone — so an old-but-critical fact
can still outrank a recent-but-marginal one.

`memory/manager.py::MemoryManager.build_context()` is the single call
agents make to get "everything relevant to say next": recent turns +
top-ranked relevant facts.

## Agents (`agents/`)

`Agent` (`agents/base.py`) owns prompt construction (system prompt +
retrieved memory facts + conversation history + new user turn →
token ids), generation (via `model.generate`/`generate_stream`), and
writing the turn back to memory. `GeneralAssistant`,
`ProgrammingAssistant`, `ResearchAssistant`, and `WritingAssistant`
subclass it with only a different `system_prompt` and
`default_settings` (e.g. programming uses low temperature for
determinism, writing uses high temperature for variety) — **all four
run the exact same underlying `AilaNanoGPT` weights**.

## Web backend (`web/backend/`)

FastAPI app (`web/backend/app/main.py`) that loads the model, tokenizer,
and memory/knowledge stores once at startup (`AilaState`, via a lifespan
handler) and shares them across requests through dependency injection.
Routers: `/health`, `/chat` (+ `/chat/stream` SSE), `/agents`, `/memory`,
`/upload` (chunk-and-index a text file into a knowledge base separate
from personal long-term memory). See [docs/API.md](API.md).

## Web frontend (`web/frontend/`)

Next.js (App Router) + TypeScript + Tailwind CSS. Talks to the backend
over plain fetch (`lib/api.ts`), including manual SSE parsing for
streaming (browsers' built-in `EventSource` can't send a POST body, so
the stream is read and parsed from a `fetch` `ReadableStream`).
