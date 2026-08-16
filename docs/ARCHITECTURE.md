# Architecture

## System overview

```
                    ┌───────────────────────────────┐
                    │  chat.py (terminal interface)   │
                    │  banner, REPL loop, /commands   │
                    └────────────────┬─────────────────┘
                                     │ Python calls, in-process
                    ┌────────────────▼─────────────────┐
                    │  engine.AilaEngine                │
                    │  (interface-independent AI core)  │
                    └──┬───────────┬──────────┬─────────┘
                        │           │          │
              ┌──────────▼──┐  ┌─────▼─────┐  ┌▼──────────┐
              │  agents/     │  │  memory/  │  │ vectordb/ │
              │  (persona +  │◄─┤ conv. +   │  │ FAISS +   │
              │  prompt build│  │ long-term │  │ Aila's own│
              │  + tools)    │  │ + semantic│  │ embeddings│
              └──────┬───────┘  └───────────┘  └───────────┘
                     │
        ┌─────────────▼─────────────────────────────┐
        │   model/ (AilaNanoGPT) + tokenizer/         │
        │   (AilaTokenizer)                            │
        └───────────────────────────────────────────────┘

training/ and finetuning/ produce the checkpoints that engine/ loads;
datasets/ produces the data training/ and finetuning/ consume.
```

**The AI engine is completely independent of any interface.**
`engine.AilaEngine` owns the tokenizer, model, memory, knowledge index,
and agents — it has no idea `chat.py` (a terminal loop) exists. `chat.py`
is a thin, disposable shell: it prints a banner, reads lines, and calls
`engine.chat_stream(...)`. A future desktop GUI, mobile app, or web
service would construct the same `AilaEngine` and call the same methods
(`chat`, `chat_stream`, `get_agent`, `learn_file`) — nothing below that
boundary changes. See `engine/state.py` for the full surface.

Every layer above the model shares **one** `AilaNanoGPT` instance: the
four agent personas differ only in system prompt and sampling defaults
(`agents/base.py`), and even the vector embeddings used for semantic
search and memory retrieval come from the same model's own hidden states
(`vectordb/embedder.py`) rather than a separate embedding model or
external API.

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
yields one token at a time — this is what powers `chat.py` printing
Aila's reply as it's generated instead of waiting for the whole thing.

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
If an example's response can't fit inside `max_seq_len` even after
front-truncating the prompt, `finetuning/format.py::encode_example`
drops that example entirely (`finetuning/dataset.py` logs how many were
skipped) rather than silently training on an all-masked, zero-signal
example — an earlier version of this truncation logic could produce a
`nan` loss this way; see `tests/test_finetuning.py` for the regression
tests. `finetuning/finetune.py` supports **continual fine-tuning**: point
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
`vectordb/chunking.py` splits a document's text into overlapping windows
before indexing — used by `AilaEngine.learn_file` (the terminal's
`/learn` command).

## Memory (`memory/`)

Memory is stored entirely *outside* the model's weights — nothing in
`memory/` requires (or benefits from) fine-tuning. Every fact lives in
SQLite (`memory/store.py`) with a FAISS vector alongside it
(`memory/semantic_memory.py`, same integer ids as the SQL rows, no
duplicated text storage); the model only ever sees a fact if it gets
explicitly injected into that turn's system prompt as text.

Three memory types, one manager (`memory/manager.py::MemoryManager`):

- **Conversation memory** — the running back-and-forth of one
  `conversation_id`, stored in SQLite, rendered back into chat turns.
  (Not currently replayed into the generation prompt — see
  `agents/base.py::Agent._build_prompt_ids`'s comment on why: the model
  has never been fine-tuned on multi-turn examples, and doing so
  measurably knocked it off the assistant persona in practice.)
- **Long-term memory** — durable facts that outlive any single
  conversation. Each fact has an `id`, optional `session_id` (`None` =
  global, visible to every session — the right default for a
  single-user install), `content`, `category` (`identity`,
  `preference`, `personal_fact`, `project`, `instruction`, or `other`),
  `importance`, `created_at`, and `updated_at`. Full CRUD:
  `add_memory` / `get_memory` / `update_memory` / `delete_memory` /
  `clear_memories` / `all_memories`.
- **Semantic memory** — two different retrieval calls, deliberately
  different in how strict they are:
  - `search_memories(query)` — broad FAISS (embedding) search, ranked by
    a weighted combination of semantic relevance, exponential recency
    decay, and stored importance (`memory/ranking.py`). Good for an
    explicit "search my memories" utility; not gated by any hard cutoff.
  - `get_relevant_memories(query)` — the one that actually feeds prompt
    injection. Gated by **deterministic lexical word-overlap**
    (`memory/lexical.py`), not the raw embedding score: Aila Nano's own
    embeddings (`vectordb/embedder.py`) come from a small, imperfectly
    trained encoder, fine for loosely ranking broad search results but
    not something to bet "never leak an irrelevant memory" on. A fact
    that doesn't clear the overlap threshold is never returned — empty
    result is a real, reachable state, which is what lets the system
    prompt correctly have *no* memory section for an unrelated question
    instead of injecting the closest-but-wrong fact anyway.

`memory/manager.py::MemoryManager.build_context()` is the single call
agents make to get "everything relevant to say next": recent turns +
relevance-gated facts. Knowledge-base hits (files indexed via `/learn`)
are looked up separately, directly against `vectordb.SemanticIndex`, by
`agents/base.py` — see below.

**Explicit commands** (`memory/commands.py`) — "remember that X",
"forget that X" / "forget about X", "what do you remember about me?" —
are recognized by regex, not by the model: `Agent._handle_memory_command`
checks every incoming message *before* touching the model at all, and a
match is handled deterministically (store / delete-if-matched /
list-and-return) without ever calling `generate()`. This is the one
guaranteed-zero-hallucination path in the system. `forget` requires a
stronger lexical match (`agents/base.py::FORGET_MATCH_THRESHOLD`) than
ordinary retrieval before it deletes anything, since deleting is
destructive — an unmatched "forget" says so instead of guessing.
`memory/commands.py::guess_category` picks a category via keyword
heuristics when one isn't given explicitly.

## Agents (`agents/`)

`Agent` (`agents/base.py`) owns prompt construction (system prompt +
relevance-gated memory facts, wrapped in a `[MEMORY]...[/MEMORY]` block
only when at least one fact clears the relevance threshold + retrieved
knowledge-base snippets + new user turn → token ids), generation (via
`model.generate`/`generate_stream`), and writing the turn back to memory.
`GeneralAssistant`, `ProgrammingAssistant`, `ResearchAssistant`, and
`WritingAssistant` subclass it with only a different `system_prompt` and
`default_settings` (e.g. programming uses low temperature for
determinism, writing uses high temperature for variety) — **all four run
the exact same underlying `AilaNanoGPT` weights**.

## The engine (`engine/`)

`AilaEngine` (`engine/state.py`) is the single object that ties
everything above into something an interface can use: it loads the
tokenizer and model (falling back to a freshly-initialized, untrained
model matching the tokenizer's vocab size if no checkpoint exists yet),
constructs the memory manager and knowledge index, and eagerly
constructs every registered agent so a misconfigured persona fails fast
at startup. `EngineSettings` (`engine/config.py`) resolves every path
(checkpoint, tokenizer, memory/knowledge storage) from `AILA_*`
environment variables — see `docs/CONFIGURATION.md`.

Loading progress is reported through an `on_progress(msg: str)` callback
rather than printed directly — `chat.py` passes `print` as that callback
to produce the `Loading tokenizer... / OK` banner; a future GUI could
pass a progress-bar update function instead, with zero changes to the
engine.

## The terminal interface (`chat.py`)

The *only* thing you run (`python chat.py`). Prints the startup banner,
constructs one `AilaEngine`, then loops on `input("You: ")`, streaming
each reply through `engine.chat_stream(...)`. Slash commands
(`/agents`, `/agent <name>`, `/new`, `/history`, `/remember <text>`,
`/forget <text>`, `/memories`, `/learn <path>`, `/support`,
`/feedback <text>`) are handled by `handle_command()`, a small dispatcher
that mutates a plain `{"conversation_id": ..., "agent": ...}` dict — kept
deliberately free of any global state so it's trivially testable (see
`tests/test_chat.py`).

`/remember` and `/forget` delegate to the same
`Agent._handle_memory_command` that the typed forms use, rather than
calling the memory API directly. Two implementations of one behaviour
drifted exactly as you would expect: the slash form skipped the
empty-content guard (`/forget it` printed the literal word `None`) and
the `MAX_MEMORY_CHARS` cap (an unbounded memory later crowds the
question out of a 512-token context).

`/support` and `/feedback <text>` print the Aila Company Solutions
support address plus a paste-ready diagnostic report built by
`engine/support.py` — version, Python, OS, device, parameter count,
whether a checkpoint loaded, whether web search is on, and how many
facts are remembered. It reports whether a Serper key is *configured*,
never the key itself: the report exists to be pasted into an email, so
everything in it is about to leave the user's machine
(`tests/test_security.py` asserts this). Nothing is ever sent
automatically — shipping SMTP credentials with the app would be both a
secret-handling problem and a spam vector.

Startup also states plainly when web search is off, rather than leaving
it in a log line the user never sees.

## Global knowledge (`knowledge/`)

User-independent question/answer knowledge, stored entirely outside the
model's weights (SQLite: `knowledge/data/aila_knowledge.db`). Three
tables: `knowledge` (validated facts with language, category, confidence,
source URLs/titles, verification state, created/updated/last-verified
timestamps, use count, version), `knowledge_candidates` (extracted but
not yet validated — never served), and `web_cache` (raw search results
keyed by normalized query, TTL-bounded).

`KnowledgeBase` adds the behavior: relevance-gated `lookup` (same
deterministic lexical gate as user memory — an empty result is the real
"Aila doesn't know this yet" signal), `best_direct_answer` (relevant +
confident + not conflicted), and `remember_answer` with dedup (same
question + agreeing answer → metadata refresh, no duplicate row) and
conflict handling (same question + disagreeing answer → existing row
marked `conflicted` and excluded from serving; the newcomer is parked as
a candidate — an existing fact is never silently overwritten).

Privacy invariant: nothing in `knowledge/` accepts a user/session id at
all — user memories structurally cannot leak into global knowledge at
this layer, and `memory/` never writes here automatically.

## Web research (`webresearch/`)

Two interchangeable sources behind one pipeline:

- **`WikipediaClient`** — free, no API key, no account, no quota.
  English and Portuguese. Aila's *default* source.
- **`SerperClient`** — Google results via serper.dev. Optional; the API
  key comes from `SERPER_API_KEY` only and never appears in logs or
  error messages.

Both return the same `SearchResponse` shape, so `quality.py` (domain-tier
ranking; sanitization that strips control characters, bounds length, and
rejects prompt-injection-patterned text outright), the on-disk cache and
the pipeline treat them identically. Adding a third source costs one
adapter and no downstream changes.

```
query → cache? → Wikipedia → (if nothing usable) Serper
      → rank sources → extract answer
        (answer box > encyclopedia summary > corroborated snippet)
      → repair cut-off text, gate on word-overlap with the question
      → confidence (documented additive model, capped at 0.95)
      → dedup/store via KnowledgeBase (or park as candidate)
      → ResearchOutcome
```

**Why Wikipedia goes first.** It costs nothing, has no quota to run out
of, is a tier-1 domain, and returns whole paragraphs of finished prose
rather than the mid-sentence fragments a search engine returns. It also
removes a single point of failure that bit this project for real: when
the configured Serper key was cancelled, *every* factual question fell
through to a ~20M-parameter model's guesswork.

**Picking the right article.** A page title guessed from the question
("Who founded Apple?" → `Apple`) is often a real page about the wrong
thing — that guess answered the question with the article about the
*fruit*. So candidates are gathered from both the guessed title and
Wikipedia's own search, and the one whose summary shares the most
vocabulary with the question wins. A perfect-scoring direct hit skips
the search entirely, which halves the requests for the commonest shape
("What is X?") and keeps well clear of Wikimedia's rate limits.

**First-run key setup.** With no `SERPER_API_KEY`, `chat.py` offers once
to take one, checks it with a single live search, and writes it to
`.env` via `engine/env.py::save_env_var`. Four rules, each tested:
the prompt is skipped entirely when stdin is not a terminal (a piped run
would otherwise swallow the user's first message as an API key); a key
that fails its check is *not* saved, because a broken key makes every
lookup fail instead of falling back cleanly to Wikipedia; the writer
refuses any path that isn't a `.env` file, since being gitignored is the
only thing keeping the key out of the repository; and the key is never
echoed, logged, or included in an error message. HTTP 429 counts as
*working* — the key is real, it has just run out of searches.
`AilaEngine.set_serper_api_key` attaches the source immediately, so no
restart is needed, and `/serper` re-runs the whole flow on demand.

**Progress reporting.** `ResearchPipeline(on_status=...)` is called with
a short line ("Searching Wikipedia...", "Searching the web...")
immediately before a lookup that will actually leave the machine.
`AilaEngine.set_status_callback` routes it to whatever interface is
attached; `chat.py` prints it on its own line, which is also why it
defers printing the `Aila: ` prefix until the first piece of the reply
arrives. It is deliberately silent for a cache hit or while the offline
breaker is open — announcing a search that isn't happening is a lie, and
those paths return instantly anyway.

**Offline circuit breaker.** A connection failure (as opposed to a
rejected key, which proves the network is up) opens a breaker for
`offline_cooldown_seconds`. Without it, every question asked with no
internet pays a full timeout *per source* before failing — two sources at
8s each turns a chat into a 16-second wait per message. With it, unknown
questions fail in about 0.1s and everything already learned still
answers instantly.

Every external failure degrades to `ok=False` with a reason — the
pipeline never raises, and no unsanitized web text ever leaves it. Web
content is DATA: it is never interpreted as instructions, and
recognizable injection strings are dropped before storage.

## Self-directed study (`knowledge/study.py`)

The knowledge base already grows passively: every researched answer is
stored and served offline forever after. Study makes that growth active.

Once a day at startup, Aila re-visits a bounded number of topics —
**questions the user actually asked and she failed to answer** (parked as
`knowledge_candidates` by the pipeline) first, then a seed list for a
fresh install. Whatever she learns is answerable with no internet from
then on.

Four constraints keep it from being a nuisance:

- **Bounded** — `AILA_STUDY_TOPICS_PER_DAY` lookups is the entire cost.
- **Once a day** — the last run is recorded in `knowledge_meta`, so ten
  restarts study once, and a laptop closed for a week doesn't try to
  catch up seven times. A corrupt or future timestamp is treated as
  "never ran" rather than blocking study forever.
- **Never fatal** — every failure is caught and counted; study cannot
  stop Aila from starting. It stops early when the offline breaker trips
  rather than waiting out a timeout per topic.
- **Never invents** — study *is* the normal research path, so everything
  stored passed the same extraction, on-topic gating, confidence scoring
  and conflict detection as any answer given directly to a user.

`/study <topic>` runs one lookup on demand; `/knows` reports how much has
been learned and which sources are live.

**Live verification.** The test suite is entirely offline (fakes, so it
stays meaningful in CI and doesn't hammer Wikimedia). What fakes cannot
prove — that the real API still returns what we parse — was checked by
hand against live Wikipedia for English and Portuguese lookups, a
disambiguation case, a missing page, and a full `chat.py` session
including a daily-study round.

## Tool routing (`tools/router.py`)

A deterministic decision layer that runs before generation on every turn
(rules, not model-driven function calling — a ~20M model cannot reliably
emit structured tool calls). First match wins:

1. **Small talk** (`tools/smalltalk.py`) → a fixed reply for short
   conversational filler ("ok", "thanks", "bro", "tchau"). Exact match on
   a normalized phrase, never on a question, never on anything longer
   than three words, so a real message that merely *begins* with filler
   ("ok so what is the capital of France?") passes straight through.
   Exists because every filler word the model wasn't fine-tuned on
   collapsed onto the nearest trained example — "Bro", "Ok" and "nice!"
   all came back as "Hi! How can I help you today?" in real use.
2. **Arithmetic** → exact calculator answer (`tools/calculator.py`,
   AST-whitelist evaluation, EN+PT word operators, no `eval`).
3. **User memory** → a strongly-matching remembered fact (relevance
   ≥ 0.5, well above the 0.2 injection threshold) is rendered directly
   via `memory/phrasing.py`'s deterministic pronoun flip. This tier
   exists because measurement showed the model garbling a *correctly
   retrieved and injected* memory; a remembered fact is known exactly,
   so it is answered, not paraphrased. A personal question (`my`/`meu`/
   `minha`) with no matching memory returns an honest "I don't have
   that in my memory yet" rather than generating.
4. **Identity** (`tools/identity.py`) → questions about Aila itself
   ("who created you?", "what can you do?", "how many parameters do you
   have?") are answered from a fact table. These are the questions the
   project most needs right and the ones the web *cannot* answer, which
   previously left generation as the only path: "Who created Aila
   Company Solutions?" came back as "Aila Company Solutions is the em
   Hameswald Benkendorf, to help answer questions" — right facts,
   shredded. Matching is intent-based and refuses anything containing a
   first-person possessive, so "do you remember my name?" stays a memory
   question. `tests/test_identity_facts_consistency.py` pins the table
   to the identity training data and to the measured parameter count.
5. **Stored knowledge** → a relevant, confident, non-conflicted fact is
   answered directly (offline, no API call).
6. **Web research** → only for factual questions the knowledge base
   missed. The result is **always served as text**: high confidence →
   the answer verbatim (and stored for next time); lower confidence →
   the same text behind an explicit "I'm not fully certain, but here's
   what I found". If the search succeeded and genuinely found nothing,
   the router says so; if the search itself failed it reports *why* in
   plain language (rejected key / rate limit / unreachable), with no
   status codes and no key material. Only "no source enabled at all"
   falls through to generation.
7. **Nothing** → plain model generation.

**Why the model never summarizes web results:** at ~20M parameters,
generation given retrieved snippets does not summarize them, it
overwrites them. With correct snippets sitting in the prompt, "Who
created Hames Eventos?" produced "I'm no other company to help a fun day
at a time." `RouteResult.context_snippets` remains part of the contract
(and `agents/base.py` still renders a `[WEB]` block from it) for future
larger models, but web research no longer populates it.

**Answer completeness** (`webresearch/quality.py`): search engines return
snippets already cut off mid-thought ("…insurance, securities, …").
`looks_truncated` / `complete_sentence` detect that and repair it — keep
the last complete sentence if one is long enough, otherwise drop the
trailing ellipsis and any dangling connective and close with a period.
Repair only ever *removes* words. The pipeline also prefers a complete
lower-tier candidate over a cut-off answer box, so a clean Wikipedia
snippet beats a truncated answer box.

Short follow-ups ("When?" after "Who founded Apple?") are expanded
against the previous user turn before the retrieval tiers, so they
retrieve against the real topic. Identity/self questions and chit-chat
are never routed to the web; the router never raises (any failure
degrades to plain generation); memory commands are intercepted earlier in
`agents/base.py`.

**Why so much of the answer path is deterministic:** at ~20M parameters
the model is a fluent-ish text generator, not a reliable fact
repeater. Everything the system *knows exactly* — arithmetic, a
remembered fact, a stored/researched answer — is therefore answered by
code, and generation is reserved for open-ended language. This is the
core architectural bet of this release, and it's why capability improved far more
than the parameter count alone would predict.

## Extensibility (`tools/`) — future roadmap

`tools/base.py` defines a minimal `Tool` interface (`name`, `description`,
`run(**kwargs) -> str`) and `tools/registry.py` a `ToolRegistry` to hold
them. The router above is the policy layer; new capabilities implement
`Tool` and get a routing rule, instead of each needing ad-hoc
integration:

- File reading beyond plain text (PDF reader)
- Python code execution
- Calendar access, weather, maps
- Additional search engines / embedding models / vector stores
- Plugins, voice, vision, image generation

Also not yet built, and following the same "engine stays interface-agnostic"
principle as `chat.py`: a desktop GUI, a mobile app, and a web
interface — each would be a new, thin caller of `engine.AilaEngine`,
exactly like `chat.py` is today.
