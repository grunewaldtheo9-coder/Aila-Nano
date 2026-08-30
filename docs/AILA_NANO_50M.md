# Aila Nano 50M — conversational upgrade

This document describes the **Aila Nano 50M** work: a ~50M-parameter
architecture plus the conversational infrastructure that turns Aila from a
search-first assistant into a conversation-first one.

## Honest status

The upgrade has two halves, and only one of them can be finished on a
machine without a GPU:

| Part | Status |
|---|---|
| 50M **architecture** (config + verified parameter count) | ✅ Done |
| Conversational **behaviour today** (personality, no-search routing) | ✅ Done, deterministic |
| Multi-turn **chat format** + datasets + validator + training path | ✅ Done, tested |
| **Trained** 50M conversational **weights** | ⏳ Needs a GPU training run |

At ~50M parameters, real conversational ability comes from *training the
weights* on the conversational data — and that is a GPU job (days on the
right hardware), not something a 4-CPU box can do. An untrained 50M network
produces noise, which is worse than the current **trained 20M** model. So:

- The **20M trained checkpoint remains the shipped model.**
- The 50M architecture, the chat pipeline, and the datasets are all in
  place and tested, so a 50M training run is a "press go on a GPU" away.
- Aila still gets **noticeably more conversational today**, because the
  personality and routing improvements live in the deterministic layer and
  need no retraining.

Nothing here fakes a training run (spec §91).

## The architecture

`model/config.py::nano_50m()`:

| | value |
|---|---|
| d_model | 512 |
| layers | 12 |
| attention heads | 8 (4 KV — grouped-query attention) |
| feed-forward | 2048 (SwiGLU, 4× expansion) |
| context length | 1024 |
| vocab | 8192 (unchanged tokenizer) |
| tied embeddings | yes |
| **parameters** | **51,393,024** (verified in the 45M–55M target) |

Same grouped-query attention, RoPE, RMSNorm, SwiGLU, and tied embeddings as
the 10M/20M presets, so the training, inference, and checkpoint code work
unchanged.

### 20M vs 50M at a glance

| | nano_20m (shipped, trained) | nano_50m (architecture only) |
|---|---|---|
| parameters | 19,796,160 | 51,393,024 |
| d_model | 320 | 512 |
| layers | 15 | 12 |
| heads / KV heads | 8 / 4 | 8 / 4 |
| head_dim | 40 | 64 |
| feed-forward (SwiGLU) | 872 (×2.72) | 2048 (×4.0) |
| context length | 512 | 1024 |
| vocab | 8192 | 8192 |
| checkpoint dir | `checkpoints/finetune_20m/` | `checkpoints/50m/` |
| trained weights | ✅ shipped | ⏳ needs a GPU run |

The 50M is wider (bigger `d_model` and feed-forward) with a longer context;
the 20M is deeper. Both share the tokenizer, the attention design, and the
checkpoint format, so the only thing standing between the 50M architecture
and a working 50M model is a training run.

### Known limitations

- **No trained 50M weights.** The 50M *architecture* is implemented and
  verified; a 50M *checkpoint* has not been trained (that needs a GPU). An
  untrained 50M network produces noise, so the trained 20M remains the
  shipped model.
- **CPU pretraining is impractical at 51M.** The pipeline runs on CPU, but
  a real pretrain is a GPU job — see "CPU compatibility" below.
- **1024-token context on CPU** roughly doubles attention cost vs the 20M's
  512; a CPU training run may use a shorter context (parameter count is
  unaffected).

Verify the count yourself:

```bash
python scripts/count_params.py --preset nano_50m
# ...
# Total parameters: 51,393,024 (51.39M)
# OK: within the 45.00M-55.00M target range.
```

## Conversational behaviour today (no retraining)

Because the ~20M/50M model can't be trusted to *generate* a consistent
personality, Aila's identity and preferences are answered from an explicit
config, `tools/personality.py::PERSONALITY`, through the router — so these
now work **now**, in English and Portuguese, with **no web search**:

```
You: Do you like Minecraft?
Aila: Yeah, I do! I like talking about Minecraft — there's so much you can
      build and so many ways to play it. Do you play Survival or Creative?

You: Are you human?
Aila: I'm not human — I'm Aila Nano, an AI assistant. ...

You: Do you search everything?
Aila: No — I answer most things myself. I only search the web when you need
      current or up-to-date information ...
```

Honesty rule (spec §39): Aila says she *enjoys talking about* things; she
never claims to have played, built, or experienced them.

### Search routing

`tests/test_personality.py` includes the spec's §78 routing matrix.
Greetings, identity, preferences, arithmetic, and stable knowledge are
answered **without** a search; only current-information questions (latest
news, today's weather, current prices) trigger one.

## The chat pipeline (ready for a GPU run)

- **Canonical chat template** (`finetuning/chat_format.py`) — the multi-turn
  `messages` schema, encoded with the *same* special tokens the single-turn
  instruction format uses, so training and inference agree. Loss falls only
  on assistant turns; the last turn is closed with EOS so the model learns
  to stop.
- **Datasets** (`datasets/conversational/*.jsonl`) — 55 curated
  conversations (English + Portuguese): greetings, identity, preferences,
  multi-turn project dialogue with follow-ups and context, knowledge,
  explanations, stories, uncertainty, and search-vs-no-search decisions.
- **Validator** (`datasets/scripts/validate_conversations.py`) — rejects
  malformed records and duplicates; exits non-zero to gate CI/training.
- **Training path** (`finetuning/chat_dataset.py` + `--format chat`).

## Conversation, memory & tools architecture

Model-agnostic layers (they work with the 20M model now and a future 50M
one — nothing branches on model size; the architecture is read from the
checkpoint):

- **`ConversationManager`** (`conversation/manager.py`) — owns conversation
  *shape*: active-topic extraction, an extractive rule-based summary of
  older turns (keeps project facts, drops filler), and prioritised context
  assembly (summary + relevant memories). Rule-based on purpose — a trained
  50M model can replace these methods without changing the interface.
- **`EntityTracker`** (`conversation/entities.py`) — tracks the entities a
  conversation mentions with lightweight semantic types (technology / game /
  project / proper_noun) and resolves pronouns ("it", "isso") to the single
  most-recently-mentioned entity — reporting ambiguity (with candidates)
  when two are equally recent instead of guessing. English + Portuguese.
- **`TopicStack`** (`conversation/topics.py`) — current / previous / dormant
  topics. Switches only on an explicit introduction ("let's talk about X")
  or transition marker ("by the way"), so follow-ups stay put; restores an
  earlier thread on "back to X" / "voltando para X" / "back to the previous
  topic". Both are rebuilt from history by `ConversationManager`
  (`entity_tracker`, `topic_stack`), so they survive a restart, and
  `resolve_reference` now chains list → ordinal → pronoun (e.g. "the second
  one" then "why is it better?" → PostgreSQL).
- **`ReferenceResolver`** (`conversation/reference.py`) — resolves short
  contextual messages against recent turns: list-item ordinals ("the second
  one", "number 2", "the last one" → the item the assistant listed),
  affirmations/negations ("yes"/"no"/"exactly"), and "the other one" (only
  when exactly two options exist). Returns a confidence and, when genuinely
  ambiguous ("number 9", 3-option "the other one"), resolves to nothing so
  the caller can ask instead of inventing. Deterministic, English +
  Portuguese, exposed via `ConversationManager.resolve_reference()`. It is
  context infrastructure — the resolved value is available to the model
  prompt; the 20M model still can't compose a free-form comparison from it,
  but a trained 50M model can.
- **`PendingQuestion`** (`conversation/pending.py`) — when Aila asks the
  user to choose or confirm ("Which one do you mean, SQLite or
  PostgreSQL?", "Do you mean PostgreSQL?"), this records the exact options
  offered and resolves the user's short reply against them: naming an option
  ("PostgreSQL"), an ordinal ("the second one"), or a bare "yes"/"no" to a
  single-option proposition. It uses clean entity extraction for the
  options, so the question's lead-in never leaks in as a choice, and it
  refuses to guess when a reply can't be tied to an option (reporting the
  candidates instead). `ConversationManager.pending_question()` finds the
  still-unanswered assistant question by replaying history. English +
  Portuguese.
- **`ConversationContext`** (`conversation/context.py`) — one structured
  object gathering everything the response layer needs for the current
  message, in priority order: the classified user intent (correction /
  clarification_response / continuation / topic_return / topic_switch /
  greeting / question / statement / …), the current topic and dormant
  topics, the active entities, the resolved reference, any pending question,
  the summary, and the relevant memories. `classify_intent()` is
  context-sensitive — a bare "yes" is an *affirmation* on its own but a
  *clarification_response* when a question is pending. `render()` emits a
  compact `[CONTEXT]…[/CONTEXT]` block for a model prompt, dropping empty
  sections. Assembled by `ConversationManager.build_conversation_context()`,
  which wires the six components together (a pending answer takes priority
  over the general reference resolver for that turn). Deterministic and
  CPU-only — nothing here calls the model; it is context infrastructure a
  trained 50M model can compose free-form replies from.
- **`ToolManager`** (`tools/manager.py`) — one contract for every tool
  (`Tool.run` → structured `ToolResult`) with error isolation: an unknown
  tool or any failure (web timeout, memory error) becomes
  `ToolResult(success=False, error=...)` instead of crashing the chat loop.
  Ships `WebSearchTool` and `MemorySearchTool`.
- **`MemoryManager`** (versioned) — relevance-gated retrieval, categories,
  dedup, natural-language `/remember`/`/forget`, **and last-writer-wins
  correction**: a statement like "my favorite game is Zelda" supersedes a
  previous "…Minecraft" (each memory carries `status`, `version`, `source`,
  `confidence`, `attribute_key`, `superseded_by`). Retrieval returns only
  the current value; superseded and deleted values are kept for history
  (`attribute_history`) but never surfaced. `/forget my favorite game`
  deactivates that attribute. `memory_audit()` counts active/superseded/
  deleted. Verified live on the 20M model: "what's my favorite game?" →
  Minecraft, then → Zelda after a correction.
- **Live context**: `ConversationManager.assemble()` is the canonical
  context source feeding the model prompt — the agent injects a `[SUMMARY]`
  block plus the current (corrected) memories. Raw turns are not replayed
  into the 20M prompt (it is single-turn); a trained 50M model can.
- CLI: `/context` (turns, summary, active topics), `/stats`, `/model`,
  `/debug`.

## Growing the conversational dataset

The datasets are built by a template + slot-filling generator, so they
scale without hand-writing:

```bash
# Generate (deterministic, deduplicated) — currently ~4,900 distinct
# conversations across 14 categories, English + Portuguese:
python datasets/scripts/generate_conversations.py --count 8000 \
    --out datasets/conversational/generated/generated.jsonl

# Validate everything and print real statistics:
python datasets/scripts/validate_conversations.py "datasets/conversational/**/*.jsonl" --stats

# Split into unseen train / validation sets:
python datasets/scripts/split_conversations.py \
    "datasets/conversational/**/*.jsonl" --out-dir datasets/conversational/split
```

Add templates and slot fillers in `generate_conversations.py` to grow
toward tens of thousands — no other code changes needed.

### Loading a checkpoint (20M now, 50M later)

Checkpoints are self-describing (they store their architecture config and
metadata), so the CLI loads whichever you point it at — no source edits:

```bash
python chat.py                                        # the shipped 20M model
python chat.py --checkpoint checkpoints/chat_50m/best.pt   # a future 50M model
```

A vocab mismatch fails with a clear message rather than a shape error, and
`/model` in chat shows the loaded model's size and architecture.

### Configuration files

The 50M architecture and its training recipes live in version-controlled
config files, so the run is reproducible and there are no magic numbers in
the training scripts:

| File | Purpose |
|---|---|
| `configs/model/nano_50m.yaml` | the architecture (matches `model/config.py::nano_50m`) |
| `configs/training/pretrain_50m.yaml` | pretraining hyperparameters (TrainingConfig) |
| `configs/training/finetune_50m.yaml` | instruction / conversation fine-tune (FinetuneConfig) |

All 50M checkpoints are written under a dedicated `checkpoints/50m/`
directory, kept **separate** from the shipped 20M checkpoints
(`checkpoints/finetune_20m/`, `checkpoints/pretrain_20m/`) so nothing is
overwritten.

### How to actually train the 50M model (on a GPU)

```bash
# 1. Pretrain the 50M model on general text (TinyStories/WikiText shards).
#    --preset nano_50m selects the built-in architecture; the equivalent
#    explicit form is --model-config configs/model/nano_50m.yaml.
python -m training.train \
    --preset nano_50m \
    --train-config configs/training/pretrain_50m.yaml
#    -> checkpoints/50m/pretrain/

# Resume an interrupted run (picks up the latest checkpoint in the dir):
python -m training.train \
    --preset nano_50m \
    --train-config configs/training/pretrain_50m.yaml --resume

# 2. Instruction / conversation fine-tune (chat format).
python -m finetuning.finetune \
    --init-checkpoint checkpoints/50m/pretrain/best.pt \
    --tokenizer tokenizer/artifacts/aila_nano.model \
    --data datasets/conversational/*.jsonl \
    --format chat \
    --config configs/training/finetune_50m.yaml
#    -> checkpoints/50m/finetune/

# 3. Point the chat at it (architecture is auto-detected from the file).
python chat.py --checkpoint checkpoints/50m/finetune/best.pt
# or: export AILA_CHECKPOINT=checkpoints/50m/finetune/best.pt
```

The pipeline is verified end to end on a small model (loss computes,
decreases, checkpoints save and resume) and the config files load into the
verified 51,393,024-parameter architecture — the only missing ingredient is
GPU time for the real 50M run.

### CPU compatibility and mixed precision

Everything above runs on CPU without a GPU or any CUDA-only dependency; the
50M *architecture, configs, param accounting, checkpoint handling, and
inference path* are all exercised on CPU in the test suite. What CPU can't
do at ~51M parameters is a *fast* pretraining run — a single step is minutes,
not seconds, so a full pretrain is a GPU job (this is stated honestly, not
worked around). Mixed precision is handled safely for CPU: the trainer only
enables the fp16 `GradScaler` on CUDA and uses bf16 autocast elsewhere, so
turning `amp` on never assumes CUDA exists (the 50M pretrain config leaves
it off, since bf16 buys little at this scale on CPU).

### Loading the wrong checkpoint fails clearly

The engine reads a checkpoint's architecture from its own stored `config`,
so a 20M or a 50M checkpoint each load correctly with no code change. A
vocab mismatch against the tokenizer is rejected up front with a plain-
language message. And a caller that explicitly demands a specific
architecture (`validate_checkpoint_compatibility(ckpt, vocab,
expected_config=nano_50m())`) gets a clear **"Checkpoint architecture
mismatch: expected a ~50M configuration, found an incompatible ~20M one
(d_model: expected 512, found 320, …)"** instead of a cryptic
`load_state_dict` shape error.

## Migration from 20M (spec §83)

The 20M architecture and checkpoints are untouched. The 50M net is a new
preset with its own checkpoint directories; nothing loads a 20M checkpoint
into the 50M architecture (a compatibility check would reject the shape
mismatch). Keep both until a 50M model is trained and validated.
