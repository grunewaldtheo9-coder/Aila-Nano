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
- **`ToolManager`** (`tools/manager.py`) — one contract for every tool
  (`Tool.run` → structured `ToolResult`) with error isolation: an unknown
  tool or any failure (web timeout, memory error) becomes
  `ToolResult(success=False, error=...)` instead of crashing the chat loop.
  Ships `WebSearchTool` and `MemorySearchTool`.
- **`MemoryManager`** (existing) — relevance-gated retrieval, categories,
  dedup, and natural-language `/remember`/`/forget` commands.
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

### How to actually train the 50M model (on a GPU)

```bash
# 1. Pretrain the 50M model on general text (TinyStories/WikiText shards).
python -m training.train --preset nano_50m --out-dir checkpoints/pretrain_50m

# 2. Instruction-tune.
python -m finetuning.finetune \
    --init-checkpoint checkpoints/pretrain_50m/best.pt \
    --tokenizer tokenizer/artifacts/aila_nano.model \
    --data datasets/aila_knowledge/*.jsonl \
    --out-dir checkpoints/instruct_50m

# 3. Conversation fine-tune (the new chat format).
python -m finetuning.finetune \
    --init-checkpoint checkpoints/instruct_50m/best.pt \
    --tokenizer tokenizer/artifacts/aila_nano.model \
    --data datasets/conversational/*.jsonl \
    --format chat \
    --out-dir checkpoints/chat_50m

# 4. Point the engine at it.
export AILA_CHECKPOINT=checkpoints/chat_50m/best.pt
```

The chat pipeline is verified end to end on a small model (loss computes,
decreases, checkpoints save) — the only missing ingredient is GPU time for
the real 50M run.

## Migration from 20M (spec §83)

The 20M architecture and checkpoints are untouched. The 50M net is a new
preset with its own checkpoint directories; nothing loads a 20M checkpoint
into the 50M architecture (a compatibility check would reject the shape
mismatch). Keep both until a 50M model is trained and validated.
