# Troubleshooting

Every entry here is a failure that was actually hit while building and
running Aila Nano — not a hypothetical list. Symptoms are quoted as they
appear on screen.

---

## Startup

### `ModuleNotFoundError: No module named 'torch'`

Dependencies aren't installed (or you're in the wrong environment).

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### `Could not start: Tokenizer model not found at 'tokenizer/artifacts/aila_nano.model'`

The tokenizer files aren't where the engine expects them. Put
`aila_nano.model` and `aila_nano.vocab` in `tokenizer/artifacts/`, or
point at them explicitly:

```bash
AILA_TOKENIZER=/path/to/aila_nano.model python chat.py
```

### `Could not start: a storage file is corrupted (file is not a database)`

A SQLite file was truncated or overwritten (interrupted copy, disk full,
a Git LFS pointer downloaded instead of the real file). These files are
user state and caches — **not** the model — so deleting them is safe;
they rebuild empty:

```
memory/data/aila_memory.db          # conversations + user memories
vectordb/index/knowledge.db         # /learn-ed documents
knowledge/data/aila_knowledge.db    # global knowledge + web cache
```

### `_pickle.UnpicklingError: invalid load key, 'v'`

The checkpoint file is not a real checkpoint. The usual cause: it was
downloaded from GitHub's **"Download ZIP"** button, which substitutes a
small text pointer for Git LFS files. Download `best.pt` from its own
file page ("Download raw file"), or `git lfs pull`. A genuine checkpoint
is ~130 MB (1.0) / ~240 MB (2.0); a pointer is a few hundred bytes.

### `RuntimeError: Error(s) in loading state_dict ... Missing key(s)`

You're loading a checkpoint into a different architecture (e.g. 1.0
weights into the 2.0 model). This is intentional — incompatible
checkpoints are never loaded silently. Each checkpoint stores its own
config, so load it through `training.checkpoint.load_checkpoint` and
build the model from `ckpt["config"]` rather than a hardcoded preset.

### Startup warns "No checkpoint found ... serving a freshly-initialized, UNTRAINED model"

Exactly what it says: `AILA_CHECKPOINT` and `AILA_FALLBACK_CHECKPOINT`
both point at files that don't exist, so responses will be noise. Check
the path — for 2.0 it's `checkpoints/finetune_20m/best.pt`.

---

## Answer quality

### Replies are fluent-ish but garble names, numbers, or facts

Expected at this scale. A ~20M-parameter model is a reasonable text
generator and an unreliable fact repeater. This is precisely why 2.0
answers everything it *knows exactly* deterministically (arithmetic,
remembered facts, stored/researched knowledge) — see
[ARCHITECTURE.md](ARCHITECTURE.md#tool-routing-toolsrouterpy--aila-20).
If a specific fact matters, teach it:

```
remember that my project deadline is March 14
```

### Replies wander off into unrelated stories

Two known causes, both already mitigated in 2.0:

1. **Sampling too loose.** Defaults are `temperature=0.2, top_k=4`
   (`agents/base.py::GenerationSettings`). Raising them noticeably
   degrades coherence at this scale.
2. **Conversation history in the prompt.** The model was fine-tuned only
   on single-turn examples, so replaying prior turns pushed it
   out-of-distribution and back onto its pretraining prior (freeform
   narrative). History is recorded but deliberately not replayed — see
   the comment in `agents/base.py::_build_prompt_ids`.

### "What is my name?" says it doesn't know, but I told it earlier

Memories are stored per phrasing, and retrieval needs a shared
significant word. `remember that my dog is called Max` then asking
`what is my pet's name?` won't match ("dog" vs "pet"). Check what's
actually stored:

```
/memories
```

### It answers a *different* remembered fact

The direct-answer threshold is deliberately strict (relevance ≥ 0.5), but
short memories with overlapping words can still collide. Use `/memories`
to inspect, `/forget <text>` to remove the wrong one, and prefer
distinctive wording when storing.

---

## Web research (Serper)

### Web research never fires

Check, in order:

```bash
grep SERPER_API_KEY .env          # key present?
echo $AILA_WEB_SEARCH_ENABLED     # not "false"?
```

With no key the engine logs `Web search enabled but SERPER_API_KEY is
not set — running offline` at startup and simply runs without the tool.
Also note the router only researches **factual questions**: it never
sends chit-chat, non-questions, or anything mentioning Aila/"you" to the
web.

### `Serper rejected the API key (HTTP 401/403)`

Wrong or revoked key. Regenerate at [serper.dev](https://serper.dev) and
update `.env`. If your key was ever pasted into a chat, a commit, or a
screenshot, treat it as compromised and rotate it.

### `Serper rate limit reached (HTTP 429)`

Free-tier quota exhausted. Aila degrades to the knowledge base and the
model automatically — nothing crashes. Answers already researched stay
available offline.

### The same question hits the API twice

It shouldn't: results are cached by normalized query for
`AILA_WEB_CACHE_TTL_HOURS` (default 168h), and once an answer is stored
the knowledge base serves it before research is ever reached. If you're
testing and want a fresh call, delete
`knowledge/data/aila_knowledge.db`.

---

## Memory and knowledge

### A fact is wrong and won't go away

If two sources disagreed, the item is marked `conflicted` and stops
being served — that's why it may seem to vanish. Inspect and clean up:

```python
from knowledge.store import KnowledgeStore
store = KnowledgeStore("knowledge/data/aila_knowledge.db")
for row in store.all_knowledge():
    print(row["id"], row["verification"], row["confidence"], row["question"])
store.delete_knowledge(<id>)
```

### `remember that` did nothing

Filler-only commands are rejected on purpose (`remember that`,
`forget about`, `remember it`) — otherwise the literal word "that" would
be stored as a memory. Include real content:
`remember that my name is Theo`.

### Memories look truncated

Single memories are capped at 500 characters
(`memory/commands.MAX_MEMORY_CHARS`). An unbounded memory is injected
verbatim into the prompt and would crowd out the system prompt and your
actual question.

---

## Training

### Training is extremely slow

Expected on CPU: no flash-attention kernel, so the math fallback
materializes full attention matrices. Measured on this project's
hardware: ~1,300 tok/s for the 20M model at `batch=16, seq=256`.
Halving the sequence length roughly doubles throughput, which is why the
real runs train at 256 rather than the 512 spec context. A GPU changes
this by orders of magnitude.

### Out of memory during training

Lower `batch_size` in the training YAML and raise `grad_accum_steps` to
keep the effective batch the same. Measured on a 15 GB machine at
seq=512: batch 4 ≈ 2.4 GB, 8 ≈ 4.6 GB, 16 ≈ 8.9 GB, 32 = OOM.

### Loss looks great but generation is garbage

The classic teacher-forcing trap, and a real bug this project hit: if
`input_ids` and `labels` aren't shifted, the model learns to echo the
token it just saw. Loss drops beautifully; autoregressive generation
never works. `finetuning/dataset.py` shifts explicitly and
`tests/test_finetuning.py::test_run_finetune_learns_real_next_token_prediction`
guards it by greedy-decoding without teacher forcing. **Always check
generation, not just loss.**

### Training stopped when the machine restarted

Resume from the last checkpoint:

```bash
python -m training.train \
  --model-config configs/model/nano20m_real_run.yaml \
  --train-config configs/training/pretrain_20m.yaml \
  --resume
```

---

## Configuration

### A setting in `.env` seems ignored

Real environment variables always win over `.env` (by design). Unset the
shell variable, or set it there instead.

### `AILA_WEB_MAX_RESULTS='five' is not an integer; using default 5`

A typo in a numeric setting. Aila warns and uses the documented default
rather than refusing to start. Fix the value to silence it.

---

## Getting more detail

Logs carry per-turn routing, latency, and tool decisions (no secrets, no
user text):

```bash
python -c "
import logging; logging.basicConfig(level=logging.INFO)
import chat; chat.main()
"
```

Look for `turn path=tool:calculator latency=0.001s`,
`knowledge hit id=3 relevance=0.71`, `web cache hit`, or
`serper search failed: ...`.
