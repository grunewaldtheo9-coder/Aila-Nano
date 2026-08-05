# API Reference: `engine.AilaEngine`

Aila Nano is terminal-first and has no HTTP/network API — `chat.py` talks
to `AilaEngine` through plain Python calls, in-process. This page
documents that Python API, which is what you'd build a new interface
(desktop GUI, mobile app, web service — see the roadmap in
`docs/ARCHITECTURE.md`) against.

```python
from engine import AilaEngine, EngineSettings

engine = AilaEngine(EngineSettings())
```

`EngineSettings()` reads its defaults from `AILA_*` environment
variables (see `docs/CONFIGURATION.md`); pass an explicit
`EngineSettings(checkpoint_path=..., ...)` to override programmatically.

Construction eagerly loads the tokenizer, model, memory, knowledge
index, and every registered agent, so it's the expensive part — construct
one `AilaEngine` per process and reuse it.

```python
engine = AilaEngine(
    EngineSettings(),
    on_progress=print,  # optional: called with a message at each loading stage
)
```

## `engine.chat(conversation_id, message, agent_name="general") -> str`

Get a full reply in one call.

```python
reply = engine.chat("session-1", "What is a transformer?", agent_name="research")
```

## `engine.chat_stream(conversation_id, message, agent_name="general")`

Generator form: yields text deltas as they're produced.

```python
for delta in engine.chat_stream("session-1", "Tell me a short story."):
    print(delta, end="", flush=True)
```

Both `chat` and `chat_stream` automatically record the turn in
conversation memory unless you call the lower-level
`engine.get_agent(name).respond(..., remember_turn=False)` directly.

## `engine.get_agent(agent_name) -> Agent`

Returns (and caches) an `agents.base.Agent` instance. Useful when you
need lower-level control than `chat`/`chat_stream` provide — e.g.
`agent.respond(..., settings=GenerationSettings(temperature=0.2))`, or
`agent.prompt_preview(conversation_id, message)` to inspect the exact
prompt that would be built without running generation.

## `engine.available_agents() -> list[str]`

`["general", "programming", "research", "writing"]`.

## `engine.learn_file(path) -> int`

Chunk a local `.txt`/`.md`/`.jsonl`/`.csv`/`.log` file (≤5 MB) and index
it into the shared knowledge base. Returns the number of chunks indexed.
Raises `ValueError` for an unsupported extension, empty content, or an
oversized file; `FileNotFoundError` if the path doesn't exist.

```python
n = engine.learn_file("notes.md")
```

Indexed content is automatically surfaced as context in later turns
(`agents/base.py` queries the knowledge base alongside long-term memory
when building each prompt) — no separate "search" call needed.

## `engine.memory` / `engine.knowledge`

Direct access to the underlying stores, for interfaces that want more
than the `chat`/`learn_file` convenience wrappers:

```python
engine.memory.remember_fact("The user prefers concise answers.", importance=0.7)
engine.memory.semantic.recall("What does the user prefer?")
engine.memory.conversation.get_history("session-1")

engine.knowledge.search("some query", k=3)
```

See `memory/manager.py` and `vectordb/semantic_index.py` for their full
APIs.

## `engine.is_trained -> bool`

`False` if no checkpoint was found at startup (an untrained,
freshly-initialized model is being served instead — see
`docs/CONFIGURATION.md`'s `AILA_CHECKPOINT` / `AILA_FALLBACK_CHECKPOINT`).

## `engine.parameter_count() -> int`

Total parameter count of the loaded model.

## `engine.save()` / `engine.close()`

`save()` flushes memory and knowledge-base indexes to disk without
tearing anything down (useful for a long-running interface that wants
periodic checkpoints of its own state). `close()` saves and then closes
the underlying SQLite connections — call this once, on shutdown.
`AilaEngine` also supports the context-manager protocol:

```python
with AilaEngine(EngineSettings()) as engine:
    print(engine.chat("s1", "Hello!"))
# engine.close() called automatically
```
