"""Guards against a real bug found while fine-tuning on real data: the
Aila knowledge / sample instruction datasets used a short, hand-typed
system prompt that didn't match what the agents actually send at
inference time (agents/*.py's `system_prompt`, built from
`AILA_KNOWLEDGE_PRIMER` + persona text). A model trained only on the
mismatched prompt saw a completely different, never-seen system prompt
the moment it was used for real, and produced degenerate output (verified
manually: coherent next-token predictions on the exact trained prompt,
garbage on the runtime one).

These tests don't catch generation quality (that needs a real trained
checkpoint), but they do guarantee training data and runtime prompts can
never silently drift apart again — every "system" value used in the
instruction data must be the *exact* string some registered agent
actually uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agents.base import Agent
from agents.registry import AGENT_REGISTRY
from memory.manager import MemoryContext

REPO_ROOT = Path(__file__).resolve().parent.parent
AILA_KNOWLEDGE = REPO_ROOT / "datasets" / "aila_knowledge" / "aila_company.jsonl"
FINETUNE_SAMPLE = REPO_ROOT / "datasets" / "sample" / "finetune_sample.jsonl"
MEMORY_RECALL = REPO_ROOT / "datasets" / "aila_knowledge" / "memory_recall.jsonl"

VALID_SYSTEM_PROMPTS = {cls.system_prompt for cls in AGENT_REGISTRY.values()}


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_aila_knowledge_system_prompts_match_a_real_agent():
    examples = _load_jsonl(AILA_KNOWLEDGE)
    assert examples, "aila_company.jsonl must not be empty"
    for i, ex in enumerate(examples):
        assert ex.get("system") in VALID_SYSTEM_PROMPTS, (
            f"{AILA_KNOWLEDGE.name}:{i} has a system prompt that doesn't exactly match "
            f"any agent's runtime system_prompt — a model fine-tuned on this will see a "
            f"different, never-trained-on prompt the moment it's used for real. "
            f"Got: {ex.get('system')!r}"
        )


def test_finetune_sample_system_prompts_match_a_real_agent():
    examples = _load_jsonl(FINETUNE_SAMPLE)
    assert examples, "finetune_sample.jsonl must not be empty"
    for i, ex in enumerate(examples):
        assert ex.get("system") in VALID_SYSTEM_PROMPTS, (
            f"{FINETUNE_SAMPLE.name}:{i} has a system prompt that doesn't exactly match "
            f"any agent's runtime system_prompt. Got: {ex.get('system')!r}"
        )


def test_memory_recall_system_prompts_match_the_real_facts_augmentation():
    # memory_recall.jsonl teaches the model to use facts injected into its
    # system prompt by the memory system (see agents.base.Agent's
    # _build_system_prompt) — the exact same failure class as the other two
    # tests above, but for the facts-augmented prompt instead of the bare
    # persona prompt. Rebuild the expected string via the real
    # `_build_system_prompt` logic (not a hardcoded copy of its format)
    # so this test breaks loudly if that method's wording ever changes
    # without the training data being regenerated to match.
    examples = _load_jsonl(MEMORY_RECALL)
    assert examples, "memory_recall.jsonl must not be empty"
    for i, ex in enumerate(examples):
        system = ex.get("system", "")
        base_prompt = next((p for p in VALID_SYSTEM_PROMPTS if system.startswith(p)), None)
        assert base_prompt is not None, (
            f"{MEMORY_RECALL.name}:{i} system prompt doesn't start with any real agent's "
            f"system_prompt. Got: {system!r}"
        )
        facts_text = system[len(base_prompt):]
        facts = [line[2:] for line in facts_text.strip("\n").splitlines()[1:]]
        fake_agent = SimpleNamespace(system_prompt=base_prompt, knowledge=None)
        expected = Agent._build_system_prompt(
            fake_agent, ex["instruction"], MemoryContext(relevant_facts=[{"content": f} for f in facts])
        )
        assert system == expected, (
            f"{MEMORY_RECALL.name}:{i} system prompt doesn't exactly match what "
            f"Agent._build_system_prompt would actually produce at runtime for the same "
            f"facts — a model fine-tuned on this drifted format would see a different, "
            f"never-trained-on prompt in production."
        )


def test_aila_knowledge_covers_every_agent_persona():
    # Not just *a* valid prompt each time — exercise all four personas so
    # the model learns its identity regardless of which agent is active.
    examples = _load_jsonl(AILA_KNOWLEDGE)
    used_prompts = {ex["system"] for ex in examples}
    assert used_prompts == VALID_SYSTEM_PROMPTS, (
        "aila_knowledge/aila_company.jsonl should include examples under every agent's "
        "system prompt, not just one persona."
    )
