#!/usr/bin/env python3
"""Generate datasets/aila_knowledge/memory_recall.jsonl: instruction
examples that teach the model to actually *use* facts injected into its
system prompt by the memory system, not just answer from what it saw
during pretraining/other fine-tuning.

Why this exists: `agents.base.Agent._build_system_prompt` appends
remembered facts to the system prompt as:

    {persona system prompt}

    Relevant background you may use if helpful:
    - fact one
    - fact two

...at inference time, whenever `/remember` (or the memory system more
generally) has something relevant to the current query. But nothing in
the original fine-tuning data ever showed the model that block, or
what to do with it — "remembering" (storing a fact) and "using a
remembered fact to answer a question" are different skills, and only
the first one had any training signal. This generator produces the
second: the *exact* system-prompt augmentation format above, paired
with a question the injected fact(s) answer, and an output that
correctly uses them.

A few "distractor" examples (facts present, but irrelevant to the
question asked) are included too, so the model learns to use facts
when relevant rather than force-inserting them into every reply.

Usage:
    python datasets/scripts/generate_memory_recall_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agents.registry import AGENT_REGISTRY

OUT_PATH = Path(__file__).resolve().parent.parent / "aila_knowledge" / "memory_recall.jsonl"

FACTS_BLOCK_HEADER = "Relevant background you may use if helpful:"


def with_facts(system_prompt: str, facts: list[str]) -> str:
    facts_text = "\n".join(f"- {f}" for f in facts)
    return f"{system_prompt}\n\n{FACTS_BLOCK_HEADER}\n{facts_text}"


# Each case: (facts injected into the system prompt, instruction, output).
# `facts` is what the memory system would have retrieved; the model must
# use it correctly (or, for distractor cases, ignore it and answer the
# actual question normally).
CASES: list[dict] = [
    {
        "facts": ["The user's name is Theo."],
        "instruction": "What's my name?",
        "output": "Your name is Theo.",
    },
    {
        "facts": ["The user's name is Theo."],
        "instruction": "Do you remember my name?",
        "output": "Yes — your name is Theo.",
    },
    {
        "facts": ["The user's name is Maria."],
        "instruction": "What is my name again?",
        "output": "Your name is Maria.",
    },
    {
        "facts": ["The user's favorite color is blue."],
        "instruction": "What's my favorite color?",
        "output": "Your favorite color is blue.",
    },
    {
        "facts": ["The user's favorite color is green."],
        "instruction": "Do you know what color I like?",
        "output": "Yes, you told me your favorite color is green.",
    },
    {
        "facts": ["The user has a dog named Max."],
        "instruction": "What's my dog's name?",
        "output": "Your dog's name is Max.",
    },
    {
        "facts": ["The user works as a teacher."],
        "instruction": "What's my job?",
        "output": "You work as a teacher.",
    },
    {
        "facts": ["The user lives in Berlin."],
        "instruction": "Where do I live?",
        "output": "You live in Berlin.",
    },
    {
        "facts": ["The user's favorite food is pizza."],
        "instruction": "What food do I like the most?",
        "output": "Your favorite food is pizza.",
    },
    {
        "facts": ["The user is learning Python."],
        "instruction": "What am I learning right now?",
        "output": "You're learning Python.",
    },
    {
        "facts": ["The user's birthday is in May."],
        "instruction": "When is my birthday?",
        "output": "Your birthday is in May.",
    },
    {
        "facts": ["The user's name is Theo.", "The user's favorite color is blue."],
        "instruction": "What's my name and favorite color?",
        "output": "Your name is Theo, and your favorite color is blue.",
    },
    {
        "facts": ["The user's name is Ana.", "The user has a cat named Luna."],
        "instruction": "What's my cat's name?",
        "output": "Your cat's name is Luna.",
    },
    {
        "facts": ["The user's name is Sam.", "The user works as a nurse."],
        "instruction": "What's my name?",
        "output": "Your name is Sam.",
    },
    # Distractors: a fact is present, but the question doesn't need it —
    # answer normally instead of shoehorning the fact in.
    {
        "facts": ["The user's name is Theo."],
        "instruction": "What is 9 plus 10?",
        "output": "9 plus 10 is 19.",
    },
    {
        "facts": ["The user's favorite color is blue."],
        "instruction": "What is the capital of Italy?",
        "output": "The capital of Italy is Rome.",
    },
    {
        "facts": ["The user has a dog named Max."],
        "instruction": "Write a Python function that adds two numbers.",
        "output": "def add(a, b):\n    return a + b",
    },
    {
        "facts": ["The user's name is Theo."],
        "instruction": "Hello!",
        "output": "Hello, Theo! What can I help you with?",
    },
]


def main() -> None:
    personas = list(AGENT_REGISTRY.keys())
    prompts = {name: cls.system_prompt for name, cls in AGENT_REGISTRY.items()}

    examples = []
    for i, case in enumerate(CASES):
        persona = personas[i % len(personas)]
        system = with_facts(prompts[persona], case["facts"])
        examples.append(
            {
                "system": system,
                "instruction": case["instruction"],
                "input": "",
                "output": case["output"],
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
