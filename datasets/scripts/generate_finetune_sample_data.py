#!/usr/bin/env python3
"""Generate datasets/sample/finetune_sample.jsonl: general-purpose
instruction examples (not Aila-identity facts — see
generate_aila_identity_data.py for those) covering everyday chat
competence — greetings, factual Q&A, math, translation, summarization,
continuation, code, and short creative writing — across all four agent
personas, with system prompts pulled directly from agents.registry so
they can never drift from what the agents actually use at inference
time (see tests/test_finetune_data_consistency.py).

Usage:
    python datasets/scripts/generate_finetune_sample_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agents.registry import AGENT_REGISTRY

OUT_PATH = Path(__file__).resolve().parent.parent / "sample" / "finetune_sample.jsonl"

# (persona, instruction, input, output)
EXAMPLES: list[tuple[str, str, str, str]] = [
    ("general", "What is the capital of France?", "", "The capital of France is Paris."),
    ("general", "What is the capital of Japan?", "", "The capital of Japan is Tokyo."),
    (
        "general",
        "Summarize the following text in one sentence.",
        "The cat sat quietly on the windowsill, watching the rain fall on the garden below, seemingly lost in thought.",
        "A cat sits on a windowsill, calmly watching the rain over the garden.",
    ),
    ("general", "Translate the following sentence to French.", "Good morning, how are you?", "Bonjour, comment allez-vous ?"),
    ("general", "Translate the following sentence to Spanish.", "Where is the train station?", "¿Dónde está la estación de tren?"),
    ("general", "What is 12 multiplied by 8?", "", "12 multiplied by 8 is 96."),
    ("general", "What is 45 divided by 9?", "", "45 divided by 9 is 5."),
    (
        "general",
        "List three benefits of regular exercise.",
        "",
        "1. Improves cardiovascular health.\n2. Boosts mood and reduces stress.\n3. Increases energy levels and sleep quality.",
    ),
    ("general", "Hello!", "", "Hello! How can I help you today?"),
    ("general", "Hi, how are you?", "", "I'm doing well, thanks for asking! What can I help you with?"),
    ("general", "Good morning!", "", "Good morning! What can I help you with today?"),
    ("general", "Thank you for your help!", "", "You're welcome! Let me know if there's anything else I can help with."),
    ("general", "That's all, goodbye.", "", "Goodbye! Have a great day."),
    ("general", "What day comes after Wednesday?", "", "Thursday comes after Wednesday."),
    ("general", "Is the sky blue?", "", "Yes, the sky appears blue during the day due to how sunlight scatters in the atmosphere."),
    (
        "programming",
        "Write a Python function that returns the square of a number.",
        "",
        "def square(x):\n    return x * x",
    ),
    (
        "programming",
        "Write a Python function that checks if a number is even.",
        "",
        "def is_even(n):\n    return n % 2 == 0",
    ),
    (
        "programming",
        "What does the acronym API stand for?",
        "",
        "API stands for Application Programming Interface — a set of rules that lets different software programs communicate with each other.",
    ),
    (
        "programming",
        "What is a variable in programming?",
        "",
        "A variable is a named piece of storage in a program that holds a value, which can be read or changed while the program runs.",
    ),
    (
        "programming",
        "Write a Python function that reverses a string.",
        "",
        "def reverse_string(s):\n    return s[::-1]",
    ),
    (
        "programming",
        "What is the difference between a list and a tuple in Python?",
        "",
        "A list is mutable (its contents can be changed after creation) while a tuple is immutable (its contents are fixed once created).",
    ),
    (
        "writing",
        "Write a short, encouraging message for someone starting a new job.",
        "",
        "Congratulations on your new role! Every expert was once a beginner — take it one day at a time, ask questions freely, and trust that you'll find your footing quickly.",
    ),
    (
        "writing",
        "Continue the story.",
        "Once upon a time, in a small village surrounded by mountains, there lived a curious young inventor.",
        "She spent her days building small machines out of wood, wire, and old clock parts, dreaming of the day one of her inventions would help the whole village.",
    ),
    (
        "writing",
        "Write a two-sentence description of a rainy afternoon.",
        "",
        "Rain tapped steadily against the window as gray clouds settled over the town. Inside, the smell of fresh coffee made the quiet afternoon feel warm and unhurried.",
    ),
    (
        "writing",
        "Rewrite this sentence to sound more formal.",
        "hey can u send me that file when u get a sec",
        "Could you please send me that file when you have a moment?",
    ),
    (
        "writing",
        "Write a short thank-you note for a birthday gift.",
        "",
        "Thank you so much for the thoughtful gift — it really made my birthday special. I'm so grateful to have you in my life.",
    ),
    (
        "research",
        "Explain what a neural network is in simple terms.",
        "",
        "A neural network is a computer system loosely inspired by the brain. It is made of layers of simple units called neurons that pass signals to each other, and by adjusting the strength of these connections during training, the network learns to recognize patterns in data.",
    ),
    (
        "research",
        "Explain what gravity is in simple terms.",
        "",
        "Gravity is the force that pulls objects with mass toward each other. It's why things fall to the ground on Earth and why planets orbit the Sun.",
    ),
    (
        "research",
        "What is the difference between weather and climate?",
        "",
        "Weather describes short-term atmospheric conditions in a specific place, like today's temperature or rain. Climate describes the average weather patterns in a region over many years.",
    ),
    (
        "research",
        "Why do leaves change color in autumn?",
        "",
        "Leaves get their green color from chlorophyll, which plants use for photosynthesis. In autumn, shorter days and cooler temperatures cause chlorophyll to break down, revealing the yellow and orange pigments that were there all along.",
    ),
    (
        "research",
        "What is the scientific method?",
        "",
        "The scientific method is a systematic approach to investigating questions: making an observation, forming a hypothesis, testing it with experiments, and drawing conclusions based on the evidence gathered.",
    ),
]


def main() -> None:
    prompts = {name: cls.system_prompt for name, cls in AGENT_REGISTRY.items()}

    examples = []
    for persona, instruction, input_text, output in EXAMPLES:
        examples.append(
            {
                "system": prompts[persona],
                "instruction": instruction,
                "input": input_text,
                "output": output,
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
