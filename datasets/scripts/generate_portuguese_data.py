#!/usr/bin/env python3
"""Generate datasets/aila_knowledge/portuguese_basic.jsonl: basic
Brazilian-Portuguese instruction examples (greetings, identity,
courtesy, simple facts) across the agent personas.

Honest scope note: the pretraining corpus is English-only (TinyStories),
so a ~20M model cannot become genuinely fluent in Portuguese from a few
dozen instruction pairs — the goal here is *graceful basic handling*
(recognize PT greetings/identity questions and reply with trained PT
phrases via byte-fallback tokenization) rather than real fluency, which
would require a bilingual pretraining corpus and a retrained tokenizer.
Documented as a limitation in docs/MODEL_CARD.md.

Usage:
    python datasets/scripts/generate_portuguese_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agents.registry import AGENT_REGISTRY

OUT_PATH = Path(__file__).resolve().parent.parent / "aila_knowledge" / "portuguese_basic.jsonl"

# (instruction, output) — every pair repeated across personas by main().
EXAMPLES: list[tuple[str, str]] = [
    ("Olá!", "Olá! Como posso ajudar você hoje?"),
    ("Oi, tudo bem?", "Tudo bem, obrigada por perguntar! Como posso ajudar?"),
    ("Bom dia!", "Bom dia! Em que posso ajudar?"),
    ("Boa tarde!", "Boa tarde! Como posso ajudar você?"),
    ("Boa noite!", "Boa noite! Em que posso ajudar?"),
    ("Quem criou você?", "Fui criada pela Aila Company Solutions, fundada por Theo Grunewald Hames e Guilherme Grunewald Benkendorf."),
    ("Quem fez você?", "A Aila Company Solutions me criou. A empresa foi fundada por Theo Grunewald Hames e Guilherme Grunewald Benkendorf."),
    ("O que é a Aila Nano?", "A Aila Nano é um modelo de linguagem pequeno criado do zero pela Aila Company Solutions."),
    ("Qual é o seu nome?", "Meu nome é Aila Nano. Como posso ajudar?"),
    ("Você fala português?", "Falo um pouco de português! Fui treinada principalmente em inglês, mas posso ajudar com frases simples."),
    ("Obrigado!", "De nada! Se precisar de mais alguma coisa, é só falar."),
    ("Obrigada pela ajuda!", "De nada! Fico feliz em ajudar."),
    ("Tchau!", "Tchau! Tenha um ótimo dia."),
    ("Até logo!", "Até logo! Volte quando quiser."),
    ("Me ajuda?", "Claro! Me diga o que você precisa."),
    ("Quanto é 2 mais 2?", "2 mais 2 é 4."),
    ("Qual é a capital do Brasil?", "A capital do Brasil é Brasília."),
    ("Qual é a capital de Portugal?", "A capital de Portugal é Lisboa."),
]


def main() -> None:
    personas = list(AGENT_REGISTRY.keys())
    prompts = {name: cls.system_prompt for name, cls in AGENT_REGISTRY.items()}

    examples = []
    for i, (instruction, output) in enumerate(EXAMPLES):
        persona = personas[i % len(personas)]
        examples.append(
            {
                "system": prompts[persona],
                "instruction": instruction,
                "input": "",
                "output": output,
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
