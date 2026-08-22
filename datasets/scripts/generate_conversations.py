#!/usr/bin/env python3
"""Synthetic conversational dataset generator.

At ~50M parameters a model needs *many thousands* of diverse conversations
to learn to converse — far more than anyone would hand-write. This builds
them from templates with slot-filling: each template is a conversation
skeleton with placeholders ({game}, {project}, {controller}, ...) and
several natural phrasings per turn, so combining slots and phrasings yields
large numbers of *distinct* conversations rather than duplicates.

The output is the same `messages` JSONL the training pipeline consumes,
plus per-record metadata (category, language, requires_memory,
requires_web, turns). Generation is deterministic (seeded) and exact
duplicates are dropped, so re-running gives the same, deduplicated set.

This is infrastructure: the templates and slot lists here produce a few
thousand conversations, and the design scales to tens of thousands by
adding templates and fillers — no code change needed. It does NOT fabricate
a trained model; it prepares data for a future GPU fine-tune.

Usage:
    python datasets/scripts/generate_conversations.py --count 4000 \
        --out datasets/conversational/generated/generated.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from finetuning.chat_format import validate_conversation

SYSTEM_EN = "You are Aila Nano, a friendly conversational AI assistant."
SYSTEM_PT = "Você é a Aila Nano, uma assistente de IA amigável e conversacional."

# -- slot fillers -------------------------------------------------------------
SLOTS_EN = {
    "greeting": ["Hi", "Hello", "Hey", "Hey there", "Hi there", "Yo", "Hiya", "Hey Aila", "Heya"],
    "greeting_reply": [
        "Hey! How are you?", "Hi! What's up?", "Hey there! How's it going?",
        "Hello! How's your day going?", "Hey! Nice to see you. What's on your mind?",
        "Hi! Good to see you. How are you doing?",
    ],
    "name": ["Alex", "Sam", "Theo", "Maria", "Lucas", "Ana", "Chris", "Jordan", "Bea", "Noah", "Kai", "Ivy", "Leo", "Mia", "Owen", "Zoe"],
    "game": ["Minecraft", "Zelda", "Mario", "Roblox", "Terraria", "Stardew Valley", "Portal", "Celeste", "Hollow Knight", "Tetris", "Among Us", "Fortnite", "Pokemon", "Sonic", "Doom", "Factorio"],
    "topic": ["technology", "robotics", "games", "programming", "science", "space", "music", "movies", "building things", "AI", "electronics", "3D printing", "chemistry", "history"],
    "controller": ["Arduino Uno", "Arduino Mega", "Raspberry Pi", "ESP32", "Micro:bit", "Raspberry Pi Pico", "STM32"],
    "project": ["robot", "website", "game", "app", "drone", "smart lamp", "weather station", "chatbot", "3D printer", "alarm clock"],
    "feature": ["a screen", "sensors", "voice control", "wheels", "LED lights", "a camera", "Bluetooth", "a speaker", "a battery", "Wi-Fi"],
    "smalltalk_q": [
        "What are you doing?", "What should we talk about?", "Tell me something interesting.",
        "I'm bored.", "What do you like to do?", "How's it going?",
    ],
    "smalltalk_a": [
        "Just here, ready to chat! What's on your mind?",
        "I'm around and happy to help. What are you up to?",
        "Here's one: octopuses have three hearts. Want another?",
        "We could talk about a project you're working on, or something you're curious about. What sounds good?",
    ],
    "obscure": [
        "What was the exact weather in Rome on March 3rd, 1421?",
        "How many red cars were sold in my city last Tuesday?",
        "What did my neighbor have for breakfast?",
    ],
    "unsure": [
        "I'm not sure enough about that to answer reliably.",
        "Honestly, I don't know that one confidently — I don't want to make something up.",
    ],
    "like_reply": [
        "Yeah, I do! {thing} is one of my favorite things to talk about.",
        "Definitely — {thing} is a great topic. What do you like about it?",
        "For sure! I really enjoy talking about {thing}.",
        "Yeah! {thing} is fun to talk about. What got you into it?",
    ],
}
SLOTS_PT = {
    "greeting": ["Oi", "Olá", "E aí", "Oi, tudo bem?", "Opa", "Oi Aila", "Bom dia", "Boa tarde", "Boa noite"],
    "greeting_reply": [
        "Oi! Tudo bem?", "Olá! Como você está?", "E aí! Como vai?",
        "Oi! Como está o seu dia?", "Opa! No que posso ajudar hoje?",
    ],
    "name": ["Théo", "Guilherme", "Maria", "Lucas", "Ana", "João", "Bia", "Rafa", "Pedro", "Júlia", "Enzo", "Laura"],
    "game": ["Minecraft", "Zelda", "Mario", "Roblox", "Terraria", "Free Fire", "Stardew Valley", "Fortnite", "Pokémon", "GTA", "Valorant"],
    "topic": ["tecnologia", "robótica", "jogos", "programação", "ciência", "espaço", "música", "filmes", "construir coisas", "eletrônica", "impressão 3D"],
    "controller": ["Arduino Uno", "Arduino Mega", "Raspberry Pi", "ESP32", "Raspberry Pi Pico"],
    "project": ["robô", "site", "jogo", "aplicativo", "drone", "estação meteorológica", "chatbot", "despertador"],
    "feature": ["uma tela", "sensores", "controle por voz", "rodas", "luzes de LED", "uma câmera", "um alto-falante", "Wi-Fi"],
    "smalltalk_q": [
        "O que você está fazendo?", "Sobre o que a gente pode conversar?", "Me conta algo interessante.",
        "Estou entediado.", "Do que você gosta de fazer?", "Como vai?",
    ],
    "smalltalk_a": [
        "Estou por aqui, pronta para conversar! No que você está pensando?",
        "Estou aqui e feliz em ajudar. O que você está fazendo?",
        "Aqui vai uma: o polvo tem três corações. Quer outra?",
        "Podemos falar sobre algum projeto seu, ou algo que te deixa curioso. O que acha?",
    ],
    "obscure": [
        "Qual foi o clima exato em Roma no dia 3 de março de 1421?",
        "Quantos carros vermelhos foram vendidos na minha cidade na terça passada?",
    ],
    "unsure": [
        "Não tenho certeza suficiente disso para responder com segurança.",
        "Sinceramente, não sei isso com confiança — não quero inventar.",
    ],
    "like_reply": [
        "Gosto sim! {thing} é um dos meus assuntos favoritos.",
        "Com certeza — {thing} é um ótimo tema. O que você curte nisso?",
        "Gosto muito! Curto bastante falar sobre {thing}.",
        "Gosto! {thing} é legal de conversar. Como você começou com isso?",
    ],
}

# -- explanation pairs (fixed, correct) --------------------------------------
EXPLANATIONS_EN = {
    "RAM": "RAM is a computer's short-term memory. It holds what the computer is actively using so it's fast to reach, and it clears when the power turns off.",
    "an API": "An API is a set of rules that lets two programs talk to each other — like a waiter taking your order to the kitchen and bringing food back.",
    "an Arduino": "An Arduino is a small, cheap circuit board you can program to control lights, motors, and sensors. It's popular for learning electronics.",
    "Python": "Python is a beginner-friendly programming language known for clear, readable code. It's used for web apps, data, AI, and automation.",
    "gravity": "Gravity is the force that pulls objects with mass toward each other. It's why things fall and why planets orbit the Sun.",
    "Wi-Fi": "Wi-Fi lets devices connect to a network and the internet using radio waves instead of cables.",
}
EXPLANATIONS_PT = {
    "RAM": "RAM é a memória de curto prazo do computador. Guarda o que ele está usando no momento, e é apagada quando desliga.",
    "uma API": "Uma API é um conjunto de regras que deixa dois programas conversarem — como um garçom levando seu pedido à cozinha e trazendo a comida.",
    "um Arduino": "Um Arduino é uma placa pequena e barata que você programa para controlar luzes, motores e sensores. É ótimo para aprender eletrônica.",
    "Python": "Python é uma linguagem de programação fácil para iniciantes, conhecida por código claro. É usada em sites, dados, IA e automação.",
    "gravidade": "Gravidade é a força que puxa objetos com massa uns para os outros. É por isso que as coisas caem e os planetas orbitam o Sol.",
}

# Current-information questions that SHOULD route to search, with the
# assistant modelling the decision to look it up.
SEARCH_QUESTIONS_EN = [
    "What's the latest news about {topic}?", "What's the newest {game} update?",
    "What's the weather today?", "What's the current price of {game}?",
    "What happened in the news today?", "What's the latest version of Python?",
]
SEARCH_REPLIES_EN = [
    "I'll look that up so I can give you the latest information.",
    "That's current info, so I'll search for it.",
    "Let me search for the up-to-date answer.",
]
SEARCH_QUESTIONS_PT = [
    "Quais são as notícias mais recentes sobre {topic}?", "Qual é a última atualização de {game}?",
    "Como está o tempo hoje?", "O que aconteceu nas notícias hoje?",
]
SEARCH_REPLIES_PT = [
    "Vou pesquisar isso para te dar a informação mais recente.",
    "Isso é informação atual, então vou buscar.",
]


def _fill(text: str, rng: random.Random, slots: dict, extra: dict | None = None) -> str:
    out = text
    pool = dict(slots)
    if extra:
        pool.update(extra)
    # Replace each {slot} with a random filler (or a provided extra value).
    for key, value in pool.items():
        token = "{" + key + "}"
        while token in out:
            choice = value if isinstance(value, str) else rng.choice(value)
            out = out.replace(token, choice, 1)
    return out


def _templates(lang: str):
    """Yield (category, requires_memory, requires_web, builder) tuples. Each
    builder(rng, slots) returns a list of (role, content) turns (excluding
    the system turn)."""
    S = SLOTS_EN if lang == "en" else SLOTS_PT
    EXPL = EXPLANATIONS_EN if lang == "en" else EXPLANATIONS_PT
    SQ = SEARCH_QUESTIONS_EN if lang == "en" else SEARCH_QUESTIONS_PT
    SR = SEARCH_REPLIES_EN if lang == "en" else SEARCH_REPLIES_PT
    name_q = "What's your name?" if lang == "en" else "Qual é o seu nome?"
    name_a = ["I'm Aila Nano!", "I'm Aila Nano — nice to meet you!", "My name's Aila Nano."] if lang == "en" \
        else ["Eu sou a Aila Nano!", "Sou a Aila Nano, prazer!", "Meu nome é Aila Nano."]
    like_q = "Do you like {thing}?" if lang == "en" else "Você gosta de {thing}?"
    fav_q = "What's my favorite game?" if lang == "en" else "Qual é o meu jogo favorito?"
    fav_pre = "My favorite game is {game}." if lang == "en" else "Meu jogo favorito é {game}."
    fav_ack = ["Nice choice!", "Cool!", "Great pick!"] if lang == "en" else ["Boa escolha!", "Legal!", "Massa!"]
    myname = "My name is {name}." if lang == "en" else "Meu nome é {name}."
    nice_meet = "Nice to meet you, {name}!" if lang == "en" else "Prazer em te conhecer, {name}!"
    whatname_q = "What's my name?" if lang == "en" else "Qual é o meu nome?"
    whatname_a = "You told me your name is {name}." if lang == "en" else "Você me disse que o seu nome é {name}."
    proj_start = "I'm building a {project}." if lang == "en" else "Estou construindo um {project}."
    proj_ask = ["That's awesome! What do you want it to do?", "Cool! What's the goal for it?"] if lang == "en" \
        else ["Que legal! O que você quer que ele faça?", "Massa! Qual é o objetivo dele?"]
    proj_ctrl_ask = ["Nice! Are you using a microcontroller?", "Cool — what are you building it with?"] if lang == "en" \
        else ["Legal! Você vai usar um microcontrolador?", "Massa — com o que você vai construir?"]
    ctrl_reply = "I'm using a {controller}." if lang == "en" else "Vou usar um {controller}."
    ctrl_ack = ["Good pick — that gives you a lot to work with. Adding {feature}?", "Nice! Are you adding {feature}?"] if lang == "en" \
        else ["Boa — dá pra fazer bastante coisa. Vai adicionar {feature}?", "Legal! Vai colocar {feature}?"]

    def greeting(rng, s):
        return [("user", _fill("{greeting}" + ("!" if rng.random() < 0.5 else ""), rng, s)),
                ("assistant", rng.choice(s["greeting_reply"]))]

    def identity(rng, s):
        return [("user", name_q), ("assistant", rng.choice(name_a))]

    def like_topic(rng, s):
        thing = rng.choice(s["topic"])
        reply = _fill(rng.choice(s["like_reply"]), rng, s, {"thing": thing})
        return [("user", _fill(like_q, rng, s, {"thing": thing})), ("assistant", reply)]

    def like_game(rng, s):
        thing = rng.choice(s["game"])
        reply = _fill(rng.choice(s["like_reply"]), rng, s, {"thing": thing})
        return [("user", _fill(like_q, rng, s, {"thing": thing})), ("assistant", reply)]

    def memory_fav(rng, s):
        g = rng.choice(s["game"])
        return [("user", _fill(fav_pre, rng, s, {"game": g})), ("assistant", rng.choice(fav_ack)),
                ("user", fav_q), ("assistant", g + ".")]

    def memory_name(rng, s):
        n = rng.choice(s["name"])
        return [("user", _fill(myname, rng, s, {"name": n})), ("assistant", _fill(nice_meet, rng, s, {"name": n})),
                ("user", whatname_q), ("assistant", _fill(whatname_a, rng, s, {"name": n}))]

    def project(rng, s):
        p = rng.choice(s["project"]); c = rng.choice(s["controller"]); f = rng.choice(s["feature"])
        return [("user", _fill(proj_start, rng, s, {"project": p})), ("assistant", rng.choice(proj_ask)),
                ("user", _fill(ctrl_reply, rng, s, {"controller": c})),
                ("assistant", _fill(rng.choice(ctrl_ack), rng, s, {"feature": f}))]

    def explanation(rng, s):
        term = rng.choice(list(EXPL))
        q = f"What is {term}?" if lang == "en" else f"O que é {term}?"
        return [("user", q), ("assistant", EXPL[term])]

    def search(rng, s):
        q = _fill(rng.choice(SQ), rng, s)
        return [("user", q), ("assistant", rng.choice(SR))]

    def small_talk(rng, s):
        return [("user", rng.choice(s["smalltalk_q"])), ("assistant", rng.choice(s["smalltalk_a"]))]

    def uncertainty(rng, s):
        return [("user", rng.choice(s["obscure"])), ("assistant", rng.choice(s["unsure"]))]

    def correction(rng, s):
        c1, c2 = rng.sample(s["controller"], 2)
        ack1 = "Nice!" if lang == "en" else "Legal!"
        line2 = f"Actually, I meant a {c2}." if lang == "en" else f"Na verdade, quis dizer um {c2}."
        got = f"Got it — a {c2}." if lang == "en" else f"Entendi — um {c2}."
        first = _fill(ctrl_reply, rng, s, {"controller": c1})
        return [("user", first), ("assistant", ack1), ("user", line2), ("assistant", got)]

    def ambiguity(rng, s):
        q = "Can you help me fix it?" if lang == "en" else "Você pode me ajudar a consertar isso?"
        a = "Sure! What are you trying to fix?" if lang == "en" else "Claro! O que você está tentando consertar?"
        return [("user", q), ("assistant", a)]

    def goodbye(rng, s):
        if lang == "en":
            q = rng.choice(["bye", "thanks!", "see you", "gotta go", "thank you"])
            a = rng.choice(["See you! Take care.", "Anytime — bye!", "Glad I could help. See you!"])
        else:
            q = rng.choice(["tchau", "valeu!", "até mais", "obrigado", "preciso ir"])
            a = rng.choice(["Tchau! Se cuida.", "De nada — até mais!", "Fico feliz em ajudar. Até!"])
        return [("user", q), ("assistant", a)]

    return [
        ("greeting", False, False, greeting),
        ("identity", False, False, identity),
        ("preference", False, False, like_topic),
        ("preference", False, False, like_game),
        ("small_talk", False, False, small_talk),
        ("memory", True, False, memory_fav),
        ("memory", True, False, memory_name),
        ("multi_turn", False, False, project),
        ("correction", True, False, correction),
        ("ambiguity", False, False, ambiguity),
        ("uncertainty", False, False, uncertainty),
        ("explanation", False, False, explanation),
        ("search", False, True, search),
        ("goodbye", False, False, goodbye),
    ]


def generate(count: int, seed: int = 1234) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    seen: set[str] = set()
    langs = ["en", "pt"]
    attempts = 0
    max_attempts = count * 40
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        lang = rng.choice(langs)
        templates = _templates(lang)
        category, req_mem, req_web, builder = rng.choice(templates)
        turns = builder(rng, SLOTS_EN if lang == "en" else SLOTS_PT)
        system = SYSTEM_EN if lang == "en" else SYSTEM_PT
        messages = [{"role": "system", "content": system}] + [
            {"role": r, "content": c} for r, c in turns
        ]
        record = {
            "category": category,
            "language": lang,
            "requires_memory": req_mem,
            "requires_web": req_web,
            "turns": len(turns),
            "messages": messages,
        }
        if validate_conversation(record) is not None:
            continue
        key = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=4000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="datasets/conversational/generated/generated.jsonl")
    args = p.parse_args()

    records = generate(args.count, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Generated {len(records)} unique conversations -> {out_path}")
    if len(records) < args.count:
        print(
            f"(requested {args.count}; the template/slot space produced "
            f"{len(records)} distinct conversations before duplicates.)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
