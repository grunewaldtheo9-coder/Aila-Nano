#!/usr/bin/env python3
"""Aila Nano — terminal chat.

The single entrypoint for talking to Aila Nano:

    python chat.py

Loads the tokenizer, model, memory, knowledge base, and agents once
(via `engine.AilaEngine`), then starts an interactive loop. Type `exit`
or `quit` to leave, or `/help` to see what else it can do.

This file is intentionally a thin terminal shell around `engine.AilaEngine`
— it does no AI work itself. That's on purpose: the engine is the product;
this is just today's interface to it. A future desktop, mobile, or web
interface would import the same `AilaEngine` and look nothing like this
file.
"""

from __future__ import annotations

import platform
import sys
import uuid

from engine import AilaEngine, EngineSettings
from engine.env import load_env

VERSION = "2.0"

COMMANDS = """\
Commands:
  /help              show this message
  /agents            list available agents
  /agent <name>      switch the active agent
  /new               start a new conversation (fresh memory context)
  /history           show the current conversation so far
  /remember <text>   save something to long-term memory
  /forget <text>     remove a matching remembered fact
  /memories          list everything currently remembered
  /learn <path>      index a local .txt/.md file into the knowledge base
  exit, quit         leave

You can also just type things naturally, e.g.:
  "Remember that my name is Theo"
  "Forget that my name is Theo"
  "What do you remember about me?"
"""


def print_banner(device: str) -> None:
    print("=====================================")
    print(f"Aila Nano v{VERSION}")
    print("Small Language Model")
    print(f"Python {platform.python_version()}")
    print(f"{'GPU' if device == 'cuda' else 'CPU'} Mode")
    print()


def new_conversation_id() -> str:
    return f"session-{uuid.uuid4().hex[:8]}"


def handle_command(engine: AilaEngine, command: str, state: dict) -> bool:
    """Returns False if the command should end the session, True otherwise."""
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("exit", "quit"):
        return False

    if cmd == "/help":
        print(COMMANDS)

    elif cmd == "/agents":
        current = state["agent"]
        for name in engine.available_agents():
            marker = "*" if name == current else " "
            print(f"  {marker} {name}")

    elif cmd == "/agent":
        if not arg:
            print(f"Current agent: {state['agent']}")
        elif arg in engine.available_agents():
            state["agent"] = arg
            print(f"Switched to '{arg}'.")
        else:
            print(f"Unknown agent '{arg}'. Try /agents to see the list.")

    elif cmd == "/new":
        state["conversation_id"] = new_conversation_id()
        print(f"Started a new conversation ({state['conversation_id']}).")

    elif cmd == "/history":
        history = engine.memory.conversation.get_history(state["conversation_id"])
        if not history:
            print("(empty)")
        for turn in history:
            print(f"  {turn['role']}: {turn['content']}")

    elif cmd == "/remember":
        if not arg:
            print("Usage: /remember <something to remember>")
        else:
            fact_id = engine.memory.remember_fact(arg)
            print(f"Remembered (id={fact_id}).")

    elif cmd == "/forget":
        if not arg:
            print("Usage: /forget <something to forget>")
        else:
            agent = engine.get_agent(state["agent"])
            print(agent._handle_memory_command(f"forget that {arg}"))

    elif cmd == "/memories":
        agent = engine.get_agent(state["agent"])
        print(agent._handle_memory_command("what do you remember about me?"))

    elif cmd == "/learn":
        if not arg:
            print("Usage: /learn <path to a .txt/.md/.jsonl/.csv/.log file>")
        else:
            try:
                n = engine.learn_file(arg)
                print(f"Indexed {n} chunk(s) from '{arg}'.")
            except (ValueError, OSError) as e:
                print(f"Could not learn '{arg}': {e}")

    else:
        print(f"Unknown command '{cmd}'. Type /help to see what's available.")

    return True


def main() -> int:
    load_env()  # .env (gitignored) → os.environ; real env vars always win
    print_banner(device=EngineSettings().resolved_device())

    def on_progress(msg: str) -> None:
        print(msg)

    try:
        engine = AilaEngine(EngineSettings(), on_progress=on_progress)
    except FileNotFoundError as e:
        print(f"\nCould not start: {e}")
        print("See docs/TRAINING.md to train a tokenizer/model first.")
        return 1

    if not engine.is_trained:
        print(
            "\n(Note: no trained checkpoint found — responses will be gibberish "
            "until you train Aila Nano. See docs/TRAINING.md.)"
        )

    print("Ready.\n")

    state = {"conversation_id": new_conversation_id(), "agent": engine.settings.default_agent}
    if state["agent"] not in engine.available_agents():
        state["agent"] = engine.available_agents()[0]

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.startswith("/") or user_input.lower() in ("exit", "quit"):
                if not handle_command(engine, user_input, state):
                    print("Goodbye!")
                    break
                continue

            print("Aila: ", end="", flush=True)
            try:
                for delta in engine.chat_stream(
                    state["conversation_id"], user_input, agent_name=state["agent"]
                ):
                    print(delta, end="", flush=True)
            except Exception as e:  # keep the session alive on a bad turn
                print(f"\n[error generating a response: {e}]", end="")
            print("\n")
    finally:
        engine.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
