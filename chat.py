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
import sqlite3
import sys
import uuid

from engine import AilaEngine, EngineSettings
from engine.env import load_env
from engine.support import SUPPORT_EMAIL, support_message
from training.checkpoint import CheckpointNotDownloadedError
from tools.identity import PRODUCT_NAME, RELEASE_STAGE

# Public release name. Comes from tools/identity.py so the banner, the
# support report and Aila's own answers can never disagree about what she
# is called.
VERSION = RELEASE_STAGE

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
  /study <topic>     look a topic up now and remember it forever
  /knows             how much Aila has learned so far
  /support           how to report a problem to Aila Company Solutions
  /feedback <text>   same, with your message included
  exit, quit         leave

You can also just type things naturally, e.g.:
  "Remember that my name is Theo"
  "Forget that my name is Theo"
  "What do you remember about me?"
"""


def print_banner(device: str) -> None:
    print("=====================================")
    print(f"{PRODUCT_NAME} {VERSION}")
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

    # /remember and /forget deliberately go through the *same*
    # deterministic handler as typing "remember that ..." in chat. Going
    # straight to the memory API instead let the two paths drift: the
    # slash command skipped the empty-content guard (so "/forget it"
    # printed the literal word "None") and the MAX_MEMORY_CHARS cap (so a
    # very long "/remember ..." was stored whole and later crowded the
    # question out of the context window).
    elif cmd == "/remember":
        if not arg:
            print("Usage: /remember <something to remember>")
        else:
            agent = engine.get_agent(state["agent"])
            print(
                agent._handle_memory_command(f"remember that {arg}")
                or "I couldn't tell what to remember there. Try: /remember my name is Theo"
            )

    elif cmd == "/forget":
        if not arg:
            print("Usage: /forget <something to forget>")
        else:
            agent = engine.get_agent(state["agent"])
            print(
                agent._handle_memory_command(f"forget that {arg}")
                or "I couldn't tell what to forget there. Try: /forget my name is Theo"
            )

    elif cmd == "/memories":
        agent = engine.get_agent(state["agent"])
        print(agent._handle_memory_command("what do you remember about me?"))

    elif cmd == "/study":
        if not arg:
            print("Usage: /study <topic>   e.g. /study photosynthesis")
        else:
            print(f"Looking up '{arg}'...")
            _, message = engine.study(arg)
            print(message)

    elif cmd == "/knows":
        count = engine.known_fact_count
        sources = engine.research_sources
        print(f"I've learned {count} fact(s) so far, and I can answer them with no internet.")
        print(
            "Live sources: " + ", ".join(sources)
            if sources
            else "Live sources: none (I can only use what I already know)."
        )

    elif cmd in ("/support", "/feedback"):
        print(support_message(engine, VERSION, note=arg))

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
    except CheckpointNotDownloadedError as e:
        # The single most common install failure. Its raw form is a
        # torch pickling traceback that says nothing useful, so it gets
        # its own branch with the actual fix.
        print(f"\nCould not start.\n\n{e}")
        print(f"\nStill stuck? Email {SUPPORT_EMAIL}")
        return 1
    except FileNotFoundError as e:
        print(f"\nCould not start: {e}")
        print("See docs/TRAINING.md to train a tokenizer/model first.")
        print(f"Still stuck? Email {SUPPORT_EMAIL}")
        return 1
    except sqlite3.DatabaseError as e:
        # A corrupted memory/knowledge database used to kill startup with
        # a raw traceback. The data files are recoverable by deleting
        # them (they're caches//user state, not the model), so say that.
        print(f"\nCould not start: a storage file is corrupted ({e}).")
        print(
            "Delete the affected database file and restart — memory and knowledge\n"
            "will be rebuilt empty. Default locations:\n"
            "  memory/data/aila_memory.db\n"
            "  vectordb/index/knowledge.db\n"
            "  knowledge/data/aila_knowledge.db"
        )
        print(f"Still stuck? Email {SUPPORT_EMAIL}")
        return 1

    if not engine.is_trained:
        print(
            "\n(Note: no trained checkpoint found — responses will be gibberish "
            "until you train Aila Nano. See docs/TRAINING.md.)"
        )

    if engine.web_search_active:
        print(f"\nLooking things up with: {', '.join(engine.research_sources)}")
    else:
        print(
            "\n(Looking things up is OFF — no sources are enabled.\n"
            " Aila can still chat, do maths, and use what she already knows,\n"
            " but she can't learn anything new.)"
        )

    # One bounded round of self-directed study, at most once a day. Aila
    # re-visits questions she previously failed to answer, so the same
    # question works next time — and works offline from then on.
    #
    # The "Studying..." line is printed *before* the round, because the
    # round blocks: announcing it afterwards left the user watching a
    # frozen screen with no explanation.
    if engine.study_due():
        print("Studying something new...")
    report = engine.run_daily_study()
    summary = report.summary()
    if summary:
        print(summary)

    print(f"I know {engine.known_fact_count} fact(s) I can use without the internet.")

    print("\nReady.\n")

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
