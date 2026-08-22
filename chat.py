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

import getpass
import platform
import sqlite3
import sys
import uuid

from engine import AilaEngine, EngineSettings
from engine.env import load_env, save_env_var
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
  /model             show the loaded model's size and architecture
  /debug             toggle debug details (memory/routing/model)
  /serper            add or replace your Serper key (search the whole web)
  /support           how to report a problem to Aila Company Solutions
  /feedback <text>   same, with your message included
  exit, quit         leave

You can also just type things naturally, e.g.:
  "Remember that my name is Theo"
  "Forget that my name is Theo"
  "What do you remember about me?"
"""


SERPER_PROMPT_SHOWN_KEY = "serper_prompt_shown"


def offer_serper_setup(engine, force: bool = False) -> None:
    """Ask once, on first run, whether the user wants to add a Serper key.

    Wikipedia answers most questions and needs no key, so this is an
    offer rather than a requirement — Enter skips it, and it is never
    asked twice (the answer is recorded in the knowledge store).

    Deliberately skipped when stdin is not a terminal: a scripted or
    piped run would otherwise swallow the first line of real input as if
    it were an API key.
    """
    if engine.settings.serper_api_key and not force:
        return  # already configured
    if not sys.stdin.isatty():
        return  # piped/scripted input — never consume a line
    store = getattr(engine, "knowledge_store", None)
    if not force and store is not None and store.get_meta(SERPER_PROMPT_SHOWN_KEY):
        return  # asked before; don't nag on every start

    print()
    print("-" * 55)
    print("For better researching, you can give me a Serper API key and")
    print("I'll be able to search the whole internet, not just Wikipedia.")
    print("It's free at https://serper.dev — or just press Enter to skip.")
    print("(You can add it later by typing /serper)")
    print("-" * 55)
    try:
        # getpass keeps the key off the screen. Some terminals don't
        # support it; a visible prompt is better than crashing.
        try:
            api_key = getpass.getpass("Paste your Serper API key (hidden), or Enter to skip: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise
        except Exception:
            api_key = input("Paste your Serper API key, or Enter to skip: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        api_key = ""

    if store is not None:
        store.set_meta(SERPER_PROMPT_SHOWN_KEY, "1")

    if not api_key:
        print("No problem — I'll use Wikipedia. Type /serper any time to add one.\n")
        return

    print("Checking that key...")
    ok, reason = engine.check_serper_api_key(api_key)
    if not ok:
        # Not saved: a key that doesn't work is worse than none, because
        # every lookup would then fail instead of falling back cleanly.
        print(f"{reason} Nothing was saved — I'll keep using Wikipedia.\n")
        return

    try:
        save_env_var("SERPER_API_KEY", api_key)
    except (OSError, ValueError) as e:
        print(f"{reason} But I couldn't save it to .env ({e}).")
        print("It will work for this session only.\n")
        engine.set_serper_api_key(api_key)
        return

    engine.set_serper_api_key(api_key)
    print(f"{reason} Saved to your .env file — I'll use it from now on.\n")


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
            engine.set_status_callback(lambda msg: print(f"({msg})", flush=True))
            try:
                _, message = engine.study(arg)
            finally:
                engine.set_status_callback(None)
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

    elif cmd == "/serper":
        offer_serper_setup(engine, force=True)

    elif cmd == "/model":
        meta = getattr(engine, "_checkpoint_metadata", None)
        if meta:
            print(
                f"Model: {meta['model_name']} {meta['model_size']} "
                f"({meta['parameters']:,} params)"
            )
            print(
                f"  d_model={meta['d_model']} layers={meta['layers']} "
                f"heads={meta['heads']} ffn={meta['ffn']} ctx={meta['context_length']}"
            )
        else:
            cfg = engine.model.cfg
            trained = "trained" if engine.is_trained else "UNTRAINED"
            print(
                f"Model: aila_nano ({trained}) — d_model={cfg.d_model} "
                f"layers={cfg.n_layers} heads={cfg.n_heads} ctx={cfg.max_seq_len}"
            )

    elif cmd == "/debug":
        state["debug"] = not state.get("debug", False)
        print(f"Debug mode {'on' if state['debug'] else 'off'}.")

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


def _parse_cli_args(argv: list[str] | None = None):
    import argparse

    p = argparse.ArgumentParser(description="Chat with Aila Nano.")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a model checkpoint (e.g. checkpoints/chat_50m/best.pt). "
        "The architecture is read from the checkpoint itself, so a future 50M "
        "checkpoint loads with no code changes.",
    )
    p.add_argument("--debug", action="store_true", help="Show memory/routing/model details each turn.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_env()  # .env (gitignored) → os.environ; real env vars always win
    args = _parse_cli_args(argv)

    # A --checkpoint overrides the configured path; the engine auto-detects
    # the architecture from the checkpoint's own stored config.
    settings = (
        EngineSettings(checkpoint_path=args.checkpoint) if args.checkpoint else EngineSettings()
    )
    print_banner(device=settings.resolved_device())

    def on_progress(msg: str) -> None:
        print(msg)

    try:
        engine = AilaEngine(settings, on_progress=on_progress)
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

    # First run: offer to add a Serper key. Before the sources line, so
    # that line reflects the answer.
    offer_serper_setup(engine)

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
    # Same reason as during a chat turn: a study round makes several
    # network calls and would otherwise look like a frozen screen.
    engine.set_status_callback(lambda msg: print(f"  ({msg})", flush=True))
    try:
        report = engine.run_daily_study()
    finally:
        engine.set_status_callback(None)
    summary = report.summary()
    if summary:
        print(summary)

    print(f"I know {engine.known_fact_count} fact(s) I can use without the internet.")

    print("\nReady.\n")

    state = {
        "conversation_id": new_conversation_id(),
        "agent": engine.settings.default_agent,
        "debug": args.debug,
    }
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

            # "Aila: " is printed on the first piece of the reply, not
            # before the turn starts. A lookup happens *inside*
            # chat_stream and can take a couple of seconds; printing the
            # prefix up front would leave "Aila: " hanging with nothing
            # after it, and any "(Searching Wikipedia...)" line would
            # land in the middle of her sentence.
            started_reply = False

            def show_reply_prefix() -> None:
                nonlocal started_reply
                if not started_reply:
                    print("Aila: ", end="", flush=True)
                    started_reply = True

            engine.set_status_callback(lambda msg: print(f"({msg})", flush=True))
            try:
                for delta in engine.chat_stream(
                    state["conversation_id"], user_input, agent_name=state["agent"]
                ):
                    show_reply_prefix()
                    print(delta, end="", flush=True)
            except Exception as e:  # keep the session alive on a bad turn
                show_reply_prefix()
                print(f"\n[error generating a response: {e}]", end="")
            finally:
                engine.set_status_callback(None)
            show_reply_prefix()  # an empty reply still gets its prefix
            print("\n")
    finally:
        engine.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
