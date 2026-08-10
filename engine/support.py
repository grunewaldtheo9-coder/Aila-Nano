"""Support / feedback report — what to send to Aila Company Solutions
when something is wrong.

Kept in the engine rather than in `chat.py` because every future
interface (desktop, mobile, web) needs the same thing, and because the
one rule that matters here is worth testing in one place: **the report
must never contain a secret.** It reports whether a Serper key is
configured, never the key itself. `tests/test_security.py` asserts that.

There is no mail sending here on purpose. Sending mail would need SMTP
credentials shipped with the app, which is both a secret-handling
problem and a spam vector. Showing the user an address and a
ready-to-paste report is the honest version, and it works offline.
"""

from __future__ import annotations

import platform
import sys

SUPPORT_EMAIL = "mailailacompanysolutions@gmail.com"


def build_support_report(engine, version: str, note: str = "") -> str:
    """A plain-text block the user can paste into an email.

    `engine` is an `AilaEngine`; passed loosely so this module stays
    importable without constructing one (and so tests can pass a stub).
    """
    lines = [
        "Aila Nano — support report",
        "=" * 40,
        f"Aila Nano version : {version}",
        f"Python            : {platform.python_version()}",
        f"System            : {platform.system()} {platform.release()}",
        f"Machine           : {platform.machine()}",
    ]

    lines.append(f"Device            : {_safe(lambda: engine.settings.resolved_device())}")
    lines.append(f"Model parameters  : {_safe(lambda: f'{engine.parameter_count():,}')}")
    lines.append(f"Trained model     : {_safe(lambda: 'yes' if engine.is_trained else 'NO')}")
    lines.append(f"Checkpoint        : {_safe(lambda: engine.model_loaded_from or '(none)')}")
    # Source *names* only — never the key. See module docstring.
    lines.append(
        f"Lookup sources    : {_safe(lambda: ', '.join(engine.research_sources) or 'none')}"
    )
    lines.append(f"Facts learned     : {_safe(lambda: engine.known_fact_count)}")
    lines.append(f"Remembered facts  : {_safe(lambda: len(engine.memory.all_memories()))}")

    if note:
        lines += ["", "What happened:", note.strip()]

    lines += [
        "",
        "(Please also paste the last few messages you sent and the replies",
        " you got — that is usually what makes a problem findable.)",
    ]
    return "\n".join(lines)


def _safe(getter) -> str:
    """Never let a broken engine break the very report meant to describe
    it — a support command that crashes is worse than useless."""
    try:
        value = getter()
    except Exception as e:  # noqa: BLE001 — reporting, not control flow
        return f"(unavailable: {type(e).__name__})"
    return str(value)


def support_message(engine, version: str, note: str = "") -> str:
    """The full thing `/support` and `/feedback` print."""
    what = "feedback" if note else "a problem report"
    return (
        f"Send {what} to: {SUPPORT_EMAIL}\n"
        f"\nCopy everything between the lines below into the email:\n\n"
        f"{build_support_report(engine, version, note)}\n"
        + "=" * 40
        + f"\n\nTip: you can also write the address by hand — {SUPPORT_EMAIL}\n"
        "Nothing is sent automatically; Aila never emails anything on its own."
    )
