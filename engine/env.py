"""Minimal .env loader — no python-dotenv dependency.

Loads KEY=VALUE lines from a .env file into os.environ (without
overwriting variables that are already set, so real environment always
wins over the file). Called by chat.py at startup; any other interface
(tests, a future server) can call `load_env()` the same way.

Secrets policy: .env is gitignored (see .gitignore); .env.example is the
committed, secret-free template. Never put real keys in tracked files.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def load_env(path: str | Path = ".env") -> int:
    """Load a .env file if it exists. Returns the number of variables
    actually set (existing environment variables are never overwritten).
    Silently does nothing if the file doesn't exist — a missing .env is
    a normal state, not an error."""
    env_path = Path(path)
    if not env_path.exists():
        return 0

    loaded = 0
    for line_no, raw in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or not key.replace("_", "").isalnum():
            logger.warning(".env:%d skipped malformed line", line_no)
            continue
        if key in os.environ:
            continue
        os.environ[key] = value
        loaded += 1
    return loaded


def save_env_var(key: str, value: str, path: str | Path = ".env") -> None:
    """Write `KEY=value` into the .env file, replacing any existing line
    for that key and leaving every other line untouched.

    Used by the first-run setup so a pasted API key survives a restart.
    Two things this must get right, because it is the only place in the
    project that writes a secret to disk:

    - The file is created with owner-only permissions (0600) where the
      platform supports it, and existing permissions are tightened the
      same way. Windows ignores the mode; there is no portable
      equivalent, so this is best-effort rather than a guarantee.
    - `.env` is gitignored (see .gitignore), which is what keeps the key
      out of the repository. This function refuses to write anywhere
      whose name isn't a .env file, so a caller cannot be tricked into
      dropping a secret into a tracked file.
    """
    if not key or not key.replace("_", "").isalnum():
        raise ValueError(f"Refusing to write malformed env key {key!r}")
    if "\n" in value or "\r" in value:
        raise ValueError("Refusing to write a multi-line env value")

    env_path = Path(path)
    if not env_path.name.startswith(".env"):
        raise ValueError(f"Refusing to write secrets to {env_path.name!r}; expected a .env file")

    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    replaced = False
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.partition("=")[0].strip() == key:
            lines[i] = f"{key}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{key}={value}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        env_path.chmod(0o600)
    except OSError:  # Windows / unusual filesystems — nothing portable to do
        logger.debug("could not tighten permissions on %s", env_path)

    os.environ[key] = value
