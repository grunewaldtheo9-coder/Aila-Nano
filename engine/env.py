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
