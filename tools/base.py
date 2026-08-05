"""Extension point for future agent capabilities: internet search, file
reading, PDF reading, Python execution, calendar access, plugins, and so
on (see the roadmap in docs/ARCHITECTURE.md).

Nothing in this module is wired into generation yet — Aila Nano has no
function-calling training today, so nothing calls `Tool.run()`
automatically. This module exists purely so that when those capabilities
are built, they have one obvious, consistent shape to implement against
(`ToolRegistry` in `tools/registry.py`) instead of each needing its own
ad-hoc integration into `agents/` and `chat.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """A single named capability an agent or interface could invoke.

    Example (future, not implemented):

        class WebSearchTool(Tool):
            name = "web_search"
            description = "Search the internet and return top results."

            def run(self, query: str) -> str:
                ...
    """

    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool and return a plain-text result suitable for
        feeding back to the model as additional context.
        """
        raise NotImplementedError
