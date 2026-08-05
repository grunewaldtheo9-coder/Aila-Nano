"""Registry of available `Tool` instances — mirrors `agents/registry.py`'s
pattern deliberately, so the two feel like one consistent system. Empty
by default; nothing in Phase 2 registers a tool here. See the roadmap in
docs/ARCHITECTURE.md for what's planned.
"""

from __future__ import annotations

from tools.base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool '{name}'. Registered: {list(self._tools)}")
        return self._tools[name]

    def list(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)
