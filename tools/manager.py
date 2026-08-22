"""ToolManager: a uniform, error-isolated interface over Aila's tools.

Today the router calls the web-research pipeline and the memory manager
directly. That works, but it means each call site re-implements error
handling and there's no single place that knows what tools exist. This
wraps every tool behind one small contract —

    tool_manager.register(WebSearchTool(pipeline))
    result = tool_manager.execute("web_search", query="...")
    if result.success: ...

— so a failing tool (a web timeout, a memory error) returns a structured
`ToolResult(success=False, error=...)` instead of raising into the chat
loop, and new tools (a calculator, future tools for the 50M model) plug in
by subclassing `Tool`. The router/orchestration decides *whether* a tool is
needed; the manager only runs it safely and reports what happened.

This is model-agnostic: it works identically with the 20M checkpoint now
and a future 50M one.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    success: bool
    tool_name: str
    data: Any = None
    error: str | None = None
    latency_seconds: float = 0.0
    meta: dict = field(default_factory=dict)


class Tool:
    """Base class. Subclasses set `name`/`description` and implement
    `run(**kwargs)`, returning any data or raising on failure — the manager
    turns a raise into `ToolResult(success=False, ...)`."""

    name: str = "tool"
    description: str = ""

    def run(self, **kwargs) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError


class ToolManager:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("a tool must have a name")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, str]]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

    def execute(self, name: str, **kwargs) -> ToolResult:
        """Run a tool by name, always returning a ToolResult — never
        raising. An unknown tool, or any exception from the tool itself, is
        reported as a failed result so the assistant can continue."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, tool_name=name, error=f"unknown tool: {name!r}")
        start = time.monotonic()
        try:
            data = tool.run(**kwargs)
        except Exception as e:  # noqa: BLE001 — isolation is the whole point
            latency = time.monotonic() - start
            logger.warning("tool %s failed: %s", name, e)
            return ToolResult(
                success=False, tool_name=name, error=str(e), latency_seconds=latency
            )
        return ToolResult(
            success=True, tool_name=name, data=data, latency_seconds=time.monotonic() - start
        )


# -- concrete tools -----------------------------------------------------------


class WebSearchTool(Tool):
    """Wraps the web-research pipeline. `run(query=...)` returns the
    pipeline's ResearchOutcome; a pipeline that is None (no sources
    configured) is reported as a clean failure rather than a crash."""

    name = "web_search"
    description = "Look up current or external information on the web."

    def __init__(self, research_pipeline):
        self._pipeline = research_pipeline

    def run(self, query: str = "", **_) -> Any:
        if not query.strip():
            raise ValueError("empty search query")
        if self._pipeline is None:
            raise RuntimeError("web search is not configured")
        return self._pipeline.research(query)


class MemorySearchTool(Tool):
    """Wraps relevance-gated memory retrieval. `run(query=..., k=...)`
    returns the list of relevant memories (possibly empty)."""

    name = "memory_search"
    description = "Retrieve the user's relevant remembered facts."

    def __init__(self, memory_manager):
        self._memory = memory_manager

    def run(self, query: str = "", k: int = 5, **_) -> Any:
        if self._memory is None:
            raise RuntimeError("memory is not configured")
        return self._memory.get_relevant_memories(query, k=k)
