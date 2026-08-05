"""Tests for the tools/ extensibility stub — no concrete tools exist yet
(see docs/ARCHITECTURE.md's roadmap), so this only verifies the registry
contract a future tool implementation would rely on.
"""

from __future__ import annotations

import pytest

from tools import Tool, ToolRegistry


class _EchoTool(Tool):
    name = "echo"
    description = "Echoes its input back."

    def run(self, text: str = "") -> str:
        return text


def test_registry_starts_empty():
    registry = ToolRegistry()
    assert len(registry) == 0
    assert registry.list() == []


def test_register_and_get_tool():
    registry = ToolRegistry()
    tool = _EchoTool()
    registry.register(tool)

    assert len(registry) == 1
    assert registry.list() == ["echo"]
    assert registry.get("echo") is tool
    assert tool.run(text="hi") == "hi"


def test_register_duplicate_name_raises():
    registry = ToolRegistry()
    registry.register(_EchoTool())
    with pytest.raises(ValueError):
        registry.register(_EchoTool())


def test_get_unknown_tool_raises():
    registry = ToolRegistry()
    with pytest.raises(KeyError):
        registry.get("nonexistent")


def test_tool_is_abstract():
    with pytest.raises(TypeError):
        Tool()  # can't instantiate without implementing run()
