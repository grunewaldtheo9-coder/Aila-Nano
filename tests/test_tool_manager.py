"""ToolManager: registration, structured results, and error isolation."""

from __future__ import annotations

from tools.manager import MemorySearchTool, Tool, ToolManager, ToolResult, WebSearchTool


class _EchoTool(Tool):
    name = "echo"
    description = "echoes its input"

    def run(self, text="", **_):
        return text.upper()


class _BoomTool(Tool):
    name = "boom"
    description = "always fails"

    def run(self, **_):
        raise RuntimeError("kaboom")


def test_register_and_list():
    tm = ToolManager()
    tm.register(_EchoTool())
    assert tm.has("echo")
    assert {t["name"] for t in tm.list_tools()} == {"echo"}


def test_successful_execution_returns_structured_result():
    tm = ToolManager()
    tm.register(_EchoTool())
    r = tm.execute("echo", text="hi")
    assert isinstance(r, ToolResult)
    assert r.success and r.tool_name == "echo" and r.data == "HI"
    assert r.error is None
    assert r.latency_seconds >= 0


def test_unknown_tool_is_a_failed_result_not_a_crash():
    tm = ToolManager()
    r = tm.execute("nope")
    assert r.success is False
    assert "unknown tool" in r.error


def test_a_failing_tool_is_isolated():
    tm = ToolManager()
    tm.register(_BoomTool())
    r = tm.execute("boom")
    assert r.success is False
    assert "kaboom" in r.error
    # The manager itself did not raise.


# -- concrete tools -----------------------------------------------------------


class _FakePipeline:
    def __init__(self):
        self.calls = []

    def research(self, query):
        self.calls.append(query)
        return {"answer": f"result for {query}"}


def test_web_search_tool_runs_the_pipeline():
    tm = ToolManager()
    pipe = _FakePipeline()
    tm.register(WebSearchTool(pipe))
    r = tm.execute("web_search", query="latest news")
    assert r.success and pipe.calls == ["latest news"]


def test_web_search_tool_reports_missing_pipeline_cleanly():
    tm = ToolManager()
    tm.register(WebSearchTool(None))
    r = tm.execute("web_search", query="x")
    assert r.success is False and "not configured" in r.error


def test_web_search_tool_rejects_empty_query():
    tm = ToolManager()
    tm.register(WebSearchTool(_FakePipeline()))
    r = tm.execute("web_search", query="   ")
    assert r.success is False and "empty" in r.error


class _FakeMemory:
    def get_relevant_memories(self, query, k=5):
        return [{"content": "The user's name is Theo."}][:k]


def test_memory_search_tool_returns_relevant_memories():
    tm = ToolManager()
    tm.register(MemorySearchTool(_FakeMemory()))
    r = tm.execute("memory_search", query="name", k=3)
    assert r.success and r.data[0]["content"].endswith("Theo.")
