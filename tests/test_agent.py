"""Agent loop mechanics.

`ScriptedLLM` replaces only the language model (a test double living in the test suite). Everything
else is real: toolbox, permissions, staged edits, the MCP client talking to a real MCP server
process, memory and the ledger. The assertions check that every claimed operation really happened.
Tests that need a real local model are marked `model` and skip when none is installed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from lma.core.agent import Agent, parse_plan, report
from lma.core.state import AppState
from lma.llm.manager import ChatResult, LLMError, ToolCall
from lma.llm.runtime import find_server
from lma.mcpx.manager import ServerConfig
from tests.conftest import DEMO


class ScriptedLLM:
    """Plays back a list of model turns. A turn is text, or a list of (tool, args) calls."""

    def __init__(self, turns: list[Any]) -> None:
        self.turns = list(turns)
        self.requests: list[dict[str, Any]] = []

    def install(self, state: AppState) -> None:
        state.llm.chat_stream = self.chat_stream  # type: ignore[method-assign]
        state.llm.chat = self.chat  # type: ignore[method-assign]
        state.llm.has = lambda role: role == "chat"  # type: ignore[method-assign]

    async def chat_stream(self, messages, tools=None, **kw):  # noqa: ANN001, ANN201
        self.requests.append({"messages": messages, "tools": [t["function"]["name"] for t in tools or []]})
        if not self.turns:
            raise AssertionError("the agent asked for more model turns than scripted")
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        if isinstance(turn, str):
            yield "text", turn
            yield "done", ChatResult(text=turn, prompt_tokens=100, completion_tokens=10)
        else:
            calls = [ToolCall(f"c{i}", name, args, "") for i, (name, args) in enumerate(turn)]
            yield "done", ChatResult(tool_calls=calls, prompt_tokens=100, completion_tokens=10)

    async def chat(self, messages, **kw):  # noqa: ANN001, ANN201
        final = ChatResult()
        async for kind, payload in self.chat_stream(messages, **kw):
            if kind == "done":
                final = payload
        return final


@pytest.fixture
def state(settings, sample_repo: Path):
    settings.agent.plan_first = False
    events: list[tuple[str, Any]] = []
    st = AppState(settings, emit=lambda k, p: events.append((k, p)))
    st.events = events  # type: ignore[attr-defined]
    st.open_project(sample_repo, persist=False)
    yield st
    st.llm.shutdown()
    st.memory.close()
    if st.index:
        st.index.close()


async def test_autonomous_filesystem_and_mcp_workflow(state: AppState, sample_repo: Path):
    """Acceptance test 9: multi-step task mixing filesystem tools and MCP tools, every claim verified."""
    conn = state.mcp.add("demo", ServerConfig(transport="stdio", command=sys.executable, args=[str(DEMO), "stdio"]))
    await conn.connect()
    state.permissions.auto_approve = True
    state.settings.agent.plan_first = True
    llm = ScriptedLLM([
        "1. Read pricing.py\n2. Fix the discount\n3. Record a note on the MCP server\n4. Verify",   # plan
        [("update_plan", {"step": 1, "status": "in_progress"}), ("read_file", {"path": "shop/orders/pricing.py"})],
        [("edit_file", {"path": "shop/orders/pricing.py", "old_text": "total * percent", "new_text": "total * percent / 100"}),
         ("write_file", {"path": "CHANGELOG.md", "content": "- discount is now 0-100\n"})],
        [("mcp__demo__create_note", {"name": "release", "text": "discount fixed"})],
        [("mcp__demo__read_note", {"name": "release"}), ("mcp__demo__read_note", {"name": "missing"})],
        "Done: staged the discount fix and changelog, stored the release note.",
    ])
    llm.install(state)
    agent = Agent(state)
    try:
        res = await agent.run("fix the discount in pricing and record a release note", "agent")
        assert res.stopped == "completed" and res.steps == 5
        assert [s.text for s in state.plan][:2] == ["Read pricing.py", "Fix the discount"]

        ledger = {(e.tool, e.ok) for e in res.ledger}
        assert ("read_file", True) in ledger and ("edit_file", True) in ledger and ("mcp__demo__create_note", True) in ledger
        failed = [e for e in res.ledger if not e.ok]
        assert [e.tool for e in failed] == ["mcp__demo__read_note"], "the failing call must be recorded as failed"

        # the model saw real results, including the real error
        tool_msgs = [m["content"] for m in state.history if m.get("role") == "tool"]
        assert any("apply_discount" in m for m in tool_msgs) and any(m == "discount fixed" for m in tool_msgs)
        assert any(m.startswith("ERROR:") for m in tool_msgs)

        # MCP modification really happened on the server
        assert (await conn.call_tool("read_note", {"name": "release"})).text == "discount fixed"
        # file edits are staged, not written, until applied
        ws = state.workspace
        assert ws is not None and set(ws.pending) == {"shop/orders/pricing.py", "CHANGELOG.md"}
        assert "percent / 100" not in (sample_repo / "shop/orders/pricing.py").read_text(encoding="utf-8")
        ws.apply()
        assert "percent / 100" in (sample_repo / "shop/orders/pricing.py").read_text(encoding="utf-8")
        assert (sample_repo / "CHANGELOG.md").read_text(encoding="utf-8") == "- discount is now 0-100\n"

        text = report(res.ledger)
        assert "FAIL mcp__demo__read_note" in text and "OK   mcp__demo__create_note" in text
        # automatic project memory of what was executed
        tasks = [m for m in state.memory.list("project") if m.kind == "task"]
        assert tasks and "mcp__demo__create_note" in tasks[0].text
    finally:
        await state.mcp.close()


async def test_read_only_modes_cannot_modify(state: AppState, sample_repo: Path):
    llm = ScriptedLLM([[("edit_file", {"path": "README.md", "old_text": "# Shop", "new_text": "# Hacked"}),
                        ("write_file", {"path": "x.txt", "content": "x"})], "I cannot modify files in this mode."])
    llm.install(state)
    res = await Agent(state).run("change the readme", "ask")
    assert "edit_file" not in llm.requests[0]["tools"] and "read_file" in llm.requests[0]["tools"]
    assert all(not e.ok and "permission denied" in e.detail for e in res.ledger)
    assert not state.workspace.pending and not (sample_repo / "x.txt").exists()
    assert "No modifying operation was executed" in report(res.ledger)


async def test_denied_approval_is_reported_not_executed(state: AppState, sample_repo: Path):
    async def deny(tool: str, risk: str, summary: str) -> str:
        return "n"

    state.permissions.approver = deny
    llm = ScriptedLLM([[("create_dir", {"path": "newdir"})], "The directory was not created because permission was denied."])
    llm.install(state)
    res = await Agent(state).run("make a folder", "agent")
    assert not (sample_repo / "newdir").exists()
    assert [(e.tool, e.ok) for e in res.ledger] == [("create_dir", False)]
    assert "NOT EXECUTED" in state.history[-2]["content"]


async def test_error_recovery_limits_and_validation(state: AppState):
    state.settings.agent.max_steps = 6
    same = [("read_file", {"path": "nope.py"})]
    llm = ScriptedLLM([
        LLMError("temporary failure"),                                   # retried
        same, same, same,                                                # third identical call is blocked
        [("write_file", {"path": "bad.py", "content": "def broken(:\n"})],
        "finished",                                                      # validation finds the syntax error
        [("write_file", {"path": "bad.py", "content": "def ok():\n    return 1\n"})],
    ])
    llm.install(state)
    res = await Agent(state).run("do things", "code")
    tool_msgs = [m["content"] for m in state.history if m.get("role") == "tool"]
    assert sum("does not exist" in m for m in tool_msgs) == 2 and any("already made twice" in m for m in tool_msgs)
    assert any("Syntax check: FAILED" in m for m in tool_msgs)
    assert any(m.get("role") == "user" and "Automatic validation" in str(m.get("content")) for m in state.history)
    assert res.stopped == "step_limit" and state.workspace.pending["bad.py"].new == "def ok():\n    return 1\n"


async def test_llm_failure_is_an_error_not_a_fake_answer(state: AppState):
    state.settings.agent.max_retries = 1
    ScriptedLLM([LLMError("no chat model installed"), LLMError("no chat model installed")]).install(state)
    res = await Agent(state).run("hello", "ask")
    assert res.stopped == "error" and "no chat model" in (res.error or "") and res.text == ""
    assert state.history == []


async def test_context_compression(state: AppState):
    from lma.core import context as ctx

    ScriptedLLM(["summary: user asked about pricing; edited pricing.py"]).install(state)
    history = []
    for i in range(6):
        history += [{"role": "user", "content": f"question {i} " + "x" * 3000}, {"role": "assistant", "content": "answer " + "y" * 3000}]
    before = ctx.message_tokens(history)
    assert await ctx.compress(state, history, limit=4000)
    assert ctx.message_tokens(history) < before / 2
    assert history[0]["content"].startswith("[Summary of the earlier conversation]") and "pricing.py" in history[0]["content"]
    assert history[-2]["content"].startswith("question 5")


def test_parse_plan():
    plan = parse_plan("Here is the plan:\n1. Inspect the **domain model**\n2) Add entity `Task`\n- Validate\nthanks")
    assert [s.text for s in plan] == ["Inspect the domain model", "Add entity Task", "Validate"]


# ---------------------------------------------------------------- real local model
@pytest.mark.model
async def test_local_model_inference_and_embeddings():
    """Acceptance test 1: a downloaded GGUF model runs locally through llama.cpp (no Ollama, no admin)."""
    from lma.config import load_settings
    from lma.llm.manager import LLMManager

    s = load_settings()
    llm = LLMManager(s)
    if find_server(s) is None or llm.model_id("chat") is None:
        pytest.skip("no llama.cpp runtime / chat model installed (run /download)")
    try:
        chunks = []
        async for kind, payload in llm.chat_stream([{"role": "user", "content": "What is 2+2? Answer with the number only."}], max_tokens=64):
            if kind == "text":
                chunks.append(payload)
            elif kind == "done":
                assert payload.completion_tokens > 0
        assert "4" in "".join(chunks)
        if llm.model_id("embed"):
            vecs = await llm.embed(["def login(user, password)", "a recipe for apple pie"])
            q = (await llm.embed(["authentication code"], is_query=True))[0]
            dot = lambda a, b: sum(x * y for x, y in zip(a, b)) / (sum(x * x for x in a) ** 0.5 * sum(y * y for y in b) ** 0.5)  # noqa: E731
            assert dot(q, vecs[0]) > dot(q, vecs[1])
    finally:
        llm.shutdown()
