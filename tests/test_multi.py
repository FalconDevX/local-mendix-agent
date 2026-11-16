"""Streaming output, loop protection, additional context folders and multi-agent runs.

As in test_agent.py only the language model is replaced (`RuleLLM` answers from a function of the
request); toolbox, permissions, staging and the ledger are the real ones.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from lma.core.agent import Agent, repetition_start
from lma.core.state import AppState
from lma.core.team import parse_subtasks, run_team
from lma.editing.editor import EditError
from lma.llm.manager import ChatResult, ToolCall
from lma.ui.rendering import THEME, Renderer, split_stable


class RuleLLM:
    """A model double whose answer is computed from the request: fn(system, messages, tools) -> text | [(tool, args)]."""

    def __init__(self, fn) -> None:  # noqa: ANN001
        self.fn = fn
        self.requests: list[dict[str, Any]] = []

    def install(self, state: AppState) -> None:
        state.llm.chat_stream = self.chat_stream  # type: ignore[method-assign]
        state.llm.chat = self.chat  # type: ignore[method-assign]
        state.llm.has = lambda role: role == "chat"  # type: ignore[method-assign]

    async def chat_stream(self, messages, tools=None, **kw):  # noqa: ANN001, ANN201
        names = [t["function"]["name"] for t in tools or []]
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        self.requests.append({"system": system, "messages": messages, "tools": names, "model": kw.get("model")})
        turn = self.fn(system, messages, names)
        if isinstance(turn, str):
            yield "text", turn
            yield "done", ChatResult(text=turn, prompt_tokens=50, completion_tokens=5)
        else:
            calls = [ToolCall(f"c{len(self.requests)}_{i}", name, args, "") for i, (name, args) in enumerate(turn)]
            yield "done", ChatResult(tool_calls=calls, prompt_tokens=50, completion_tokens=5)

    async def chat(self, messages, **kw):  # noqa: ANN001, ANN201
        final = ChatResult()
        async for kind, payload in self.chat_stream(messages, **kw):
            if kind == "done":
                final = payload
        return final


def tool_results(messages: list[dict[str, Any]]) -> list[str]:
    return [m["content"] for m in messages if m.get("role") == "tool"]


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
    st._close_indexes()


@pytest.fixture
def library(tmp_path: Path) -> Path:
    """A second folder outside the project."""
    root = tmp_path / "shared-lib"
    (root / "lib").mkdir(parents=True)
    (root / "lib" / "money.py").write_text('"""Money helpers."""\n\n\ndef to_cents(amount: float) -> int:\n    return round(amount * 100)\n', encoding="utf-8")
    (root / "NOTES.md").write_text("# Shared library\nto_cents converts a float amount to integer cents.\n", encoding="utf-8")
    return root


# ---------------------------------------------------------------- streaming output
def _terminal(height: int = 20) -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=True, width=80, height=height, theme=THEME, color_system=None), buf


def test_streamed_answer_is_printed_exactly_once():
    """Regression: an answer taller than the window used to be re-printed on every refresh."""
    import time

    console, buf = _terminal(height=20)
    ui = Renderer(console, glyphs="ascii")
    ui.turn_start()
    for i in range(80):
        for piece in (f"Paragraph {i:02d} ", "of the ", "streamed answer.\n\n"):
            ui.emit("text", piece)
        if i % 10 == 0:
            time.sleep(0.15)  # let the live region refresh a few times
    ui.emit("text_end", None)
    ui.turn_end()
    out = buf.getvalue()
    for i in range(80):
        assert out.count(f"Paragraph {i:02d} of the streamed answer.") == 1, f"paragraph {i} printed {out.count(f'Paragraph {i:02d} ')} times"


def test_long_code_block_is_not_repeated_and_not_split():
    console, buf = _terminal(height=12)
    ui = Renderer(console, glyphs="ascii")
    ui.turn_start()
    ui.emit("text", "Here is the file:\n\n```python\n")
    for i in range(60):
        ui.emit("text", f"value_{i:02d} = {i}\n" + ("\n" if i % 7 == 0 else ""))
    ui.emit("text", "```\n\nDone.\n")
    ui.turn_end()
    out = buf.getvalue()
    assert all(out.count(f"value_{i:02d} = {i}") == 1 for i in range(60))
    assert out.count("Done.") == 1


def test_split_stable_keeps_open_fences_and_unfinished_paragraphs():
    assert split_stable("one\n\ntwo") == ("one\n\n", "two")
    assert split_stable("no blank line yet\nstill going") == ("", "no blank line yet\nstill going")
    text = "intro\n\n```py\na = 1\n\nb = 2\n"
    assert split_stable(text) == ("intro\n\n", "```py\na = 1\n\nb = 2\n")       # blank line inside an open fence is not a cut
    assert split_stable(text + "```\n\nafter") == (text + "```\n\n", "after")


def test_sub_agent_text_is_not_streamed_to_the_screen():
    from lma.core.state import current_agent

    console, buf = _terminal()
    ui = Renderer(console, glyphs="ascii")
    token = current_agent.set("explorer-1")
    try:
        ui.emit("text", "private notes of the sub-agent\n\n")
        ui.emit("text_end", None)
        ui.emit("tool_call", ("read_file", "path=a.py", "READING PROJECT"))
    finally:
        current_agent.reset(token)
    out = buf.getvalue()
    assert "private notes" not in out and "explorer-1" in out and "read_file" in out


# ---------------------------------------------------------------- loop protection
def test_repetition_detection():
    intro = "The function validates the order and then computes the total with tax applied.\n"
    assert repetition_start(intro * 1) is None
    looping_lines = intro + "I will now read the file.\n" * 9
    cut = repetition_start(looping_lines)
    assert cut is not None and looping_lines[:cut].count("I will now read the file.") == 1
    block = "First I check the configuration file, then I look at the settings module to be sure. "
    cut = repetition_start(intro + block * 4)
    assert cut is not None and (intro + block * 4)[:cut].count("First I check") == 1
    code = "def a():\n    return 1\n\n\ndef b():\n    return 2\n\n\ndef c():\n    return 3\n\n\ndef d():\n    return 4\n\n\ndef e():\n    return 5\n"
    assert repetition_start(code) is None
    table = "| name | value |\n|---|---|\n" + "".join(f"| item {i} | {i * 7} |\n" for i in range(30))
    assert repetition_start(table) is None


async def test_agent_stops_a_model_that_repeats_itself(state: AppState):
    stuck = "Let me look at the pricing module to understand the discount.\n" * 40
    llm = RuleLLM(lambda system, messages, tools: stuck)
    llm.install(state)
    res = await Agent(state).run("explain the discount", "ask")
    assert res.stopped == "loop" and len(llm.requests) == 1
    assert res.text.count("Let me look at the pricing module") == 1
    assert state.history[-1]["content"].count("Let me look at") == 1, "the loop must not be kept in the conversation"


async def test_agent_stops_when_the_same_tool_call_keeps_coming(state: AppState):
    llm = RuleLLM(lambda system, messages, tools: [("read_file", {"path": "README.md"})])
    llm.install(state)
    res = await Agent(state).run("read the readme", "ask")
    assert res.stopped == "loop" and "repeating the same tool call" in res.text
    assert len(llm.requests) == 5, "2 real calls, then 3 refused repeats, then stop (not the 25-step limit)"
    assert sum(1 for e in res.ledger if e.tool == "read_file") == 2


# ---------------------------------------------------------------- additional context folders
async def test_additional_folder_read_search_edit_undo(state: AppState, sample_repo: Path, library: Path, tmp_path: Path):
    ws = state.workspace
    assert ws is not None
    money = (library / "lib" / "money.py").resolve()
    with pytest.raises(EditError):
        ws.read(str(money))                                   # not authorised yet
    state.add_dir(library)
    assert state.settings.extra_dirs[str(sample_repo.resolve())] == [str(library.resolve())]
    box = Agent(state).toolbox

    assert "to_cents" in (await box.execute("read_file", {"path": str(money)})).text
    assert "to_cents" in (await box.execute("read_file", {"path": "shared-lib/lib/money.py"})).text   # folder-name alias
    assert "to_cents" in (await box.execute("read_file", {"path": "lib/money.py"})).text              # falls back to the added folder
    hits = (await box.execute("grep", {"pattern": "def to_cents|def order_total"})).text
    assert f"{money.as_posix()}:4:" in hits and "shop/orders/pricing.py:7:" in hits
    assert "additional context folder" in (await box.execute("list_dir", {})).text

    for _, idx, _ in state.indexes():
        await idx.update(None)
    found = (await box.execute("search_code", {"query": "convert amount to cents"})).text
    assert money.as_posix() in found
    assert any(path == money.as_posix() for _, _, path, _ in state.find_symbol("to_cents"))

    out = await box.execute("edit_file", {"path": str(money), "old_text": "round(amount * 100)", "new_text": "int(round(amount * 100))"})
    assert out.ok and list(ws.pending) == [money.as_posix()]
    assert "int(round" not in money.read_text(encoding="utf-8"), "staged only"
    ws.apply()
    assert "int(round" in money.read_text(encoding="utf-8")
    ws.undo()
    assert "int(round" not in money.read_text(encoding="utf-8")

    outside = tmp_path / "secret.txt"
    outside.write_text("nope", encoding="utf-8")
    assert not (await box.execute("read_file", {"path": str(outside)})).ok
    assert not (await box.execute("read_file", {"path": "../secret.txt"})).ok

    state.remove_dir("shared-lib")
    assert not (await box.execute("read_file", {"path": str(money)})).ok
    assert str(sample_repo.resolve()) not in state.settings.extra_dirs


def test_additional_folders_are_restored_with_the_project(settings, sample_repo: Path, library: Path):
    settings.extra_dirs[str(sample_repo.resolve())] = [str(library.resolve()), str(library / "gone")]
    st = AppState(settings)
    try:
        st.open_project(sample_repo, persist=False)
        assert st.workspace.extra == [library.resolve()], "a folder that no longer exists is skipped"
        assert "Additional context folders" in __import__("lma.core.context", fromlist=["x"]).build_system(st, "ask", "")
    finally:
        st.llm.shutdown()
        st.memory.close()
        st._close_indexes()


# ---------------------------------------------------------------- multi-agent
def lead_and_subs(system: str, messages: list[dict[str, Any]], tools: list[str]):
    results = tool_results(messages)
    if "ROLE explorer" in system:
        if not results:
            return [("read_file", {"path": "shop/orders/pricing.py"}), ("edit_file", {"path": "README.md", "old_text": "# Shop", "new_text": "# Hacked"})]
        return "REPORT explorer: apply_discount is in shop/orders/pricing.py:13 and subtracts total * percent."
    if "ROLE coder" in system:
        if not results:
            return [("edit_file", {"path": "shop/orders/pricing.py", "old_text": "total * percent", "new_text": "total * percent / 100"})]
        return "REPORT coder: staged the fix in shop/orders/pricing.py."
    if "ROLE reviewer" in system:
        return "REPORT reviewer: no problems found."
    if not results:  # the lead
        return [("delegate", {"role": "explorer", "task": "Find where the discount is computed."}),
                ("delegate", {"role": "coder", "task": "In shop/orders/pricing.py make apply_discount treat percent as 0-100."})]
    return "Both sub-agents are done: " + " | ".join(r[:40] for r in results)


async def test_main_agent_delegates_to_parallel_sub_agents(state: AppState, sample_repo: Path):
    llm = RuleLLM(lead_and_subs)
    llm.install(state)
    res = await Agent(state).run("fix the discount handling in the shop project please", "agent")
    assert res.stopped == "completed"

    lead = [r for r in llm.requests if "ROLE " not in r["system"]]
    subs = [r for r in llm.requests if "ROLE " in r["system"]]
    assert "delegate" in lead[0]["tools"] and "update_plan" in lead[0]["tools"]
    assert all("delegate" not in r["tools"] and "update_plan" not in r["tools"] for r in subs), "sub-agents cannot spawn agents"
    explorer_tools = next(r["tools"] for r in subs if "ROLE explorer" in r["system"])
    assert "read_file" in explorer_tools and "edit_file" not in explorer_tools

    # each sub-agent has its own conversation: it starts from the task, not from the lead's history
    first_sub = next(r for r in subs if "ROLE coder" in r["system"])
    assert [m["role"] for m in first_sub["messages"]] == ["system", "user"] and "treat percent as 0-100" in first_sub["messages"][1]["content"]
    # the lead got both reports back as tool results
    delegated = tool_results(state.history)
    assert any("REPORT explorer" in t for t in delegated) and any("REPORT coder" in t for t in delegated)
    assert not any("ROLE " in str(m.get("content")) for m in state.history)

    by_agent = {(e.agent.split("-")[0] if e.agent else "lead", e.tool, e.ok) for e in res.ledger}
    assert ("explorer", "read_file", True) in by_agent and ("coder", "edit_file", True) in by_agent
    assert ("explorer", "edit_file", False) in by_agent, "a read-only sub-agent must be refused even while the lead is in /agent mode"
    assert ("lead", "delegate", True) in by_agent
    assert set(state.workspace.pending) == {"shop/orders/pricing.py"}
    assert "# Hacked" not in (sample_repo / "README.md").read_text(encoding="utf-8")

    kinds = [k for k, _ in state.events]
    assert kinds.count("agent_start") == 2 and kinds.count("agent_end") == 2


async def test_delegation_can_be_switched_off_and_is_absent_in_read_only_modes(state: AppState):
    box = Agent(state).toolbox
    names = lambda mode, **kw: [t["function"]["name"] for t in box.schemas(mode, **kw)]  # noqa: E731
    assert "delegate" in names("agent") and "delegate" not in names("ask") and "delegate" not in names("agent", sub=True)
    state.settings.agent.multi_agent = False
    assert "delegate" not in names("agent")


def test_parse_subtasks():
    fenced = 'Sure:\n```json\n[{"role": "explorer", "task": "find X"}, {"role": "CODER", "task": "change Y"}, {"role": "pilot", "task": "fly"}]\n```'
    assert [(t.role, t.task) for t in parse_subtasks(fenced)] == [("explorer", "find X"), ("coder", "change Y")]
    listed = "1. explorer: look at the auth module\n2. **coder** - add the check\nreviewer: verify it"
    assert [t.role for t in parse_subtasks(listed)] == ["explorer", "coder", "reviewer"]
    assert parse_subtasks("I do not know how to split this [see above].") == []


async def test_team_pipeline_explores_builds_reviews_and_reports(state: AppState, sample_repo: Path):
    def rules(system: str, messages: list[dict[str, Any]], tools: list[str]):
        user = str(messages[-1].get("content"))
        if "Split the user's task" in user:
            return '[{"role": "explorer", "task": "Find where the discount is computed."}, {"role": "coder", "task": "Fix apply_discount in shop/orders/pricing.py."}]'
        if "Your team finished" in system:
            assert "REPORT explorer" in user and "REPORT coder" in user and "REPORT reviewer" in user
            return "Final: the discount fix is staged in shop/orders/pricing.py and was reviewed."
        return lead_and_subs(system, messages, tools)

    llm = RuleLLM(rules)
    llm.install(state)
    state.settings.agent.role_models["explorer"] = "not-installed-model"
    res = await run_team(state, Agent(state).toolbox, "fix the discount handling")
    assert res.stopped == "completed" and res.text.startswith("Final:")

    order = [p[1] for k, p in state.events if k == "agent_start"]
    assert order == ["explorer", "coder", "reviewer"], "investigate, then build, then an automatic review of the staged edits"
    coder = next(r for r in llm.requests if "ROLE coder" in r["system"] and len(r["messages"]) == 2)
    assert "REPORT explorer" in coder["messages"][1]["content"], "workers receive what the explorers found"
    reviewer = next(r for r in llm.requests if "ROLE reviewer" in r["system"])
    assert "shop/orders/pricing.py" in reviewer["messages"][1]["content"]
    assert all(r["model"] is None for r in llm.requests), "a role model that is not installed falls back to the chat model"
    assert set(state.workspace.pending) == {"shop/orders/pricing.py"}
    assert [m["role"] for m in state.history] == ["user", "assistant"] and state.history[1]["content"].startswith("Final:")


async def test_team_falls_back_to_one_worker_when_the_split_is_unusable(state: AppState):
    def rules(system: str, messages: list[dict[str, Any]], tools: list[str]):
        if "Split the user's task" in str(messages[-1].get("content")):
            return "I think we should carefully consider the options."
        return lead_and_subs(system, messages, tools)

    RuleLLM(rules).install(state)
    res = await run_team(state, Agent(state).toolbox, "fix the discount handling")
    roles = [p[1] for k, p in state.events if k == "agent_start"]
    assert roles == ["coder", "reviewer"] and res.stopped == "completed"


async def test_role_model_is_passed_to_the_model_server(state: AppState):
    from lma.llm.registry import Installed, ModelSpec

    state.llm.registry.installed["tiny"] = Installed(spec=ModelSpec(id="tiny", role="chat", file="tiny.gguf"), path="tiny.gguf")
    state.settings.agent.role_models["explorer"] = "tiny"
    llm = RuleLLM(lambda system, messages, tools: "REPORT explorer: nothing to do.")
    llm.install(state)
    from lma.core.team import run_subagent

    res = await run_subagent(state, Agent(state).toolbox, "explorer", "look around")
    assert res.text.startswith("REPORT") and llm.requests[-1]["model"] == "tiny"
