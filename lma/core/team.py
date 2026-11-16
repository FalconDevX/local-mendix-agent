"""Multi-agent mode.

Two ways to use several agents:
  * `delegate` tool  - the main agent decides by itself to hand a sub-task to a sub-agent,
  * `/team <task>`   - a fixed pipeline that does not depend on the model choosing to delegate
                       (small local models rarely do): the lead splits the task, explorers
                       investigate in parallel, workers build, a reviewer checks, the lead reports.

Sub-agents share the project, the staging area and the permission system, but each has a private
conversation, so a long investigation does not eat the lead's context window.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import re
import time
from dataclasses import dataclass
from typing import Any

from lma.core import context as ctx
from lma.core.agent import Agent, TurnResult, record_turn
from lma.core.roles import REPORT_RULES, ROLES
from lma.core.state import AppState
from lma.llm.manager import LLMError
from lma.project.scanner import tree
from lma.tools.toolbox import Toolbox

_counter = itertools.count(1)
_slots: dict[int, asyncio.Semaphore] = {}

SPLIT_PROMPT = """Split the user's task into sub-tasks for your team. Roles:
{roles}
Rules:
- 1 to {max_tasks} sub-tasks. Use explorers first when the code must be understood, then coder/mendix tasks, optionally one reviewer.
- Sub-tasks of the same role run in parallel, so they must not depend on each other and must not edit the same file.
- A sub-agent knows NOTHING about this conversation: each task must be self-contained (goal, files, constraints).
- A simple task needs a single sub-task.
Answer with ONLY a JSON array, no other text:
[{{"role": "explorer", "task": "..."}}, {{"role": "coder", "task": "..."}}]"""

FINAL_PROMPT = ("Your team finished. Write the final answer for the user in the user's language: what was found, "
                "what was changed (file edits are STAGED for review, not saved), what failed, what is still open. "
                "Use only the reports below; do not invent anything.")


@dataclass
class SubTask:
    role: str
    task: str


def _semaphore(state: AppState) -> asyncio.Semaphore:
    """At most `agent.parallel_agents` sub-agents generate at the same time (one semaphore per event loop)."""
    loop_id = id(asyncio.get_running_loop())
    if loop_id not in _slots:
        _slots.clear()
        _slots[loop_id] = asyncio.Semaphore(max(1, state.settings.agent.parallel_agents))
    return _slots[loop_id]


async def run_subagent(state: AppState, toolbox: Toolbox, role_name: str, task: str, background: str = "") -> TurnResult:
    """Run one sub-agent to completion and return its report."""
    role = ROLES[role_name]
    name = f"{role_name}-{next(_counter)}"
    model = state.settings.agent.role_models.get(role_name)
    if model and model not in state.llm.registry.installed:
        state.emit("activity", ("WARNING", f"model {model} for role {role_name} is not installed: using the chat model"))
        model = None
    agent = Agent(state, toolbox, name=name, history=[], model=model, instructions=role.prompt + "\n" + REPORT_RULES,
                  max_steps=state.settings.agent.subagent_max_steps)
    prompt = task if not background else f"{task}\n\nBackground from earlier work of the team:\n{background}"
    async with _semaphore(state):
        state.emit("agent_start", (name, role_name, task))
        try:
            res = await agent.run(prompt, role.mode)
        except asyncio.CancelledError:
            state.emit("agent_end", (name, TurnResult("", stopped="cancelled")))
            raise
        state.emit("agent_end", (name, res))
    return res


def parse_subtasks(text: str, max_tasks: int = 5) -> list[SubTask]:
    """Read the lead's split. Accepts a JSON array (possibly wrapped in prose or a code fence) or
    `role: task` lines; anything else yields an empty list and the caller falls back to one task."""
    tasks: list[SubTask] = []
    for m in re.finditer(r"\[", text):
        try:
            data, _ = json.JSONDecoder().raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(data, list) and data and all(isinstance(x, dict) for x in data):
            for item in data:
                role, task = str(item.get("role", "")).strip().lower(), str(item.get("task", "")).strip()
                if role in ROLES and task:
                    tasks.append(SubTask(role, task))
            if tasks:
                return tasks[:max_tasks]
    for line in text.splitlines():
        m2 = re.match(r"^\s*(?:\d+[.)]|[-*])?\s*\**(" + "|".join(ROLES) + r")\**\s*[:\-]\s*(.+\S)", line, re.I)
        if m2:
            tasks.append(SubTask(m2.group(1).lower(), m2.group(2).strip()))
    return tasks[:max_tasks]


async def _split(state: AppState, toolbox: Toolbox, text: str, retrieved: str) -> list[SubTask]:
    roles = {n: r for n, r in ROLES.items() if n != "mendix" or state.mendix.connected}
    facts = ""
    if state.root:
        facts = "\n\nProject tree:\n" + tree(state.root, state.settings.index.ignore_dirs, max_depth=2, max_entries=60)
    system = ctx.build_system(state, "team", retrieved, with_plan=False) + facts
    prompt = SPLIT_PROMPT.format(roles="\n".join(f"- {n}: {r.summary}" for n, r in roles.items()), max_tasks=5)
    try:
        res = await state.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": f"Task: {text}\n\n{prompt}"}],
                                   max_tokens=700, temperature=0.1)
        tasks = [t for t in parse_subtasks(res.text) if t.role in roles]
    except LLMError as exc:
        state.emit("activity", ("WARNING", f"could not split the task ({str(exc).splitlines()[0][:100]}); running it as one sub-task"))
        tasks = []
    if not tasks:  # the lead produced nothing usable: one worker does the whole task
        tasks = [SubTask("mendix" if state.mode == "mendix" and state.mendix.connected else "coder", text)]
    return tasks


def _reports(done: list[tuple[SubTask, TurnResult]], limit: int = 2500) -> str:
    parts = []
    for task, res in done:
        body = (res.text or f"(no report; stopped: {res.stopped} {res.error or ''})").strip()
        parts.append(f"### {task.role}: {task.task[:160]}\n{body[:limit]}")
    return "\n\n".join(parts)


async def run_team(state: AppState, toolbox: Toolbox, text: str) -> TurnResult:
    """The /team pipeline. Appends the request and the final answer to the main conversation."""
    t0, ledger_mark = time.time(), len(state.ledger)
    toolbox.refresh_mcp()
    state.emit("activity", ("PLANNING", "splitting the task for the team"))
    try:
        retrieved = await ctx.retrieve(state, text, "team", first_turn=False)
        tasks = await _split(state, toolbox, text, retrieved)
        state.emit("phase", "Team: " + ", ".join(f"{sum(t.role == r for t in tasks)} {r}" for r in ROLES if any(t.role == r for t in tasks)))
        done: list[tuple[SubTask, TurnResult]] = []
        for phase in (1, 2, 3):
            batch = [t for t in tasks if ROLES[t.role].phase == phase]
            if phase == 3 and not batch and state.workspace and state.workspace.pending:
                # edits were staged and nobody was asked to check them: always add a review
                files = ", ".join(list(state.workspace.pending)[:12])
                diff = state.workspace.diff()[:6000]  # handed over directly: staged edits are not in git yet
                batch = [SubTask("reviewer", f"Review the staged edits in: {files}. They were made for this request: {text}\n"
                                             f"The edits are not written to disk or git yet; this is the full diff:\n```diff\n{diff}\n```")]
            if not batch:
                continue
            background = _reports(done)
            results = await asyncio.gather(*(run_subagent(state, toolbox, t.role, t.task, background) for t in batch))
            done += list(zip(batch, results))
        final = await _final(state, text, done)
    except asyncio.CancelledError:
        state.emit("activity", ("CANCELLED", "team run cancelled; completed tool calls are listed below"))
        res = TurnResult("", stopped="cancelled", ledger=state.ledger[ledger_mark:], seconds=time.time() - t0)
        state.history += [{"role": "user", "content": text}, {"role": "assistant", "content": "[cancelled by the user]"}]
        return res
    failed = [f"{t.role}: {r.error or r.stopped}" for t, r in done if r.stopped not in ("completed", "step_limit")]
    res = TurnResult(final, ledger=state.ledger[ledger_mark:], steps=sum(r.steps for _, r in done), seconds=time.time() - t0)
    if done and len(failed) == len(done):
        res.stopped, res.error = "error", "every sub-agent failed: " + "; ".join(failed)
    state.history += [{"role": "user", "content": text}, {"role": "assistant", "content": final or "(no result)"}]
    record_turn(state, text, "team", res)
    return res


async def _final(state: AppState, text: str, done: list[tuple[SubTask, TurnResult]]) -> str:
    """The lead writes the answer from the reports (streamed to the user)."""
    reports = _reports(done)
    if len(done) == 1 and done[0][1].text:
        # one sub-agent: its report already is the answer, a rewrite would only cost time
        state.emit("text", done[0][1].text)
        state.emit("text_end", None)
        return done[0][1].text
    staged = ""
    if state.workspace and state.workspace.pending:
        staged = "\n\nStaged file changes: " + ", ".join(f"{c.kind} {c.rel}" for c in state.workspace.pending.values())
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": ctx.BASE + "\n\n" + FINAL_PROMPT},
        {"role": "user", "content": f"User request: {text}\n\nTeam reports:\n{reports}{staged}"}]
    state.emit("activity", ("THINKING", "writing the final answer"))
    out = ""
    try:
        async for kind, payload in state.llm.chat_stream(messages, max_tokens=state.settings.llm.max_output_tokens):
            if kind == "text":
                out += payload
                state.emit("text", payload)
        state.emit("text_end", None)
    except LLMError as exc:
        state.emit("text_end", None)
        state.emit("activity", ("WARNING", f"final summary failed ({str(exc).splitlines()[0][:100]}); showing the raw reports"))
        out = reports
        state.emit("text", out)
        state.emit("text_end", None)
    return out.strip()
