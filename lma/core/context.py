"""Prompt construction, retrieval of relevant context and context-window management."""
from __future__ import annotations

import json
from typing import Any

from lma.core.state import AppState
from lma.llm.manager import LLMError, estimate_tokens
from lma.tools import git

BASE = """You are Local Mendix Agent, an AI software engineer running fully locally on the user's Windows laptop.
Rules that always apply:
- Use the provided tools to inspect things instead of guessing. Never invent file contents, tool names or results.
- Only report actions that a tool call actually performed. If a tool failed or was denied, say so.
- File edits made with write_file/edit_file are STAGED for the user's review; describe them as staged, not as saved.
- Be concise. Answer in the language the user writes in.
- When the task is finished, stop calling tools and give a short final summary."""

MODES: dict[str, str] = {
    "ask": "MODE /ask: answer questions. You may read and search the project. You cannot modify anything.",
    "plan": ("MODE /plan: investigate with read-only tools if needed, then output ONLY a numbered implementation plan "
             "(max 8 concrete steps, one line each). Do not modify anything."),
    "agent": ("MODE /agent: complete the task autonomously, step by step. Follow the plan, call update_plan as steps "
              "start/finish, inspect results of every tool call and adapt when something fails."),
    "code": ("MODE /code: implement the requested code change. Read the relevant files first, then make minimal, "
             "correct edits with edit_file/write_file. Keep the existing style. Fix any reported syntax errors."),
    "review": ("MODE /review: review the code or the current git diff. Report concrete problems as "
               "`file:line - severity - issue - suggested fix`, most severe first. Do not modify anything."),
    "mendix": "MODE /mendix: you are a Mendix expert working on the app open in Studio Pro through its MCP server.",
    "vision": "MODE /vision: analyse the attached image(s) and answer the user's question about them.",
}

PLAN_PROMPT = ("Before acting, write the plan for this task as a numbered list of at most 8 short, concrete steps. "
               "Output only the list, no introduction.")


def build_system(state: AppState, mode: str, retrieved: str) -> str:
    parts = [BASE, MODES.get(mode, MODES["agent"])]
    if mode == "mendix":
        parts.append(state.mendix.guidance())
    if state.root:
        facts = [f"Open project: {state.root}"]
        if state.mendix_project:
            facts.append(state.mendix_project.summary())
        if g := git.summary(state.root):
            facts.append(f"Git: {g}")
        if summary := state.memory.get_kind("project_summary"):
            facts.append("Project summary (from memory):\n" + summary[:1500])
        parts.append("\n".join(facts))
    else:
        parts.append("No project folder is open (the user can run /open <folder>). File tools are unavailable.")
    others = [n for n, c in state.mcp.connections.items() if c.connected and n != "mendix"]
    if others:
        parts.append("Other connected MCP servers: " + ", ".join(others))
    if state.plan and mode != "plan":
        marks = {"pending": " ", "in_progress": ">", "done": "x", "failed": "!", "skipped": "-"}
        parts.append("Current plan:\n" + "\n".join(f"{i}. [{marks.get(s.status, ' ')}] {s.text}" for i, s in enumerate(state.plan, 1)))
    if retrieved:
        parts.append(retrieved)
    return "\n\n".join(parts)


async def retrieve(state: AppState, query: str, mode: str, first_turn: bool) -> str:
    """Relevant memory + code snippets for this request (never the whole history or repo)."""
    blocks: list[str] = []
    try:
        mems = await state.memory.search(query, state.embedder, limit=4)
    except Exception:  # noqa: BLE001
        mems = []
    mems = [m for m in mems if m.kind != "project_summary"]
    if mems:
        blocks.append("Relevant memory:\n" + "\n".join(f"- [{m.scope}/{m.kind}] {m.text[:400]}" for m in mems))
    if first_turn:
        prev = state.memory.previous_context(state.memory.project_id, 3)
        if prev:
            lines = [f"- {title}" + (f" => {summary[:300]}" if summary else "") for _, title, summary in prev]
            blocks.append("Earlier sessions on this project:\n" + "\n".join(lines))
    if state.index is not None and mode in ("ask", "code", "review", "agent", "plan") and state.index.counts()["files"]:
        try:
            hits = await state.index.search(query, state.embedder, limit=3)
        except Exception:  # noqa: BLE001
            hits = []
        if hits:
            snippets = [f"--- {h.path}:{h.start}-{h.end}\n" + "\n".join(h.text.splitlines()[:18]) for h in hits]
            blocks.append("Possibly relevant code (verify with read_file before relying on it):\n" + "\n".join(snippets))
    return "\n\n".join(blocks)


def message_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            total += sum(estimate_tokens(p.get("text", "")) if p.get("type") == "text" else 800 for p in content)
        else:
            total += estimate_tokens(content or "")
        if m.get("tool_calls"):
            total += estimate_tokens(json.dumps(m["tool_calls"]))
        total += 4
    return total


def budget(state: AppState, tools: list[dict[str, Any]]) -> int:
    ctx = state.settings.llm.context_size
    return max(1024, ctx - state.settings.llm.max_output_tokens - estimate_tokens(json.dumps(tools)) - 256)


async def compress(state: AppState, history: list[dict[str, Any]], limit: int, force: bool = False) -> bool:
    """Keep `history` under `limit` tokens. Returns True if anything was changed.

    Stage 1: shorten old tool results. Stage 2: summarise everything before the last user turn(s).
    """
    if message_tokens(history) <= limit * 0.85 and not force:
        return False
    changed = False
    for m in history[:-4]:
        if m.get("role") == "tool" and isinstance(m.get("content"), str) and len(m["content"]) > 400:
            m["content"] = m["content"][:300] + "\n... [older tool output shortened]"
            changed = True
    if message_tokens(history) <= limit * 0.7 and not force:
        return changed
    user_idx = [i for i, m in enumerate(history) if m.get("role") == "user"]
    if len(user_idx) < 2:
        # a single huge turn: drop the oldest tool exchanges, keeping the request itself
        while len(history) > 6 and message_tokens(history) > limit * 0.7:
            del history[1]
            while len(history) > 1 and history[1].get("role") == "tool":
                del history[1]
            changed = True
        return changed
    cut = user_idx[-1] if len(user_idx) < 4 else user_idx[-2]
    old = history[:cut]
    if not old:
        return changed
    state.emit("activity", ("THINKING", f"compressing context ({len(old)} older messages)"))
    transcript = "\n".join(f"{m.get('role')}: {str(m.get('content'))[:700]}" for m in old if m.get("content"))[-9000:]
    try:
        res = await state.llm.chat([
            {"role": "system", "content": "Summarise this conversation for your own later use: goals, decisions, files changed, "
                                          "open problems. Max 200 words. Facts only."},
            {"role": "user", "content": transcript}], max_tokens=400, temperature=0.1)
        summary = res.text.strip() or "(summary unavailable)"
    except LLMError:
        summary = "(earlier conversation dropped to fit the context window)"
    state.memory.set_summary(summary)
    history[:cut] = [{"role": "user", "content": "[Summary of the earlier conversation]\n" + summary},
                     {"role": "assistant", "content": "Understood, continuing from that summary."}]
    return True
