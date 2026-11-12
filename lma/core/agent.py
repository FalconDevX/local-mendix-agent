"""The agent execution loop: retrieve -> plan -> call tools -> inspect -> revise -> validate -> report."""
from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from lma.core import context as ctx
from lma.core.state import AppState, LedgerEntry, PlanStep
from lma.editing.editor import validate
from lma.llm.manager import ChatResult, LLMError
from lma.tools.image import SCREENSHOT_PROMPT, LoadedImage
from lma.tools.toolbox import Toolbox

PLANNING_MODES = {"agent", "code", "mendix"}
_STEP = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.*\S)")


@dataclass
class TurnResult:
    text: str
    ledger: list[LedgerEntry] = field(default_factory=list)
    steps: int = 0
    stopped: str = "completed"  # completed | step_limit | cancelled | error
    error: str | None = None
    seconds: float = 0.0


def parse_plan(text: str) -> list[PlanStep]:
    steps = []
    for line in text.splitlines():
        m = _STEP.match(line)
        if m:
            steps.append(PlanStep(re.sub(r"\*\*|`", "", m.group(1)).strip()[:200]))
    return steps[:10]


def report(ledger: list[LedgerEntry]) -> str:
    """Factual summary of what the tools did this turn. Generated from the ledger, not by the model."""
    if not ledger:
        return ""
    acts = [e for e in ledger if e.risk != "read"]
    lines = [f"Operations: {len(ledger)} tool call(s), {sum(1 for e in ledger if not e.ok)} failed."]
    for e in acts:
        lines.append(f"  {'OK  ' if e.ok else 'FAIL'} {e.tool}({e.summary})" + ("" if e.ok else f" - {e.detail[:160]}"))
    for e in ledger:
        if not e.ok and e.risk == "read":
            lines.append(f"  FAIL {e.tool}({e.summary}) - {e.detail[:160]}")
    if not any(e.ok for e in acts):
        lines.append("  No modifying operation was executed.")
    return "\n".join(lines)


class Agent:
    def __init__(self, state: AppState, toolbox: Toolbox | None = None) -> None:
        self.s = state
        self.toolbox = toolbox or Toolbox(state)
        self._turns = 0

    # ---- public --------------------------------------------------------
    async def run(self, text: str, mode: str | None = None, images: list[LoadedImage] | None = None) -> TurnResult:
        mode = mode or self.s.mode
        previous_mode = self.s.permissions.mode
        self.s.permissions.mode = mode
        mark, ledger_mark, t0 = len(self.s.history), len(self.s.ledger), time.time()
        try:
            result = await self._turn(text, mode, images or [])
        except asyncio.CancelledError:
            del self.s.history[mark:]
            self.s.history += [{"role": "user", "content": text}, {"role": "assistant", "content": "[cancelled by the user]"}]
            self.s.emit("activity", ("CANCELLED", "turn cancelled; completed tool calls are listed below"))
            result = TurnResult("", stopped="cancelled")
        except LLMError as exc:
            del self.s.history[mark:]
            result = TurnResult("", stopped="error", error=str(exc))
        finally:
            self.s.permissions.mode = previous_mode
        result.ledger = self.s.ledger[ledger_mark:]
        result.seconds = time.time() - t0
        self._persist(text, mode, result)
        return result

    # ---- one turn ------------------------------------------------------
    async def _turn(self, text: str, mode: str, images: list[LoadedImage]) -> TurnResult:
        s = self.s
        self.toolbox.refresh_mcp()
        s.emit("activity", ("THINKING", "retrieving project context and memory"))
        retrieved = await ctx.retrieve(s, text, mode, first_turn=self._turns == 0)
        self._turns += 1
        if mode == "vision" and images and not s.llm.chat_supports_vision() and s.llm.model_id("vision"):
            # Pure image question: let the vision model answer directly instead of relaying its
            # description through a (possibly much weaker) text model.
            for img in images:
                s.emit("activity", ("THINKING", "image " + img.meta()))
            s.emit("activity", ("THINKING", f"vision model {s.llm.model_id('vision')} is reading {len(images)} image(s)"))
            prompt = SCREENSHOT_PROMPT + f"\nThen answer the user's question: {text}"
            answer = await s.llm.describe_images(prompt, [img.data for img in images], max_tokens=1200)
            s.emit("text", answer)
            s.emit("text_end", None)
            labels = ", ".join(img.label for img in images)
            s.history += [{"role": "user", "content": f"{text}\n[attached image(s): {labels}]"}, {"role": "assistant", "content": answer}]
            return TurnResult(answer, steps=1)
        user_msg = await self._user_message(text, images)
        s.history.append(user_msg)
        tools = [] if mode == "vision" else self.toolbox.schemas(mode)

        # Plan real tasks only: short follow-ups ("continue", "yes", a quick question) keep the current plan.
        if mode in PLANNING_MODES and s.settings.agent.plan_first and len(text.split()) >= 6:
            s.plan = []
            await self._make_plan(mode, retrieved)

        counts: dict[str, int] = {}
        validated = False
        final = ""
        max_steps = s.settings.agent.max_steps
        for step in range(1, max_steps + 1):
            limit = ctx.budget(s, tools)
            await ctx.compress(s, s.history, limit)
            messages = [{"role": "system", "content": ctx.build_system(s, mode, retrieved)}, *s.history]
            s.context_tokens = ctx.message_tokens(messages)
            res = await self._generate(messages, tools, limit)
            if res.prompt_tokens:
                s.context_tokens = res.prompt_tokens + res.completion_tokens
            if not res.tool_calls:
                final = res.text.strip()
                s.history.append({"role": "assistant", "content": final})
                if mode == "plan":
                    s.plan = parse_plan(final) or s.plan
                    s.emit("plan", s.plan)
                # validate staged modifications once before finishing
                problems = self._validate_staged() if not validated else []
                validated = True
                if problems and step < max_steps:
                    s.emit("activity", ("VALIDATING", f"{len(problems)} staged file(s) have syntax errors, asking the model to fix"))
                    s.history.append({"role": "user", "content": "Automatic validation found problems in staged files. Fix them:\n" + "\n".join(problems)})
                    continue
                self._close_plan()
                return TurnResult(final, steps=step)
            s.history.append({
                "role": "assistant", "content": res.text or None,
                "tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.raw_arguments or json.dumps(c.arguments)}}
                               for c in res.tool_calls],
            })
            for call in res.tool_calls:
                key = call.name + json.dumps(call.arguments, sort_keys=True)
                counts[key] = counts.get(key, 0) + 1
                if call.parse_error:
                    out = f"ERROR: could not parse the arguments of {call.name}: {call.parse_error}. Send valid JSON."
                elif counts[key] > 2:
                    out = ("ERROR: this exact call was already made twice with the same result. Do not repeat it; "
                           "use the earlier result, try a different approach, or finish with a summary.")
                else:
                    out = (await self.toolbox.execute(call.name, call.arguments)).text
                s.history.append({"role": "tool", "tool_call_id": call.id, "content": out})
        self._close_plan(interrupted=True)
        note = f"Stopped after the step limit ({max_steps}). The task may be incomplete; say 'continue' to go on."
        s.history.append({"role": "assistant", "content": note})
        return TurnResult(note, steps=max_steps, stopped="step_limit")

    async def _user_message(self, text: str, images: list[LoadedImage]) -> dict[str, Any]:
        """Attach images directly if the chat model can see; otherwise route them through the vision model."""
        s = self.s
        if not images:
            return {"role": "user", "content": text}
        for img in images:
            s.emit("activity", ("THINKING", "image " + img.meta()))
        if s.llm.chat_supports_vision():
            parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
            for img in images:
                parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(img.data).decode()}})
            return {"role": "user", "content": parts}
        if s.llm.model_id("vision") is None:
            note = (f"[{len(images)} image(s) were attached, but no local vision model is installed, so they could NOT be "
                    "analysed. Tell the user to run /download with a vision model from /models.]")
            return {"role": "user", "content": f"{text}\n\n{note}"}
        s.emit("activity", ("THINKING", f"vision model {s.llm.model_id('vision')} is reading {len(images)} image(s)"))
        prompt = SCREENSHOT_PROMPT + (f"\nThe user's question: {text}" if text else "")
        described = []
        for img in images:
            described.append(f"[Vision model analysis of {img.label}]\n" + await s.llm.describe_images(prompt, [img.data]))
        return {"role": "user", "content": text + "\n\n" + "\n\n".join(described)}

    async def _make_plan(self, mode: str, retrieved: str) -> None:
        s = self.s
        s.emit("activity", ("PLANNING", "creating a plan"))
        tool_names = ", ".join(t["function"]["name"] for t in self.toolbox.schemas(mode))
        messages = [{"role": "system", "content": ctx.build_system(s, mode, retrieved) + f"\n\nTools you will be able to use: {tool_names}"},
                    *_plain(s.history[-6:]), {"role": "user", "content": ctx.PLAN_PROMPT}]
        try:
            res = await s.llm.chat(messages, max_tokens=500)
        except LLMError as exc:
            s.emit("activity", ("PLANNING", f"planning skipped: {exc}"))
            return
        plan = parse_plan(res.text)
        if plan:
            s.plan = plan
            s.emit("plan", s.plan)

    async def _generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], limit: int) -> ChatResult:
        """One streamed model call with retries and context-overflow recovery."""
        s = self.s
        last: LLMError | None = None
        for attempt in range(s.settings.agent.max_retries + 1):
            try:
                result = ChatResult()
                thinking = False
                async for kind, payload in s.llm.chat_stream(messages, tools=tools or None):
                    if kind == "text":
                        s.emit("text", payload)
                    elif kind == "reasoning" and not thinking:
                        thinking = True
                        s.emit("activity", ("THINKING", "model is reasoning"))
                    elif kind == "done":
                        result = payload
                s.emit("text_end", result)
                return result
            except LLMError as exc:
                last = exc
                s.emit("text_end", None)
                if "context" in str(exc).lower():
                    await ctx.compress(s, s.history, limit, force=True)
                    messages[1:] = s.history
                s.emit("activity", ("THINKING", f"model call failed ({str(exc).splitlines()[0][:120]}), retry {attempt + 1}"))
                await asyncio.sleep(1 + attempt)
        raise last or LLMError("model call failed")

    def _validate_staged(self) -> list[str]:
        ws = self.s.workspace
        if ws is None or not ws.pending:
            return []
        problems = []
        for ch in ws.pending.values():
            if ch.new is not None and (err := validate(ch.rel, ch.new)):
                problems.append(f"{ch.rel}: {err}")
        self.s.emit("activity", ("VALIDATING", f"{len(ws.pending)} staged file(s): " + ("all pass syntax checks" if not problems else "; ".join(problems))))
        return problems

    def _close_plan(self, interrupted: bool = False) -> None:
        for step in self.s.plan:
            if step.status == "in_progress":
                step.status = "pending" if interrupted else "done"
        if any(step.status != "pending" for step in self.s.plan):  # only show progress the model actually reported
            self.s.emit("plan", self.s.plan)

    def _persist(self, text: str, mode: str, result: TurnResult) -> None:
        """Automatic memory update: conversation log + a factual task record for this project."""
        mem = self.s.memory
        mem.add_message("user", text, mode)
        if result.text:
            mem.add_message("assistant", result.text, mode)
        acts = [e for e in result.ledger if e.risk != "read" and e.ok]
        if acts:
            ops = "; ".join(f"{e.tool}({e.summary[:60]})" for e in acts[:12])
            record = f"Task ({mode}, {time.strftime('%Y-%m-%d')}): {text[:200]} | executed: {ops} | outcome: {result.stopped}"
            mem.db.execute("INSERT INTO memories(scope, project_id, kind, text, ts, session_id) VALUES(?,?,?,?,?,?)",
                           ("project" if mem.project_id else "global", mem.project_id, "task", record, time.time(), mem.session_id))
            mem.db.commit()


def _plain(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """History without tool plumbing, for auxiliary calls (planning)."""
    out = []
    for m in history:
        if m.get("role") in ("user", "assistant") and m.get("content") and not m.get("tool_calls"):
            content = m["content"]
            if isinstance(content, list):
                content = " ".join(p.get("text", "[image]") for p in content)
            out.append({"role": m["role"], "content": content})
    return out
