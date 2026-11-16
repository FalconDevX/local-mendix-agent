"""The agent execution loop: retrieve -> plan -> call tools -> inspect -> revise -> validate -> report."""
from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from lma.core import context as ctx
from lma.core.permissions import turn_mode
from lma.core.state import AppState, LedgerEntry, PlanStep, current_agent
from lma.editing.editor import validate
from lma.llm.manager import ChatResult, LLMError, ToolCall
from lma.tools.image import SCREENSHOT_PROMPT, LoadedImage
from lma.tools.toolbox import Toolbox

PLANNING_MODES = {"agent", "code", "mendix"}
PARALLEL_TOOLS = {"delegate"}  # calls that run side by side when the model issues several at once
_STEP = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.*\S)")


@dataclass
class TurnResult:
    text: str
    ledger: list[LedgerEntry] = field(default_factory=list)
    steps: int = 0
    stopped: str = "completed"  # completed | step_limit | cancelled | error | loop
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
        who = f"[{e.agent}] " if e.agent else ""
        lines.append(f"  {'OK  ' if e.ok else 'FAIL'} {who}{e.tool}({e.summary})" + ("" if e.ok else f" - {e.detail[:160]}"))
    for e in ledger:
        if not e.ok and e.risk == "read":
            lines.append(f"  FAIL {e.tool}({e.summary}) - {e.detail[:160]}")
    if not any(e.ok for e in acts):
        lines.append("  No modifying operation was executed.")
    return "\n".join(lines)


def repetition_start(text: str) -> int | None:
    """Detect a model that got stuck repeating itself.

    Returns the offset where the text should be cut (one copy of the repeated part is kept), or None.
    Two shapes are recognised: the same line many times in a row, and any block of 20-600 characters
    repeated back to back (3 times for blocks of 60+ characters, 5 times for short ones).
    """
    if len(text) < 100:
        return None
    complete = text.split("\n")[:-1]  # the last element is the line still being written
    solid = [ln.strip() for ln in complete if ln.strip()]
    if len(solid) >= 6 and len(solid[-1]) >= 4 and all(x == solid[-1] for x in solid[-6:]):
        return _run_start(complete, solid[-1])
    tail = text[-2400:]
    for period in range(20, 601):
        need = 3 if period >= 60 else 5
        if len(tail) < period * need:
            break
        unit = tail[-period:]
        if not unit.strip() or len(set(unit)) < 4:
            continue
        if all(tail[-period * (k + 1):len(tail) - period * k] == unit for k in range(1, need)):
            # walk back to where the repetition began and keep exactly one copy
            start = len(text) - period * need
            while start - period >= 0 and text[start - period:start] == unit:
                start -= period
            return start + period
    return None


def _run_start(lines: list[str], last: str) -> int:
    """Offset just after the first line of the trailing run of `last` lines."""
    i = len(lines)
    first = i
    while i > 0 and (lines[i - 1].strip() == last or not lines[i - 1].strip()):
        i -= 1
        if lines[i].strip() == last:
            first = i
    return sum(len(line) + 1 for line in lines[:first + 1])


class Agent:
    """One agent = one conversation. The main agent talks to the user and owns `state.history`;
    sub-agents (`name` set) get a private history, a restricted tool set and report back with text."""

    def __init__(self, state: AppState, toolbox: Toolbox | None = None, *, name: str | None = None,
                 history: list[dict[str, Any]] | None = None, model: str | None = None,
                 tool_filter: Callable[[str], bool] | None = None, instructions: str = "", max_steps: int | None = None) -> None:
        self.s = state
        self.toolbox = toolbox or Toolbox(state)
        self.name = name
        self._history = history
        self.model = model
        self.tool_filter = tool_filter
        self.instructions = instructions
        self.max_steps = max_steps
        self._turns = 0

    @property
    def history(self) -> list[dict[str, Any]]:
        return self.s.history if self._history is None else self._history

    @property
    def is_sub(self) -> bool:
        return self.name is not None

    # ---- public --------------------------------------------------------
    async def run(self, text: str, mode: str | None = None, images: list[LoadedImage] | None = None) -> TurnResult:
        mode = mode or self.s.mode
        mode_token = turn_mode.set(mode)
        agent_token = current_agent.set(self.name)
        mark, ledger_mark, t0 = len(self.history), len(self.s.ledger), time.time()
        try:
            result = await self._turn(text, mode, images or [])
        except asyncio.CancelledError:
            del self.history[mark:]
            self.history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": "[cancelled by the user]"}])
            if self.is_sub:
                raise  # the lead agent is being cancelled too; let it report
            self.s.emit("activity", ("CANCELLED", "turn cancelled; completed tool calls are listed below"))
            result = TurnResult("", stopped="cancelled")
        except LLMError as exc:
            del self.history[mark:]
            result = TurnResult("", stopped="error", error=str(exc))
        finally:
            turn_mode.reset(mode_token)
            current_agent.reset(agent_token)
        entries = self.s.ledger[ledger_mark:]
        result.ledger = [e for e in entries if e.agent == self.name] if self.is_sub else entries
        result.seconds = time.time() - t0
        if not self.is_sub:
            self._persist(text, mode, result)
        return result

    def _tools(self, mode: str) -> list[dict[str, Any]]:
        if mode == "vision":
            return []
        tools = self.toolbox.schemas(mode, sub=self.is_sub)
        if self.tool_filter:
            tools = [t for t in tools if self.tool_filter(t["function"]["name"])]
        return tools

    def _system(self, mode: str, retrieved: str) -> str:
        text = ctx.build_system(self.s, mode, retrieved, with_plan=not self.is_sub)
        return text + ("\n\n" + self.instructions if self.instructions else "")

    # ---- one turn ------------------------------------------------------
    async def _turn(self, text: str, mode: str, images: list[LoadedImage]) -> TurnResult:
        s, history = self.s, self.history
        self.toolbox.refresh_mcp()
        s.emit("activity", ("THINKING", "retrieving project context and memory"))
        retrieved = await ctx.retrieve(s, text, mode, first_turn=self._turns == 0 and not self.is_sub)
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
            history.extend([{"role": "user", "content": f"{text}\n[attached image(s): {labels}]"}, {"role": "assistant", "content": answer}])
            return TurnResult(answer, steps=1)
        history.append(await self._user_message(text, images))
        tools = self._tools(mode)

        # Plan real tasks only: short follow-ups ("continue", "yes", a quick question) keep the current plan.
        if not self.is_sub and mode in PLANNING_MODES and s.settings.agent.plan_first and len(text.split()) >= 6:
            s.plan = []
            await self._make_plan(mode, retrieved)

        counts: dict[str, int] = {}
        blocked = 0
        validated = False
        final = ""
        max_steps = self.max_steps or s.settings.agent.max_steps
        for step in range(1, max_steps + 1):
            limit = ctx.budget(s, tools)
            await ctx.compress(s, history, limit)
            messages = [{"role": "system", "content": self._system(mode, retrieved)}, *history]
            if not self.is_sub:
                s.context_tokens = ctx.message_tokens(messages)
            res = await self._generate(messages, tools, limit)
            if res.prompt_tokens and not self.is_sub:
                s.context_tokens = res.prompt_tokens + res.completion_tokens
            if not res.tool_calls:
                final = res.text.strip()
                history.append({"role": "assistant", "content": final})
                if res.finish_reason == "repetition":
                    self._close_plan(interrupted=True)
                    return TurnResult(final, steps=step, stopped="loop",
                                      error="the model started repeating itself and was stopped; try a larger model or rephrase the request")
                if mode == "plan" and not self.is_sub:
                    s.plan = parse_plan(final) or s.plan
                    s.emit("plan", s.plan)
                # validate staged modifications once before finishing
                problems = self._validate_staged() if not validated and not self.is_sub else []
                validated = True
                if problems and step < max_steps:
                    s.emit("activity", ("VALIDATING", f"{len(problems)} staged file(s) have syntax errors, asking the model to fix"))
                    history.append({"role": "user", "content": "Automatic validation found problems in staged files. Fix them:\n" + "\n".join(problems)})
                    continue
                self._close_plan()
                return TurnResult(final, steps=step)
            history.append({
                "role": "assistant", "content": res.text or None,
                "tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.raw_arguments or json.dumps(c.arguments)}}
                               for c in res.tool_calls],
            })
            outputs, newly_blocked = await self._run_calls(res.tool_calls, counts)
            for call, out in zip(res.tool_calls, outputs):
                history.append({"role": "tool", "tool_call_id": call.id, "content": out})
            blocked += newly_blocked
            if blocked >= 3:
                # the model ignores "do not repeat this call": going on would only print the same thing again
                self._close_plan(interrupted=True)
                note = "Stopped: the model kept repeating the same tool call. Rephrase the request or switch to a larger model (/model)."
                history.append({"role": "assistant", "content": note})
                return TurnResult(note, steps=step, stopped="loop", error=note)
        self._close_plan(interrupted=True)
        note = f"Stopped after the step limit ({max_steps}). The task may be incomplete; say 'continue' to go on."
        history.append({"role": "assistant", "content": note})
        return TurnResult(note, steps=max_steps, stopped="step_limit")

    async def _run_calls(self, calls: list[ToolCall], counts: dict[str, int]) -> tuple[list[str], int]:
        """Execute the tool calls of one model message. Returns their outputs (same order) and how many
        were refused as repeats. Neighbouring `delegate` calls run concurrently."""
        outputs: list[str] = [""] * len(calls)
        blocked = 0
        batch: list[int] = []

        async def flush() -> None:
            if batch:
                done = await asyncio.gather(*(self.toolbox.execute(calls[i].name, calls[i].arguments) for i in batch))
                for i, outcome in zip(batch, done):
                    outputs[i] = outcome.text
                batch.clear()

        for i, call in enumerate(calls):
            key = call.name + json.dumps(call.arguments, sort_keys=True)
            counts[key] = counts.get(key, 0) + 1
            if call.parse_error:
                outputs[i] = f"ERROR: could not parse the arguments of {call.name}: {call.parse_error}. Send valid JSON."
            elif counts[key] > 2:
                blocked += 1
                outputs[i] = ("ERROR: this exact call was already made twice with the same result. Do not repeat it; "
                              "use the earlier result, try a different approach, or finish with a summary.")
            elif call.name in PARALLEL_TOOLS:
                batch.append(i)
            else:
                await flush()
                outputs[i] = (await self.toolbox.execute(call.name, call.arguments)).text
        await flush()
        return outputs, blocked

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
        tool_names = ", ".join(t["function"]["name"] for t in self._tools(mode))
        messages = [{"role": "system", "content": self._system(mode, retrieved) + f"\n\nTools you will be able to use: {tool_names}"},
                    *_plain(self.history[-6:]), {"role": "user", "content": ctx.PLAN_PROMPT}]
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
        """One streamed model call with retries, context-overflow recovery and loop detection."""
        s = self.s
        last: LLMError | None = None
        kw: dict[str, Any] = {"model": self.model} if self.model else {}
        for attempt in range(s.settings.agent.max_retries + 1):
            stream = s.llm.chat_stream(messages, tools=tools or None, **kw)
            try:
                result = ChatResult()
                thinking = False
                text, checked = "", 0
                async for kind, payload in stream:
                    if kind == "text":
                        s.emit("text", payload)
                        text += payload
                        if len(text) - checked >= 80:  # look for a stuck model every ~80 characters
                            checked = len(text)
                            cut = repetition_start(text)
                            if cut is not None:
                                result = ChatResult(text=text[:cut].rstrip(), finish_reason="repetition")
                                break
                    elif kind == "reasoning" and not thinking:
                        thinking = True
                        s.emit("activity", ("THINKING", "model is reasoning"))
                    elif kind == "done":
                        result = payload
                s.emit("text_end", result)
                if result.finish_reason == "repetition":
                    s.emit("activity", ("WARNING", "the model began repeating itself: generation stopped"))
                return result
            except LLMError as exc:
                last = exc
                s.emit("text_end", None)
                if "context" in str(exc).lower():
                    await ctx.compress(s, self.history, limit, force=True)
                    messages[1:] = self.history
                s.emit("activity", ("THINKING", f"model call failed ({str(exc).splitlines()[0][:120]}), retry {attempt + 1}"))
                await asyncio.sleep(1 + attempt)
            finally:
                await stream.aclose()  # also ends the HTTP request when we stopped reading early
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
        if self.is_sub:
            return
        for step in self.s.plan:
            if step.status == "in_progress":
                step.status = "pending" if interrupted else "done"
        if any(step.status != "pending" for step in self.s.plan):  # only show progress the model actually reported
            self.s.emit("plan", self.s.plan)

    def _persist(self, text: str, mode: str, result: TurnResult) -> None:
        """Automatic memory update: conversation log + a factual task record for this project."""
        record_turn(self.s, text, mode, result)


def record_turn(state: AppState, text: str, mode: str, result: TurnResult) -> None:
    mem = state.memory
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
