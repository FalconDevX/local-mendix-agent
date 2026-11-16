"""Rich rendering of agent events in a Claude Code like layout.

Streaming works in two parts so nothing is ever printed twice:
  * finished markdown blocks are printed once and never touched again,
  * only the unfinished tail lives in a small, height-capped live region together with the spinner.
(The previous implementation re-rendered the whole answer in a `Live` with visible overflow; once the
answer was taller than the window every refresh printed it again.)
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.segment import Segment
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from lma.core.state import LedgerEntry, PlanStep, current_agent

THEME = Theme({
    "accent": "bold #d97757", "accent2": "#d97757", "dim": "grey58", "ok": "green", "warn": "yellow", "fail": "bold red",
    "tool": "bold", "agent": "#5fafd7", "user": "grey70",
    "markdown.code": "#d7af87", "markdown.h1": "bold #d97757", "markdown.h2": "bold", "markdown.link": "#5fafd7",
})
STATUS_STYLE = {"ok": "ok", "warn": "warn", "fail": "fail", "info": "dim"}
STATUS_MARK = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL", "info": "info"}

GLYPHS = {
    # Windows Terminal / VS Code / modern terminals
    "unicode": {"bullet": "●", "branch": "⎿", "todo": "☐", "done": "☒", "now": "◼", "failed": "✗", "skipped": "⊘",
                "spin": "✻✽✶✳✢·", "check": "✓", "prompt": "❯", "sep": "·"},
    # classic conhost (cmd.exe / powershell.exe window): only glyphs every Windows console font has
    "safe": {"bullet": "●", "branch": "└", "todo": "□", "done": "■", "now": "►", "failed": "x", "skipped": "-",
             "spin": "|/-\\", "check": "√", "prompt": ">", "sep": "·"},
    "ascii": {"bullet": "*", "branch": "L", "todo": "[ ]", "done": "[x]", "now": "[>]", "failed": "[!]", "skipped": "[-]",
              "spin": "|/-\\", "check": "ok", "prompt": ">", "sep": "-"},
}
# activity labels that only describe what is going on right now: shown in the spinner, not as lines
TRANSIENT = {"THINKING", "PLANNING", "READING PROJECT", "SEARCHING FILES"}


def pick_glyphs(choice: str = "auto") -> dict[str, str]:
    if choice in GLYPHS:
        return GLYPHS[choice]
    enc = (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "")
    if "utf" not in enc:
        return GLYPHS["ascii"]
    modern = sys.platform != "win32" or any(os.environ.get(v) for v in ("WT_SESSION", "TERM_PROGRAM", "ConEmuANSI"))
    return GLYPHS["unicode" if modern else "safe"]


def make_console() -> Console:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    console = Console(theme=THEME, highlight=False, soft_wrap=False)
    if not console.is_terminal:
        console = Console(theme=THEME, highlight=False, width=150)
    return console


def split_stable(text: str) -> tuple[str, str]:
    """Split streamed markdown into (finished blocks, unfinished tail).

    The cut is the last blank line that is not inside an open ``` fence, so a block that was printed
    can never change afterwards.
    """
    cut, fenced, pos = 0, False, 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fenced = not fenced
        pos += len(line)
        if not fenced and stripped == "" and line.endswith("\n"):
            cut = pos
    return text[:cut], text[cut:]


class _Tail:
    """Render a renderable but keep only its last `max_lines` lines (the live region must stay small)."""

    def __init__(self, renderable: Any, max_lines: int) -> None:
        self.renderable, self.max_lines = renderable, max_lines

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        lines = console.render_lines(self.renderable, options, pad=False)
        if len(lines) > self.max_lines:
            lines = lines[-self.max_lines:]
            yield Segment("  ...\n")
        for line in lines:
            yield from line
            yield Segment.line()


class Renderer:
    """Receives events from the agent (`state.emit`) and draws them."""

    def __init__(self, console: Console | None = None, stream_markdown: bool = True, glyphs: str = "auto") -> None:
        self.console = console or make_console()
        self.interactive = stream_markdown and self.console.is_terminal
        self.g = pick_glyphs(glyphs)
        self._live: Live | None = None
        self._turn = False          # a turn is running: the live region may be (re)started
        self._turn_t0 = 0.0
        self._status = ""
        self._buffer = ""           # streamed text that is not printed permanently yet
        self._blocks = 0            # blocks of the current message already printed
        self._raw_open = False      # non-interactive: a raw text line is open
        self._agents: dict[str, float] = {}   # running sub-agents -> start time
        self._tokens = 0

    # ---- turn / live region ---------------------------------------------
    def turn_start(self, status: str = "Thinking") -> None:
        self._turn, self._turn_t0, self._status, self._tokens = True, time.monotonic(), status, 0
        self._ensure_live()

    def turn_end(self) -> None:
        self._flush_text()
        self._turn = False
        self._agents.clear()
        self._stop_live()

    def finish(self) -> None:
        """Flush pending text and release the terminal (before prompts and direct printing)."""
        self._flush_text()
        self._stop_live()

    def _ensure_live(self) -> None:
        if self._live is None and self._turn and self.interactive:
            self._live = Live(get_renderable=self._region, console=self.console, refresh_per_second=8, transient=True)
            self._live.start()

    def _stop_live(self) -> None:
        if self._live is not None:
            live, self._live = self._live, None
            live.stop()

    def _region(self) -> Any:
        parts: list[Any] = []
        _, tail = split_stable(self._buffer)
        if tail.strip() and current_agent.get() is None:
            room = max(3, self.console.size.height - 6)
            parts.append(_Tail(self._message(tail, first=self._blocks == 0), room))
        parts.append(self._spinner())
        return Group(*parts)

    def _spinner(self) -> Text:
        frames = self.g["spin"]
        elapsed = time.monotonic() - self._turn_t0
        frame = frames[int(time.monotonic() * 6) % len(frames)]
        status = self._status
        if self._agents:
            status = f"{len(self._agents)} agent(s) working: " + ", ".join(sorted(self._agents))
        extra = f"{elapsed:.0f}s {self.g['sep']} Ctrl+C to interrupt"
        return Text.assemble((f"{frame} ", "accent2"), (f"{status[:100]}… ", "accent2"), (f"({extra})", "dim"))

    def _print(self, *renderables: Any, **kw: Any) -> None:
        self.console.print(*renderables, **kw)

    # ---- events ---------------------------------------------------------
    def emit(self, kind: str, payload: Any) -> None:
        agent = current_agent.get()
        if kind == "text":
            if agent is None:
                self._text(payload)
            return
        if kind == "text_end":
            if agent is None:
                self._flush_text()
            return
        if agent is None:
            self._flush_text()
        self._ensure_live()
        if kind == "activity":
            self._activity(*payload, agent=agent)
        elif kind == "tool_call":
            self._tool_call(payload, agent)
        elif kind == "tool_result":
            self._tool_result(payload, agent)
        elif kind == "plan":
            if agent is None:
                self.plan(payload)
        elif kind == "agent_start":
            name, role, task = payload
            self._agents[name] = time.monotonic()
            self._print(Text.assemble((f"{self.g['bullet']} ", "agent"), ("Agent", "tool"), (f"({name})", "agent"), (f" {_one_line(task, 110)}", "dim")))
        elif kind == "agent_end":
            name, res = payload
            self._agents.pop(name, None)
            ok = res.stopped == "completed"
            tail = f"{len(res.ledger)} tool use(s) {self.g['sep']} {res.seconds:.0f}s" + ("" if ok else f" {self.g['sep']} {res.stopped}: {res.error or ''}"[:160])
            self._print(Text.assemble((f"  {self.g['branch']} ", "dim"), (f"{name} ", "agent"), ("done " if ok else "stopped ", "ok" if ok else "warn"), (tail, "dim")))
        elif kind == "phase":
            self._print()
            self._print(Text.assemble((f"{self.g['bullet']} ", "accent2"), (str(payload), "bold")))

    def _activity(self, label: str, detail: Any, agent: str | None) -> None:
        detail = str(detail)
        if label in TRANSIENT or label.startswith("CALLING"):
            self._status = detail[:1].upper() + detail[1:] if agent is None else self._status
            if self._live is None and agent is None:  # no spinner to show it in (slash command, piped output)
                self._print(Text(f"  {self.g['sep']} {detail}", style="dim"))
            return
        style = {"COMPLETED": "ok", "CANCELLED": "warn", "WARNING": "warn"}.get(label, "dim")
        mark = self.g["check"] if label == "COMPLETED" else self.g["sep"]
        who = f"{agent} " if agent else ""
        self._print(Text(f"  {mark} {who}{detail}", style=style))

    def _tool_call(self, payload: tuple[str, str, str], agent: str | None) -> None:
        name, args, activity = payload
        if agent is None:
            self._status = f"Running {name}"
            self._print(Text.assemble((f"{self.g['bullet']} ", "ok"), (_pretty_tool(name), "tool"), (f"({args})", "dim")))
        else:
            self._print(Text.assemble(("    ", ""), (f"{agent} ", "agent"), (_pretty_tool(name), "default"), (f"({args[:70]})", "dim")))

    def _tool_result(self, e: LedgerEntry, agent: str | None) -> None:
        if e.tool == "delegate" and e.ok:
            return  # agent_end already reported it
        if e.ok:
            if agent is not None:
                return  # sub-agents: one line per call is enough
            first = _one_line(e.detail, 110)
            more = e.lines - 1
            notes = ([f"+{more} line{'s' if more > 1 else ''}"] if more > 0 else []) + ([f"{e.seconds:.1f}s"] if e.seconds >= 0.1 else [])
            self._print(Text.assemble((f"  {self.g['branch']} ", "dim"), (first, "dim"), (f"  ({', '.join(notes)})" if notes else "", "dim")))
        else:
            pad = "      " if agent else "  "
            self._print(Text.assemble((f"{pad}{self.g['branch']} ", "fail"), (_one_line(e.detail, 300), "warn")))

    # ---- streaming ------------------------------------------------------
    def _message(self, text: str, first: bool) -> Any:
        """Assistant text: bullet in the gutter, markdown beside it."""
        grid = Table.grid(padding=0)
        grid.add_column(width=2, no_wrap=True)
        grid.add_column(ratio=1)
        grid.add_row(Text(self.g["bullet"] if first else "", style="default"), Markdown(text.strip("\n")))
        return grid

    def _text(self, delta: str) -> None:
        if not self.interactive:
            self.console.file.write(delta)
            self.console.file.flush()
            self._raw_open = True
            return
        self._buffer += delta
        self._tokens += 1
        self._status = "Writing"
        self._ensure_live()
        stable, tail = split_stable(self._buffer)
        if stable.strip():
            self._commit(stable)
            self._buffer = tail

    def _commit(self, text: str) -> None:
        self._print()
        self._print(self._message(text, first=self._blocks == 0))
        self._blocks += 1

    def _flush_text(self) -> None:
        if self._raw_open:
            self.console.file.write("\n")
            self.console.file.flush()
            self._raw_open = False
        if self._buffer.strip():
            text, self._buffer = self._buffer, ""
            self._commit(text)
        self._buffer = ""
        if self._blocks:
            self._blocks = 0
            self._status = "Thinking"

    # ---- pieces ---------------------------------------------------------
    def user(self, text: str) -> None:
        """Echo of a prompt that did not come from the interactive input line."""
        self._print(Text(f"{self.g['prompt']} {text}", style="user"))

    def plan(self, steps: list[PlanStep]) -> None:
        if not steps:
            return
        marks = {"pending": self.g["todo"], "in_progress": self.g["now"], "done": self.g["done"], "failed": self.g["failed"], "skipped": self.g["skipped"]}
        styles = {"done": "dim", "failed": "fail", "in_progress": "bold", "skipped": "dim"}
        done = sum(s.status == "done" for s in steps)
        out = Text.assemble((f"{self.g['bullet']} ", "accent2"), ("Plan", "tool"), (f" {done}/{len(steps)}", "dim"))
        for i, s in enumerate(steps):
            lead = f"  {self.g['branch']} " if i == 0 else "    "
            out.append("\n" + lead, style="dim")
            out.append(f"{marks.get(s.status, self.g['todo'])} {s.text}" + (f"  ({s.note})" if s.note else ""), style=styles.get(s.status, "default"))
        self._print(out)

    def banner(self, title: str, lines: list[str]) -> None:
        body = Text.assemble((f"{self.g['spin'][0]} ", "accent2"), (title, "bold"))
        for line in lines:
            body.append("\n  " + line, style="dim")
        self._print(Panel(body, box=box.ROUNDED if self.g is not GLYPHS["ascii"] else box.ASCII, border_style="accent2", expand=False, padding=(0, 1)))

    def diff(self, text: str) -> None:
        if text.strip():
            self._print(Padding(Syntax(text, "diff", theme="ansi_dark", background_color="default", word_wrap=True), (0, 0, 0, 2)))

    def code(self, text: str, lexer: str, start_line: int = 1) -> None:
        self._print(Syntax(text, lexer, theme="ansi_dark", background_color="default", line_numbers=True, start_line=start_line, word_wrap=True))

    def markdown(self, text: str) -> None:
        self._print(Markdown(text))

    def table(self, title: str, columns: list[str], rows: list[list[Any]]) -> None:
        t = Table(title=title, title_justify="left", title_style="accent", header_style="bold", border_style="dim",
                  show_lines=False, box=box.SIMPLE_HEAD if self.g is not GLYPHS["ascii"] else box.ASCII)
        for c in columns:
            t.add_column(c, overflow="fold")
        for r in rows:
            t.add_row(*[x if isinstance(x, Text) else Text(str(x)) for x in r])  # Text: brackets are data, not markup
        self._print(t)

    def checks(self, checks: list[Any]) -> None:
        group = None
        for c in checks:
            if c.group != group:
                group = c.group
                self._print(f"\n[accent]{group}[/accent]")
            self._print(Text.assemble((f"  {STATUS_MARK[c.status]} ", STATUS_STYLE[c.status]), (f"{c.name}: ", "bold"), c.detail))

    def info(self, msg: str) -> None:
        self._print(Text(msg, style="dim"))

    def ok(self, msg: str) -> None:
        self._print(Text(msg, style="ok"))

    def warn(self, msg: str) -> None:
        self._print(Text(msg, style="warn"))

    def error(self, msg: str) -> None:
        self._print(Text(msg, style="fail"))


def _one_line(text: str, limit: int) -> str:
    stripped = text.strip()
    first = stripped.splitlines()[0] if stripped else ""
    return first[:limit] + ("…" if len(first) > limit else "")


def _pretty_tool(name: str) -> str:
    """mcp__mendix__get_modules -> mendix:get_modules"""
    if name.startswith("mcp__"):
        server, _, tool = name[5:].partition("__")
        return f"{server}:{tool}"
    return name
