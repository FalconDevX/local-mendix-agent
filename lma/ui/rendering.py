"""Rich rendering of agent events: streaming markdown, activity lines, plans, diffs, tables."""
from __future__ import annotations

import sys
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from lma.core.state import LedgerEntry, PlanStep

THEME = Theme({
    "accent": "bold #d97757", "dim": "grey58", "ok": "green", "warn": "yellow", "fail": "bold red",
    "act.THINKING": "magenta", "act.PLANNING": "cyan", "act.READING PROJECT": "blue", "act.SEARCHING FILES": "blue",
    "act.CALLING MAIA MCP": "bold #d97757", "act.EXECUTING TOOL": "yellow", "act.EDITING": "green",
    "act.VALIDATING": "cyan", "act.COMPLETED": "bold green", "act.CANCELLED": "bold red",
})
STATUS_STYLE = {"ok": "ok", "warn": "warn", "fail": "fail", "info": "dim"}
STATUS_MARK = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL", "info": "info"}
PLAN_MARK = {"pending": "[ ]", "in_progress": "[>]", "done": "[x]", "failed": "[!]", "skipped": "[-]"}


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


class Renderer:
    """Receives events from the agent (`state.emit`) and draws them."""

    def __init__(self, console: Console | None = None, stream_markdown: bool = True) -> None:
        self.console = console or make_console()
        self.stream_markdown = stream_markdown and self.console.is_terminal
        self._live: Live | None = None
        self._buffer = ""

    def emit(self, kind: str, payload: Any) -> None:
        if kind == "text":
            self._text(payload)
            return
        self._end_stream()
        if kind == "activity":
            label, detail = payload
            style = f"act.{label}" if f"act.{label}" in THEME.styles else ("act.CALLING MAIA MCP" if label.startswith("CALLING") else "dim")
            self.console.print(Text.assemble((f"[{label}]", style), " ", (str(detail), "dim")))
        elif kind == "tool_result":
            self._tool_result(payload)
        elif kind == "plan":
            self.plan(payload)

    # ---- streaming -----------------------------------------------------
    def _text(self, delta: str) -> None:
        self._buffer += delta
        if not self.stream_markdown:
            self.console.file.write(delta)
            self.console.file.flush()
            return
        if self._live is None:
            self._live = Live(Markdown(self._buffer), console=self.console, refresh_per_second=8, vertical_overflow="visible")
            self._live.start()
        else:
            self._live.update(Markdown(self._buffer))

    def _end_stream(self) -> None:
        if self._live is not None:
            self._live.update(Markdown(self._buffer))
            self._live.stop()
            self._live = None
        elif self._buffer and not self.stream_markdown:
            self.console.file.write("\n")
            self.console.file.flush()
        self._buffer = ""

    def finish(self) -> None:
        self._end_stream()

    # ---- pieces --------------------------------------------------------
    def _tool_result(self, e: LedgerEntry) -> None:
        if e.ok:
            first = e.detail.strip().splitlines()[0][:110] if e.detail.strip() else ""
            self.console.print(Text.assemble(("  ok ", "ok"), (f"{e.tool} ", "default"), (f"{e.seconds:.1f}s ", "dim"), (first, "dim")))
        else:
            self.console.print(Text.assemble(("  !! ", "fail"), (f"{e.tool} ", "default"), (e.detail.strip()[:300], "warn")))

    def plan(self, steps: list[PlanStep]) -> None:
        if not steps:
            return
        body = Text()
        for i, s in enumerate(steps, 1):
            style = {"done": "ok", "failed": "fail", "in_progress": "accent", "skipped": "dim"}.get(s.status, "default")
            body.append(f"{PLAN_MARK.get(s.status, '[ ]')} {i}. {s.text}" + (f"  ({s.note})" if s.note else "") + "\n", style=style)
        self.console.print(Panel(body, title="Plan", title_align="left", border_style="cyan", expand=False))

    def diff(self, text: str) -> None:
        if text.strip():
            self.console.print(Syntax(text, "diff", theme="ansi_dark", background_color="default", word_wrap=True))

    def code(self, text: str, lexer: str, start_line: int = 1) -> None:
        self.console.print(Syntax(text, lexer, theme="ansi_dark", background_color="default", line_numbers=True, start_line=start_line, word_wrap=True))

    def markdown(self, text: str) -> None:
        self.console.print(Markdown(text))

    def table(self, title: str, columns: list[str], rows: list[list[Any]]) -> None:
        t = Table(title=title, title_justify="left", title_style="accent", header_style="bold", border_style="dim", show_lines=False)
        for c in columns:
            t.add_column(c, overflow="fold")
        for r in rows:
            t.add_row(*[x if isinstance(x, Text) else Text(str(x)) for x in r])  # Text: brackets are data, not markup
        self.console.print(t)

    def checks(self, checks: list[Any]) -> None:
        group = None
        for c in checks:
            if c.group != group:
                group = c.group
                self.console.print(f"\n[accent]{group}[/accent]")
            self.console.print(Text.assemble((f"  {STATUS_MARK[c.status]} ", STATUS_STYLE[c.status]), (f"{c.name}: ", "bold"), c.detail))

    def info(self, msg: str) -> None:
        self.console.print(Text(msg, style="dim"))

    def ok(self, msg: str) -> None:
        self.console.print(Text(msg, style="ok"))

    def warn(self, msg: str) -> None:
        self.console.print(Text(msg, style="warn"))

    def error(self, msg: str) -> None:
        self.console.print(Text(msg, style="fail"))
