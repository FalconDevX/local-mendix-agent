"""Interactive terminal: prompt_toolkit input (history, completion, multiline, status bar) + Rich output."""
from __future__ import annotations

import asyncio
import re
import signal
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

from lma import __version__
from lma.core.agent import Agent
from lma.core.state import AppState
from lma.llm.runtime import find_server
from lma.tools.image import IMAGE_SUFFIXES, load_image
from lma.ui.commands import MODES, Commands
from lma.ui.rendering import Renderer

STYLE = Style.from_dict({"bottom-toolbar": "bg:#1c1c1c #8a8a8a", "prompt": "#d97757 bold", "mode": "#5fafd7"})
SUBCOMMANDS = {
    "/mendix": ["status", "connect", "tools", "inspect", "test", "capabilities", "call"],
    "/mcp": ["list", "add", "connect", "disconnect", "tools", "inspect", "test", "call", "prompt", "resource"],
    "/memory": ["search", "project", "global", "add", "forget", "clear-session"],
    "/models": ["delete", "add"], "/download": ["runtime"], "/model": ["chat", "vision", "embed"],
    "/attach": ["paste", "preview", "clear"], "/vision": ["paste"], "/index": ["--full"], "/config": ["secret"],
}
PATH_COMMANDS = ("/read", "/write", "/edit", "/open", "/attach", "/vision", "/reject")


class SlashCompleter(Completer):
    def __init__(self, commands: Commands, state: AppState) -> None:
        self.commands, self.state = commands, state

    def get_completions(self, document, complete_event):  # noqa: ANN001, ANN201
        text = document.text_before_cursor
        if text.startswith("/") and " " not in text:
            for name in self.commands.names:
                if name.startswith(text):
                    yield Completion(name, start_position=-len(text))
            return
        cmd, _, rest = text.partition(" ")
        word = rest.split(" ")[-1] if rest else ""
        if cmd in SUBCOMMANDS and " " not in rest:
            for sub in SUBCOMMANDS[cmd]:
                if sub.startswith(rest):
                    yield Completion(sub, start_position=-len(rest))
            if cmd in ("/download", "/model", "/models"):
                reg = self.state.llm.registry
                for mid in sorted({*reg.catalogue, *reg.installed}):
                    if mid.startswith(word):
                        yield Completion(mid, start_position=-len(word))
        if cmd in ("/mcp",) and rest.count(" ") == 1:
            for name in self.state.mcp.connections:
                if name.startswith(word):
                    yield Completion(name, start_position=-len(word))
        at = re.search(r"@([\w./\\-]*)$", text)
        if (cmd in PATH_COMMANDS and rest) or at:
            frag = at.group(1) if at else word
            yield from self._paths(frag)

    def _paths(self, frag: str):  # noqa: ANN202
        root = self.state.root or Path.cwd()
        base = (root / frag).parent if frag and not frag.endswith(("/", "\\")) else root / frag
        prefix = Path(frag).name if frag and not frag.endswith(("/", "\\")) else ""
        try:
            entries = sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))[:200]
        except OSError:
            return
        for p in entries:
            if p.name.lower().startswith(prefix.lower()) and p.name not in self.state.settings.index.ignore_dirs:
                yield Completion(p.name + ("/" if p.is_dir() else ""), start_position=-len(prefix))


def toolbar(state: AppState):  # noqa: ANN201
    def render():  # noqa: ANN202
        models = state.llm.status()
        mx = state.mendix
        others = sum(1 for n, c in state.mcp.connections.items() if c.connected and n != "mendix")
        pending = len(state.workspace.pending) if state.workspace else 0
        ctx = state.settings.llm.context_size
        parts = [
            f"<b>/{state.mode}</b>", f"model: {models['chat']}",
            f"Maia: {'<style fg=\"#5faf5f\">connected</style>' if mx.connected else '<style fg=\"#d75f5f\">offline</style>'}" + (f" +{others} MCP" if others else ""),
            f"ctx: {state.context_tokens}/{ctx}",
            f"project: {state.root.name if state.root else 'none'}",
        ]
        if pending:
            parts.append(f"<style fg=\"#d7af5f\">{pending} staged</style>")
        if state.attachments:
            parts.append(f"{len(state.attachments)} image(s)")
        return HTML(" | ".join(parts))
    return render


def expand_mentions(text: str, state: AppState, ui: Renderer) -> str:
    """`@path` attaches images and inlines small text files into the message."""
    if state.workspace is None:
        return text
    extra: list[str] = []
    for m in re.finditer(r"(?<!\S)@([\w./\\-]+)", text):
        rel = m.group(1)
        try:
            full = state.workspace.resolve(rel)
        except Exception:  # noqa: BLE001
            continue
        if not full.is_file():
            continue
        if full.suffix.lower() in IMAGE_SUFFIXES:
            img = load_image(full)
            state.attachments.append(img)
            ui.info(f"attached {img.meta()}")
        else:
            try:
                content = state.workspace.read(rel)
            except Exception:  # noqa: BLE001
                continue
            lines = content.splitlines()
            shown = "\n".join(lines[:300])
            extra.append(f"\n\n[Attached file {rel}, {len(lines)} lines" + (", first 300 shown" if len(lines) > 300 else "") + f"]\n```\n{shown}\n```")
            ui.info(f"attached file {rel} ({len(lines)} lines)")
    return text + "".join(extra)


def banner(state: AppState, ui: Renderer) -> None:
    c = ui.console
    c.print(f"[accent]Local Mendix Agent[/accent] [dim]v{__version__} - local models, Maia MCP, offline capable[/dim]")
    hw = state.llm.hardware
    c.print(f"[dim]{hw.cpu} | RAM {hw.ram_total_gb} GB | {hw.gpu_name or 'CPU only'}"
            f"{f' {hw.vram_total_gb} GB' if hw.gpu_name else ''} | /help for commands[/dim]")


async def first_run_hints(state: AppState, ui: Renderer) -> None:
    if find_server(state.settings) is None:
        ui.warn("llama.cpp runtime not installed -> /download runtime")
    if state.llm.model_id("chat") is None:
        rec = state.llm.registry.recommend(state.llm.hardware)["chat"]
        picks = ", ".join(f"{tier}: {item[0].id} ({item[0].size_gb:.1f} GB, {item[1]})" for tier, item in rec.items() if item)
        ui.warn(f"no chat model installed. Recommended for this hardware -> {picks}. Install with /download <id> (see /models).")
    if state.root is None:
        ui.info("no project open -> /open <folder>")


async def connect_servers(state: AppState, ui: Renderer, agent: Agent) -> None:
    results = await state.mcp.connect_all() if state.mcp.connections else {}
    for name, err in results.items():
        if err:
            ui.warn(f"MCP {name}: not connected ({err.splitlines()[0][:140]})")
        else:
            ui.info(f"MCP {name}: connected, {len(state.mcp.get(name).tools)} tools")
    d = await state.mendix.connect()
    if d.connected:
        ui.ok(f"Maia MCP: connected at {d.url}, {len(state.mendix.tools())} tools discovered")
    elif state.mendix.cfg.enabled:
        ui.info("Maia MCP: not connected (Studio Pro MCP server not found). Details: /mendix status")
    agent.toolbox.refresh_mcp()


async def repl(state: AppState) -> None:
    ui = Renderer()
    state.emit = ui.emit
    agent = Agent(state)
    hist = state.settings.data_path / "input_history.txt"
    kb = KeyBindings()

    @kb.add("escape", "enter")
    def _(event):  # noqa: ANN001, ANN202
        event.current_buffer.insert_text("\n")

    session: PromptSession[str] | None
    side: PromptSession[str] | None
    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            raise OSError("not a console")
        session = PromptSession(history=FileHistory(str(hist)), key_bindings=kb, style=STYLE)
        side = PromptSession()
    except Exception as exc:  # noqa: BLE001 - e.g. mintty/Git Bash or redirected stdin: plain line mode
        session = side = None
        ui.info(f"(plain input mode: {type(exc).__name__}; completion, history and the status bar need cmd/PowerShell/Windows Terminal)")

    async def read_plain(prompt: str) -> str:
        ui.console.file.write(prompt)
        ui.console.file.flush()
        line = await asyncio.to_thread(sys.stdin.readline)
        if line == "":
            raise EOFError
        return line.rstrip("\r\n")

    async def ask(question: str) -> str:
        ui.finish()
        if side is None:
            return await read_plain(question)
        with patch_stdout():
            return await side.prompt_async(question)

    async def approver(tool: str, risk: str, summary: str) -> str:
        ui.finish()
        ui.console.print(f"[warn]Permission needed[/warn] ({risk}): [bold]{tool}[/bold]")
        ui.console.print(f"  {summary}", markup=False)
        ans = (await ask("  allow? [y]es / [n]o / [a]lways this session: ")).strip().lower()
        return "a" if ans.startswith("a") else "y" if ans in ("y", "yes", "t", "tak") else "n"

    state.permissions.approver = approver
    commands = Commands(state, agent, ui, ask=ask)
    if session is not None:
        session.completer = SlashCompleter(commands, state)
    banner(state, ui)
    await connect_servers(state, ui, agent)
    await first_run_hints(state, ui)
    prev = state.memory.previous_context(state.memory.project_id, 2)
    if prev:
        ui.info("previous sessions here: " + " | ".join((t or "")[:50] for _, t, _ in prev) + "  (/history, /memory)")

    watcher = asyncio.create_task(_watch(state)) if state.settings.index.watch_interval_s > 0 else None
    loop = asyncio.get_running_loop()
    while commands.running:
        try:
            if session is None:
                line = await read_plain(f"{state.mode} > ")
            else:
                with patch_stdout():
                    line = await session.prompt_async(HTML(f"<mode>{state.mode}</mode> <prompt>&gt;</prompt> "),
                                                      bottom_toolbar=toolbar(state), complete_while_typing=True)
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        line = line.strip()
        if not line:
            continue
        work = commands.dispatch(line) if line.startswith("/") else commands.chat(expand_mentions(line, state, ui), None)
        task = asyncio.ensure_future(work)
        old = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda *_: loop.call_soon_threadsafe(task.cancel))
        try:
            await task
        except asyncio.CancelledError:
            ui.finish()
            ui.warn("cancelled")
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive
            ui.finish()
            ui.error(f"unexpected error: {type(exc).__name__}: {exc}")
        finally:
            signal.signal(signal.SIGINT, old)
    if watcher:
        watcher.cancel()
    await _goodbye(state, ui)


async def _watch(state: AppState) -> None:
    """Polling file watcher: keeps the index fresh without extra dependencies."""
    while True:
        await asyncio.sleep(state.settings.index.watch_interval_s)
        if state.index is not None and state.index.counts()["files"]:
            try:
                await state.index.update(state.embedder)
            except Exception:  # noqa: BLE001
                pass


async def _goodbye(state: AppState, ui: Renderer) -> None:
    """Store a session summary so the next start can pick up the context."""
    turns = [m for m in state.history if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str) and m["content"]]
    if len(turns) >= 2 and state.llm.servers.get("chat") and state.llm.servers["chat"].alive:
        try:
            transcript = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in turns[-12:])
            res = await asyncio.wait_for(state.llm.chat([
                {"role": "system", "content": "Summarise this session in max 80 words: what was asked, what was done, what is still open."},
                {"role": "user", "content": transcript}], max_tokens=200), 60)
            if res.text.strip():
                state.memory.set_summary(res.text.strip())
        except Exception:  # noqa: BLE001
            pass
    await state.close()
    ui.info("session saved. bye.")
