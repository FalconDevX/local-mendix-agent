"""Slash commands. Every command here does real work against the shared AppState."""
from __future__ import annotations

import asyncio
import json
import shlex
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn

from lma import doctor
from lma.config import load_settings, save_settings, set_setting, store_secret
from lma.core.agent import Agent, TurnResult, report
from lma.core.context import message_tokens
from lma.core.state import AppState
from lma.editing.editor import EditError
from lma.llm import runtime
from lma.llm.downloader import DownloadError
from lma.llm.hardware import recommend_runtime
from lma.llm.manager import LLMError
from lma.llm.registry import fit
from lma.mcpx.manager import ServerConfig, describe_error, save_mcp_config
from lma.project.scanner import read_text, scan, stats, tree
from lma.tools import git
from lma.tools.image import IMAGE_SUFFIXES, from_clipboard, load_image
from lma.ui.rendering import Renderer

MODES = ("ask", "plan", "agent", "code", "review", "mendix", "vision")
HELP: list[tuple[str, str]] = [
    ("/help", "this list"),
    ("/ask /plan /agent /code /review /mendix /vision", "switch mode, or run once: /ask <question>"),
    ("/model [chat|vision|embed] [id]", "show or switch the active model"),
    ("/models [delete <id> | add <gguf> <role> [mmproj]]", "catalogue, hardware fit, installed models"),
    ("/download [<model-id> | runtime [cpu|vulkan|cuda-12.4 | <zip>...]]", "download a model or the llama.cpp runtime"),
    ("/project", "active project facts"), ("/open <folder>", "open a project folder"),
    ("/index [--full]", "index the project (incremental) and refresh its summary"),
    ("/search <query>", "hybrid semantic + keyword search"), ("/read <file> [from-to]", "show a file"),
    ("/write <file> [:: text]", "stage a new file content"), ("/edit <file> <instruction>", "let the model edit one file (staged)"),
    ("/diff", "show staged changes"), ("/apply", "write staged changes (snapshot first)"), ("/reject [file]", "discard staged changes"),
    ("/undo [id]", "roll back the last applied change set"), ("/history [show|resume <id>]", "applied change sets and past sessions"),
    ("/mcp [list|add|connect|disconnect|tools|inspect|test|call|prompt|resource] [server]", "MCP servers"),
    ("/mendix [status|connect [port|url]|tools [name]|inspect|test|capabilities|call <tool> <json>]", "Studio Pro MCP (Maia)"),
    ("/memory [search <q>|project|global|add <text>|forget <id>|clear-session]", "persistent memory"),
    ("/context", "context window usage"), ("/attach <image>|paste|preview|clear", "attach an image to the next message, preview what the model gets"),
    ("/config [key value | secret <name>]", "show or change settings"), ("/doctor", "full diagnostics"),
    ("/clear", "clear the conversation (memory is kept)"), ("/exit", "quit"),
]

AskFn = Callable[[str], Awaitable[str]]


class Commands:
    def __init__(self, state: AppState, agent: Agent, ui: Renderer, ask: AskFn | None = None, assume_yes: bool = False) -> None:
        self.s, self.agent, self.ui = state, agent, ui
        self.ask = ask  # interactive question -> answer; None when non-interactive
        self.assume_yes = assume_yes
        self.running = True

    @property
    def names(self) -> list[str]:
        return sorted("/" + n[4:].replace("_", "-") for n in dir(self) if n.startswith("cmd_"))

    async def confirm(self, question: str, default: bool = False) -> bool:
        if self.assume_yes:
            return True
        if self.ask is None:
            return default
        ans = (await self.ask(f"{question} [y/N] ")).strip().lower()
        return ans in ("y", "yes", "t", "tak")

    async def dispatch(self, line: str) -> None:
        name, _, arg = line.strip().partition(" ")
        handler = getattr(self, "cmd_" + name[1:].replace("-", "_"), None)
        if handler is None:
            self.ui.error(f"unknown command {name}. Type /help.")
            return
        try:
            await handler(arg.strip())
        except (EditError, LLMError, DownloadError, KeyError, FileNotFoundError, NotADirectoryError, ValueError, ConnectionError) as exc:
            self.ui.error(str(exc).strip("'\""))

    # ------------------------------------------------------------------ chat
    async def chat(self, text: str, mode: str | None = None) -> TurnResult:
        """Send a message to the agent and handle the aftermath (report, staged edits)."""
        images, self.s.attachments = self.s.attachments, []
        res = await self.agent.run(text, mode, images)
        self.ui.finish()
        if res.error:
            self.ui.error(res.error)
        rep = report(res.ledger)
        if rep:
            self.ui.info(rep)
        if res.stopped == "completed":
            self.ui.emit("activity", ("COMPLETED", f"{res.steps} step(s), {res.seconds:.0f}s, context {self.s.context_tokens}/{self.s.settings.llm.context_size} tokens"))
        await self.review_pending()
        return res

    async def review_pending(self) -> None:
        ws = self.s.workspace
        if ws is None or not ws.pending:
            return
        self.ui.console.print(f"\n[accent]{len(ws.pending)} staged change(s)[/accent]")
        self.ui.diff(ws.diff())
        if self.s.settings.agent.auto_apply_edits or self.assume_yes:
            await self.cmd_apply("")
        elif self.ask is not None:
            if await self.confirm("Apply these changes?"):
                await self.cmd_apply("")
            else:
                self.ui.info("Changes stay staged. /apply to write them, /reject to discard, /diff to view.")
        else:
            self.ui.info("Changes are staged only (non-interactive run without --yes): nothing was written.")

    async def _mode(self, mode: str, arg: str) -> None:
        if arg:
            await self.chat(arg, mode)
        else:
            self.s.set_mode(mode)
            self.s.settings.mode = mode
            self.ui.ok(f"mode: /{mode}")

    async def cmd_ask(self, arg: str) -> None:
        await self._mode("ask", arg)

    async def cmd_plan(self, arg: str) -> None:
        await self._mode("plan", arg)

    async def cmd_agent(self, arg: str) -> None:
        await self._mode("agent", arg)

    async def cmd_code(self, arg: str) -> None:
        await self._mode("code", arg)

    async def cmd_review(self, arg: str) -> None:
        if arg or self.s.root is None:
            await self._mode("review", arg)
            return
        ok, diff = git.run_git(self.s.root, "diff", "HEAD")
        staged = self.s.workspace.diff() if self.s.workspace and self.s.workspace.pending else ""
        material = (staged + "\n" + (diff if ok else "")).strip()
        if not material:
            self.s.set_mode("review")
            self.ui.ok("mode: /review (no git diff or staged changes found; ask e.g. `/review src/app.py`)")
            return
        await self.chat("Review these changes:\n```diff\n" + material[:12000] + "\n```", "review")

    # ------------------------------------------------------------------ basics
    async def cmd_help(self, arg: str) -> None:
        self.ui.table("Commands", ["command", "what it does"], [[c, d] for c, d in HELP])
        self.ui.info("Type a message to talk to the agent. Alt+Enter = new line, Ctrl+C = cancel the running task, @file = mention a file, Ctrl+D = exit.")

    async def cmd_exit(self, arg: str) -> None:
        self.running = False

    async def cmd_quit(self, arg: str) -> None:
        self.running = False

    async def cmd_clear(self, arg: str) -> None:
        self.s.history.clear()
        self.s.plan.clear()
        self.s.context_tokens = 0
        self.ui.console.clear()
        self.ui.info("conversation cleared (persistent memory kept)")

    async def cmd_context(self, arg: str) -> None:
        cfg = self.s.settings
        used = message_tokens(self.s.history)
        tools = len(self.agent.toolbox.schemas(self.s.mode))
        tool_tokens = len(json.dumps(self.agent.toolbox.schemas(self.s.mode))) // 3
        rows = [["context window", f"{cfg.llm.context_size} tokens"], ["conversation", f"~{used} tokens in {len(self.s.history)} messages"],
                ["tool definitions", f"~{tool_tokens} tokens ({tools} tools available in /{self.s.mode})"],
                ["reserved for answer", f"{cfg.llm.max_output_tokens} tokens"],
                ["last request (measured)", f"{self.s.context_tokens} tokens"],
                ["staged edits", str(len(self.s.workspace.pending) if self.s.workspace else 0)],
                ["attachments", ", ".join(a.label for a in self.s.attachments) or "none"]]
        self.ui.table("Context", ["item", "value"], rows)
        self.ui.plan(self.s.plan)

    async def cmd_config(self, arg: str) -> None:
        parts = arg.split(None, 1)
        if not parts:
            import yaml

            self.ui.code(yaml.safe_dump(self.s.settings.model_dump(), sort_keys=False), "yaml")
            self.ui.info("change with: /config llm.context_size 8192   |   store a secret: /config secret <name>")
            return
        if parts[0] == "secret":
            if self.ask is None or len(parts) < 2:
                raise ValueError("usage (interactive only): /config secret <name>")
            value = await self.ask(f"value for secret {parts[1]!r} (stored in Windows Credential Manager): ")
            store_secret(parts[1], value.strip())
            self.ui.ok(f"stored. Reference it in config/mcp_servers.yaml as keyring:{parts[1]}")
            return
        if len(parts) < 2:
            raise ValueError("usage: /config <key> <value>")
        new = set_setting(self.s.settings, parts[0], parts[1])
        for field in type(new).model_fields:
            setattr(self.s.settings, field, getattr(new, field))
        self.ui.ok(f"{parts[0]} = {parts[1]} (saved)")
        if parts[0].startswith("llm."):
            self.s.llm.shutdown()
            self.ui.info("model servers will restart with the new settings on next use")

    async def cmd_doctor(self, arg: str) -> None:
        self.ui.info("running diagnostics...")
        checks = await doctor.run(self.s, inference="--quick" not in arg)
        self.ui.checks(checks)
        fails = sum(c.status == "fail" for c in checks)
        warns = sum(c.status == "warn" for c in checks)
        self.ui.console.print(f"\n{'[fail]' if fails else '[ok]'}{fails} failed[/], [warn]{warns} warning(s)[/warn]")

    # ------------------------------------------------------------------ models
    async def cmd_models(self, arg: str) -> None:
        parts = shlex.split(arg, posix=False)
        reg, hw = self.s.llm.registry, self.s.llm.hardware
        if parts and parts[0] == "delete":
            if len(parts) < 2:
                raise ValueError("usage: /models delete <id>")
            if await self.confirm(f"Delete model {parts[1]} from disk?"):
                self.s.llm.stop()
                self.ui.ok(f"deleted {parts[1]}, freed {reg.delete(parts[1]):.2f} GB")
            return
        if parts and parts[0] == "add":
            if len(parts) < 3 or parts[2] not in ("chat", "vision", "embed"):
                raise ValueError("usage: /models add <path.gguf> <chat|vision|embed> [mmproj.gguf]")
            inst = reg.register_local(Path(parts[1].strip('"')), parts[2], mmproj=Path(parts[3].strip('"')) if len(parts) > 3 else None)  # type: ignore[arg-type]
            self.ui.ok(f"registered {inst.spec.id}")
            return
        self.ui.console.print(f"[accent]Hardware[/accent]  {hw.cpu} | RAM {hw.ram_total_gb} GB | "
                              f"{hw.gpu_name or 'no NVIDIA GPU'}{f' {hw.vram_total_gb} GB VRAM' if hw.gpu_name else ''} | disk free {hw.disk_free_gb} GB")
        active = {self.s.llm.model_id(r) for r in ("chat", "vision", "embed")}  # type: ignore[arg-type]
        rows = []
        specs = {**reg.catalogue, **{k: v.spec for k, v in reg.installed.items()}}
        for m in sorted(specs.values(), key=lambda m: (["chat", "vision", "embed"].index(m.role), m.size_gb)):
            state = "active" if m.id in active else "installed" if m.id in reg.installed else ""
            rows.append([m.id, m.role, m.tier, f"{m.size_gb:.2f} GB", m.quant, f"{m.ctx_max // 1024}k", m.tools, "yes" if m.vision else "no",
                         f"{m.min_ram_gb:g}/{m.min_vram_gb:g}", fit(m, hw), m.license, state])
        self.ui.table("Models", ["id", "role", "tier", "size", "quant", "ctx", "tools", "vision", "RAM/VRAM GB", "fit", "licence", "state"], rows)
        rec = reg.recommend(hw)
        lines = []
        for role in ("chat", "vision", "embed"):
            picks = [f"{tier}: {item[0].id} ({item[1]})" for tier, item in rec[role].items() if item]
            lines.append(f"  {role:<7}" + " | ".join(picks))
        self.ui.console.print("[accent]Recommended for this machine[/accent]\n" + "\n".join(lines))
        variant, why = recommend_runtime(hw)
        exe = runtime.find_server(self.s.settings)
        self.ui.info(f"runtime: {exe or 'NOT INSTALLED'} (recommended variant: {variant} - {why})")
        self.ui.info("download with /download <id>; nothing is downloaded without your command.")

    async def cmd_model(self, arg: str) -> None:
        parts = arg.split()
        if not parts:
            self.ui.table("Active models", ["role", "model"], [[r, v] for r, v in self.s.llm.status().items()])
            return
        role, mid = (parts[0], parts[1]) if len(parts) > 1 else (None, parts[0])
        if role is None:
            role = self.s.llm.registry.spec(mid).role
        if role not in ("chat", "vision", "embed"):
            raise ValueError("usage: /model [chat|vision|embed] <id>")
        self.s.llm.set_model(role, mid)  # type: ignore[arg-type]
        self.ui.ok(f"{role} model -> {mid} (loads on next use)")
        if role == "embed" and self.s.index and self.s.index.counts()["embedded"]:
            self.ui.warn("embedding model changed: run /index --full so stored vectors match the new model")

    async def _with_progress(self, label: str, fn: Callable[[Callable[[int, int], None]], Any]) -> Any:
        with Progress(TextColumn("[accent]{task.description}"), BarColumn(), DownloadColumn(), TransferSpeedColumn(),
                      TimeRemainingColumn(), console=self.ui.console, transient=False) as bar:
            task = bar.add_task(label, total=None)
            return await asyncio.to_thread(fn, lambda done, total: bar.update(task, completed=done, total=total))

    async def cmd_download(self, arg: str) -> None:
        parts = shlex.split(arg, posix=False)
        reg, hw = self.s.llm.registry, self.s.llm.hardware
        if not parts:
            await self.cmd_models("")
            return
        if parts[0] == "runtime":
            zips = [Path(p.strip('"')) for p in parts[1:] if p.lower().endswith(".zip")]
            if zips:
                exe = runtime.install_from_zips(self.s.settings, zips, "manual-" + zips[0].stem)
                self.ui.ok(f"runtime installed from local archive: {exe}")
                return
            variant = parts[1] if len(parts) > 1 else (self.s.settings.llm.runtime_variant or recommend_runtime(hw)[0])
            if not await self.confirm(f"Download the official llama.cpp '{variant}' build from github.com/ggml-org/llama.cpp into {self.s.settings.runtime_path}?"):
                return
            exe = await self._with_progress(f"llama.cpp {variant}", lambda cb: runtime.install(self.s.settings, variant, cb))
            self.s.settings.llm.runtime_variant = variant
            save_settings(self.s.settings)
            self.s.llm.shutdown()
            self.ui.ok(f"runtime installed: {exe}")
            return
        for mid in parts:
            spec = reg.catalogue.get(mid)
            if spec is None:
                raise KeyError(f"{mid!r} is not in the catalogue (see /models)")
            how = fit(spec, hw)
            if how == "too-big":
                self.ui.warn(f"{mid} needs about {spec.min_ram_gb:g} GB RAM; this machine has {hw.ram_total_gb} GB. It will probably not run.")
            if not await self.confirm(f"Download {mid} ({spec.size_gb:.2f} GB, licence {spec.license}) from {self.s.settings.llm.hf_endpoint}/{spec.repo}?"):
                continue
            inst = await self._with_progress(mid, lambda cb, mid=mid: reg.download(mid, cb))
            self.ui.ok(f"{mid} installed at {inst.path} (SHA-256 {'verified' if inst.verified else 'not available'})")
        if runtime.find_server(self.s.settings) is None:
            self.ui.warn("the llama.cpp runtime is not installed yet: run /download runtime")

    # ------------------------------------------------------------------ project
    async def cmd_open(self, arg: str) -> None:
        if not arg:
            raise ValueError("usage: /open <folder>")
        self.s.open_project(Path(arg.strip('"')))
        self.ui.ok(f"project: {self.s.root}")
        await self.cmd_project("")
        if self.s.index and self.s.index.counts()["files"] == 0:
            self.ui.info("not indexed yet: run /index for search and project-wide questions")

    async def cmd_project(self, arg: str) -> None:
        if self.s.root is None:
            self.ui.warn("no project open. /open <folder>")
            return
        files = list(scan(self.s.root, self.s.settings.index.ignore_dirs))
        self.ui.console.print(f"[accent]{self.s.root}[/accent]\n{stats(files)}")
        if g := git.summary(self.s.root):
            self.ui.console.print(f"git: {g}")
        if self.s.mendix_project:
            self.ui.console.print(self.s.mendix_project.summary())
        if self.s.index:
            c = self.s.index.counts()
            self.ui.console.print(f"index: {c['files']} files, {c['chunks']} chunks ({c['embedded']} embedded), {c['symbols']} symbols, {c['deps']} imports")
        if summary := self.s.memory.get_kind("project_summary"):
            self.ui.markdown("**Summary**\n\n" + summary)
        self.ui.console.print(tree(self.s.root, self.s.settings.index.ignore_dirs, max_depth=2, max_entries=60), style="dim")

    async def cmd_index(self, arg: str) -> None:
        if self.s.index is None or self.s.root is None:
            raise ValueError("no project open. /open <folder>")
        self.ui.emit("activity", ("READING PROJECT", f"indexing {self.s.root}"))
        t0 = time.time()
        last = {"t": 0.0}

        def progress(stage: str, done: int, total: int) -> None:
            if time.time() - last["t"] > 2 or done == total:
                last["t"] = time.time()
                self.ui.info(f"  {stage}: {done}/{total}")

        embedder = self.s.embedder
        if embedder is None:
            self.ui.warn("no embedding model/runtime installed: building keyword index only (/download nomic-embed-v1.5 for semantic search)")
        res = await self.s.index.update(embedder, full="--full" in arg, progress=progress)
        c = self.s.index.counts()
        self.ui.ok(f"indexed in {time.time() - t0:.1f}s: {res['scanned']} files scanned, {res['changed']} changed, {res['removed']} removed, "
                   f"{res['embedded']} chunks embedded. Totals: {c['chunks']} chunks, {c['symbols']} symbols, {c['deps']} imports")
        if res["changed"] or not self.s.memory.get_kind("project_summary"):
            await self._summarize_project()

    async def _summarize_project(self) -> None:
        assert self.s.index and self.s.root
        facts = self.s.index.overview()
        if self.s.mendix_project:
            facts += "\n" + self.s.mendix_project.summary()
        summary = facts
        if self.s.llm.has("chat"):
            readme = next((p for p in (self.s.root / "README.md", self.s.root / "readme.md", self.s.root / "README.txt") if p.exists()), None)
            material = facts + "\n\nTree:\n" + tree(self.s.root, self.s.settings.index.ignore_dirs, 2, 80)
            if readme:
                material += "\n\nREADME excerpt:\n" + read_text(readme, 3000)
            try:
                res = await self.s.llm.chat([
                    {"role": "system", "content": "Write a factual project summary in at most 150 words: purpose, technologies, main "
                                                  "modules/folders, entry points. Use only the given material; do not speculate."},
                    {"role": "user", "content": material[:9000]}], max_tokens=350)
                if res.text.strip():
                    summary = res.text.strip() + "\n\n" + facts
            except LLMError as exc:
                self.ui.warn(f"LLM summary skipped: {str(exc).splitlines()[0]}")
        self.s.memory.upsert_kind("project_summary", summary)
        self.ui.info("project summary stored in project memory (/memory project)")

    async def cmd_search(self, arg: str) -> None:
        if not arg:
            raise ValueError("usage: /search <query>")
        self.ui.emit("activity", ("SEARCHING FILES", arg))
        if self.s.index is None or self.s.index.counts()["files"] == 0:
            out = await self.agent.toolbox.execute("grep", {"pattern": arg})
            self.ui.console.print(out.text)
            self.ui.info("(plain text search: project not indexed, run /index)")
            return
        hits = await self.s.index.search(arg, self.s.embedder, 8)
        syms = self.s.index.find_symbol(arg, 5) if " " not in arg else []
        for name, kind, path, line in syms:
            self.ui.console.print(f"[ok]symbol[/ok] {kind} {name}  {path}:{line}")
        for h in hits:
            self.ui.console.print(f"[accent]{h.path}:{h.start}-{h.end}[/accent] [dim]{h.source}[/dim]")
            self.ui.console.print("\n".join("    " + ln[:160] for ln in h.text.strip().splitlines()[:6]), style="dim", markup=False)
        if not hits and not syms:
            self.ui.warn("no results")

    async def cmd_read(self, arg: str) -> None:
        parts = shlex.split(arg, posix=False)
        if not parts or self.s.workspace is None:
            raise ValueError("usage: /read <file> [from-to]   (needs an open project)")
        text = self.s.workspace.read(parts[0].strip('"'))
        lines = text.splitlines()
        start, end = 1, min(len(lines), 200)
        if len(parts) > 1 and "-" in parts[1]:
            a, b = parts[1].split("-", 1)
            start, end = max(1, int(a)), min(len(lines), int(b))
        self.ui.code("\n".join(lines[start - 1:end]), Path(parts[0]).suffix.lstrip(".") or "text", start)
        if end < len(lines):
            self.ui.info(f"... {len(lines)} lines total; /read {parts[0]} {end + 1}-{min(len(lines), end + 200)}")

    # ------------------------------------------------------------------ editing
    def _ws(self):  # noqa: ANN202
        if self.s.workspace is None:
            raise ValueError("no project open. /open <folder>")
        return self.s.workspace

    async def cmd_write(self, arg: str) -> None:
        ws = self._ws()
        path, sep, inline = arg.partition(" :: ")
        if not path.strip():
            raise ValueError("usage: /write <file> [:: content]")
        if sep:
            content = inline.replace("\\n", "\n")
        elif self.ask is not None:
            content = await self.ask(f"content of {path.strip()} (Alt+Enter for new lines, Enter to finish):\n")
        else:
            raise ValueError("non-interactive use needs: /write <file> :: <content>")
        ch = ws.stage_write(path.strip().strip('"'), content if content.endswith("\n") else content + "\n")
        self.ui.ok(f"staged {ch.kind} {ch.rel}")
        await self.review_pending()

    async def cmd_edit(self, arg: str) -> None:
        self._ws()
        path, _, instruction = arg.partition(" ")
        if not path or not instruction:
            raise ValueError("usage: /edit <file> <what to change>")
        await self.chat(f"Edit the file `{path}`: {instruction}\nRead it first, then apply the change with edit_file.", "code")

    async def cmd_diff(self, arg: str) -> None:
        ws = self._ws()
        if not ws.pending:
            self.ui.info("no staged changes")
            return
        self.ui.diff(ws.diff())

    async def cmd_apply(self, arg: str) -> None:
        ws = self._ws()
        manifest = ws.apply(label=(self.s.history[-2]["content"][:80] if len(self.s.history) >= 2 and isinstance(self.s.history[-2].get("content"), str) else ""))
        for f in manifest["files"]:  # verify the result on disk
            full = ws.root / f["rel"]
            state = "deleted" if f["kind"] == "DELETE" and not full.exists() else "written" if full.exists() else "MISSING"
            self.ui.console.print(f"  [ok]{f['kind']}[/ok] {f['rel']} ({state})")
        self.ui.ok(f"applied change set {manifest['id']} ({len(manifest['files'])} file(s)); /undo rolls it back")
        if self.s.index and self.s.index.counts()["files"]:
            await self.s.index.update(self.s.embedder)

    async def cmd_reject(self, arg: str) -> None:
        n = self._ws().reject(arg.strip('"') or None)
        self.ui.ok(f"discarded {n} staged change(s)")

    async def cmd_undo(self, arg: str) -> None:
        manifest = self._ws().undo(arg or None)
        self.ui.ok(f"rolled back change set {manifest['id']}: " + ", ".join(f["rel"] for f in manifest["files"]))
        for w in manifest.get("warnings", []):
            self.ui.warn(w)

    async def cmd_history(self, arg: str) -> None:
        sub, _, sid = arg.partition(" ")
        if sub == "resume":
            rows = self.s.memory.session_messages(sid.strip(), 40)
            if not rows:
                raise KeyError(f"no stored messages for session {sid!r} (ids are listed by /history)")
            self.s.history = [{"role": role, "content": content} for role, content in rows if role in ("user", "assistant")]
            self.ui.ok(f"loaded {len(self.s.history)} message(s) from session {sid} into the conversation")
            return
        if sub == "show":
            for role, content in self.s.memory.session_messages(sid.strip(), 60):
                self.ui.console.print(f"[accent]{role}[/accent]: ", end="")
                self.ui.console.print(content[:1500], markup=False)
            return
        if self.s.workspace:
            rows = [[h["id"], time.strftime("%Y-%m-%d %H:%M", time.localtime(h["time"])), "undone" if h["undone"] else "applied",
                     ", ".join(f"{f['kind'][0]} {f['rel']}" for f in h["files"])[:70], h.get("label", "")[:40]] for h in self.s.workspace.history()[-15:]]
            self.ui.table("Applied change sets", ["id", "time", "state", "files", "request"], rows)
        sess = self.s.memory.sessions(self.s.memory.project_id)
        rows = [[sid, time.strftime("%Y-%m-%d %H:%M", time.localtime(started)), n, (title or "")[:60], (summary or "")[:60]]
                for sid, started, title, summary, _path, n in sess]
        self.ui.table("Sessions for this project", ["id", "started", "msgs", "first message", "summary"], rows)
        self.ui.info("/history show <id> prints a session, /history resume <id> continues it")

    # ------------------------------------------------------------------ memory
    async def cmd_memory(self, arg: str) -> None:
        sub, _, rest = arg.partition(" ")
        mem = self.s.memory
        if sub == "search":
            hits = await mem.search(rest, self.s.embedder, 10)
            self.ui.table(f"Memory search: {rest}", ["id", "scope", "kind", "text"], [[m.id, m.scope, m.kind, m.text[:300]] for m in hits])
        elif sub == "clear-session":
            n = mem.clear_session()
            self.s.history.clear()
            self.ui.ok(f"removed {n} stored message(s) of this session and cleared the conversation")
        elif sub == "add":
            mid = await mem.remember(rest, "note", "project", self.s.embedder)
            self.ui.ok(f"saved memory #{mid}")
        elif sub == "forget":
            self.ui.ok("deleted" if mem.forget(int(rest)) else "no such memory in this project")
        elif sub in ("project", "global", ""):
            items = mem.list(sub or None, 40)
            title = {"project": f"Project memory ({self.s.root or 'no project'})", "global": "Long-term (global) memory", "": "Memory visible in this project"}[sub]
            self.ui.table(title, ["id", "scope", "kind", "when", "text"],
                          [[m.id, m.scope, m.kind, time.strftime("%m-%d %H:%M", time.localtime(m.ts)), m.text[:400]] for m in items])
            prev = mem.previous_context(mem.project_id, 5)
            if prev and sub != "global":
                self.ui.table("Earlier sessions", ["started", "first message", "summary"],
                              [[time.strftime("%Y-%m-%d %H:%M", time.localtime(st)), (t or "")[:60], (sm or "")[:100]] for st, t, sm in prev])
        else:
            raise ValueError("usage: /memory [search <q>|project|global|add <text>|forget <id>|clear-session]")

    # ------------------------------------------------------------------ images
    async def cmd_attach(self, arg: str) -> None:
        if arg == "clear":
            self.s.attachments.clear()
            self.ui.ok("attachments removed")
            return
        if arg == "preview":
            if not self.s.attachments:
                raise ValueError("nothing attached")
            import os
            import tempfile

            for img in self.s.attachments:  # open exactly what the model will receive
                fd, tmp = tempfile.mkstemp(suffix=".png", prefix="lma-preview-")
                with os.fdopen(fd, "wb") as fh:
                    fh.write(img.data)
                self.ui.info(f"{img.meta()} -> {tmp}")
                if hasattr(os, "startfile"):
                    os.startfile(tmp)  # noqa: S606 - default image viewer of the user
            return
        if arg in ("paste", "clipboard", ""):
            img = from_clipboard()
            if img is None:
                raise ValueError("no image on the clipboard (take a screenshot with Win+Shift+S first), or give a file path")
        else:
            p = Path(arg.strip('"'))
            img = load_image(p if p.is_absolute() or self.s.root is None else self.s.root / p)
        self.s.attachments.append(img)
        self.ui.ok(f"attached {img.meta()} - it goes with your next message")

    async def cmd_vision(self, arg: str) -> None:
        first, _, rest = arg.partition(" ")
        if first in ("paste", "clipboard") or Path(first.strip('"')).suffix.lower() in IMAGE_SUFFIXES:
            await self.cmd_attach(first)
            arg = rest
        if not self.s.attachments:
            self.s.set_mode("vision")
            self.ui.ok("mode: /vision - attach with /attach <file> or /attach paste, then ask")
            return
        await self.chat(arg or "Describe this screenshot in detail.", "vision")

    # ------------------------------------------------------------------ MCP
    async def cmd_mcp(self, arg: str) -> None:
        parts = shlex.split(arg, posix=False)
        sub = parts[0] if parts else "list"
        mcp = self.s.mcp
        if sub == "list":
            rows = []
            for name, c in mcp.connections.items():
                rows.append([name, c.cfg.transport, c.target[:70], "connected" if c.connected else "disconnected",
                             len(c.tools), len(c.resources), len(c.prompts), (c.last_error or "")[:60]])
            self.ui.table("MCP servers", ["name", "transport", "target", "state", "tools", "resources", "prompts", "last error"], rows)
            if not rows:
                self.ui.info("no servers. Add them to config/mcp_servers.yaml or: /mcp add <name> <url | command args...>")
            return
        if sub == "add":
            if len(parts) < 3:
                raise ValueError("usage: /mcp add <name> <http://url/mcp | command [args...]>")
            target = parts[2]
            cfg = ServerConfig(transport="http", url=target) if target.startswith("http") else ServerConfig(transport="stdio", command=target.strip('"'), args=[p.strip('"') for p in parts[3:]])
            mcp.config.servers[parts[1]] = cfg
            save_mcp_config(mcp.config)
            mcp.add(parts[1], cfg)
            self.ui.ok(f"added {parts[1]}; connecting...")
            sub, parts = "connect", ["connect", parts[1]]
        if len(parts) < 2:
            raise ValueError(f"usage: /mcp {sub} <server>")
        conn = mcp.get(parts[1])
        if sub == "connect":
            self.ui.emit("activity", ("EXECUTING TOOL", f"connecting to {conn.target}"))
            try:
                await conn.connect()
            except Exception as exc:  # noqa: BLE001
                raise ConnectionError(f"{parts[1]}: {conn.last_error or describe_error(exc)}") from exc
            info = conn.init_result.server_info if conn.init_result else None
            self.ui.ok(f"connected to {getattr(info, 'name', parts[1])} {getattr(info, 'version', '')}: {len(conn.tools)} tools, {len(conn.resources)} resources, {len(conn.prompts)} prompts")
            self.agent.toolbox.refresh_mcp()
        elif sub == "disconnect":
            await conn.disconnect()
            self.agent.toolbox.refresh_mcp()
            self.ui.ok(f"disconnected {parts[1]}")
        elif sub == "tools":
            self._tools_table(parts[1], list(conn.tools.values()))
        elif sub == "inspect":
            info = conn.init_result
            self.ui.console.print(f"[accent]{parts[1]}[/accent] {conn.target} - {'connected' if conn.connected else 'disconnected'}")
            if info:
                self.ui.console.print(f"server: {info.server_info.name} {info.server_info.version}, protocol {info.protocol_version}")
                self.ui.console.print(f"capabilities: {', '.join(k for k, v in info.capabilities.model_dump(exclude_none=True).items()) or 'none'}")
                if info.instructions:
                    self.ui.console.print(f"instructions: {info.instructions[:600]}", markup=False)
            self._tools_table(parts[1], list(conn.tools.values()))
            if conn.resources:
                self.ui.table("Resources", ["uri", "name", "description"], [[str(r.uri), r.name, (r.description or "")[:80]] for r in conn.resources])
            if conn.prompts:
                self.ui.table("Prompts", ["name", "arguments", "description"],
                              [[p.name, ", ".join(a.name for a in (p.arguments or [])), (p.description or "")[:80]] for p in conn.prompts])
        elif sub == "test":
            if not conn.connected:
                await conn.connect()
            ms = (await conn.ping()) * 1000
            self.ui.ok(f"ping {ms:.0f} ms, {len(conn.tools)} tools discovered")
            safe = [t for t in conn.tools.values() if t.policy == "read" and not t.required]
            if safe:
                res = await conn.call_tool(safe[0].name, {})
                (self.ui.ok if res.ok else self.ui.error)(f"called read-only tool {safe[0].name}: {'OK' if res.ok else res.error} ({res.duration_s:.2f}s)")
                self.ui.console.print(res.text[:800], style="dim", markup=False)
            else:
                self.ui.info("no argument-free read-only tool to call automatically; use /mcp call <server> <tool> <json>")
        elif sub == "prompt":
            if len(parts) < 3:
                raise ValueError("usage: /mcp prompt <server> <prompt-name> [json-arguments]")
            raw = arg.split(None, 3)[3] if len(arg.split(None, 3)) > 3 else "{}"
            text = await conn.get_prompt(parts[2], json.loads(raw))
            self.ui.console.print(text, markup=False)
            if await self.confirm("Send this prompt to the agent?"):
                await self.chat(text)
        elif sub == "resource":
            if len(parts) < 3:
                raise ValueError("usage: /mcp resource <server> <uri>")
            self.ui.console.print((await conn.read_resource(parts[2]))[:6000], markup=False)
        elif sub == "call":
            if len(parts) < 3:
                raise ValueError("usage: /mcp call <server> <tool> [json-arguments]")
            raw = arg.split(None, 3)[3] if len(arg.split(None, 3)) > 3 else "{}"
            await self._manual_call(parts[1], parts[2], raw)
        else:
            raise ValueError("usage: /mcp [list|add|connect|disconnect|tools|inspect|test|call] <server>")

    def _tools_table(self, server: str, tools: list[Any]) -> None:
        rows = [[t.name, f"{t.policy} ({t.policy_source})", ", ".join(f"{k}{'*' if k in t.required else ''}" for k in (t.input_schema.get("properties") or {})),
                 t.description.strip().splitlines()[0][:90] if t.description.strip() else ""] for t in tools]
        self.ui.table(f"Tools of {server} ({len(tools)})", ["tool", "access", "arguments (* required)", "description"], rows)

    async def _manual_call(self, server: str, tool: str, raw: str) -> None:
        conn = self.s.mcp.get(server)
        try:
            args = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"arguments must be a JSON object, e.g. {{\"name\": \"x\"}} ({exc})") from exc
        info = conn.tools.get(tool)
        if info is None:
            raise KeyError(f"{server} has no tool {tool!r} (see /mcp tools {server})")
        if info.policy != "read" and not await self.confirm(f"{tool} can MODIFY state on {server}. Run it with {raw}?"):
            self.ui.info("not executed")
            return
        self.ui.emit("activity", ("CALLING MAIA MCP" if server == "mendix" else "EXECUTING TOOL", f"{tool}({raw[:80]})"))
        res = await conn.call_tool(tool, args)
        (self.ui.ok if res.ok else self.ui.error)(f"{tool}: {'OK' if res.ok else 'ERROR'} in {res.duration_s:.2f}s")
        self.ui.console.print((res.text or res.error or "")[:6000], markup=False)

    # ------------------------------------------------------------------ Mendix
    async def cmd_mendix(self, arg: str) -> None:
        parts = arg.split(None, 1)
        sub = parts[0] if parts else ""
        rest = parts[1] if len(parts) > 1 else ""
        mx = self.s.mendix
        if sub == "":
            if rest == "" and not arg:
                self.s.set_mode("mendix")
                self.ui.ok("mode: /mendix" + ("" if mx.connected else " (Studio Pro MCP not connected: /mendix connect)"))
            return
        if sub == "connect":
            if rest:
                if rest.isdigit():
                    mx.cfg.port, mx.cfg.url = int(rest), None
                else:
                    mx.cfg.url = rest
                save_mcp_config(self.s.mcp.config)
                self.ui.info("saved to config/mcp_servers.yaml")
            self.ui.emit("activity", ("CALLING MAIA MCP", "locating and initializing the Studio Pro MCP server"))
            d = await mx.connect()
            self._diagnosis(d)
            self.agent.toolbox.refresh_mcp()
        elif sub == "status":
            if mx.last_diagnosis is None or not mx.connected:
                await mx.connect()
                self.agent.toolbox.refresh_mcp()
            self._diagnosis(mx.last_diagnosis)
            if mx.connected and mx.conn:
                try:
                    self.ui.ok(f"health: ping {(await mx.conn.ping()) * 1000:.0f} ms, connected for {time.time() - (mx.conn.connected_at or time.time()):.0f}s")
                except Exception as exc:  # noqa: BLE001
                    self.ui.error(f"health check failed: {describe_error(exc)} (try /mendix connect)")
        elif sub in ("tools", "capabilities", "inspect", "test", "call"):
            if not mx.connected:
                d = await mx.connect()
                self.agent.toolbox.refresh_mcp()
                if not d.connected:
                    self._diagnosis(d)
                    return
            assert mx.conn is not None
            if sub == "tools":
                if rest:
                    t = mx.conn.tools.get(rest)
                    if t is None:
                        raise KeyError(f"Maia has no tool {rest!r}")
                    self.ui.console.print(f"[accent]{t.name}[/accent] ({t.policy}, {t.policy_source})\n{t.description}", markup=True)
                    self.ui.code(json.dumps(t.input_schema, indent=2), "json")
                else:
                    self._tools_table("mendix", mx.tools())
                    self.ui.info("schema of one tool: /mendix tools <name>. Access is from server annotations where given, otherwise a name heuristic; "
                                 "override per tool in config/mcp_servers.yaml (mendix.tool_policy).")
            elif sub == "capabilities":
                caps = mx.capabilities()
                self.ui.console.print(f"[accent]Discovered from the running server ({mx.last_diagnosis.url if mx.last_diagnosis else ''})[/accent]")
                self.ui.console.print(f"[ok]Read-only ({len(caps['read'])}):[/ok] {', '.join(caps['read']) or 'none'}")
                self.ui.console.print(f"[warn]Modifying, approval required ({len(caps['write'])}):[/warn] {', '.join(caps['write']) or 'none'}")
                if caps["deny"]:
                    self.ui.console.print(f"[fail]Denied by config:[/fail] {', '.join(caps['deny'])}")
                self.ui.console.print(f"Resources: {len(mx.conn.resources)}, prompts: {len(mx.conn.prompts)}")
                self.ui.info("Only what is listed here is possible. Official docs note for 11.10: no page generation; web fetch waits for approval in Studio Pro.")
            elif sub == "inspect":
                for name, res in await mx.inspect():
                    self.ui.console.print(f"\n[accent]{name}[/accent] {'[ok]OK[/ok]' if res.ok else '[fail]ERROR[/fail]'} [dim]{res.duration_s:.2f}s[/dim]")
                    self.ui.console.print((res.text or res.error or "")[:2500], markup=False)
                if not mx.safe_read_tools():
                    self.ui.info("the server exposes no argument-free read tools; ask in /mendix mode, e.g. 'show me the modules of this app'")
            elif sub == "test":
                self.ui.console.print("1. connection  [ok]OK[/ok]")
                self.ui.console.print(f"2. discovery   [ok]OK[/ok] {len(mx.tools())} tools")
                name, res = await mx.smoke_test()
                if name is None or res is None:
                    self.ui.warn("3. read call    skipped: no argument-free read-only tool; use /mendix call <tool> <json>")
                elif res.ok:
                    self.ui.console.print(f"3. read call   [ok]OK[/ok] {name} in {res.duration_s:.2f}s")
                    self.ui.console.print(res.text[:1200], style="dim", markup=False)
                else:
                    self.ui.error(f"3. read call   FAILED {name}: {res.error}")
            else:
                tool, _, raw = rest.partition(" ")
                await self._manual_call("mendix", tool, raw or "{}")
        else:
            await self.chat(arg, "mendix")

    def _diagnosis(self, d: Any) -> None:
        if d is None:
            return
        for step in d.steps:
            mark = "[ok]OK  [/ok]" if step.ok else "[dim]info[/dim]" if step.ok is None else "[fail]FAIL[/fail]"
            self.ui.console.print(f"  {mark} [bold]{step.name}[/bold]: ", end="")
            self.ui.console.print(step.detail, markup=False)
        if d.connected:
            self.ui.ok(f"Maia MCP connected at {d.url}")
        else:
            self.ui.error("Maia MCP is not connected.")
            for h in d.hints:
                self.ui.console.print(f"  - {h}", markup=False)
            self.ui.info(f"docs: {doctor_docs()}")


def doctor_docs() -> str:
    from lma.mcpx.mendix import DOCS_URL

    return DOCS_URL


def reload_settings_into(state: AppState) -> None:
    new = load_settings()
    for field in type(new).model_fields:
        setattr(state.settings, field, getattr(new, field))
