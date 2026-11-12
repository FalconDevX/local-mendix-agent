"""Tool registry and executor.

Every tool call from the model goes through `Toolbox.execute`: argument validation, permission
check, timeout, truncation and a ledger entry describing what actually happened.
"""
from __future__ import annotations

import asyncio
import base64
import fnmatch
import json
import re
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lma.core.state import AppState, LedgerEntry
from lma.editing.editor import Change, EditError, validate
from lma.project.scanner import classify, scan, tree
from lma.tools import git
from lma.tools.image import SCREENSHOT_PROMPT, load_bytes, load_image

Handler = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    risk: str  # read | stage | write | exec
    handler: Handler
    activity: str = "EXECUTING TOOL"

    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


@dataclass
class Outcome:
    ok: bool
    text: str


def obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required or []}


S = {"type": "string"}
I = {"type": "integer"}  # noqa: E741
B = {"type": "boolean"}


def summarize_args(args: dict[str, Any], limit: int = 90) -> str:
    parts = []
    for k, v in args.items():
        text = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        text = text.replace("\n", "\\n")
        parts.append(f"{k}={text[:40] + '...' if len(text) > 40 else text}")
    out = ", ".join(parts)
    return out[:limit] + ("..." if len(out) > limit else "")


class Toolbox:
    def __init__(self, state: AppState) -> None:
        self.s = state
        self.tools: dict[str, ToolSpec] = {}
        self._register_builtin()

    # ---- registry ------------------------------------------------------
    def add(self, name: str, description: str, parameters: dict[str, Any], risk: str, handler: Handler, activity: str = "EXECUTING TOOL") -> None:
        self.tools[name] = ToolSpec(name, description, parameters, risk, handler, activity)

    def refresh_mcp(self) -> None:
        """(Re)build MCP tool entries from what the connected servers actually report."""
        for name in [n for n in self.tools if n.startswith("mcp__")]:
            del self.tools[name]
        for info in self.s.mcp.all_tools():
            if info.policy == "deny":
                continue
            full = f"mcp__{re.sub(r'[^A-Za-z0-9_-]', '_', info.server)}__{info.name}"
            label = "CALLING MAIA MCP" if info.server == "mendix" else f"CALLING MCP {info.server}"

            async def handler(args: dict[str, Any], _info=info) -> str:
                return await self._call_mcp(_info.server, _info.name, args)

            schema = dict(info.input_schema)
            schema.setdefault("type", "object")
            schema.setdefault("properties", {})
            self.add(full, (info.description or info.name)[:1000], schema, "read" if info.policy == "read" else "write", handler, label)

    def schemas(self, mode: str) -> list[dict[str, Any]]:
        read_only = mode in ("ask", "plan", "review", "vision")
        out = []
        for t in self.tools.values():
            if read_only and t.risk != "read":
                continue
            if t.name == "run_command" and not self.s.settings.agent.allow_shell:
                continue
            if self.s.workspace is None and not (t.name.startswith("mcp__") or t.name in ("remember", "recall", "update_plan", "mcp_read_resource")):
                continue
            if mode == "mendix" and t.name in ("run_command", "git_commit"):
                continue
            out.append(t.schema())
        return out

    # ---- execution -----------------------------------------------------
    async def execute(self, name: str, args: dict[str, Any]) -> Outcome:
        spec = self.tools.get(name)
        if spec is None:
            close = ", ".join(sorted(self.tools)[:40])
            return self._record(name, args, False, 0, f"unknown tool {name!r}. Available tools: {close}", "read")
        missing = [r for r in spec.parameters.get("required", []) if r not in args]
        if missing:
            return self._record(name, args, False, 0, f"missing required argument(s): {', '.join(missing)}", spec.risk)
        decision = await self.s.permissions.check(name, spec.risk, summarize_args(args, 200))
        if not decision.allowed:
            return self._record(name, args, False, 0, f"NOT EXECUTED - permission denied: {decision.reason}", spec.risk)
        self.s.emit("activity", (spec.activity, f"{name}({summarize_args(args)})"))
        t0 = time.perf_counter()
        try:
            text = await asyncio.wait_for(spec.handler(args), self.s.settings.agent.tool_timeout_s)
            ok = True
        except asyncio.TimeoutError:
            ok, text = False, f"tool timed out after {self.s.settings.agent.tool_timeout_s}s"
        except (EditError, FileNotFoundError, PermissionError, ValueError, KeyError, OSError) as exc:
            ok, text = False, f"{type(exc).__name__}: {exc}"
        except ToolFailure as exc:
            ok, text = False, str(exc)
        except Exception as exc:  # noqa: BLE001 - a tool must never crash the agent loop
            ok, text = False, f"{type(exc).__name__}: {exc}"
        return self._record(name, args, ok, time.perf_counter() - t0, text, spec.risk)

    def _record(self, name: str, args: dict[str, Any], ok: bool, seconds: float, text: str, risk: str) -> Outcome:
        limit = self.s.settings.agent.tool_result_chars
        shown = text if len(text) <= limit else text[:limit] + f"\n... [truncated {len(text) - limit} of {len(text)} chars; narrow the request]"
        entry = LedgerEntry(name, summarize_args(args), ok, seconds, text[:300], risk)
        self.s.ledger.append(entry)
        self.s.emit("tool_result", entry)
        return Outcome(ok, shown if ok else "ERROR: " + shown)

    # ---- MCP -----------------------------------------------------------
    async def _call_mcp(self, server: str, tool: str, args: dict[str, Any]) -> str:
        res = await self.s.mcp.get(server).call_tool(tool, args)
        text = res.text
        if res.images:
            if self.s.llm.model_id("vision"):
                self.s.emit("activity", ("THINKING", f"analysing {len(res.images)} image(s) returned by {tool}"))
                imgs = [load_bytes(base64.b64decode(b64), f"{tool}-{i}").data for i, (_, b64) in enumerate(res.images)]
                text += "\n[vision model description of returned image(s)]\n" + await self.s.llm.describe_images(SCREENSHOT_PROMPT, imgs)
            else:
                text += f"\n[{len(res.images)} image(s) returned but no local vision model is installed to read them]"
        if not res.ok:
            raise ToolFailure(res.error or text or "MCP tool reported an error")
        return text or "(tool returned no content)"

    # ---- built-in tools ------------------------------------------------
    def _ws(self):  # noqa: ANN202
        if self.s.workspace is None:
            raise ToolFailure("no project is open; the user must run /open <folder> first")
        return self.s.workspace

    def _register_builtin(self) -> None:
        a = self.add
        a("list_dir", "List files and folders of the project as a tree.", obj({"path": S, "depth": I}), "read", self.list_dir, "READING PROJECT")
        a("read_file", "Read a text file with line numbers. Use start_line/end_line for large files (max 400 lines per call).",
          obj({"path": S, "start_line": I, "end_line": I}, ["path"]), "read", self.read_file, "READING PROJECT")
        a("file_info", "Size, type, modification time and line count of a file.", obj({"path": S}, ["path"]), "read", self.file_info, "READING PROJECT")
        a("grep", "Regex search in file contents. Optional glob filters file paths (e.g. '*.py').",
          obj({"pattern": S, "glob": S, "max_results": I}, ["pattern"]), "read", self.grep, "SEARCHING FILES")
        a("search_code", "Semantic + keyword search over the indexed project. Best for 'where is X implemented' questions.",
          obj({"query": S, "limit": I}, ["query"]), "read", self.search_code, "SEARCHING FILES")
        a("find_symbol", "Find where a class/function/type is defined.", obj({"name": S}, ["name"]), "read", self.find_symbol, "SEARCHING FILES")
        a("file_dependencies", "What a file imports and which project files import it.", obj({"path": S}, ["path"]), "read", self.file_dependencies, "SEARCHING FILES")
        a("project_overview", "File statistics, most-imported files, Mendix project facts and the stored project summary.", obj({}), "read", self.project_overview, "READING PROJECT")
        a("write_file", "Create or fully overwrite a text file. The change is STAGED and shown to the user as a diff; it is written to disk only after approval.",
          obj({"path": S, "content": S}, ["path", "content"]), "stage", self.write_file, "EDITING")
        a("edit_file", "Replace an exact text fragment in a file (staged). old_text must match the file exactly and be unique unless replace_all is true.",
          obj({"path": S, "old_text": S, "new_text": S, "replace_all": B}, ["path", "old_text", "new_text"]), "stage", self.edit_file, "EDITING")
        a("delete_file", "Stage deletion of a file.", obj({"path": S}, ["path"]), "stage", self.delete_file, "EDITING")
        a("rename_file", "Stage moving/renaming a file.", obj({"source": S, "destination": S}, ["source", "destination"]), "stage", self.rename_file, "EDITING")
        a("create_dir", "Create a directory inside the project.", obj({"path": S}, ["path"]), "write", self.create_dir, "EDITING")
        a("git_status", "Git branch and changed files.", obj({}), "read", self.git_status, "READING PROJECT")
        a("git_diff", "Git diff of the working tree (optionally one path).", obj({"path": S}), "read", self.git_diff, "READING PROJECT")
        a("git_log", "Recent commits.", obj({"count": I}), "read", self.git_log, "READING PROJECT")
        a("git_commit", "Commit all applied changes with a message (needs user approval).", obj({"message": S}, ["message"]), "exec", self.git_commit)
        a("run_command", "Run a shell command in the project root (needs user approval every time). Use for tests/builds.",
          obj({"command": S, "timeout_s": I}, ["command"]), "exec", self.run_command)
        a("analyze_image", "Let the local vision model look at an image file (screenshot) of the project and answer a question about it.",
          obj({"path": S, "question": S}, ["path"]), "read", self.analyze_image, "THINKING")
        a("remember", "Save an important fact, decision or convention to persistent memory. scope: project or global.",
          obj({"text": S, "kind": S, "scope": {"type": "string", "enum": ["project", "global"]}}, ["text"]), "read", self.remember)
        a("recall", "Search persistent memory (earlier sessions, decisions, project facts).", obj({"query": S}, ["query"]), "read", self.recall)
        a("update_plan", "Update the status of a plan step. status: in_progress, done, failed or skipped.",
          obj({"step": I, "status": {"type": "string", "enum": ["in_progress", "done", "failed", "skipped"]}, "note": S}, ["step", "status"]), "read", self.update_plan, "PLANNING")
        a("mcp_read_resource", "Read a resource exposed by a connected MCP server.", obj({"server": S, "uri": S}, ["server", "uri"]), "read", self.mcp_read_resource)

    # filesystem ---------------------------------------------------------
    async def list_dir(self, args: dict[str, Any]) -> str:
        ws = self._ws()
        target = ws.resolve(args.get("path") or ".")
        if not target.is_dir():
            raise ToolFailure(f"{args.get('path')} is not a directory")
        return tree(target, self.s.settings.index.ignore_dirs, max_depth=int(args.get("depth") or 2))

    async def read_file(self, args: dict[str, Any]) -> str:
        ws = self._ws()
        text = ws.read(args["path"])
        lines = text.splitlines()
        start = max(1, int(args.get("start_line") or 1))
        end = min(len(lines), int(args.get("end_line") or start + 399), start + 399)
        body = "\n".join(f"{i:>5}| {lines[i - 1]}" for i in range(start, end + 1))
        staged = " (showing STAGED, not yet applied content)" if ws.rel(args["path"]) in ws.pending else ""
        more = f"\n... file has {len(lines)} lines; call again with start_line={end + 1}" if end < len(lines) else ""
        return f"{ws.rel(args['path'])} lines {start}-{end} of {len(lines)}{staged}\n{body}{more}"

    async def file_info(self, args: dict[str, Any]) -> str:
        ws = self._ws()
        p = ws.resolve(args["path"])
        if not p.exists():
            raise ToolFailure(f"{args['path']} does not exist")
        st = p.stat()
        kind = "directory" if p.is_dir() else classify(p)
        info = f"{ws.rel(p)}: {kind}, {st.st_size} bytes, modified {time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}"
        if p.is_file() and kind not in ("binary", "image") and not kind.startswith("mendix") and st.st_size < 5_000_000:
            info += f", {len(ws.read(p).splitlines())} lines"
        return info

    async def grep(self, args: dict[str, Any]) -> str:
        ws = self._ws()
        try:
            rx = re.compile(args["pattern"], re.I)
        except re.error as exc:
            raise ToolFailure(f"invalid regex: {exc}") from exc
        glob, limit = args.get("glob"), int(args.get("max_results") or 40)
        hits: list[str] = []
        for f in scan(ws.root, self.s.settings.index.ignore_dirs):
            if not f.is_text or f.size > self.s.settings.index.max_file_bytes:
                continue
            if glob and not (fnmatch.fnmatch(f.rel, glob) or fnmatch.fnmatch(Path(f.rel).name, glob)):
                continue
            try:
                content = ws.read(f.rel)
            except EditError:
                continue
            for n, line in enumerate(content.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{f.rel}:{n}: {line.strip()[:200]}")
                    if len(hits) >= limit:
                        return "\n".join(hits) + f"\n... stopped at {limit} matches"
        return "\n".join(hits) if hits else "no matches"

    async def search_code(self, args: dict[str, Any]) -> str:
        idx = self._index()
        hits = await idx.search(args["query"], self.s.embedder, int(args.get("limit") or 6))
        if not hits:
            return "no results (try grep with a regex, or other keywords)"
        mode = "hybrid (keyword + embeddings)" if idx.counts()["embedded"] and self.s.embedder else "keyword only (no embedding model installed)"
        parts = [f"search mode: {mode}"]
        for h in hits:
            snippet = "\n".join(h.text.splitlines()[:25])
            parts.append(f"--- {h.path}:{h.start}-{h.end} [{h.source}]\n{snippet}")
        return "\n".join(parts)

    def _index(self):  # noqa: ANN202
        self._ws()
        idx = self.s.index
        if idx is None or idx.counts()["files"] == 0:
            raise ToolFailure("the project is not indexed yet; ask the user to run /index (grep and read_file still work)")
        return idx

    async def find_symbol(self, args: dict[str, Any]) -> str:
        rows = self._index().find_symbol(args["name"])
        return "\n".join(f"{kind} {name} - {path}:{line}" for name, kind, path, line in rows) or "symbol not found in the index"

    async def file_dependencies(self, args: dict[str, Any]) -> str:
        idx = self._index()
        rel = self._ws().rel(args["path"])
        deps = idx.dependencies(rel)
        users = idx.dependents(rel)
        imports = "\n".join(f"  {t}" + (f"  -> {r}" if r else "  (external)") for t, r in deps) or "  none found"
        return f"{rel} imports:\n{imports}\nimported by:\n" + ("\n".join("  " + u for u in users) or "  no project file")

    async def project_overview(self, args: dict[str, Any]) -> str:
        ws = self._ws()
        parts = [f"Project root: {ws.root}"]
        if self.s.index and self.s.index.counts()["files"]:
            parts.append(self.s.index.overview())
        else:
            parts.append("(not indexed yet)")
        if self.s.mendix_project:
            parts.append(self.s.mendix_project.summary())
        if g := git.summary(ws.root):
            parts.append(f"Git: {g}")
        if summary := self.s.memory.get_kind("project_summary"):
            parts.append("Stored project summary:\n" + summary)
        parts.append(tree(ws.root, self.s.settings.index.ignore_dirs, max_depth=2, max_entries=80))
        return "\n".join(parts)

    # editing ------------------------------------------------------------
    def _staged(self, ch: Change) -> str:
        if ch.old == ch.new:
            return f"no change: {ch.rel} already has this content"
        old_n, new_n = len((ch.old or "").splitlines()), len((ch.new or "").splitlines())
        msg = f"STAGED {ch.kind} {ch.rel} ({old_n} -> {new_n} lines). Not written to disk until the user approves."
        if ch.new is not None:
            err = validate(ch.rel, ch.new)
            self.s.emit("activity", ("VALIDATING", f"{ch.rel}: {'syntax OK' if err is None else err}"))
            msg += f" Syntax check: {'OK' if err is None else 'FAILED - ' + err + ' - fix it with another edit.'}"
        return msg

    async def write_file(self, args: dict[str, Any]) -> str:
        return self._staged(self._ws().stage_write(args["path"], args["content"]))

    async def edit_file(self, args: dict[str, Any]) -> str:
        return self._staged(self._ws().stage_edit(args["path"], args["old_text"], args["new_text"], bool(args.get("replace_all"))))

    async def delete_file(self, args: dict[str, Any]) -> str:
        return self._staged(self._ws().stage_delete(args["path"]))

    async def rename_file(self, args: dict[str, Any]) -> str:
        new, old = self._ws().stage_rename(args["source"], args["destination"])
        return f"STAGED rename {old.rel} -> {new.rel}. Not written to disk until the user approves."

    async def create_dir(self, args: dict[str, Any]) -> str:
        p = self._ws().resolve(args["path"])
        p.mkdir(parents=True, exist_ok=True)
        return f"directory {self._ws().rel(p)} exists now"

    # git / shell --------------------------------------------------------
    async def git_status(self, args: dict[str, Any]) -> str:
        ok, out = git.run_git(self._ws().root, "status", "--short", "--branch")
        if not ok:
            raise ToolFailure(out)
        return out or "clean working tree"

    async def git_diff(self, args: dict[str, Any]) -> str:
        cmd = ["diff", "--stat", "--patch"] + (["--", self._ws().rel(args["path"])] if args.get("path") else [])
        ok, out = git.run_git(self._ws().root, *cmd)
        if not ok:
            raise ToolFailure(out)
        return out or "no unstaged changes"

    async def git_log(self, args: dict[str, Any]) -> str:
        ok, out = git.run_git(self._ws().root, "log", f"-{int(args.get('count') or 10)}", "--pretty=format:%h %ad %an: %s", "--date=short")
        if not ok:
            raise ToolFailure(out)
        return out

    async def git_commit(self, args: dict[str, Any]) -> str:
        ws = self._ws()
        if ws.pending:
            raise ToolFailure("there are staged edits that are not applied yet; they must be applied (/apply) before committing")
        ok, out = git.run_git(ws.root, "add", "-A")
        if ok:
            ok, out = git.run_git(ws.root, "commit", "-m", args["message"])
        if not ok:
            raise ToolFailure(out)
        return out

    async def run_command(self, args: dict[str, Any]) -> str:
        ws = self._ws()
        if not self.s.settings.agent.allow_shell:
            raise ToolFailure("shell commands are disabled (agent.allow_shell is false)")
        timeout = min(int(args.get("timeout_s") or 60), self.s.settings.agent.tool_timeout_s - 5)
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

        def run() -> tuple[int, str]:
            try:
                p = subprocess.run(args["command"], shell=True, cwd=ws.root, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=timeout, creationflags=flags)
                return p.returncode, (p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr.strip() else "")).strip()
            except subprocess.TimeoutExpired:
                return -1, f"command timed out after {timeout}s"

        code, out = await asyncio.to_thread(run)
        text = f"exit code {code}\n{out}"
        if code != 0:
            raise ToolFailure(text)
        return text

    # vision / memory / plan ----------------------------------------------
    async def analyze_image(self, args: dict[str, Any]) -> str:
        img = load_image(self._ws().resolve(args["path"]))
        question = args.get("question") or ""
        prompt = SCREENSHOT_PROMPT + (f"\nThe user asks: {question}" if question else "")
        return f"[{img.meta()}]\n" + await self.s.llm.describe_images(prompt, [img.data])

    async def remember(self, args: dict[str, Any]) -> str:
        mid = await self.s.memory.remember(args["text"], args.get("kind") or "note", args.get("scope") or "project", self.s.embedder)
        return f"saved memory #{mid}"

    async def recall(self, args: dict[str, Any]) -> str:
        hits = await self.s.memory.search(args["query"], self.s.embedder)
        return "\n".join(f"#{m.id} [{m.scope}/{m.kind}] {m.text}" for m in hits) or "nothing relevant in memory"

    async def update_plan(self, args: dict[str, Any]) -> str:
        i = int(args["step"]) - 1
        if not 0 <= i < len(self.s.plan):
            raise ToolFailure(f"plan has {len(self.s.plan)} steps; step numbers start at 1")
        self.s.plan[i].status, self.s.plan[i].note = args["status"], args.get("note") or ""
        self.s.emit("plan", self.s.plan)
        return f"step {i + 1} marked {args['status']}"

    async def mcp_read_resource(self, args: dict[str, Any]) -> str:
        return await self.s.mcp.get(args["server"]).read_resource(args["uri"])


class ToolFailure(RuntimeError):
    """A tool ran (or was refused) and failed in an expected way."""
