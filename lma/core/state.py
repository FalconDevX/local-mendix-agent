"""Shared application state: one object wiring settings, models, MCP, project, edits and memory."""
from __future__ import annotations

import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lma.config import Settings, save_settings
from lma.core.permissions import Permissions
from lma.editing.editor import Workspace
from lma.llm.manager import LLMManager
from lma.mcpx.manager import McpManager
from lma.mcpx.mendix import MendixClient
from lma.memory.store import MemoryStore
from lma.project.indexer import Hit, ProjectIndex
from lma.project.scanner import MendixProject, detect_mendix, project_id
from lma.tools.image import LoadedImage

# Name of the sub-agent running in the current asyncio task (None = the main agent).
# Events and ledger entries are attributed with it, so concurrent agents never get mixed up.
current_agent: ContextVar[str | None] = ContextVar("lma_current_agent", default=None)


@dataclass
class PlanStep:
    text: str
    status: str = "pending"  # pending | in_progress | done | failed | skipped
    note: str = ""


@dataclass
class LedgerEntry:
    """A tool execution that really happened (or really failed). The final report is built from these."""

    tool: str
    summary: str
    ok: bool
    seconds: float
    detail: str = ""
    risk: str = "read"
    ts: float = field(default_factory=time.time)
    agent: str | None = None
    lines: int = 1  # size of the full tool output (detail keeps only its beginning)


# Event sink: (kind, payload). The UI renders these; tests collect them.
Emit = Callable[[str, Any], None]


class AppState:
    def __init__(self, settings: Settings, emit: Emit | None = None) -> None:
        self.settings = settings
        self.emit: Emit = emit or (lambda kind, payload: None)
        self.llm = LLMManager(settings)
        self.mcp = McpManager(op_log=self._log_dir() / "mcp-operations.jsonl")
        self.mendix = MendixClient(self.mcp)
        self.memory = MemoryStore(settings.db_path)
        self.permissions = Permissions(mode=settings.mode)
        self.workspace: Workspace | None = None
        self.index: ProjectIndex | None = None
        self.extra_indexes: dict[Path, ProjectIndex] = {}  # one index per additional context folder
        self.mendix_project: MendixProject | None = None
        self.plan: list[PlanStep] = []
        self.ledger: list[LedgerEntry] = []
        self.history: list[dict[str, Any]] = []  # chat messages of this session (model-facing)
        self.attachments: list[LoadedImage] = []
        self.mode = settings.mode
        self.context_tokens = 0
        self.memory.start_session(None, None)
        if settings.project and Path(settings.project).is_dir():
            self.open_project(Path(settings.project), persist=False)

    def _log_dir(self) -> Path:
        d = self.settings.data_path / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- project -------------------------------------------------------
    @property
    def root(self) -> Path | None:
        return self.workspace.root if self.workspace else None

    def open_project(self, path: Path, persist: bool = True) -> None:
        path = path.expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"{path} is not a directory")
        self._close_indexes()
        pid = project_id(path)
        self.workspace = Workspace(path, self.settings.data_path / "snapshots" / pid)
        self.index = ProjectIndex(path, self.settings)
        self.mendix_project = detect_mendix(path)
        self.memory.set_project(pid, str(path))
        for extra in self.settings.extra_dirs.get(str(path), []):
            if Path(extra).is_dir():  # a folder that vanished is skipped, not an error
                self._attach(Path(extra).resolve())
        if persist:
            self.settings.project = str(path)
            save_settings(self.settings)

    # ---- additional context folders --------------------------------------
    @property
    def roots(self) -> list[Path]:
        return self.workspace.roots if self.workspace else []

    def _attach(self, path: Path) -> None:
        assert self.workspace is not None
        self.workspace.add_root(path)
        self.extra_indexes[path] = ProjectIndex(path, self.settings)

    def add_dir(self, path: Path) -> Path:
        """Authorise one more folder: its files become readable, searchable and editable (staged) too."""
        path = path.expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"{path} is not a directory")
        if self.workspace is None:
            self.open_project(path)
            return path
        for r in self.roots:
            if path == r or r in path.parents:
                raise ValueError(f"{path} is already covered by {r}")
        self._attach(path)
        self._save_dirs()
        return path

    def remove_dir(self, path: str) -> Path:
        assert self.workspace is not None
        want = Path(path).expanduser()
        match = next((r for r in self.workspace.extra if r == want.resolve() or r.name.lower() == path.strip().lower()), None)
        if match is None:
            raise KeyError(f"{path} is not an additional context folder (see /dirs)")
        self.workspace.remove_root(match)
        if idx := self.extra_indexes.pop(match, None):
            idx.close()
        self._save_dirs()
        return match

    def _save_dirs(self) -> None:
        assert self.workspace is not None
        key = str(self.workspace.root)
        if self.workspace.extra:
            self.settings.extra_dirs[key] = [str(p) for p in self.workspace.extra]
        else:
            self.settings.extra_dirs.pop(key, None)
        save_settings(self.settings)

    def indexes(self) -> list[tuple[Path, ProjectIndex, str]]:
        """(root, index, path prefix shown to the model). Main project paths stay relative."""
        out: list[tuple[Path, ProjectIndex, str]] = []
        if self.index is not None and self.root is not None:
            out.append((self.root, self.index, ""))
        out += [(root, idx, root.as_posix() + "/") for root, idx in self.extra_indexes.items()]
        return out

    def indexed(self) -> bool:
        return any(idx.counts()["files"] for _, idx, _ in self.indexes())

    async def search(self, query: str, limit: int = 8) -> list[Hit]:
        """Hybrid search over the project and every additional context folder."""
        hits: list[Hit] = []
        for _, idx, prefix in self.indexes():
            if not idx.counts()["files"]:
                continue
            for h in await idx.search(query, self.embedder, limit):
                hits.append(Hit(prefix + h.path, h.start, h.end, h.text, h.score, h.source))
        hits.sort(key=lambda h: -h.score)
        return hits[:limit]

    def find_symbol(self, name: str, limit: int = 30) -> list[tuple[str, str, str, int]]:
        rows: list[tuple[str, str, str, int]] = []
        for _, idx, prefix in self.indexes():
            rows += [(n, kind, prefix + path, line) for n, kind, path, line in idx.find_symbol(name, limit)]
        return rows[:limit]

    def _close_indexes(self) -> None:
        if self.index:
            self.index.close()
        for idx in self.extra_indexes.values():
            idx.close()
        self.extra_indexes = {}

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.permissions.mode = mode

    # ---- embedding helper shared by index and memory --------------------
    @property
    def embedder(self) -> Any:
        return self.llm.embed if self.llm.has("embed") else None

    async def close(self) -> None:
        await self.mcp.close()
        self.llm.shutdown()
        self._close_indexes()
        self.memory.close()
