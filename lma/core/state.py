"""Shared application state: one object wiring settings, models, MCP, project, edits and memory."""
from __future__ import annotations

import time
from collections.abc import Callable
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
from lma.project.indexer import ProjectIndex
from lma.project.scanner import MendixProject, detect_mendix, project_id
from lma.tools.image import LoadedImage


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
        if self.index:
            self.index.close()
        pid = project_id(path)
        self.workspace = Workspace(path, self.settings.data_path / "snapshots" / pid)
        self.index = ProjectIndex(path, self.settings)
        self.mendix_project = detect_mendix(path)
        self.memory.set_project(pid, str(path))
        if persist:
            self.settings.project = str(path)
            save_settings(self.settings)

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
        if self.index:
            self.index.close()
        self.memory.close()
