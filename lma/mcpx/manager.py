"""Generic multi-server MCP client built on the official MCP Python SDK.

Each server connection lives in its own asyncio task so the SDK's context managers are
entered and exited in the same task. Everything reported here comes from the server:
tools, schemas, resources and prompts are discovered, never assumed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
from pydantic import BaseModel, Field

from lma import __version__
from lma.config import CONFIG_DIR, resolve_secret

log = logging.getLogger("lma.mcp")

Policy = Literal["read", "write", "deny"]
READ_WORDS = ("read", "list", "get", "find", "search", "check", "glob", "inspect", "describe", "status", "show", "fetch_schema")


class ServerConfig(BaseModel):
    transport: Literal["http", "stdio"] = "http"
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)  # values may be env:/keyring: refs
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    enabled: bool = True
    timeout_s: float = 60
    connect_timeout_s: float = 15
    tool_policy: dict[str, Policy] = Field(default_factory=dict)


class MendixConfig(BaseModel):
    """Dedicated section for the Studio Pro MCP server (Maia)."""

    enabled: bool = True
    url: str | None = None  # full URL wins, e.g. http://localhost:7782/mcp
    port: int | None = None  # the port set in Studio Pro > Preferences > AI > MCP Server
    host: str = "localhost"
    path: str = "/mcp"
    auto_discover: bool = True
    # Not from official docs (they state no default). 7782 is what the mendixlabs Kiro power
    # reports for Studio Pro 11.13/11.14. Only used as a probe candidate.
    candidate_ports: list[int] = Field(default_factory=lambda: [7782])
    timeout_s: float = 120
    connect_timeout_s: float = 10
    tool_policy: dict[str, Policy] = Field(default_factory=dict)


class McpConfig(BaseModel):
    mendix: MendixConfig = Field(default_factory=MendixConfig)
    servers: dict[str, ServerConfig] = Field(default_factory=dict)


MCP_CONFIG_FILE = CONFIG_DIR / "mcp_servers.yaml"


def load_mcp_config(path: Path | None = None) -> McpConfig:
    path = path or MCP_CONFIG_FILE
    if not path.exists():
        return McpConfig()
    with path.open("r", encoding="utf-8") as fh:
        return McpConfig.model_validate(yaml.safe_load(fh) or {})


CONFIG_HEADER = """# MCP configuration (rewritten by /mcp add and /mendix connect; comments here are regenerated).
#
# mendix: the official Mendix Studio Pro MCP server (Maia capabilities), Studio Pro 11.10+.
#   Enable it in Studio Pro: Preferences > AI > MCP Server > Enable MCP Server, and note the port.
#   Official docs: https://docs.mendix.com/refguide/studio-pro-mcp-server/
#   Keys: enabled, port (as shown in Preferences / status bar), url (full URL, wins over port),
#         auto_discover (probe the loopback ports of the running studiopro.exe, then candidate_ports),
#         candidate_ports (default [7782]: community-reported, not stated in the official docs),
#         timeout_s, tool_policy {tool: read|write|deny}.
#   Only loopback hosts are ever accepted for Maia.
#
# servers: any other MCP servers.
#   name:
#     transport: http | stdio
#     url: http://localhost:9000/mcp            (http)
#     headers: {Authorization: "keyring:NAME"}  (secrets only as env:NAME or keyring:NAME references)
#     command: C:/tools/server.exe, args: [...], env: {...}, cwd: ...   (stdio)
#     enabled: true, timeout_s: 60, tool_policy: {tool: read|write|deny}
"""


def save_mcp_config(cfg: McpConfig, path: Path | None = None) -> None:
    path = path or MCP_CONFIG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.model_dump(exclude_defaults=True)
    data.setdefault("mendix", {}).setdefault("enabled", cfg.mendix.enabled)
    data.setdefault("servers", {})
    with path.open("w", encoding="utf-8") as fh:
        fh.write(CONFIG_HEADER)
        yaml.safe_dump(data, fh, sort_keys=False)


@dataclass
class ToolInfo:
    server: str
    name: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any]
    policy: Policy
    policy_source: str  # config | annotation | name-heuristic | default

    @property
    def required(self) -> list[str]:
        return list(self.input_schema.get("required") or [])


@dataclass
class CallResult:
    ok: bool
    text: str
    images: list[tuple[str, str]] = field(default_factory=list)  # (mime, base64)
    structured: Any = None
    duration_s: float = 0.0
    error: str | None = None


def classify(tool: types.Tool, overrides: dict[str, Policy]) -> tuple[Policy, str]:
    if tool.name in overrides:
        return overrides[tool.name], "config"
    ann = tool.annotations
    if ann is not None and ann.read_only_hint is not None:
        return ("read" if ann.read_only_hint else "write"), "annotation"
    low = tool.name.lower()
    tokens = low.replace("-", "_").split("_")
    if any(w in tokens for w in READ_WORDS) and not any(w in tokens for w in ("write", "create", "update", "delete", "patch", "install", "set")):
        return "read", "name-heuristic"
    return "write", "default"


def describe_error(exc: BaseException) -> str:
    """Flatten ExceptionGroups coming out of anyio task groups into one readable line."""
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(describe_error(e) for e in exc.exceptions)
    name = type(exc).__name__
    msg = str(exc)
    return f"{name}: {msg}" if msg else name


class ServerConnection:
    def __init__(self, name: str, cfg: ServerConfig, op_log: Path | None = None) -> None:
        self.name = name
        self.cfg = cfg
        self.op_log = op_log
        self.session: ClientSession | None = None
        self.init_result: types.InitializeResult | None = None
        self.tools: dict[str, ToolInfo] = {}
        self.resources: list[types.Resource] = []
        self.prompts: list[types.Prompt] = []
        self.last_error: str | None = None
        self.connected_at: float | None = None
        self._task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()

    # ---- lifecycle -----------------------------------------------------
    @property
    def connected(self) -> bool:
        return self.session is not None and self._task is not None and not self._task.done()

    @property
    def target(self) -> str:
        if self.cfg.transport == "http":
            return self.cfg.url or "?"
        return " ".join([self.cfg.command or "?", *self.cfg.args])

    async def connect(self) -> None:
        await self.disconnect()
        self._ready, self._stop = asyncio.Event(), asyncio.Event()
        self.last_error = None
        self._task = asyncio.create_task(self._run(), name=f"mcp-{self.name}")
        try:
            await asyncio.wait_for(self._ready.wait(), self.cfg.connect_timeout_s)
        except asyncio.TimeoutError:
            self.last_error = f"timed out after {self.cfg.connect_timeout_s:.0f}s connecting to {self.target}"
            await self.disconnect()
        if not self.connected:
            self._log("connect", ok=False, error=self.last_error)
            raise ConnectionError(self.last_error or "connection failed")
        self._log("connect", ok=True)
        await self.discover()

    async def _run(self) -> None:
        try:
            if self.cfg.transport == "http":
                if not self.cfg.url:
                    raise ValueError("no url configured")
                headers = {k: resolve_secret(v) for k, v in self.cfg.headers.items()}
                async with create_mcp_http_client(headers=headers or None) as http:
                    async with streamable_http_client(self.cfg.url, http_client=http) as streams:
                        await self._serve(streams[0], streams[1])
            else:
                if not self.cfg.command:
                    raise ValueError("no command configured")
                params = StdioServerParameters(
                    command=self.cfg.command, args=self.cfg.args,
                    env={k: resolve_secret(v) for k, v in self.cfg.env.items()} or None, cwd=self.cfg.cwd,
                )
                errlog = open(self.op_log.with_name(f"mcp-{self.name}.stderr.log"), "a", encoding="utf-8") if self.op_log else open(os.devnull, "w")  # noqa: SIM115
                self._errlog = errlog
                async with stdio_client(params, errlog=errlog) as streams:
                    await self._serve(streams[0], streams[1])
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - surfaced through last_error
            self.last_error = describe_error(exc)
        finally:
            self.session = None
            self._ready.set()
            errlog = getattr(self, "_errlog", None)
            if errlog is not None:
                errlog.close()
                self._errlog = None

    async def _serve(self, read: Any, write: Any) -> None:
        info = types.Implementation(name="local-mendix-agent", version=__version__)
        async with ClientSession(read, write, client_info=info) as session:
            self.init_result = await session.initialize()
            self.session = session
            self.connected_at = time.time()
            self._ready.set()
            await self._stop.wait()

    async def disconnect(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        self._stop.set()
        try:
            await asyncio.wait_for(asyncio.shield(task), 5)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            task.cancel()
            try:
                await task
            except BaseException:  # noqa: BLE001
                pass
        self.session = None

    # ---- discovery -----------------------------------------------------
    async def discover(self) -> None:
        assert self.session is not None
        caps = self.init_result.capabilities if self.init_result else None
        self.tools, self.resources, self.prompts = {}, [], []
        if caps is None or caps.tools is not None:
            cursor = None
            while True:
                res = await self._timed(self.session.list_tools(cursor=cursor) if cursor else self.session.list_tools())
                for t in res.tools:
                    policy, source = classify(t, self.cfg.tool_policy)
                    self.tools[t.name] = ToolInfo(
                        server=self.name, name=t.name, description=t.description or "",
                        input_schema=dict(t.input_schema or {"type": "object", "properties": {}}),
                        annotations=t.annotations.model_dump(exclude_none=True) if t.annotations else {},
                        policy=policy, policy_source=source,
                    )
                cursor = getattr(res, "next_cursor", None)
                if not cursor:
                    break
        if caps is not None and caps.resources is not None:
            try:
                self.resources = list((await self._timed(self.session.list_resources())).resources)
            except Exception as exc:  # noqa: BLE001
                log.debug("list_resources failed: %s", exc)
        if caps is not None and caps.prompts is not None:
            try:
                self.prompts = list((await self._timed(self.session.list_prompts())).prompts)
            except Exception as exc:  # noqa: BLE001
                log.debug("list_prompts failed: %s", exc)
        self._log("discover", ok=True, tools=len(self.tools), resources=len(self.resources), prompts=len(self.prompts))

    async def _timed(self, coro: Any, timeout: float | None = None) -> Any:
        return await asyncio.wait_for(coro, timeout or self.cfg.timeout_s)

    # ---- operations ----------------------------------------------------
    async def ping(self) -> float:
        if not self.connected:
            raise ConnectionError("not connected")
        t0 = time.perf_counter()
        await self._timed(self.session.send_ping(), 10)  # type: ignore[union-attr]
        return time.perf_counter() - t0

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> CallResult:
        info = self.tools.get(name)
        if info is None:
            return CallResult(False, "", error=f"server {self.name!r} has no tool named {name!r}")
        if info.policy == "deny":
            return CallResult(False, "", error=f"tool {name!r} is denied by configuration")
        t0 = time.perf_counter()
        attempts = 2 if info.policy == "read" else 1  # never blindly re-send a modification
        last = ""
        for attempt in range(attempts):
            try:
                if not self.connected:
                    await self.connect()
                res = await self._timed(self.session.call_tool(name, arguments or {}))  # type: ignore[union-attr]
                out = _parse_result(res)
                out.duration_s = time.perf_counter() - t0
                self._log("call_tool", ok=out.ok, tool=name, args=arguments, seconds=round(out.duration_s, 3), error=out.error)
                return out
            except asyncio.TimeoutError:
                last = f"timed out after {self.cfg.timeout_s:.0f}s"
                break
            except Exception as exc:  # noqa: BLE001
                last = describe_error(exc)
                # The transport may be dead or the server restarted (stale session id).
                # Drop the session so the next attempt/call performs a fresh initialize.
                await self.disconnect()
                if attempt + 1 < attempts:
                    log.info("call to %s failed (%s), reconnecting", self.name, last)
                    continue
                if info.policy != "read":
                    last += " (connection reset; the modification was NOT retried automatically - verify state before repeating)"
        self._log("call_tool", ok=False, tool=name, args=arguments, error=last)
        return CallResult(False, "", error=last, duration_s=time.perf_counter() - t0)

    async def read_resource(self, uri: str) -> str:
        if not self.connected:
            await self.connect()
        res = await self._timed(self.session.read_resource(uri))  # type: ignore[union-attr]
        parts = [getattr(c, "text", None) or f"<binary {getattr(c, 'mime_type', '?')}>" for c in res.contents]
        return "\n".join(parts)

    async def get_prompt(self, name: str, arguments: dict[str, str] | None = None) -> str:
        if not self.connected:
            await self.connect()
        res = await self._timed(self.session.get_prompt(name, arguments or {}))  # type: ignore[union-attr]
        return "\n\n".join(f"[{m.role}] {getattr(m.content, 'text', m.content)}" for m in res.messages)

    def _log(self, op: str, **fields: Any) -> None:
        if self.op_log is None:
            return
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "server": self.name, "op": op, **fields}
        try:
            with self.op_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str, ensure_ascii=False)[:4000] + "\n")
        except OSError:
            pass


def _parse_result(res: Any) -> CallResult:
    texts: list[str] = []
    images: list[tuple[str, str]] = []
    for block in getattr(res, "content", None) or []:
        kind = getattr(block, "type", "")
        if kind == "text":
            texts.append(block.text)
        elif kind == "image":
            images.append((block.mime_type, block.data))
            texts.append(f"[image {block.mime_type}]")
        elif kind == "resource":
            inner = block.resource
            texts.append(getattr(inner, "text", None) or f"[embedded resource {inner.uri}]")
        else:
            texts.append(f"[{kind or type(block).__name__} content]")
    structured = getattr(res, "structured_content", None)
    if not texts and structured is not None:
        texts.append(json.dumps(structured, indent=2, ensure_ascii=False))
    text = "\n".join(texts)
    is_error = bool(getattr(res, "is_error", False))
    return CallResult(ok=not is_error, text=text, images=images, structured=structured, error=text if is_error else None)


class McpManager:
    """Holds every configured server (Maia is registered under the name `mendix`)."""

    def __init__(self, config: McpConfig | None = None, op_log: Path | None = None) -> None:
        self.config = config or load_mcp_config()
        self.op_log = op_log
        self.connections: dict[str, ServerConnection] = {}
        for name, cfg in self.config.servers.items():
            if cfg.enabled:
                self.connections[name] = ServerConnection(name, cfg, op_log)

    def add(self, name: str, cfg: ServerConfig) -> ServerConnection:
        conn = ServerConnection(name, cfg, self.op_log)
        self.connections[name] = conn
        return conn

    def get(self, name: str) -> ServerConnection:
        if name not in self.connections:
            raise KeyError(f"unknown MCP server {name!r}; configured: {', '.join(self.connections) or 'none'}")
        return self.connections[name]

    async def connect_all(self) -> dict[str, str | None]:
        """Connect every server concurrently. Returns name -> error (None on success)."""
        async def one(conn: ServerConnection) -> str | None:
            try:
                await conn.connect()
                return None
            except Exception as exc:  # noqa: BLE001
                return conn.last_error or describe_error(exc)

        names = list(self.connections)
        results = await asyncio.gather(*(one(self.connections[n]) for n in names))
        return dict(zip(names, results))

    async def close(self) -> None:
        await asyncio.gather(*(c.disconnect() for c in self.connections.values()), return_exceptions=True)

    def all_tools(self) -> list[ToolInfo]:
        return [t for c in self.connections.values() if c.connected for t in c.tools.values()]

    async def health(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name, conn in self.connections.items():
            entry: dict[str, Any] = {"connected": conn.connected, "target": conn.target, "error": conn.last_error}
            if conn.connected:
                try:
                    entry["ping_ms"] = round((await conn.ping()) * 1000, 1)
                except Exception as exc:  # noqa: BLE001
                    entry["ping_error"] = describe_error(exc)
                entry["tools"] = len(conn.tools)
            out[name] = entry
        return out
