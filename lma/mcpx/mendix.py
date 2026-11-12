"""Mendix Studio Pro MCP server (Maia) integration.

Facts taken from the official documentation (docs.mendix.com/refguide/studio-pro-mcp-server,
last modified 2026-10-19):
  * available from Studio Pro 11.10
  * enabled in Preferences > AI > MCP Server, where the port is configured as well
  * endpoint  http://localhost:<port>/mcp  over the HTTP (Streamable HTTP) transport
  * runs locally, no Studio Pro sign-in needed, but Studio Pro itself needs internet access
  * acts on the app that is currently open in Studio Pro
  * 11.10 only: cannot generate pages; web-fetch tool calls wait for approval inside Studio Pro
The docs publish NO tool list and NO default port, so both are discovered at runtime.
"""
from __future__ import annotations

import re
import socket
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import psutil

from lma.mcpx.manager import CallResult, McpManager, MendixConfig, ServerConfig, ServerConnection, describe_error

SERVER_NAME = "mendix"
MIN_VERSION = (11, 10)
DOCS_URL = "https://docs.mendix.com/refguide/studio-pro-mcp-server/"


@dataclass
class Step:
    name: str
    ok: bool | None  # None = informational / skipped
    detail: str


@dataclass
class Diagnosis:
    steps: list[Step] = field(default_factory=list)
    url: str | None = None
    connected: bool = False
    hints: list[str] = field(default_factory=list)

    def add(self, name: str, ok: bool | None, detail: str) -> None:
        self.steps.append(Step(name, ok, detail))


@dataclass
class StudioProProcess:
    pid: int
    exe: str
    version: tuple[int, ...] | None
    ports: list[int]


def find_studio_pro() -> list[StudioProProcess]:
    """Locate running Studio Pro instances and the loopback ports they listen on."""
    found: list[StudioProProcess] = []
    for proc in psutil.process_iter(["pid", "name", "exe"]):
        name = (proc.info.get("name") or "").lower()
        if "studiopro" not in name.replace(" ", ""):
            continue
        exe = proc.info.get("exe") or ""
        m = re.search(r"(\d+)\.(\d+)\.(\d+)(?:\.(\d+))?", exe)
        version = tuple(int(x) for x in m.groups() if x) if m else None
        ports: list[int] = []
        try:
            for c in proc.net_connections(kind="tcp"):
                if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port not in ports:
                    ports.append(c.laddr.port)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
        found.append(StudioProProcess(proc.info["pid"], exe, version, sorted(ports)))
    return found


def port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def candidate_urls(cfg: MendixConfig, procs: list[StudioProProcess]) -> list[tuple[str, str]]:
    """Ordered (url, why) candidates. Explicit configuration always wins."""
    out: list[tuple[str, str]] = []

    def add(url: str, why: str) -> None:
        if url not in [u for u, _ in out]:
            out.append((url, why))

    if cfg.url:
        add(cfg.url, "configured url")
    if cfg.port:
        add(f"http://{cfg.host}:{cfg.port}{cfg.path}", "configured port")
    if cfg.auto_discover:
        for p in procs:
            for port in p.ports:
                add(f"http://{cfg.host}:{port}{cfg.path}", f"port {port} is listened on by Studio Pro (pid {p.pid})")
        for port in cfg.candidate_ports:
            add(f"http://{cfg.host}:{port}{cfg.path}", f"candidate port {port} (community-reported, not in official docs)")
    return out


class MendixClient:
    """Thin, Maia-specific layer over the generic manager."""

    def __init__(self, manager: McpManager) -> None:
        self.manager = manager
        self.cfg = manager.config.mendix
        self.last_diagnosis: Diagnosis | None = None

    @property
    def conn(self) -> ServerConnection | None:
        return self.manager.connections.get(SERVER_NAME)

    @property
    def connected(self) -> bool:
        return self.conn is not None and self.conn.connected

    async def connect(self) -> Diagnosis:
        """Find the endpoint, connect, discover tools. Never raises: read the Diagnosis."""
        d = Diagnosis()
        self.last_diagnosis = d
        if not self.cfg.enabled:
            d.add("configuration", False, "mendix.enabled is false in config/mcp_servers.yaml")
            return d

        procs = find_studio_pro()
        if procs:
            for p in procs:
                ver = ".".join(map(str, p.version)) if p.version else "unknown (not in exe path)"
                d.add("Studio Pro process", True, f"pid {p.pid}, version {ver}, listening ports {p.ports or 'none visible'}")
                if p.version and p.version[:2] < MIN_VERSION:
                    d.add("Studio Pro version", False, f"{ver} is older than 11.10, which is the first version with the MCP Server")
                    d.hints.append("Upgrade to Studio Pro 11.10 or newer; the MCP Server does not exist in older versions.")
        else:
            d.add("Studio Pro process", False, "no running studiopro.exe found")
            d.hints.append("Start Mendix Studio Pro (11.10+) and open an app; the MCP Server only runs while Studio Pro runs.")

        cands = candidate_urls(self.cfg, procs)
        if not cands:
            d.add("endpoint", False, "no url/port configured and auto_discover is off")
            d.hints.append("Set mendix.port in config/mcp_servers.yaml to the port shown in Preferences > AI > MCP Server.")
            return d

        for url, why in cands:
            parsed = urlparse(url)
            host, port = parsed.hostname or "localhost", parsed.port or 80
            if host not in ("localhost", "127.0.0.1", "::1"):
                d.add(f"probe {url}", False, "refused: the Studio Pro MCP server is local-only, non-loopback hosts are not allowed")
                continue
            if not port_open(host, port):
                d.add(f"probe {url}", False, f"nothing is listening ({why})")
                continue
            conn = self.manager.add(SERVER_NAME, ServerConfig(
                transport="http", url=url, timeout_s=self.cfg.timeout_s,
                connect_timeout_s=self.cfg.connect_timeout_s, tool_policy=dict(self.cfg.tool_policy),
            ))
            try:
                await conn.connect()
            except Exception as exc:  # noqa: BLE001
                d.add(f"probe {url}", False, f"port is open but MCP initialize failed: {conn.last_error or describe_error(exc)} ({why})")
                self.manager.connections.pop(SERVER_NAME, None)
                continue
            info = conn.init_result.server_info if conn.init_result else None
            d.add(f"probe {url}", True, f"MCP initialize OK ({why})")
            d.add("server", True, f"{getattr(info, 'name', '?')} {getattr(info, 'version', '')}, protocol {conn.init_result.protocol_version if conn.init_result else '?'}")
            d.add("discovery", True, f"{len(conn.tools)} tools, {len(conn.resources)} resources, {len(conn.prompts)} prompts")
            d.url, d.connected = url, True
            return d

        d.hints += [
            "In Studio Pro open Preferences > AI > MCP Server, tick 'Enable MCP Server' and note the port.",
            "Put that port in config/mcp_servers.yaml under mendix.port (or run: /mendix connect <port>).",
            "From Studio Pro 11.13 the active port is shown in the status bar and may differ from the configured one if it was taken.",
            "Studio Pro itself needs internet access for the MCP Server to work (official docs).",
            "If a corporate proxy is set, make sure NO_PROXY contains localhost,127.0.0.1 (this app sets it for its own process).",
        ]
        return d

    # ---- capability views ---------------------------------------------
    def tools(self) -> list[Any]:
        return list(self.conn.tools.values()) if self.connected and self.conn else []

    def capabilities(self) -> dict[str, list[str]]:
        """Group the *discovered* tools. Nothing is listed that the server did not report."""
        groups: dict[str, list[str]] = {"read": [], "write": [], "deny": []}
        for t in self.tools():
            groups[t.policy].append(t.name)
        return groups

    def safe_read_tools(self) -> list[Any]:
        """Read-only tools callable without arguments: usable for an automatic smoke test."""
        return [t for t in self.tools() if t.policy == "read" and not t.required]

    async def smoke_test(self) -> tuple[str | None, CallResult | None]:
        """Execute one real read-only operation. Returns (tool name, result)."""
        if not self.connected or self.conn is None:
            return None, None
        for t in self.safe_read_tools():
            res = await self.conn.call_tool(t.name, {})
            return t.name, res
        return None, None

    async def inspect(self, limit: int = 6) -> list[tuple[str, CallResult]]:
        """Call every argument-free read tool to build a project overview."""
        out: list[tuple[str, CallResult]] = []
        if not self.connected or self.conn is None:
            return out
        for t in self.safe_read_tools()[:limit]:
            out.append((t.name, await self.conn.call_tool(t.name, {})))
        return out

    def guidance(self) -> str:
        """System-prompt fragment for /mendix mode, generated from live discovery only."""
        if not self.connected:
            return ("The Mendix Studio Pro MCP server is NOT connected. You cannot inspect or change the Mendix "
                    "model. Say so plainly and suggest `/mendix connect`; do not describe changes as done.")
        caps = self.capabilities()
        lines = [
            "You are connected to the official Mendix Studio Pro MCP server (Maia capabilities) for the app open in Studio Pro.",
            f"Read-only tools: {', '.join('mcp__mendix__' + n for n in caps['read']) or 'none discovered'}.",
            f"Modifying tools (each call needs user approval): {', '.join('mcp__mendix__' + n for n in caps['write']) or 'none discovered'}.",
            "Workflow: inspect the existing model first, propose the design, apply changes with the modifying tools, "
            "then re-read the changed documents and run any error-checking tool to validate. Fix reported errors.",
            "Only use tools from this list with arguments matching their schema. If no tool can do something, say it is "
            "not supported by this Studio Pro version instead of pretending.",
        ]
        return "\n".join(lines)
