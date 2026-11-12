"""MCP client tests against a REAL MCP server process (tests/fixtures/demo_server.py)."""
from __future__ import annotations

import sys

import pytest

from lma.mcpx.manager import McpConfig, McpManager, MendixConfig, ServerConfig
from lma.mcpx.mendix import MendixClient
from tests.conftest import DEMO, free_port, start_demo_http


def stdio_cfg() -> ServerConfig:
    return ServerConfig(transport="stdio", command=sys.executable, args=[str(DEMO), "stdio"])


async def test_stdio_discovery_and_calls(tmp_path):
    m = McpManager(McpConfig(servers={"demo": stdio_cfg()}), op_log=tmp_path / "ops.jsonl")
    assert await m.connect_all() == {"demo": None}
    conn = m.get("demo")
    try:
        assert set(conn.tools) == {"list_notes", "read_note", "create_note", "add"}
        assert conn.tools["add"].policy == "read" and conn.tools["add"].policy_source == "annotation"
        assert conn.tools["create_note"].policy == "write"
        assert conn.tools["read_note"].required == ["name"]
        assert len(conn.resources) == 1 and len(conn.prompts) == 1

        res = await conn.call_tool("add", {"a": 2, "b": 40})
        assert res.ok and res.text == "42"

        # a real modification, verified by reading the state back
        assert (await conn.call_tool("create_note", {"name": "spec", "text": "hello"})).ok
        assert (await conn.call_tool("read_note", {"name": "spec"})).text == "hello"
        assert "spec" in (await conn.call_tool("list_notes")).text

        bad = await conn.call_tool("read_note", {"name": "missing"})
        assert not bad.ok and bad.error
        unknown = await conn.call_tool("does_not_exist", {})
        assert not unknown.ok and "no tool named" in (unknown.error or "")

        assert "Demo MCP server" in await conn.read_resource("demo://about")
        assert "mcp" in await conn.get_prompt("summarize", {"topic": "mcp"})
        assert (await m.health())["demo"]["connected"] is True
        assert (tmp_path / "ops.jsonl").read_text().count('"op": "call_tool"') >= 5
    finally:
        await m.close()
    assert not conn.connected


async def test_http_transport_and_reconnect():
    port = free_port()
    proc = start_demo_http(port)
    m = McpManager(McpConfig(servers={"demo": ServerConfig(transport="http", url=f"http://127.0.0.1:{port}/mcp")}))
    try:
        assert (await m.connect_all())["demo"] is None
        conn = m.get("demo")
        assert (await conn.call_tool("add", {"a": 1, "b": 1})).text == "2"
        proc.kill()
        proc.wait()
        down = await conn.call_tool("add", {"a": 1, "b": 1})
        assert not down.ok
        proc = start_demo_http(port)
        up = await conn.call_tool("add", {"a": 1, "b": 1})  # reconnects transparently
        assert up.ok and up.text == "2"
    finally:
        proc.kill()
        await m.close()


async def test_policy_overrides_and_deny():
    cfg = stdio_cfg()
    cfg.tool_policy = {"add": "deny", "list_notes": "write"}
    m = McpManager(McpConfig(servers={"demo": cfg}))
    await m.connect_all()
    try:
        conn = m.get("demo")
        assert conn.tools["list_notes"].policy == "write" and conn.tools["list_notes"].policy_source == "config"
        res = await conn.call_tool("add", {"a": 1, "b": 1})
        assert not res.ok and "denied" in (res.error or "")
    finally:
        await m.close()


async def test_connection_failure_is_reported():
    m = McpManager(McpConfig(servers={"dead": ServerConfig(transport="http", url=f"http://127.0.0.1:{free_port()}/mcp", connect_timeout_s=5)}))
    errors = await m.connect_all()
    assert errors["dead"]
    assert not m.get("dead").connected
    await m.close()


async def test_mendix_client_diagnosis_without_studio_pro():
    """No Studio Pro here: the diagnosis must say so precisely and never pretend to be connected."""
    m = McpManager(McpConfig(mendix=MendixConfig(candidate_ports=[free_port()])))
    mx = MendixClient(m)
    d = await mx.connect()
    assert not d.connected and not mx.connected
    assert any("nothing is listening" in s.detail for s in d.steps)
    assert any("Preferences > AI > MCP Server" in h for h in d.hints)
    assert mx.tools() == [] and (await mx.smoke_test()) == (None, None)
    assert "NOT connected" in mx.guidance()


async def test_mendix_client_rejects_non_loopback():
    m = McpManager(McpConfig(mendix=MendixConfig(url="http://example.com:7782/mcp", auto_discover=False)))
    d = await MendixClient(m).connect()
    assert not d.connected
    assert any("local-only" in s.detail for s in d.steps)


async def test_mendix_client_transport_path(demo_http):
    """Exercises the Maia code path (endpoint probing, initialize, discovery, smoke test) over
    Streamable HTTP. The peer is the demo server, NOT Studio Pro: this proves the client
    mechanics only. The real check is tests/test_maia_live.py."""
    url, port = demo_http
    m = McpManager(McpConfig(mendix=MendixConfig(port=port, host="127.0.0.1", auto_discover=False)))
    mx = MendixClient(m)
    d = await mx.connect()
    try:
        assert d.connected and d.url == url
        caps = mx.capabilities()
        assert "create_note" in caps["write"] and "list_notes" in caps["read"]
        name, res = await mx.smoke_test()
        assert name == "list_notes" and res is not None and res.ok
        assert "mcp__mendix__create_note" in mx.guidance()
    finally:
        await m.close()


@pytest.mark.parametrize("name,expected", [("ped_read_document", "read"), ("list_modules", "read"), ("ped_update_document", "write"),
                                           ("write_file", "write"), ("pg_patch_page", "write"), ("mystery", "write")])
def test_name_heuristic_is_conservative(name, expected):
    from mcp import types

    from lma.mcpx.manager import classify

    tool = types.Tool(name=name, input_schema={"type": "object"})
    assert classify(tool, {})[0] == expected
