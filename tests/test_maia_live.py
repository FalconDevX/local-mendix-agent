"""Live tests against the real Mendix Studio Pro MCP server.

They run only when Studio Pro (11.10+) is running with the MCP Server enabled:

    set LMA_MAIA_URL=http://localhost:<port>/mcp      (or leave unset to use auto-discovery)
    .venv\\Scripts\\python -m pytest tests/test_maia_live.py -m maia -v

Optional modification test (acceptance test 4). It needs an explicit, human-chosen call because the
tool names and schemas depend on the Studio Pro version:

    set LMA_MAIA_WRITE_TOOL=<tool name from /mendix tools>
    set LMA_MAIA_WRITE_ARGS={"...": "..."}             (JSON arguments)
    set LMA_MAIA_VERIFY_TOOL=<read tool>               (optional)
    set LMA_MAIA_VERIFY_ARGS={"...": "..."}
    set LMA_MAIA_VERIFY_CONTAINS=<text that must appear in the verify result>
"""
from __future__ import annotations

import json
import os

import pytest

from lma.mcpx.manager import McpConfig, McpManager, MendixConfig, load_mcp_config
from lma.mcpx.mendix import MendixClient

pytestmark = pytest.mark.maia


async def _client() -> tuple[McpManager, MendixClient]:
    url = os.environ.get("LMA_MAIA_URL")
    cfg = McpConfig(mendix=MendixConfig(url=url)) if url else load_mcp_config()
    m = McpManager(cfg)
    mx = MendixClient(m)
    d = await mx.connect()
    if not d.connected:
        await m.close()
        pytest.skip("Studio Pro MCP server not reachable: " + "; ".join(f"{s.name}: {s.detail}" for s in d.steps))
    return m, mx


async def test_connect_and_discover():
    m, mx = await _client()
    try:
        assert mx.conn is not None and mx.conn.init_result is not None
        tools = mx.tools()
        assert tools, "the server reported no tools"
        for t in tools:
            assert t.name and isinstance(t.input_schema, dict)
        print("\nserver:", mx.conn.init_result.server_info)
        for t in tools:
            print(f"  {t.policy:5} {t.name}  required={t.required}")
    finally:
        await m.close()


async def test_read_operation():
    m, mx = await _client()
    try:
        name, res = await mx.smoke_test()
        if name is None:
            pytest.skip("no argument-free read-only tool; run /mendix call <tool> <json> manually")
        assert res is not None and res.ok, f"{name} failed: {res.error if res else ''}"
        print(f"\n{name} ->\n{res.text[:1500]}")
    finally:
        await m.close()


async def test_modification_and_verification():
    tool = os.environ.get("LMA_MAIA_WRITE_TOOL")
    if not tool:
        pytest.skip("set LMA_MAIA_WRITE_TOOL / LMA_MAIA_WRITE_ARGS to run a real modification")
    m, mx = await _client()
    try:
        assert mx.conn is not None
        res = await mx.conn.call_tool(tool, json.loads(os.environ.get("LMA_MAIA_WRITE_ARGS", "{}")))
        assert res.ok, res.error
        verify = os.environ.get("LMA_MAIA_VERIFY_TOOL")
        if verify:
            check = await mx.conn.call_tool(verify, json.loads(os.environ.get("LMA_MAIA_VERIFY_ARGS", "{}")))
            assert check.ok, check.error
            needle = os.environ.get("LMA_MAIA_VERIFY_CONTAINS")
            if needle:
                assert needle in check.text
    finally:
        await m.close()
