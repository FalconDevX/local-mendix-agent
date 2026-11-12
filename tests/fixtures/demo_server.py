"""A real MCP server used ONLY by tests and the manual MCP check.

It keeps a tiny in-memory "model" so tests can verify that a write tool really changed state.
It is not a Mendix simulator and is never used by the application at runtime.

Run:  python tests/fixtures/demo_server.py stdio
      python tests/fixtures/demo_server.py http 8765
"""
from __future__ import annotations

import sys

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
server = MCPServer("lma-demo", log_level="ERROR")
NOTES: dict[str, str] = {"welcome": "hello from the demo server"}


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def list_notes() -> list[str]:
    """List the names of all stored notes."""
    return sorted(NOTES)


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def read_note(name: str) -> str:
    """Read one note by name."""
    if name not in NOTES:
        raise ValueError(f"no note named {name!r}")
    return NOTES[name]


@server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False))
def create_note(name: str, text: str) -> str:
    """Create or replace a note."""
    NOTES[name] = text
    return f"stored note {name!r} ({len(text)} chars)"


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@server.resource("demo://about")
def about() -> str:
    """About this server."""
    return "Demo MCP server for local-mendix-agent tests."


@server.prompt()
def summarize(topic: str) -> str:
    """Prompt asking for a summary."""
    return f"Summarize what you know about {topic}."


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if mode == "http":
        server.run(transport="streamable-http", host="127.0.0.1", port=PORT)
    else:
        server.run(transport="stdio")
