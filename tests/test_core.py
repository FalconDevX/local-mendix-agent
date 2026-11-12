"""Editing engine, project index, memory, permissions and tool-call parsing (no model needed)."""
from __future__ import annotations

from pathlib import Path

import pytest

from lma.core.permissions import Permissions
from lma.editing.editor import EditError, Workspace, validate
from lma.llm.manager import extract_text_tool_calls
from lma.memory.store import MemoryStore
from lma.project.indexer import ProjectIndex
from lma.project.scanner import classify, detect_mendix, project_id, scan


# ---------------------------------------------------------------- editing
def test_multi_file_edit_diff_apply_undo(sample_repo: Path, tmp_path: Path):
    ws = Workspace(sample_repo, tmp_path / "snap")
    pricing = sample_repo / "shop/orders/pricing.py"
    original = pricing.read_text(encoding="utf-8")

    ws.stage_edit("shop/orders/pricing.py", "return total - total * percent", "return total - total * percent / 100")
    ws.stage_write("shop/orders/NOTES.md", "discount is 0-100\n")
    ws.stage_delete("web/util.ts")
    assert pricing.read_text(encoding="utf-8") == original, "nothing may touch the disk before apply"
    assert "percent / 100" in ws.read("shop/orders/pricing.py"), "the agent must see its own staged edit"

    diff = ws.diff()
    assert "MODIFY: shop/orders/pricing.py" in diff and "CREATE: shop/orders/NOTES.md" in diff and "DELETE: web/util.ts" in diff
    assert "-    return total - total * percent\n+    return total - total * percent / 100" in diff

    manifest = ws.apply("discount fix")
    assert len(manifest["files"]) == 3 and not ws.pending
    assert "percent / 100" in pricing.read_text(encoding="utf-8")
    assert (sample_repo / "shop/orders/NOTES.md").exists() and not (sample_repo / "web/util.ts").exists()

    undone = ws.undo()
    assert undone["id"] == manifest["id"]
    assert pricing.read_text(encoding="utf-8") == original
    assert not (sample_repo / "shop/orders/NOTES.md").exists() and (sample_repo / "web/util.ts").exists()
    assert [h["undone"] for h in ws.history()] == [True]
    with pytest.raises(EditError):
        ws.undo()


def test_edit_safety(sample_repo: Path, tmp_path: Path):
    ws = Workspace(sample_repo, tmp_path / "snap")
    for bad in ("../outside.txt", str(tmp_path / "elsewhere.txt"), "shop/../../x"):
        with pytest.raises(EditError):
            ws.stage_write(bad, "x")
    with pytest.raises(EditError, match="not found"):
        ws.stage_edit("shop/orders/api.py", "this text is not there", "x")
    with pytest.raises(EditError, match="occurs"):
        ws.stage_edit("shop/auth/tokens.py", "str", "bytes")
    with pytest.raises(EditError, match="binary"):
        ws.stage_write("logo.png", "text")
    # conflict: file changes on disk after staging
    ws.stage_edit("shop/orders/api.py", "discount_percent=0", "discount_percent=5")
    (sample_repo / "shop/orders/api.py").write_text("# changed by someone else\n", encoding="utf-8")
    with pytest.raises(EditError, match="changed on disk"):
        ws.apply()
    assert ws.reject() == 1


def test_crlf_files_are_preserved(sample_repo: Path, tmp_path: Path):
    f = sample_repo / "win.txt"
    f.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    ws = Workspace(sample_repo, tmp_path / "snap")
    ws.stage_edit("win.txt", "one\ntwo", "one\nTWO")  # model sends LF
    ws.apply()
    assert f.read_bytes() == b"one\r\nTWO\r\nthree\r\n"


def test_validate():
    assert validate("a.py", "def f(:\n  pass") is not None
    assert validate("a.py", "x = 1\n") is None
    assert validate("a.json", "{bad}") is not None and validate("a.yaml", "a: [1, 2") is not None
    assert validate("a.unknown", "anything") is None


# ---------------------------------------------------------------- index
async def test_index_search_symbols_deps_incremental(sample_repo: Path, settings):
    idx = ProjectIndex(sample_repo, settings)
    res = await idx.update()
    assert res["changed"] == res["scanned"] and res["scanned"] >= 9
    paths = {r[0] for r in idx.db.execute("SELECT path FROM files")}
    assert not any(p.startswith("node_modules") for p in paths), "ignored directories must be skipped"
    assert idx.db.execute("SELECT kind FROM files WHERE path='logo.png'").fetchone()[0] == "image"

    hits = idx.keyword("hash_password")
    assert hits and hits[0].path == "shop/auth/tokens.py"
    assert ("order_total", "function", "shop/orders/pricing.py", 7) in idx.find_symbol("order_total")
    assert any(name == "Cart" for name, *_ in idx.find_symbol("Cart"))

    assert ("shop.auth.tokens", "shop/auth/tokens.py") in idx.dependencies("shop/orders/pricing.py")
    assert idx.dependents("shop/auth/tokens.py") == ["shop/orders/pricing.py"]
    assert idx.dependents("shop/orders/pricing.py") == ["shop/orders/api.py"]
    assert idx.dependents("web/util.ts") == ["web/app.ts"]

    assert (await idx.update())["changed"] == 0, "second run must be incremental"
    (sample_repo / "shop/orders/api.py").write_text("def brand_new_function():\n    return 1\n", encoding="utf-8")
    (sample_repo / "web/util.ts").unlink()
    res = await idx.update()
    assert res["changed"] == 1 and res["removed"] == 1
    assert idx.find_symbol("brand_new_function") and not idx.find_symbol("checkout")
    idx.close()


async def test_semantic_search_with_embedder(sample_repo: Path, settings):
    """Hybrid ranking with a deterministic bag-of-words embedder (the real model is covered by test_local_model)."""
    vocab = ["password", "token", "discount", "price", "login", "total"]

    async def embedder(texts: list[str], is_query: bool) -> list[list[float]]:
        return [[float(t.lower().count(w)) + 0.01 for w in vocab] for t in texts]

    idx = ProjectIndex(sample_repo, settings)
    res = await idx.update(embedder)
    assert res["embedded"] == idx.counts()["chunks"] > 0
    hits = await idx.search("where is the password login checked", embedder, limit=3)
    assert hits[0].path == "shop/auth/tokens.py"
    idx.close()


def test_scanner_and_mendix_detection(tmp_path: Path):
    root = tmp_path / "mx"
    (root / "javasource/sales/actions").mkdir(parents=True)
    (root / "javasource/sales/actions/CalcTotal.java").write_text("public class CalcTotal {}", encoding="utf-8")
    (root / "themesource/sales").mkdir(parents=True)
    (root / "widgets").mkdir()
    (root / "widgets/DataGrid.mpk").write_bytes(b"PK\0\0")
    (root / "mprcontents").mkdir()
    (root / "App.mpr").write_bytes(b"SQLite format 3\0")
    (root / "deployment/x").mkdir(parents=True)
    (root / "deployment/x/big.js").write_text("ignored", encoding="utf-8")
    assert classify(root / "App.mpr") == "mendix-project" and classify(root / "widgets/DataGrid.mpk") == "mendix-package"
    mx = detect_mendix(root)
    assert mx is not None and mx.java_modules == ["sales"] and mx.java_actions == 1 and mx.widgets == ["DataGrid"]
    assert "mpr v2" in mx.storage and "MCP" in mx.summary()
    rels = {f.rel for f in scan(root, ["deployment"])}
    assert "App.mpr" in rels and not any(r.startswith("deployment") for r in rels)
    assert detect_mendix(tmp_path) is None


# ---------------------------------------------------------------- memory
async def test_memory_persists_and_isolates_projects(tmp_path: Path):
    db = tmp_path / "agent.sqlite"
    a, b = project_id(tmp_path / "a"), project_id(tmp_path / "b")

    mem = MemoryStore(db)
    mem.start_session(a, "a")
    mem.add_message("user", "add a discount feature", "code")
    await mem.remember("Discounts are stored as percent 0-100", "decision", "project")
    await mem.remember("Always run pytest before committing", "workflow", "global")
    mem.upsert_kind("project_summary", "Project A is an order service")
    mem.set_summary("Implemented the discount feature in pricing.py")
    mem.close()

    mem = MemoryStore(db)  # "restart the application"
    mem.start_session(a, "a")
    assert mem.get_kind("project_summary") == "Project A is an order service"
    assert [m.text for m in await mem.search("how are discounts stored")][0].startswith("Discounts are stored")
    prev = mem.previous_context(a)
    assert prev and prev[0][1] == "add a discount feature" and "discount feature" in prev[0][2]

    mem.start_session(b, "b")  # another project must not see A's memory
    texts = [m.text for m in mem.list()]
    assert "Always run pytest before committing" in texts
    assert not any("Discounts" in t or "Project A" in t for t in texts)
    assert mem.get_kind("project_summary") is None and mem.previous_context(b) == []
    assert not [m for m in await mem.search("discounts percent") if m.scope == "project"]
    mem.close()


# ---------------------------------------------------------------- permissions / parsing
async def test_permissions():
    asked: list[str] = []

    async def approver(tool: str, risk: str, summary: str) -> str:
        asked.append(tool)
        return {"mcp__mendix__update": "y", "git_commit": "n", "run_command": "a"}[tool]

    p = Permissions(mode="agent", approver=approver)
    assert (await p.check("read_file", "read", "")).allowed and (await p.check("edit_file", "stage", "")).allowed
    assert (await p.check("mcp__mendix__update", "write", "")).allowed
    assert not (await p.check("git_commit", "exec", "")).allowed
    assert (await p.check("run_command", "exec", "")).allowed and (await p.check("run_command", "exec", "")).allowed
    assert asked == ["mcp__mendix__update", "git_commit", "run_command"], "'always' must not ask twice"

    for mode in ("ask", "plan", "review"):
        ro = Permissions(mode=mode, approver=approver)
        assert not (await ro.check("edit_file", "stage", "")).allowed
        assert not (await ro.check("mcp__mendix__update", "write", "")).allowed
    assert not (await Permissions(mode="agent").check("create_dir", "write", "")).allowed, "no approver => deny"
    assert (await Permissions(mode="agent", auto_approve=True).check("create_dir", "write", "")).allowed


def test_text_tool_call_extraction():
    known = {"read_file", "grep"}
    for text in (
        '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>',
        '<tools>{"type": "function", "function": {"name": "read_file", "arguments": {"path": "a.py"}}}</tools>',
        'Sure:\n```json\n{"name": "read_file", "arguments": "{\\"path\\": \\"a.py\\"}"}\n```',
    ):
        rest, calls = extract_text_tool_calls(text, known)
        assert [(c.name, c.arguments) for c in calls] == [("read_file", {"path": "a.py"})], text
        assert "{" not in rest
    # not calls: unknown tool, echoed schema, ordinary JSON
    for text in ('{"name": "delete_everything", "arguments": {}}',
                 '{"name": "grep", "description": "Regex search", "parameters": {"type": "object", "properties": {}}}',
                 'The config is {"name": "x", "port": 1}.'):
        assert extract_text_tool_calls(text, known) == (text, [])
