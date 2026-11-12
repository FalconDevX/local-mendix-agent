"""Incremental project index: chunks + FTS5 keyword search + local embeddings + symbols + dependencies.

One SQLite file per project (data/index/<project_id>.sqlite), so projects never mix.
"""
from __future__ import annotations

import ast
import re
import sqlite3
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lma.config import Settings
from lma.project.scanner import CODE_KINDS, FileInfo, project_id, read_text, scan

SCHEMA = """
CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, kind TEXT, size INTEGER, mtime REAL, indexed_at REAL);
CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, path TEXT, start INTEGER, end INTEGER, text TEXT, emb BLOB);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, path UNINDEXED, content='chunks', content_rowid='id', tokenize="unicode61");
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN INSERT INTO chunks_fts(rowid, text, path) VALUES (new.id, new.text, new.path); END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN INSERT INTO chunks_fts(chunks_fts, rowid, text, path) VALUES('delete', old.id, old.text, old.path); END;
CREATE TABLE IF NOT EXISTS symbols(name TEXT, kind TEXT, path TEXT, line INTEGER);
CREATE INDEX IF NOT EXISTS symbols_name ON symbols(name);
CREATE TABLE IF NOT EXISTS deps(src TEXT, target TEXT, resolved TEXT);
CREATE INDEX IF NOT EXISTS deps_src ON deps(src);
CREATE INDEX IF NOT EXISTS deps_resolved ON deps(resolved);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""

Embedder = Callable[[list[str], bool], Awaitable[list[list[float]]]]

_SYMBOL_RES: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    "typescript": [("class", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)")),
                   ("function", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(\w+)")),
                   ("const", re.compile(r"^\s*export\s+(?:const|let|var)\s+(\w+)")),
                   ("type", re.compile(r"^\s*(?:export\s+)?(?:interface|type|enum)\s+(\w+)"))],
    "java": [("class", re.compile(r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(?:abstract\s+)?(?:class|interface|enum|record)\s+(\w+)")),
             ("method", re.compile(r"^\s*(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?[\w<>\[\], ?.]+\s+(\w+)\s*\("))],
    "csharp": [("class", re.compile(r"^\s*(?:public|internal|private|protected)?\s*(?:static\s+|sealed\s+|abstract\s+|partial\s+)*(?:class|interface|enum|struct|record)\s+(\w+)"))],
    "go": [("function", re.compile(r"^func\s+(?:\([^)]*\)\s*)?(\w+)")), ("type", re.compile(r"^type\s+(\w+)"))],
}
_SYMBOL_RES["javascript"] = _SYMBOL_RES["typescript"]
_IMPORT_RES: dict[str, re.Pattern[str]] = {
    "typescript": re.compile(r"""(?:import\s[^'"]*?from\s*|import\s*\(\s*|require\s*\(\s*|^\s*import\s*)['"]([^'"]+)['"]""", re.M),
    "java": re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)\s*;", re.M),
    "csharp": re.compile(r"^\s*using\s+([\w.]+)\s*;", re.M),
}
_IMPORT_RES["javascript"] = _IMPORT_RES["typescript"]


@dataclass
class Hit:
    path: str
    start: int
    end: int
    text: str
    score: float
    source: str  # keyword | semantic | hybrid


def chunk_lines(text: str, size: int, overlap: int) -> list[tuple[int, int, str]]:
    lines = text.splitlines()
    if not lines:
        return []
    out: list[tuple[int, int, str]] = []
    step = max(1, size - overlap)
    for start in range(0, len(lines), step):
        part = lines[start:start + size]
        body = "\n".join(part).strip()
        if body:
            out.append((start + 1, start + len(part), "\n".join(part)[:6000]))
        if start + size >= len(lines):
            break
    return out


def extract_python(text: str) -> tuple[list[tuple[str, str, int]], list[str]]:
    symbols: list[tuple[str, str, int]] = []
    imports: list[str] = []
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return symbols, imports
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            symbols.append((node.name, "class", node.lineno))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append((node.name, "function", node.lineno))
        elif isinstance(node, ast.Import):
            imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            imports += [f"{base}.{a.name}" if base and not base.endswith(".") else base + a.name for a in node.names]
            if node.module:
                imports.append(base)
    return symbols, sorted(set(imports))


def extract_generic(kind: str, text: str) -> tuple[list[tuple[str, str, int]], list[str]]:
    symbols: list[tuple[str, str, int]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for skind, rx in _SYMBOL_RES.get(kind, []):
            m = rx.match(line)
            if m and m.group(1) not in ("if", "for", "while", "switch", "return", "new"):
                symbols.append((m.group(1), skind, lineno))
                break
    rx = _IMPORT_RES.get(kind)
    return symbols, sorted(set(rx.findall(text))) if rx else []


class ProjectIndex:
    def __init__(self, root: Path, settings: Settings) -> None:
        self.root = root.resolve()
        self.settings = settings
        self.id = project_id(self.root)
        d = settings.data_path / "index"
        d.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(d / f"{self.id}.sqlite")
        self.db.executescript(SCHEMA)
        self._matrix: tuple[np.ndarray, list[int]] | None = None

    def close(self) -> None:
        self.db.close()

    # ---- meta ----------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, value))
        self.db.commit()

    def counts(self) -> dict[str, int]:
        q = lambda sql: int(self.db.execute(sql).fetchone()[0])  # noqa: E731
        return {"files": q("SELECT count(*) FROM files"), "chunks": q("SELECT count(*) FROM chunks"),
                "embedded": q("SELECT count(*) FROM chunks WHERE emb IS NOT NULL"),
                "symbols": q("SELECT count(*) FROM symbols"), "deps": q("SELECT count(*) FROM deps")}

    # ---- indexing ------------------------------------------------------
    async def update(self, embedder: Embedder | None = None, full: bool = False,
                     progress: Callable[[str, int, int], None] | None = None) -> dict[str, int]:
        """Incremental: only new/changed files are re-read; removed files are dropped."""
        cfg = self.settings.index
        if full:
            for t in ("files", "chunks", "symbols", "deps"):
                self.db.execute(f"DELETE FROM {t}")
        known = {r[0]: (r[1], r[2]) for r in self.db.execute("SELECT path, size, mtime FROM files")}
        files = list(scan(self.root, cfg.ignore_dirs))
        seen = {f.rel for f in files}
        removed = [p for p in known if p not in seen]
        for p in removed:
            self._drop(p)
        changed = [f for f in files if known.get(f.rel) != (f.size, f.mtime)]
        for i, f in enumerate(changed):
            if progress:
                progress("reading", i + 1, len(changed))
            self._drop(f.rel)
            self._add(f)
        self.db.commit()
        self._resolve_deps([f.rel for f in files])
        embedded = 0
        if embedder is not None:
            embedded = await self.embed_missing(embedder, progress)
        self._matrix = None
        self.set_meta("indexed_at", str(time.time()))
        return {"scanned": len(files), "changed": len(changed), "removed": len(removed), "embedded": embedded}

    def _drop(self, rel: str) -> None:
        for t, col in (("files", "path"), ("chunks", "path"), ("symbols", "path"), ("deps", "src")):
            self.db.execute(f"DELETE FROM {t} WHERE {col}=?", (rel,))

    def _add(self, f: FileInfo) -> None:
        self.db.execute("INSERT OR REPLACE INTO files VALUES(?,?,?,?,?)", (f.rel, f.kind, f.size, f.mtime, time.time()))
        if not f.is_text or f.size > self.settings.index.max_file_bytes:
            return
        try:
            text = read_text(f.path)
        except OSError:
            return
        cfg = self.settings.index
        self.db.executemany("INSERT INTO chunks(path,start,end,text) VALUES(?,?,?,?)",
                            [(f.rel, s, e, t) for s, e, t in chunk_lines(text, cfg.chunk_lines, cfg.chunk_overlap)])
        if f.kind in CODE_KINDS:
            symbols, imports = extract_python(text) if f.kind == "python" else extract_generic(f.kind, text)
            self.db.executemany("INSERT INTO symbols VALUES(?,?,?,?)", [(n, k, f.rel, ln) for n, k, ln in symbols])
            self.db.executemany("INSERT INTO deps VALUES(?,?,NULL)", [(f.rel, imp) for imp in imports])

    def _resolve_deps(self, rels: list[str]) -> None:
        """Map import strings to project files where possible (python modules, relative JS/TS, Java classes)."""
        by_module: dict[str, str] = {}
        for rel in rels:
            p = Path(rel)
            if p.suffix == ".py":
                parts = list(p.with_suffix("").parts)
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                for i in range(len(parts)):
                    by_module.setdefault(".".join(parts[i:]), rel)
            elif p.suffix == ".java":
                parts = list(p.with_suffix("").parts)
                for i in range(len(parts)):
                    by_module.setdefault(".".join(parts[i:]), rel)
        relset = set(rels)
        rows = self.db.execute("SELECT rowid, src, target FROM deps WHERE resolved IS NULL").fetchall()
        for rowid, src, target in rows:
            resolved = None
            if target.startswith("./") or target.startswith("../"):
                base = (Path(src).parent / target).as_posix()
                norm = Path(*[x for x in Path(base).parts]).as_posix()
                import posixpath

                norm = posixpath.normpath(base)
                for cand in (norm, norm + ".ts", norm + ".tsx", norm + ".js", norm + ".jsx", norm + "/index.ts", norm + "/index.js"):
                    if cand in relset:
                        resolved = cand
                        break
            else:
                name = target.lstrip(".")
                if target.startswith("."):  # relative python import
                    pkg = Path(src).parent.parts
                    level = len(target) - len(name)
                    name = ".".join([*pkg[:len(pkg) - (level - 1)], name]) if name else ".".join(pkg)
                while name and resolved is None:
                    resolved = by_module.get(name)
                    name = name.rpartition(".")[0]
            if resolved and resolved != src:
                self.db.execute("UPDATE deps SET resolved=? WHERE rowid=?", (resolved, rowid))
        self.db.commit()

    async def embed_missing(self, embedder: Embedder, progress: Callable[[str, int, int], None] | None = None) -> int:
        rows = self.db.execute("SELECT id, path, text FROM chunks WHERE emb IS NULL").fetchall()
        done = 0
        for i in range(0, len(rows), 32):
            batch = rows[i:i + 32]
            vecs = await embedder([f"{path}\n{text}" for _, path, text in batch], False)
            for (cid, _, _), vec in zip(batch, vecs):
                arr = np.asarray(vec, dtype=np.float32)
                arr /= (np.linalg.norm(arr) or 1.0)
                self.db.execute("UPDATE chunks SET emb=? WHERE id=?", (arr.tobytes(), cid))
            self.db.commit()
            done += len(batch)
            if progress:
                progress("embedding", done, len(rows))
        self._matrix = None
        return done

    # ---- search --------------------------------------------------------
    def keyword(self, query: str, limit: int = 10) -> list[Hit]:
        terms = re.findall(r"[\w]+", query)
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        try:
            rows = self.db.execute(
                "SELECT c.path, c.start, c.end, c.text, bm25(chunks_fts) FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
                "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?", (match, limit)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [Hit(p, s, e, t, -score, "keyword") for p, s, e, t, score in rows]

    def _load_matrix(self) -> tuple[np.ndarray, list[int]]:
        if self._matrix is None:
            rows = self.db.execute("SELECT id, emb FROM chunks WHERE emb IS NOT NULL").fetchall()
            if rows:
                mat = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
            else:
                mat = np.zeros((0, 1), dtype=np.float32)
            self._matrix = (mat, [r[0] for r in rows])
        return self._matrix

    def semantic(self, query_vec: list[float], limit: int = 10) -> list[Hit]:
        mat, ids = self._load_matrix()
        if not ids:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        q /= (np.linalg.norm(q) or 1.0)
        if mat.shape[1] != q.shape[0]:
            return []  # embedding model changed since indexing: run /index --full
        scores = mat @ q
        top = np.argsort(-scores)[:limit]
        hits: list[Hit] = []
        for i in top:
            row = self.db.execute("SELECT path, start, end, text FROM chunks WHERE id=?", (ids[int(i)],)).fetchone()
            hits.append(Hit(row[0], row[1], row[2], row[3], float(scores[int(i)]), "semantic"))
        return hits

    async def search(self, query: str, embedder: Embedder | None = None, limit: int = 8) -> list[Hit]:
        """Hybrid retrieval: reciprocal-rank fusion of BM25 and cosine similarity."""
        kw = self.keyword(query, limit * 3)
        sem: list[Hit] = []
        if embedder is not None and self.counts()["embedded"]:
            try:
                sem = self.semantic((await embedder([query], True))[0], limit * 3)
                if sem:  # keep only chunks close to the best match; the tail is noise
                    sem = [h for h in sem if h.score >= sem[0].score - 0.12]
            except Exception:  # noqa: BLE001 - keyword results are still valid
                sem = []
        fused: dict[tuple[str, int], tuple[float, Hit, set[str]]] = {}
        for hits in (kw, sem):
            for rank, h in enumerate(hits):
                key = (h.path, h.start)
                prev = fused.get(key, (0.0, h, set()))
                fused[key] = (prev[0] + 1.0 / (60 + rank), h, prev[2] | {h.source})
        out = []
        for score, h, sources in sorted(fused.values(), key=lambda x: -x[0])[:limit]:
            out.append(Hit(h.path, h.start, h.end, h.text, score, "hybrid" if len(sources) > 1 else next(iter(sources))))
        return out

    def find_symbol(self, name: str, limit: int = 30) -> list[tuple[str, str, str, int]]:
        return self.db.execute(
            "SELECT name, kind, path, line FROM symbols WHERE name = ? COLLATE NOCASE "
            "UNION SELECT name, kind, path, line FROM symbols WHERE name LIKE ? LIMIT ?", (name, f"%{name}%", limit)).fetchall()

    def dependencies(self, rel: str) -> list[tuple[str, str | None]]:
        return self.db.execute("SELECT target, resolved FROM deps WHERE src=? ORDER BY target", (rel,)).fetchall()

    def dependents(self, rel: str) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT DISTINCT src FROM deps WHERE resolved=? ORDER BY src", (rel,))]

    def overview(self) -> str:
        """Deterministic facts for project summaries (no LLM involved)."""
        kinds = self.db.execute("SELECT kind, count(*), sum(size) FROM files GROUP BY kind ORDER BY 2 DESC").fetchall()
        hubs = self.db.execute("SELECT resolved, count(*) c FROM deps WHERE resolved IS NOT NULL GROUP BY resolved ORDER BY c DESC LIMIT 10").fetchall()
        big = self.db.execute("SELECT path, count(*) c FROM symbols GROUP BY path ORDER BY c DESC LIMIT 10").fetchall()
        lines = ["File types: " + ", ".join(f"{k} {n}" for k, n, _ in kinds)]
        if hubs:
            lines.append("Most imported files: " + ", ".join(f"{p} ({c})" for p, c in hubs))
        if big:
            lines.append("Files with most symbols: " + ", ".join(f"{p} ({c})" for p, c in big))
        return "\n".join(lines)
