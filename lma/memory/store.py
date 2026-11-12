"""Persistent memory in SQLite: sessions, messages, and retrievable memory items.

Scopes:
  conversation - messages and summaries of one session
  project      - facts tied to one project id (architecture, conventions, decisions, task log)
  global       - reusable knowledge valid across projects
Project memories are only ever read with their own project id, so projects cannot mix.
"""
from __future__ import annotations

import re
import sqlite3
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, project_id TEXT, project_path TEXT, started REAL, updated REAL, title TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, session_id TEXT, ts REAL, role TEXT, content TEXT, mode TEXT);
CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id);
CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY, scope TEXT, project_id TEXT, kind TEXT, text TEXT, ts REAL, emb BLOB, session_id TEXT);
CREATE INDEX IF NOT EXISTS memories_scope ON memories(scope, project_id);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(text, content='memories', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text); END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN INSERT INTO memories_fts(memories_fts, rowid, text) VALUES('delete', old.id, old.text); END;
"""

Embedder = Callable[[list[str], bool], Awaitable[list[list[float]]]]


@dataclass
class Memory:
    id: int
    scope: str
    kind: str
    text: str
    ts: float
    score: float = 0.0


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.executescript(SCHEMA)
        self.session_id: str | None = None
        self.project_id: str | None = None

    def close(self) -> None:
        self.db.close()

    # ---- sessions ------------------------------------------------------
    def start_session(self, project_id: str | None, project_path: str | None) -> str:
        self.session_id = uuid.uuid4().hex[:12]
        self.project_id = project_id
        now = time.time()
        self.db.execute("INSERT INTO sessions VALUES(?,?,?,?,?,?,?)", (self.session_id, project_id, project_path, now, now, None, None))
        self.db.commit()
        return self.session_id

    def set_project(self, project_id: str | None, project_path: str | None) -> None:
        self.project_id = project_id
        self.db.execute("UPDATE sessions SET project_id=?, project_path=? WHERE id=?", (project_id, project_path, self.session_id))
        self.db.commit()

    def add_message(self, role: str, content: str, mode: str = "") -> None:
        now = time.time()
        self.db.execute("INSERT INTO messages(session_id, ts, role, content, mode) VALUES(?,?,?,?,?)", (self.session_id, now, role, content, mode))
        if role == "user":
            self.db.execute("UPDATE sessions SET title=COALESCE(title, ?), updated=? WHERE id=?", (content[:80], now, self.session_id))
        self.db.commit()

    def set_summary(self, summary: str) -> None:
        self.db.execute("UPDATE sessions SET summary=?, updated=? WHERE id=?", (summary, time.time(), self.session_id))
        self.db.commit()

    def clear_session(self) -> int:
        n = self.db.execute("DELETE FROM messages WHERE session_id=?", (self.session_id,)).rowcount
        self.db.execute("UPDATE sessions SET summary=NULL WHERE id=?", (self.session_id,))
        self.db.commit()
        return n

    def sessions(self, project_id: str | None = None, limit: int = 15) -> list[tuple]:
        sql = ("SELECT s.id, s.started, s.title, s.summary, s.project_path, (SELECT count(*) FROM messages m WHERE m.session_id=s.id) "
               "FROM sessions s WHERE (SELECT count(*) FROM messages m WHERE m.session_id=s.id) > 0")
        args: list = []
        if project_id:
            sql += " AND s.project_id=?"
            args.append(project_id)
        return self.db.execute(sql + " ORDER BY s.updated DESC LIMIT ?", (*args, limit)).fetchall()

    def session_messages(self, session_id: str, limit: int = 200) -> list[tuple[str, str]]:
        return self.db.execute("SELECT role, content FROM messages WHERE session_id=? ORDER BY id LIMIT ?", (session_id, limit)).fetchall()

    def previous_context(self, project_id: str | None, limit: int = 3) -> list[tuple[float, str, str | None]]:
        """Recent earlier sessions of the same project: (started, title, summary)."""
        rows = self.db.execute(
            "SELECT started, title, summary FROM sessions WHERE project_id IS ? AND id != ? AND title IS NOT NULL ORDER BY updated DESC LIMIT ?",
            (project_id, self.session_id or "", limit)).fetchall()
        return rows

    # ---- memories ------------------------------------------------------
    async def remember(self, text: str, kind: str = "note", scope: str = "project", embedder: Embedder | None = None) -> int:
        text = text.strip()
        if scope == "project" and self.project_id is None:
            scope = "global"
        pid = self.project_id if scope == "project" else None
        dup = self.db.execute("SELECT id FROM memories WHERE scope=? AND project_id IS ? AND text=?", (scope, pid, text)).fetchone()
        if dup:
            return int(dup[0])
        emb = None
        if embedder is not None:
            try:
                arr = np.asarray((await embedder([text], False))[0], dtype=np.float32)
                emb = (arr / (np.linalg.norm(arr) or 1.0)).tobytes()
            except Exception:  # noqa: BLE001 - memory still works by keyword
                emb = None
        cur = self.db.execute("INSERT INTO memories(scope, project_id, kind, text, ts, emb, session_id) VALUES(?,?,?,?,?,?,?)",
                              (scope, pid, kind, text, time.time(), emb, self.session_id))
        self.db.commit()
        return int(cur.lastrowid or 0)

    def upsert_kind(self, kind: str, text: str, scope: str = "project") -> None:
        """Keep exactly one memory of this kind per project (e.g. the project summary)."""
        pid = self.project_id if scope == "project" else None
        self.db.execute("DELETE FROM memories WHERE scope=? AND project_id IS ? AND kind=?", (scope, pid, kind))
        self.db.execute("INSERT INTO memories(scope, project_id, kind, text, ts, session_id) VALUES(?,?,?,?,?,?)",
                        (scope, pid, kind, text, time.time(), self.session_id))
        self.db.commit()

    def get_kind(self, kind: str, scope: str = "project") -> str | None:
        pid = self.project_id if scope == "project" else None
        row = self.db.execute("SELECT text FROM memories WHERE scope=? AND project_id IS ? AND kind=? ORDER BY ts DESC LIMIT 1", (scope, pid, kind)).fetchone()
        return row[0] if row else None

    def _visible(self) -> tuple[str, list]:
        return "(scope='global' OR (scope='project' AND project_id IS ?))", [self.project_id]

    def list(self, scope: str | None = None, limit: int = 50) -> list[Memory]:
        where, args = self._visible()
        if scope:
            where += " AND scope=?"
            args.append(scope)
        rows = self.db.execute(f"SELECT id, scope, kind, text, ts FROM memories WHERE {where} ORDER BY ts DESC LIMIT ?", (*args, limit)).fetchall()
        return [Memory(*r) for r in rows]

    async def search(self, query: str, embedder: Embedder | None = None, limit: int = 6) -> list[Memory]:
        """Hybrid keyword + semantic retrieval over memories visible to the current project."""
        where, args = self._visible()
        scores: dict[int, float] = {}
        terms = re.findall(r"\w+", query)
        if terms:
            match = " OR ".join(f'"{t}"' for t in terms)
            try:
                rows = self.db.execute(
                    f"SELECT m.id FROM memories_fts f JOIN memories m ON m.id=f.rowid WHERE memories_fts MATCH ? AND {where} ORDER BY bm25(memories_fts) LIMIT ?",
                    (match, *args, limit * 3)).fetchall()
                for rank, (mid,) in enumerate(rows):
                    scores[mid] = scores.get(mid, 0) + 1 / (60 + rank)
            except sqlite3.OperationalError:
                pass
        if embedder is not None:
            rows = self.db.execute(f"SELECT id, emb FROM memories WHERE emb IS NOT NULL AND {where}", args).fetchall()
            if rows:
                try:
                    q = np.asarray((await embedder([query], True))[0], dtype=np.float32)
                    q /= (np.linalg.norm(q) or 1.0)
                    vecs = [(mid, np.frombuffer(e, dtype=np.float32)) for mid, e in rows]
                    sims = sorted(((float(v @ q), mid) for mid, v in vecs if v.shape == q.shape), reverse=True)[:limit * 3]
                    for rank, (sim, mid) in enumerate(sims):
                        if sim > 0.3:
                            scores[mid] = scores.get(mid, 0) + 1 / (60 + rank)
                except Exception:  # noqa: BLE001
                    pass
        out: list[Memory] = []
        for mid, score in sorted(scores.items(), key=lambda x: -x[1])[:limit]:
            r = self.db.execute("SELECT id, scope, kind, text, ts FROM memories WHERE id=?", (mid,)).fetchone()
            out.append(Memory(*r, score=score))
        return out

    def forget(self, memory_id: int) -> bool:
        where, args = self._visible()
        n = self.db.execute(f"DELETE FROM memories WHERE id=? AND {where}", (memory_id, *args)).rowcount
        self.db.commit()
        return n > 0
