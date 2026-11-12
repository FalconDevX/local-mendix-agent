"""Staged edits, unified diffs, transactional apply with snapshots, undo and history.

Nothing touches the disk until `apply()`. Every path is confined to the project root.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from lma.project.scanner import looks_binary, read_text


class EditError(RuntimeError):
    pass


def digest(text: str | None) -> str | None:
    return None if text is None else hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


@dataclass
class Change:
    rel: str
    old: str | None  # None = file did not exist
    new: str | None  # None = delete

    @property
    def kind(self) -> str:
        return "CREATE" if self.old is None else "DELETE" if self.new is None else "MODIFY"


def validate(rel: str, content: str) -> str | None:
    """Cheap syntax validation. Returns an error message or None."""
    suffix = Path(rel).suffix.lower()
    try:
        if suffix == ".py":
            ast.parse(content, filename=rel)
        elif suffix == ".json":
            json.loads(content)
        elif suffix in (".yaml", ".yml"):
            import yaml

            list(yaml.safe_load_all(content))
        elif suffix == ".xml":
            import xml.etree.ElementTree as ET

            ET.fromstring(content)
        elif suffix == ".toml":
            import tomllib

            tomllib.loads(content)
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


class Workspace:
    """The authorized project directory plus the set of pending changes."""

    def __init__(self, root: Path, snapshot_dir: Path) -> None:
        self.root = root.resolve()
        self.snapshot_dir = snapshot_dir
        self.pending: dict[str, Change] = {}

    # ---- path safety ---------------------------------------------------
    def resolve(self, path: str | Path) -> Path:
        p = Path(path)
        full = (p if p.is_absolute() else self.root / p).resolve()
        if full != self.root and self.root not in full.parents:
            raise EditError(f"path {path!s} is outside the project directory {self.root}")
        return full

    def rel(self, path: str | Path) -> str:
        return self.resolve(path).relative_to(self.root).as_posix()

    # ---- overlay reads -------------------------------------------------
    def exists(self, path: str | Path) -> bool:
        rel = self.rel(path)
        if rel in self.pending:
            return self.pending[rel].new is not None
        return (self.root / rel).is_file()

    def read(self, path: str | Path) -> str:
        """Read a file as the agent should see it: staged content wins over disk."""
        rel = self.rel(path)
        if rel in self.pending:
            ch = self.pending[rel]
            if ch.new is None:
                raise EditError(f"{rel} is staged for deletion")
            return ch.new
        full = self.root / rel
        if not full.is_file():
            raise EditError(f"{rel} does not exist")
        if looks_binary(full):
            raise EditError(f"{rel} is a binary file and cannot be read as text")
        return read_text(full)

    def _disk(self, rel: str) -> str | None:
        full = self.root / rel
        if not full.exists():
            return None
        if full.is_dir():
            raise EditError(f"{rel} is a directory")
        if looks_binary(full):
            raise EditError(f"{rel} is binary; refusing to edit it as text")
        return read_text(full)

    # ---- staging -------------------------------------------------------
    def _stage(self, rel: str, new: str | None) -> Change:
        old = self.pending[rel].old if rel in self.pending else self._disk(rel)
        if old == new:
            self.pending.pop(rel, None)
            return Change(rel, old, new)
        if old is None and new is None:
            raise EditError(f"{rel} does not exist")
        ch = Change(rel, old, new)
        self.pending[rel] = ch
        return ch

    def stage_write(self, path: str | Path, content: str) -> Change:
        return self._stage(self.rel(path), content)

    def stage_edit(self, path: str | Path, old_text: str, new_text: str, replace_all: bool = False) -> Change:
        rel = self.rel(path)
        current = self.read(rel)
        if old_text == "":
            raise EditError("old_text must not be empty (use write_file to create files)")
        # tolerate LF/CRLF differences between the model's text and the file
        if old_text not in current and "\r\n" in current:
            old_text, new_text = old_text.replace("\r\n", "\n").replace("\n", "\r\n"), new_text.replace("\r\n", "\n").replace("\n", "\r\n")
        count = current.count(old_text)
        if count == 0:
            raise EditError(f"old_text not found in {rel}; read the file again and copy the exact text")
        if count > 1 and not replace_all:
            raise EditError(f"old_text occurs {count} times in {rel}; add more surrounding lines or set replace_all")
        return self._stage(rel, current.replace(old_text, new_text) if replace_all else current.replace(old_text, new_text, 1))

    def stage_delete(self, path: str | Path) -> Change:
        return self._stage(self.rel(path), None)

    def stage_rename(self, src: str | Path, dst: str | Path) -> tuple[Change, Change]:
        content = self.read(src)
        if self.exists(dst):
            raise EditError(f"{self.rel(dst)} already exists")
        return self._stage(self.rel(dst), content), self._stage(self.rel(src), None)

    def reject(self, path: str | None = None) -> int:
        if path is None:
            n = len(self.pending)
            self.pending.clear()
            return n
        return 1 if self.pending.pop(self.rel(path), None) else 0

    # ---- diff ----------------------------------------------------------
    def diff(self, rel: str | None = None) -> str:
        parts: list[str] = []
        for ch in ([self.pending[rel]] if rel else self.pending.values()):
            parts.append(change_diff(ch))
        return "\n".join(parts)

    # ---- transactional apply ------------------------------------------
    def apply(self, label: str = "") -> dict:
        """Write all pending changes atomically-ish: snapshot first, roll back on any failure."""
        if not self.pending:
            raise EditError("nothing to apply")
        changes = list(self.pending.values())
        for ch in changes:  # conflict detection: disk must still match what the diff was based on
            if digest(self._disk(ch.rel)) != digest(ch.old):
                raise EditError(f"{ch.rel} changed on disk after it was staged; /reject it and redo the edit")
        txn = time.strftime("%Y%m%d-%H%M%S") + f"-{len(list(self.snapshot_dir.glob('*'))) if self.snapshot_dir.exists() else 0:04d}"
        snap = self.snapshot_dir / txn
        (snap / "files").mkdir(parents=True, exist_ok=True)
        manifest = {"id": txn, "label": label, "time": time.time(), "root": str(self.root), "undone": False,
                    "files": [{"rel": c.rel, "kind": c.kind, "old_sha": digest(c.old), "new_sha": digest(c.new)} for c in changes]}
        for ch in changes:
            if ch.old is not None:
                dst = snap / "files" / ch.rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.root / ch.rel, dst)
        (snap / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        written: list[Change] = []
        try:
            for ch in changes:
                full = self.root / ch.rel
                if ch.new is None:
                    full.unlink()
                else:
                    full.parent.mkdir(parents=True, exist_ok=True)
                    with full.open("w", encoding="utf-8", newline="") as fh:
                        fh.write(ch.new)
                written.append(ch)
        except OSError as exc:
            for ch in written:
                self._restore(snap, ch.rel, ch.old is not None)
            shutil.rmtree(snap, ignore_errors=True)
            raise EditError(f"apply failed on {ch.rel} ({exc}); all files were rolled back") from exc
        self.pending.clear()
        return manifest

    def _restore(self, snap: Path, rel: str, existed: bool) -> None:
        full = self.root / rel
        if existed:
            full.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snap / "files" / rel, full)
        elif full.exists():
            full.unlink()

    def history(self) -> list[dict]:
        if not self.snapshot_dir.exists():
            return []
        out = []
        for m in sorted(self.snapshot_dir.glob("*/manifest.json")):
            try:
                out.append(json.loads(m.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return out

    def undo(self, txn_id: str | None = None) -> dict:
        """Roll back the most recent (or a given) applied transaction."""
        candidates = [h for h in self.history() if not h["undone"]]
        if txn_id:
            candidates = [h for h in candidates if h["id"] == txn_id]
        if not candidates:
            raise EditError("nothing to undo")
        manifest = candidates[-1]
        snap = self.snapshot_dir / manifest["id"]
        warnings = []
        for f in manifest["files"]:
            try:
                current = digest(self._disk(f["rel"]))
            except EditError:
                current = "?"
            if current != f["new_sha"]:
                warnings.append(f"{f['rel']} was modified after the change was applied; restored to the snapshot anyway")
            self._restore(snap, f["rel"], f["kind"] != "CREATE")
        manifest["undone"] = True
        (snap / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["warnings"] = warnings
        return manifest


def change_diff(ch: Change) -> str:
    # line endings are normalised for display only; files keep their own endings on disk
    old = (ch.old or "").replace("\r\n", "\n").splitlines(keepends=True)
    new = (ch.new or "").replace("\r\n", "\n").splitlines(keepends=True)
    body = "".join(
        line if line.endswith("\n") else line + "\n"
        for line in difflib.unified_diff(old, new, f"a/{ch.rel}" if ch.old is not None else "/dev/null",
                                         f"b/{ch.rel}" if ch.new is not None else "/dev/null", n=3))
    return f"{ch.kind}: {ch.rel}\n{body}"
