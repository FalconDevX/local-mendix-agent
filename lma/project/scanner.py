"""Recursive project scanning, file classification and Mendix project detection."""
from __future__ import annotations

import fnmatch
import hashlib
import os
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

KINDS: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".ts": "typescript", ".tsx": "typescript", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".java": "java", ".cs": "csharp", ".go": "go", ".rs": "rust", ".kt": "kotlin", ".sql": "sql",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp", ".ps1": "powershell", ".sh": "shell", ".bat": "batch",
    ".json": "json", ".xml": "xml", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".ini": "config",
    ".properties": "config", ".gradle": "config", ".env": "config",
    ".html": "html", ".css": "css", ".scss": "css",
    ".md": "docs", ".rst": "docs", ".txt": "docs",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image", ".bmp": "image", ".webp": "image", ".svg": "image",
    # Mendix
    ".mpr": "mendix-project", ".mprcontents": "mendix-project", ".mxunit": "mendix-unit", ".mpk": "mendix-package",
    ".mxmodule": "mendix-package", ".mendixsnapshot": "mendix-package", ".mpr.lock": "mendix-project",
}
TEXT_KINDS = {
    "python", "typescript", "javascript", "java", "csharp", "go", "rust", "kotlin", "sql", "c", "cpp", "powershell",
    "shell", "batch", "json", "xml", "yaml", "toml", "config", "html", "css", "docs",
}
CODE_KINDS = TEXT_KINDS - {"docs", "json", "xml", "yaml", "toml", "config"}


@dataclass
class FileInfo:
    path: Path
    rel: str  # posix-style path relative to the project root
    kind: str
    size: int
    mtime: float

    @property
    def is_text(self) -> bool:
        return self.kind in TEXT_KINDS


def classify(path: Path) -> str:
    kind = KINDS.get(path.suffix.lower())
    if kind:
        return kind
    if path.name.lower() in ("dockerfile", "makefile", "license", "readme"):
        return "docs" if path.name.lower() in ("license", "readme") else "config"
    return "binary" if looks_binary(path) else "text"


def looks_binary(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            head = fh.read(2048)
    except OSError:
        return True
    return b"\0" in head


def read_text(path: Path, limit: int | None = None) -> str:
    data = path.read_bytes() if limit is None else path.open("rb").read(limit)
    for enc in ("utf-8-sig", "utf-16" if data[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8", "cp1250", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def project_id(root: Path) -> str:
    return hashlib.sha1(str(root.resolve()).lower().encode()).hexdigest()[:12]


def scan(root: Path, ignore_dirs: list[str], max_bytes: int | None = None) -> Iterator[FileInfo]:
    """Walk `root`, skipping ignored directories (names or glob patterns)."""
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not any(fnmatch.fnmatch(d, pat) for pat in ignore_dirs))
        for name in sorted(filenames):
            p = Path(dirpath) / name
            try:
                st = p.stat()
            except OSError:
                continue
            kind = classify(p)
            if kind == "text":
                kind = "docs"
            yield FileInfo(p, p.relative_to(root).as_posix(), kind, st.st_size, st.st_mtime)


def tree(root: Path, ignore_dirs: list[str], max_depth: int = 3, max_entries: int = 200) -> str:
    """Compact directory tree for prompts and /project."""
    lines: list[str] = [root.name + "/"]
    count = 0

    def walk(d: Path, depth: int, prefix: str) -> None:
        nonlocal count
        try:
            entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            return
        entries = [e for e in entries if not (e.is_dir() and any(fnmatch.fnmatch(e.name, pat) for pat in ignore_dirs))]
        for e in entries:
            if count >= max_entries:
                lines.append(prefix + "...")
                return
            count += 1
            lines.append(f"{prefix}{e.name}{'/' if e.is_dir() else ''}")
            if e.is_dir() and depth < max_depth:
                walk(e, depth + 1, prefix + "  ")

    walk(root, 1, "  ")
    return "\n".join(lines)


@dataclass
class MendixProject:
    mpr: Path
    storage: str  # "mpr v2 (mprcontents folder)" | "mpr v1 (single SQLite file)"
    java_modules: list[str]
    java_actions: int
    js_actions: int
    theme_modules: list[str]
    widgets: list[str]
    userlibs: int

    def summary(self) -> str:
        return "\n".join([
            f"Mendix project file: {self.mpr.name} [{self.storage}]",
            f"Modules with Java source ({len(self.java_modules)}): {', '.join(self.java_modules[:30]) or 'none'}",
            f"Java actions: {self.java_actions}, JavaScript actions: {self.js_actions}",
            f"Theme source modules: {', '.join(self.theme_modules[:30]) or 'none'}",
            f"Widgets ({len(self.widgets)}): {', '.join(self.widgets[:25]) or 'none'}",
            f"userlib JARs: {self.userlibs}",
            "Note: domain models, microflows and pages are stored in binary model units (.mpr/.mxunit). They are "
            "read and changed through the Studio Pro MCP server, never by editing these files.",
        ])


def detect_mendix(root: Path) -> MendixProject | None:
    mprs = sorted(root.glob("*.mpr"))
    if not mprs:
        return None

    def names(sub: str) -> list[str]:
        d = root / sub
        return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else []

    def count(sub: str, pattern: str) -> int:
        d = root / sub
        return sum(1 for _ in d.rglob(pattern)) if d.is_dir() else 0

    java_modules = [m for m in names("javasource") if m not in ("system",)]
    return MendixProject(
        mpr=mprs[0],
        storage="mpr v2 (mprcontents folder)" if (root / "mprcontents").is_dir() else "mpr v1 (single SQLite file)",
        java_modules=java_modules,
        java_actions=sum(1 for p in (root / "javasource").rglob("actions/*.java")) if (root / "javasource").is_dir() else 0,
        js_actions=count("javascriptsource", "*.js"),
        theme_modules=names("themesource"),
        widgets=sorted(p.stem for p in (root / "widgets").glob("*.mpk")) if (root / "widgets").is_dir() else [],
        userlibs=count("userlib", "*.jar"),
    )


def stats(files: list[FileInfo]) -> str:
    kinds = Counter(f.kind for f in files)
    total = sum(f.size for f in files)
    top = ", ".join(f"{k}: {n}" for k, n in kinds.most_common(12))
    return f"{len(files)} files, {total/1e6:.1f} MB. By type: {top}"
