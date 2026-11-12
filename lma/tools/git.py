"""Small git wrapper (read operations + commit). Uses the user's installed git, if any."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def run_git(root: Path, *args: str, timeout: int = 30) -> tuple[bool, str]:
    exe = shutil.which("git")
    if exe is None:
        return False, "git is not installed or not on PATH"
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        p = subprocess.run([exe, "-C", str(root), "-c", "core.quotepath=off", *args], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout, creationflags=flags)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"git failed: {exc}"
    out = (p.stdout + (("\n" + p.stderr) if p.returncode else "")).strip()
    return p.returncode == 0, out


def is_repo(root: Path) -> bool:
    ok, out = run_git(root, "rev-parse", "--is-inside-work-tree")
    return ok and out.strip() == "true"


def summary(root: Path) -> str | None:
    """One-line state for the status bar / project header, or None outside a repository."""
    if not is_repo(root):
        return None
    _, branch = run_git(root, "rev-parse", "--abbrev-ref", "HEAD")
    _, status = run_git(root, "status", "--porcelain")
    changed = len([ln for ln in status.splitlines() if ln.strip()])
    return f"{branch.strip() or 'detached'}, {changed} changed file(s)"
