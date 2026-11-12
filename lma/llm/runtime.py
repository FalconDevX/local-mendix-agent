"""llama.cpp runtime management.

The official llama.cpp release ZIPs contain a self-contained `llama-server.exe`. Unzipping one
into the user data directory needs no administrator rights, no compiler and no PATH change.
(This replaces llama-cpp-python, whose Windows GPU wheels require a local C++/CUDA toolchain.)
"""
from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

import httpx

from lma.config import Settings
from lma.llm.downloader import DownloadError, Progress, RemoteFile, download, explain_network_error

REPO = "ggml-org/llama.cpp"
VARIANTS = ("cpu", "vulkan", "cuda-12.4")
EXE = "llama-server.exe" if sys.platform == "win32" else "llama-server"


def find_server(settings: Settings) -> Path | None:
    """Locate llama-server: explicit env override, then the managed runtime directory."""
    override = os.environ.get("LMA_LLAMA_SERVER")
    if override and Path(override).exists():
        return Path(override)
    meta = settings.runtime_path / "runtime.json"
    if meta.exists():
        p = Path(json.loads(meta.read_text(encoding="utf-8")).get("server", ""))
        if p.exists():
            return p
    hits = sorted(settings.runtime_path.rglob(EXE), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def runtime_info(settings: Settings) -> dict[str, str]:
    meta = settings.runtime_path / "runtime.json"
    return json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}


def _release(settings: Settings, tag: str | None, suffix: str) -> dict:
    """Newest release that actually ships the wanted asset (`releases/latest` is not always a binary build)."""
    base = f"{settings.llm.github_api.rstrip('/')}/repos/{REPO}/releases"
    url = f"{base}/tags/{tag}" if tag else f"{base}?per_page=15"
    try:
        with httpx.Client(timeout=30, follow_redirects=True) as c:
            resp = c.get(url, headers={"Accept": "application/vnd.github+json"})
    except httpx.HTTPError as exc:
        raise DownloadError(explain_network_error(exc, url) + MANUAL_HINT) from exc
    if resp.status_code != 200:
        raise DownloadError(f"GitHub API returned HTTP {resp.status_code} for {url}" + MANUAL_HINT)
    releases = [resp.json()] if tag else resp.json()
    for rel in releases:
        if any(a["name"].endswith(suffix) for a in rel.get("assets", [])):
            return rel
    raise DownloadError(f"no llama.cpp release with an asset ending in {suffix!r} was found" + MANUAL_HINT)


MANUAL_HINT = (
    "\n  Manual alternative: download `llama-<tag>-bin-win-<cpu|vulkan|cuda-12.4>-x64.zip` from "
    "https://github.com/ggml-org/llama.cpp/releases on any allowed channel, then run "
    "`/download runtime <path-to-zip>` (for CUDA also pass the cudart zip), or set LMA_LLAMA_SERVER "
    "to an existing llama-server.exe."
)


def _asset_named(release: dict, name: str) -> RemoteFile | None:
    for a in release.get("assets", []):
        if a["name"] == name:
            digest = a.get("digest") or ""
            return RemoteFile(a["browser_download_url"], int(a["size"]), digest[7:] if digest.startswith("sha256:") else None)
    return None


def install_from_zips(settings: Settings, zips: list[Path], label: str) -> Path:
    target = settings.runtime_path / label
    target.mkdir(parents=True, exist_ok=True)
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for member in zf.infolist():
                dest = (target / member.filename).resolve()
                if target.resolve() not in dest.parents and dest != target.resolve():
                    raise DownloadError(f"unsafe path in archive: {member.filename}")
            zf.extractall(target)
    hits = list(target.rglob(EXE))
    if not hits:
        raise DownloadError(f"{EXE} not found in {', '.join(z.name for z in zips)}")
    (settings.runtime_path / "runtime.json").write_text(
        json.dumps({"server": str(hits[0]), "label": label, "source": [z.name for z in zips]}, indent=2), encoding="utf-8")
    return hits[0]


def install(settings: Settings, variant: str, progress: Progress | None = None) -> Path:
    """Download and unpack an official llama.cpp Windows build (verified with the release digest)."""
    if variant not in VARIANTS:
        raise DownloadError(f"unknown runtime variant {variant!r}; choose from {', '.join(VARIANTS)}")
    if sys.platform != "win32":
        raise DownloadError("automatic runtime install is implemented for Windows; set LMA_LLAMA_SERVER on other systems")
    suffix = f"-bin-win-{variant}-x64.zip"
    rel = _release(settings, settings.llm.runtime_tag, suffix)
    tag = rel["tag_name"]
    assets = [a for a in [_asset_named(rel, f"llama-{tag}{suffix}")] if a]
    if not assets:
        raise DownloadError(f"release {tag} has no llama-{tag}{suffix}")
    if variant.startswith("cuda"):
        cud = _asset_named(rel, f"cudart-llama{suffix}")
        if cud is None:
            raise DownloadError(f"release {tag} has no CUDA runtime archive cudart-llama{suffix}")
        assets.append(cud)
    total = sum(a.size for a in assets)
    cache = settings.runtime_path / "_downloads"
    zips: list[Path] = []
    base = 0
    for a in assets:
        cb = (lambda d, _t, base=base: progress(base + d, total)) if progress else None
        zips.append(download(a, cache / a.url.rsplit("/", 1)[1], cb))
        base += a.size
    exe = install_from_zips(settings, zips, f"{tag}-{variant}")
    for z in zips:  # keep disk usage low
        z.unlink(missing_ok=True)
    return exe
