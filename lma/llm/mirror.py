"""Model mirror on GitHub Releases, for networks where Hugging Face is blocked but github.com is not.

Layout of the release (default: tag `models` in the project's own repository):

  manifest.json                         which models the mirror has, with sizes and SHA-256
  <model-id>--<file>.001, .002, ...     the GGUF files split into parts below GitHub's 2 GiB asset limit

`scripts/publish_models.py` builds and uploads the release. Downloads go through the normal
resumable downloader part by part; the parts are joined and the whole file is checked against the
SHA-256 from the manifest before it is installed.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import httpx

from lma.llm.downloader import CHUNK, DownloadError, Progress, RemoteFile, download, explain_network_error

PART_SIZE = 1900 * 2**20  # GitHub release assets must be < 2 GiB
MANIFEST = "manifest.json"


def release_base(web: str, repo: str, tag: str) -> str:
    return f"{web.rstrip('/')}/{repo.strip('/')}/releases/download/{tag}"


def fetch_manifest(base: str, timeout: float = 30) -> dict:
    url = f"{base}/{MANIFEST}"
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as c:
            resp = c.get(url)
    except httpx.HTTPError as exc:
        raise DownloadError(explain_network_error(exc, url)) from exc
    if resp.status_code == 404:
        raise DownloadError(f"no model mirror at {url} (publish one with scripts/publish_models.py)")
    if resp.status_code != 200:
        raise DownloadError(f"HTTP {resp.status_code} for {url}")
    data = resp.json()
    if not isinstance(data, dict) or "models" not in data:
        raise DownloadError(f"{url} is not a model manifest")
    return data


def part_names(model_id: str, filename: str, size: int, part_size: int = PART_SIZE) -> list[str]:
    n = max(1, -(-size // part_size))
    return [f"{model_id}--{filename}.{i:03d}" for i in range(1, n + 1)]


def split(path: Path, out_dir: Path, model_id: str, part_size: int = PART_SIZE) -> tuple[dict, list[Path]]:
    """Cut `path` into release assets; returns the manifest file entry and the part paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    size = path.stat().st_size
    h = hashlib.sha256()
    parts: list[dict] = []
    paths: list[Path] = []
    with path.open("rb") as src:
        for name in part_names(model_id, path.name, size, part_size):
            dest = out_dir / name
            written = 0
            with dest.open("wb") as fh:
                while written < part_size and (chunk := src.read(min(CHUNK, part_size - written))):
                    fh.write(chunk)
                    h.update(chunk)
                    written += len(chunk)
            parts.append({"asset": name, "size": written})
            paths.append(dest)
    return {"name": path.name, "size": size, "sha256": h.hexdigest(), "parts": parts}, paths


def _iter_parts(paths: list[Path]) -> Iterator[bytes]:
    for p in paths:
        with p.open("rb") as fh:
            while chunk := fh.read(CHUNK):
                yield chunk


def fetch_file(base: str, entry: dict, dest: Path, progress: Progress | None = None) -> Path:
    """Download one manifest file entry (all its parts), join, verify SHA-256, move to `dest`."""
    size, sha = int(entry["size"]), entry["sha256"]
    if dest.exists() and dest.stat().st_size == size:
        h = hashlib.sha256()
        for chunk in _iter_parts([dest]):
            h.update(chunk)
        if h.hexdigest() == sha:
            if progress:
                progress(size, size)
            return dest
        dest.unlink()
    work = dest.parent / f".{dest.name}.parts"
    paths: list[Path] = []
    done = 0
    for part in entry["parts"]:
        cb = (lambda d, _t, base_done=done: progress(base_done + d, size)) if progress else None
        paths.append(download(RemoteFile(f"{base}/{part['asset']}", int(part["size"]), None), work / part["asset"], cb))
        done += int(part["size"])
    joined = dest.with_name(dest.name + ".joining")
    h = hashlib.sha256()
    with joined.open("wb") as out:
        for chunk in _iter_parts(paths):
            out.write(chunk)
            h.update(chunk)
    if h.hexdigest() != sha:
        joined.unlink()
        shutil.rmtree(work, ignore_errors=True)
        raise DownloadError(f"SHA-256 mismatch for {dest.name} from the GitHub mirror; parts deleted, rerun to download again")
    joined.replace(dest)
    shutil.rmtree(work, ignore_errors=True)
    return dest


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
