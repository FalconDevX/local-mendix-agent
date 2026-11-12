"""Resumable, verified downloads (Hugging Face model files and llama.cpp release assets).

Uses the process proxy settings (HTTPS_PROXY / HTTP_PROXY / NO_PROXY) and the OS trust store,
i.e. exactly the corporate network configuration. Nothing is downloaded without an explicit call.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

Progress = Callable[[int, int], None]  # (bytes_done, bytes_total)
CHUNK = 1 << 20


class DownloadError(RuntimeError):
    pass


@dataclass
class RemoteFile:
    url: str
    size: int
    sha256: str | None


def explain_network_error(exc: Exception, url: str) -> str:
    host = httpx.URL(url).host
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    if "certificate" in low or "ssl" in low:
        hint = ("TLS verification failed. Corporate TLS inspection usually needs the company root CA; "
                "keep use_system_trust_store: true, or ask IT for the CA bundle and set SSL_CERT_FILE.")
    elif "proxy" in low or "407" in low:
        hint = "The proxy rejected the request. Set HTTPS_PROXY to the corporate proxy (with credentials if required)."
    elif isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        hint = (f"Cannot reach {host}. It may be blocked by the firewall/VPN, or a proxy is required "
                "(set HTTPS_PROXY). Offline alternative: copy the file manually and register it.")
    else:
        hint = "Check VPN/proxy settings; the download can be resumed by running the same command again."
    return f"{text}\n  -> {hint}"


def hf_file_info(endpoint: str, repo: str, filename: str, timeout: float = 30) -> RemoteFile:
    """Ask the Hub for the real size and SHA-256 of a file (LFS metadata)."""
    api = f"{endpoint.rstrip('/')}/api/models/{repo}"
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as c:
            resp = c.get(api, params={"blobs": "true"})
    except httpx.HTTPError as exc:
        raise DownloadError(explain_network_error(exc, api)) from exc
    if resp.status_code == 401 or resp.status_code == 403:
        raise DownloadError(f"{repo}: access denied (HTTP {resp.status_code}); the repository is gated or private")
    if resp.status_code != 200:
        raise DownloadError(f"{repo}: Hugging Face API returned HTTP {resp.status_code}")
    for s in resp.json().get("siblings", []):
        if s.get("rfilename") == filename:
            lfs = s.get("lfs") or {}
            return RemoteFile(
                url=f"{endpoint.rstrip('/')}/{repo}/resolve/main/{filename}",
                size=int(lfs.get("size") or s.get("size") or 0),
                sha256=lfs.get("sha256"),
            )
    raise DownloadError(f"{repo} has no file named {filename!r} (catalogue entry is stale)")


def sha256_of(path: Path, progress: Progress | None = None) -> str:
    h = hashlib.sha256()
    total, done = path.stat().st_size, 0
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    return h.hexdigest()


def download(remote: RemoteFile, dest: Path, progress: Progress | None = None, timeout: float = 60) -> Path:
    """Download to `dest` with resume (`dest.part`) and SHA-256 verification."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (not remote.size or dest.stat().st_size == remote.size):
        if not remote.sha256 or sha256_of(dest) == remote.sha256:
            if progress:
                progress(dest.stat().st_size, dest.stat().st_size)
            return dest
        dest.unlink()  # corrupt
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if remote.size and have > remote.size:
        part.unlink()
        have = 0
    if not remote.size or have < remote.size:
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with httpx.Client(timeout=httpx.Timeout(timeout, read=120), follow_redirects=True) as c:
                with c.stream("GET", remote.url, headers=headers) as resp:
                    if resp.status_code == 200 and have:
                        have = 0  # server ignored Range: start over
                    elif resp.status_code not in (200, 206):
                        raise DownloadError(f"HTTP {resp.status_code} for {remote.url}")
                    total = remote.size or have + int(resp.headers.get("content-length", 0))
                    with part.open("ab" if have else "wb") as fh:
                        done = have
                        for chunk in resp.iter_bytes(CHUNK):
                            fh.write(chunk)
                            done += len(chunk)
                            if progress:
                                progress(done, total)
        except httpx.HTTPError as exc:
            raise DownloadError(explain_network_error(exc, remote.url) + "\n  (partial file kept, rerun to resume)") from exc
    if remote.size and part.stat().st_size != remote.size:
        raise DownloadError(f"size mismatch for {dest.name}: got {part.stat().st_size}, expected {remote.size} (rerun to resume)")
    if remote.sha256:
        actual = sha256_of(part)
        if actual != remote.sha256:
            part.unlink()
            raise DownloadError(f"SHA-256 mismatch for {dest.name}: expected {remote.sha256}, got {actual}; file deleted")
    part.replace(dest)
    return dest
