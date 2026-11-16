"""GitHub model mirror: split -> serve as release assets -> download, join, verify, register."""
from __future__ import annotations

import functools
import hashlib
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from lma.config import Settings
from lma.llm import mirror
from lma.llm.downloader import DownloadError
from lma.llm.registry import ModelRegistry

REPO, TAG = "owner/agent", "models"


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D401
        pass


@pytest.fixture
def release(tmp_path: Path):
    """A local web server laid out like github.com/<repo>/releases/download/<tag>/."""
    root = tmp_path / "web"
    assets = root / REPO / "releases" / "download" / TAG
    assets.mkdir(parents=True)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(root)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", assets
    srv.shutdown()


def publish(assets: Path, src: Path, model_id: str, spec: dict, part_size: int, extra: list[Path] = ()) -> dict:
    entries = []
    for f in [src, *extra]:
        entry, _ = mirror.split(f, assets, model_id, part_size)
        entries.append(entry)
    manifest = {"version": 1, "models": {model_id: {"spec": spec, "files": entries}}}
    mirror.write_manifest(assets / mirror.MANIFEST, manifest)
    return manifest


def mirror_settings(settings: Settings, web: str, source: str) -> Settings:
    settings.llm.github_web = web
    settings.llm.models_repo = REPO
    settings.llm.models_tag = TAG
    settings.llm.model_source = source
    settings.llm.hf_endpoint = "http://127.0.0.1:9"  # nothing listens there: Hugging Face is "blocked"
    return settings


def test_split_names_and_sizes(tmp_path: Path):
    src = tmp_path / "m.gguf"
    src.write_bytes(os.urandom(2500))
    entry, paths = mirror.split(src, tmp_path / "out", "m", part_size=1000)
    assert [p["asset"] for p in entry["parts"]] == ["m--m.gguf.001", "m--m.gguf.002", "m--m.gguf.003"]
    assert [p["size"] for p in entry["parts"]] == [1000, 1000, 500]
    assert b"".join(p.read_bytes() for p in paths) == src.read_bytes()
    assert entry["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()


def test_catalogue_model_falls_back_to_mirror_when_hf_is_blocked(settings: Settings, release, tmp_path: Path):
    web, assets = release
    reg = ModelRegistry(mirror_settings(settings, web, "auto"))
    spec = reg.catalogue["qwen3-vl-4b"]  # a vision model: two files (model + mmproj)
    model, proj = tmp_path / spec.file, tmp_path / spec.mmproj
    model.write_bytes(os.urandom(5000))
    proj.write_bytes(os.urandom(1200))
    publish(assets, model, spec.id, spec.model_dump(), part_size=2048, extra=[proj])

    inst = reg.download(spec.id)
    assert Path(inst.path).read_bytes() == model.read_bytes()
    assert Path(inst.mmproj_path).read_bytes() == proj.read_bytes()
    assert inst.verified and inst.sha256 == hashlib.sha256(model.read_bytes()).hexdigest()
    assert not list(Path(inst.path).parent.glob(".*parts")), "temporary parts must be removed"
    assert spec.id in ModelRegistry(settings).installed, "the install must be persisted"


def test_mirror_only_model_and_corrupt_part(settings: Settings, release, tmp_path: Path):
    web, assets = release
    reg = ModelRegistry(mirror_settings(settings, web, "github"))
    src = tmp_path / "custom.gguf"
    src.write_bytes(os.urandom(3000))
    spec = {"id": "custom", "role": "chat", "file": "custom.gguf", "license": "mit", "size_gb": 0.0}
    publish(assets, src, "custom", spec, part_size=1024)
    (assets / "custom--custom.gguf.002").write_bytes(os.urandom(1024))  # tampered asset

    with pytest.raises(DownloadError, match="SHA-256 mismatch"):
        reg.download("custom")
    assert "custom" not in reg.installed

    publish(assets, src, "custom", spec, part_size=1024)  # fixed release
    inst = reg.download("custom")
    assert Path(inst.path).read_bytes() == src.read_bytes()
    with pytest.raises(DownloadError, match="not in the GitHub mirror"):
        reg.download("missing")


def test_hf_only_source_does_not_use_mirror(settings: Settings, release):
    web, _ = release
    reg = ModelRegistry(mirror_settings(settings, web, "hf"))
    with pytest.raises(DownloadError) as exc:
        reg.download("qwen3-4b")
    assert "GitHub mirror" not in str(exc.value)


def test_auto_candidates_prefer_default_then_mirror(settings: Settings, release, tmp_path: Path):
    from lma.llm.hardware import Hardware

    web, assets = release
    hw = Hardware(os="w", python="3", cpu="c", cores_physical=4, cores_logical=8, ram_total_gb=16, ram_free_gb=8,
                  gpu_name=None, vram_total_gb=0, vram_free_gb=0, cuda_driver=None, disk_free_gb=100)
    reg = ModelRegistry(mirror_settings(settings, web, "auto"))
    default = reg.default_choice(hw, "chat")
    src = tmp_path / "small.gguf"
    src.write_bytes(os.urandom(2000))
    spec = {"id": "small", "role": "chat", "file": "small.gguf", "license": "mit", "size_gb": 1.0, "min_ram_gb": 4}
    publish(assets, src, "small", spec, part_size=1024)

    assert [m.id for m in reg.auto_candidates(hw, "chat")] == [default.id, "small"]  # default is not mirrored: fallback follows
    assert [m.id for m in reg.auto_candidates(hw, "vision")] == [reg.default_choice(hw, "vision").id]
    reg.settings.llm.model_source = "github"
    assert [m.id for m in reg.auto_candidates(hw, "chat")] == ["small"]
    reg.download("small")
    assert reg.auto_candidates(hw, "chat") == [], "a role that has a model is left alone"
