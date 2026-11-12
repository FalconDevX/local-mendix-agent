"""Model catalogue (config/models.yaml), installed-model registry and recommendations."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from lma.config import CONFIG_DIR, Settings, load_yaml
from lma.llm.downloader import DownloadError, Progress, download, hf_file_info, sha256_of
from lma.llm.hardware import Hardware

Role = Literal["chat", "vision", "embed"]


class ModelSpec(BaseModel):
    id: str
    role: Role
    tier: Literal["small", "balanced", "quality"] = "small"
    repo: str | None = None
    file: str
    mmproj: str | None = None
    quant: str = "?"
    size_gb: float = 0
    license: str = "unknown"
    ctx_max: int = 8192
    tools: Literal["template", "none", "unknown"] = "unknown"
    vision: bool = False
    min_ram_gb: float = 0
    min_vram_gb: float = 0
    query_prefix: str = ""
    doc_prefix: str = ""
    notes: str = ""


class Installed(BaseModel):
    spec: ModelSpec
    path: str
    mmproj_path: str | None = None
    sha256: str | None = None
    verified: bool = False
    installed_at: float = 0


def fit(spec: ModelSpec, hw: Hardware) -> str:
    """How a model would run on this machine."""
    if spec.min_vram_gb and hw.vram_total_gb >= spec.min_vram_gb:
        return "gpu"
    if hw.ram_total_gb >= spec.min_ram_gb + (2 if spec.role == "chat" else 0):
        return "gpu+cpu" if hw.vram_total_gb >= 2 and spec.min_vram_gb else "cpu"
    return "too-big"


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dir = settings.models_path
        self.file = self.dir / "registry.json"
        self.catalogue: dict[str, ModelSpec] = {
            m["id"]: ModelSpec.model_validate(m) for m in load_yaml(CONFIG_DIR / "models.yaml").get("models", [])
        }
        self.installed: dict[str, Installed] = {}
        self._load()

    def _load(self) -> None:
        if self.file.exists():
            raw = json.loads(self.file.read_text(encoding="utf-8"))
            for mid, item in raw.items():
                inst = Installed.model_validate(item)
                if Path(inst.path).exists():  # drop entries whose file vanished
                    self.installed[mid] = inst

    def _save(self) -> None:
        self.file.write_text(json.dumps({k: v.model_dump() for k, v in self.installed.items()}, indent=2), encoding="utf-8")

    # ---- queries -------------------------------------------------------
    def spec(self, model_id: str) -> ModelSpec:
        if model_id in self.installed:
            return self.installed[model_id].spec
        if model_id in self.catalogue:
            return self.catalogue[model_id]
        raise KeyError(f"unknown model {model_id!r}; see /models")

    def by_role(self, role: Role, installed_only: bool = False) -> list[ModelSpec]:
        pool = [i.spec for i in self.installed.values()] if installed_only else list(
            {**self.catalogue, **{k: v.spec for k, v in self.installed.items()}}.values())
        return [m for m in pool if m.role == role or (role == "chat" and m.role == "vision" and m.tools == "template")]

    def recommend(self, hw: Hardware) -> dict[str, dict[str, tuple[ModelSpec, str] | None]]:
        """Per role and tier: the catalogue model and how it fits ('gpu', 'gpu+cpu', 'cpu')."""
        out: dict[str, dict[str, tuple[ModelSpec, str] | None]] = {}
        for role in ("chat", "vision", "embed"):
            out[role] = {}
            for tier in ("small", "balanced", "quality"):
                cands = [m for m in self.catalogue.values() if m.role == role and m.tier == tier]
                usable = [(m, fit(m, hw)) for m in cands]
                usable = [x for x in usable if x[1] != "too-big"]
                usable.sort(key=lambda x: (x[1] != "gpu", -x[0].size_gb))
                out[role][tier] = usable[0] if usable else None
        return out

    def default_choice(self, hw: Hardware, role: Role) -> ModelSpec | None:
        rec = self.recommend(hw)[role]
        for tier in ("balanced", "quality", "small"):
            item = rec.get(tier)
            if item and (item[1] == "gpu" or tier == "small" or role == "embed"):
                return item[0]
        return next((v[0] for v in rec.values() if v), None)

    # ---- mutations -----------------------------------------------------
    def download(self, model_id: str, progress: Progress | None = None) -> Installed:
        spec = self.catalogue.get(model_id)
        if spec is None or spec.repo is None:
            raise DownloadError(f"{model_id!r} is not a downloadable catalogue model")
        target = self.dir / model_id
        files = [spec.file] + ([spec.mmproj] if spec.mmproj else [])
        infos = [hf_file_info(self.settings.llm.hf_endpoint, spec.repo, f) for f in files]
        need = sum(i.size for i in infos)
        import shutil

        free = shutil.disk_usage(self.dir).free
        if need > free - 2**30:
            raise DownloadError(f"not enough disk space: need {need/2**30:.1f} GB (+1 GB reserve), free {free/2**30:.1f} GB")
        done_before = 0
        paths: list[Path] = []
        for name, info in zip(files, infos):
            base = done_before
            cb = (lambda d, _t, base=base: progress(base + d, need)) if progress else None
            paths.append(download(info, target / name, cb))
            done_before += info.size
        inst = Installed(
            spec=spec, path=str(paths[0]), mmproj_path=str(paths[1]) if len(paths) > 1 else None,
            sha256=infos[0].sha256, verified=bool(infos[0].sha256), installed_at=time.time(),
        )
        self.installed[model_id] = inst
        self._save()
        return inst

    def register_local(self, path: Path, role: Role, model_id: str | None = None, mmproj: Path | None = None) -> Installed:
        """Register a GGUF that was copied in manually (offline / blocked-download fallback)."""
        path = path.expanduser().resolve()  # llama-server runs from its own folder: paths must be absolute
        mmproj = mmproj.expanduser().resolve() if mmproj else None
        if not path.exists() or path.suffix.lower() != ".gguf":
            raise FileNotFoundError(f"{path} is not an existing .gguf file")
        if mmproj is not None and not mmproj.exists():
            raise FileNotFoundError(f"{mmproj} does not exist")
        mid = model_id or path.stem.lower()
        spec = ModelSpec(id=mid, role=role, file=path.name, size_gb=round(path.stat().st_size / 1e9, 2),
                         vision=mmproj is not None, license="unknown (user supplied)", notes="registered from local file")
        inst = Installed(spec=spec, path=str(path), mmproj_path=str(mmproj) if mmproj else None,
                         sha256=sha256_of(path), verified=False, installed_at=time.time())
        self.installed[mid] = inst
        self._save()
        return inst

    def delete(self, model_id: str) -> float:
        inst = self.installed.pop(model_id, None)
        if inst is None:
            raise KeyError(f"{model_id!r} is not installed")
        freed = 0
        for p in (inst.path, inst.mmproj_path):
            if p and Path(p).exists() and self.dir in Path(p).resolve().parents:  # only delete what we own
                freed += Path(p).stat().st_size
                Path(p).unlink()
        folder = self.dir / model_id
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
        self._save()
        return freed / 1e9
