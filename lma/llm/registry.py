"""Model catalogue (config/models.yaml), installed-model registry and recommendations."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from lma.config import CONFIG_DIR, Settings, load_yaml
from lma.llm import mirror
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
        self._manifest: dict | None = None
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
                # big models are only suggested when they fit the GPU; on CPU they are too slow to recommend
                usable = [x for x in usable if x[1] == "gpu" or (x[1] != "too-big" and x[0].size_gb <= 10)]
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

    def auto_candidates(self, hw: Hardware, role: Role) -> list[ModelSpec]:
        """Models to try, in order, for a hands-off first install of `role` (`/download auto`).

        Empty when a model of that role is already installed. The hardware default comes first; the
        models the GitHub mirror has for that role follow (largest that fits first), so a network
        that blocks Hugging Face still ends up with a working model.
        """
        if any(i.spec.role == role for i in self.installed.values()):
            return []
        out: list[ModelSpec] = []
        first = self.default_choice(hw, role)
        if first is not None and self.settings.llm.model_source != "github":
            out.append(first)
        if self.settings.llm.model_source != "hf":
            try:
                entries = self.mirror_manifest()["models"]
            except DownloadError:
                entries = {}
            mirrored = [ModelSpec.model_validate(e["spec"]) for e in entries.values()]
            mirrored = [m for m in mirrored if m.role == role and fit(m, hw) != "too-big"]
            if first is not None and any(m.id == first.id for m in mirrored):
                out = [first]  # the default is in the mirror: nothing else is needed
            else:
                # without a GPU a big model is too slow to be a sensible automatic pick
                ok = [m for m in mirrored if fit(m, hw) == "gpu" or m.size_gb <= 3.5] or mirrored
                out += sorted(ok, key=lambda m: -m.size_gb)
        return out

    # ---- mutations -----------------------------------------------------
    def hf_spec(self, ref: str, role: Role = "chat") -> ModelSpec:
        """Spec for any GGUF on the Hub, given as `owner/repo/file.gguf` (optionally `hf:` prefixed).

        Lets the user install models that are not in the catalogue. The file must exist and the
        repository must not be gated; size and SHA-256 come from the Hub as for catalogue models.
        """
        parts = ref.removeprefix("hf:").strip("/").split("/")
        if len(parts) < 3 or not parts[-1].lower().endswith(".gguf"):
            raise DownloadError("expected hf:<owner>/<repo>/<file>.gguf, e.g. hf:Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf")
        repo, file = "/".join(parts[:2]), "/".join(parts[2:])
        info = hf_file_info(self.settings.llm.hf_endpoint, repo, file)
        size = round(info.size / 1e9, 2)
        return ModelSpec(id=Path(file).stem.lower(), role=role, tier="balanced", repo=repo, file=file, size_gb=size,
                         license="see the model card", ctx_max=32768, tools="unknown" if role == "chat" else "none",
                         min_ram_gb=round(size + 2, 1), min_vram_gb=round(size + 1.5, 1), notes=f"added by the user from {repo}")

    # ---- GitHub mirror (for networks that block Hugging Face) ----------
    @property
    def mirror_base(self) -> str:
        llm = self.settings.llm
        return mirror.release_base(llm.github_web, llm.models_repo, llm.models_tag)

    def mirror_manifest(self) -> dict:
        if self._manifest is None:
            self._manifest = mirror.fetch_manifest(self.mirror_base)
        return self._manifest

    def source_label(self) -> str:
        llm = self.settings.llm
        gh = f"github.com/{llm.models_repo} (release '{llm.models_tag}')"
        return {"hf": llm.hf_endpoint, "github": gh}.get(llm.model_source, f"{llm.hf_endpoint}, fallback {gh}")

    def _download_from_mirror(self, model_id: str, spec: ModelSpec | None, progress: Progress | None) -> Installed:
        entry = self.mirror_manifest()["models"].get(model_id)
        if entry is None:
            have = ", ".join(sorted(self.mirror_manifest()["models"])) or "nothing"
            raise DownloadError(f"{model_id!r} is not in the GitHub mirror (it has: {have})")
        spec = spec or ModelSpec.model_validate(entry["spec"])
        files = entry["files"]
        need = sum(int(f["size"]) for f in files)
        self._check_disk(2 * need)  # parts + joined file exist side by side for a moment
        target = self.dir / model_id
        paths: list[Path] = []
        done_before = 0
        for f in files:
            cb = (lambda d, _t, base=done_before: progress(base + d, need)) if progress else None
            paths.append(mirror.fetch_file(self.mirror_base, f, target / f["name"], cb))
            done_before += int(f["size"])
        mm = next((p for p in paths if spec.mmproj and p.name == spec.mmproj), None)
        main = next(p for p in paths if p.name == spec.file)
        inst = Installed(spec=spec, path=str(main), mmproj_path=str(mm) if mm else None,
                         sha256=files[0]["sha256"], verified=True, installed_at=time.time())
        self.installed[model_id] = inst
        self._save()
        return inst

    def _check_disk(self, need: int) -> None:
        import shutil

        self.dir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.dir).free
        if need > free - 2**30:
            raise DownloadError(f"not enough disk space: need {need/2**30:.1f} GB (+1 GB reserve), free {free/2**30:.1f} GB")

    def download(self, model_id: str, progress: Progress | None = None, spec: ModelSpec | None = None) -> Installed:
        spec = spec or self.catalogue.get(model_id)
        source = self.settings.llm.model_source
        if source == "github" or spec is None:  # models published only in the mirror have no catalogue entry
            if source == "hf":
                raise DownloadError(f"{model_id!r} is not a downloadable catalogue model")
            return self._download_from_mirror(model_id, spec, progress)
        if spec.repo is None:
            raise DownloadError(f"{model_id!r} is not a downloadable catalogue model")
        files = [spec.file] + ([spec.mmproj] if spec.mmproj else [])
        try:
            infos = [hf_file_info(self.settings.llm.hf_endpoint, spec.repo, f) for f in files]
        except (DownloadError, ValueError) as exc:  # ValueError: a proxy block page instead of JSON
            if source != "auto":
                raise
            try:
                return self._download_from_mirror(model_id, spec, progress)
            except DownloadError as mexc:
                raise DownloadError(f"Hugging Face: {exc}\nGitHub mirror: {mexc}") from mexc
        target = self.dir / model_id
        need = sum(i.size for i in infos)
        self._check_disk(need)
        done_before = 0
        paths: list[Path] = []
        for name, info in zip(files, infos):
            base = done_before
            cb = (lambda d, _t, base=base: progress(base + d, need)) if progress else None
            paths.append(download(info, target / Path(name).name, cb))
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
