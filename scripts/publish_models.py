"""Publish models to the GitHub mirror (a release in this repository) for networks that block Hugging Face.

Run on a machine that CAN reach Hugging Face, with the GitHub CLI logged in (`gh auth login`):

  .venv\\Scripts\\python.exe scripts\\publish_models.py qwen3-8b nomic-embed-v1.5 qwen3-vl-4b
  .venv\\Scripts\\python.exe scripts\\publish_models.py --installed        every model installed here
  .venv\\Scripts\\python.exe scripts\\publish_models.py qwen3-4b --dry-run  split only, upload nothing

Models that are not installed yet are first downloaded from Hugging Face (SHA-256 verified). Each
GGUF is split into parts below GitHub's 2 GiB asset limit and uploaded with `gh release upload`;
manifest.json is uploaded last, so the mirror never lists a model whose parts are missing.
Only models whose licence allows redistribution (Apache-2.0 / MIT in the catalogue) are accepted.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lma.config import load_settings, prepare_network_env  # noqa: E402
from lma.llm import mirror  # noqa: E402
from lma.llm.downloader import DownloadError  # noqa: E402
from lma.llm.registry import ModelRegistry  # noqa: E402

REDISTRIBUTABLE = {"apache-2.0", "mit"}


def gh(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], check=check, text=True, capture_output=True)


def bar(label: str):
    state = {"last": -1}

    def cb(done: int, total: int) -> None:
        pct = int(done * 100 / total) if total else 0
        if pct // 10 != state["last"]:
            state["last"] = pct // 10
            print(f"  {label}: {pct}% ({done/1e6:.0f}/{total/1e6:.0f} MB)", flush=True)
    return cb


def notes(manifest: dict, repo: str) -> str:
    lines = [
        "GGUF model mirror for Local Mendix Agent, for networks where huggingface.co is blocked.",
        "",
        "Do not download the parts by hand: run `/download <id>` in the app (or `/download mirror` to list).",
        "The app joins the parts and verifies the SHA-256 of every file.",
        "",
        "| id | role | size | licence | original source |",
        "|---|---|---|---|---|",
    ]
    for mid, e in sorted(manifest["models"].items()):
        sp = e["spec"]
        size = sum(int(f["size"]) for f in e["files"]) / 1e9
        src = f"https://huggingface.co/{sp['repo']}" if sp.get("repo") else "-"
        lines.append(f"| {mid} | {sp['role']} | {size:.2f} GB | {sp['license']} | {src} |")
    lines += ["", "The models are redistributed unmodified under their original licences (Apache-2.0 / MIT); "
              "see the original repositories for the licence texts and model cards.",
              f"Published with scripts/publish_models.py from {repo}."]
    return "\n".join(lines)


def main() -> int:
    s = load_settings()
    prepare_network_env(s)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="*", help="catalogue ids (see /models)")
    p.add_argument("--installed", action="store_true", help="publish every model installed on this machine")
    p.add_argument("--repo", default=s.llm.models_repo)
    p.add_argument("--tag", default=s.llm.models_tag)
    p.add_argument("--dry-run", action="store_true", help="download and split, but upload nothing")
    a = p.parse_args()

    s.llm.model_source = "hf"  # the source for publishing is always the original on Hugging Face
    reg = ModelRegistry(s)
    ids = list(dict.fromkeys(a.models + (sorted(reg.installed) if a.installed else [])))
    if not ids:
        p.error("name the models to publish, or pass --installed")
    for mid in ids:
        spec = reg.installed[mid].spec if mid in reg.installed else reg.catalogue.get(mid)
        if spec is None:
            p.error(f"{mid!r} is neither installed nor in config/models.yaml")
        if spec.license.lower() not in REDISTRIBUTABLE:
            p.error(f"{mid}: licence {spec.license!r} is not known to allow redistribution; not publishing it")

    if not a.dry_run:
        if shutil.which("gh") is None:
            p.error("GitHub CLI (gh) not found")
        if gh("release", "view", a.tag, "--repo", a.repo, check=False).returncode != 0:
            print(f"creating release '{a.tag}' in {a.repo}")
            gh("release", "create", a.tag, "--repo", a.repo, "--title", "Models (GGUF mirror)",
               "--notes", "GGUF model mirror for Local Mendix Agent (being uploaded).")
    base = mirror.release_base(s.llm.github_web, a.repo, a.tag)
    try:
        manifest = mirror.fetch_manifest(base)
        print(f"existing mirror: {', '.join(sorted(manifest['models'])) or 'empty'}")
    except DownloadError:
        manifest = {"version": 1, "models": {}}

    work = Path(tempfile.mkdtemp(prefix="lma-mirror-", dir=s.models_path))
    try:
        for mid in ids:
            if mid not in reg.installed:
                print(f"{mid}: downloading from Hugging Face first")
                reg.download(mid, bar(mid))
            inst = reg.installed[mid]
            files = [Path(inst.path)] + ([Path(inst.mmproj_path)] if inst.mmproj_path else [])
            entries, parts = [], []
            for f in files:
                print(f"{mid}: splitting {f.name} ({f.stat().st_size/1e9:.2f} GB)")
                entry, paths = mirror.split(f, work, mid)
                if f == files[0] and inst.sha256 and entry["sha256"] != inst.sha256:
                    raise SystemExit(f"{mid}: local file does not match its recorded SHA-256; delete and re-download it")
                entries.append(entry)
                parts += paths
            if a.dry_run:
                print(f"{mid}: {len(parts)} parts in {work} (dry run, not uploaded)")
                continue
            print(f"{mid}: uploading {len(parts)} parts to {a.repo} release '{a.tag}' ...", flush=True)
            for part in parts:
                gh("release", "upload", a.tag, str(part), "--repo", a.repo, "--clobber")
                print(f"  uploaded {part.name}", flush=True)
                part.unlink()
            manifest["models"][mid] = {"spec": inst.spec.model_dump(), "files": entries}
            mfile = work / mirror.MANIFEST
            mirror.write_manifest(mfile, manifest)
            gh("release", "upload", a.tag, str(mfile), "--repo", a.repo, "--clobber")
            nfile = work / "notes.md"
            nfile.write_text(notes(manifest, a.repo), encoding="utf-8")
            gh("release", "edit", a.tag, "--repo", a.repo, "--notes-file", str(nfile))
            print(f"{mid}: published")
    except subprocess.CalledProcessError as exc:
        print(f"gh failed: {' '.join(exc.cmd)}\n{exc.stderr}", file=sys.stderr)
        return 1
    finally:
        if not a.dry_run:
            shutil.rmtree(work, ignore_errors=True)
    print(f"done. Mirror: https://github.com/{a.repo}/releases/tag/{a.tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
