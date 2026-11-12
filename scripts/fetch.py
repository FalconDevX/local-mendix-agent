"""Non-interactive downloader: python scripts/fetch.py runtime vulkan | model <id> [...]"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lma.config import load_settings, prepare_network_env  # noqa: E402
from lma.llm import runtime  # noqa: E402
from lma.llm.registry import ModelRegistry  # noqa: E402


def bar(label: str):
    state = {"last": -1}

    def cb(done: int, total: int) -> None:
        pct = int(done * 100 / total) if total else 0
        if pct // 10 != state["last"]:
            state["last"] = pct // 10
            print(f"  {label}: {pct}% ({done/1e6:.0f}/{total/1e6:.0f} MB)", flush=True)
    return cb


def main() -> None:
    s = load_settings()
    prepare_network_env(s)
    kind, *names = sys.argv[1:]
    if kind == "runtime":
        print("installed:", runtime.install(s, names[0], bar("runtime")))
    else:
        reg = ModelRegistry(s)
        for n in names:
            inst = reg.download(n, bar(n))
            print("installed:", inst.path, "sha256 verified" if inst.verified else "unverified")


if __name__ == "__main__":
    main()
