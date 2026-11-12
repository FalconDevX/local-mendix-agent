"""/doctor: environment, model, network and MCP diagnostics. Read-only; never changes or bypasses anything."""
from __future__ import annotations

import importlib.metadata
import os
import socket
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from lma.core.state import AppState
from lma.llm.hardware import detect, recommend_runtime
from lma.llm.runtime import find_server, runtime_info

PACKAGES = ["mcp", "pydantic", "pyyaml", "rich", "prompt_toolkit", "httpx", "psutil", "numpy", "pillow", "keyring", "truststore"]


@dataclass
class Check:
    group: str
    name: str
    status: str  # ok | warn | fail | info
    detail: str


def _http_probe(url: str, timeout: float = 8) -> tuple[str, str]:
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as c:
            r = c.head(url)
        return ("ok" if r.status_code < 500 else "warn"), f"HTTP {r.status_code} in {(time.perf_counter() - t0) * 1000:.0f} ms"
    except httpx.HTTPError as exc:
        low = f"{type(exc).__name__}: {exc}".lower()
        if "certificate" in low or "ssl" in low:
            why = "TLS error - likely corporate TLS inspection; the company root CA must be in the Windows trust store"
        elif "proxy" in low:
            why = "proxy error - check HTTPS_PROXY"
        elif "timeout" in low:
            why = "timed out - blocked by firewall/VPN or proxy required"
        else:
            why = "unreachable - blocked, offline, or proxy required"
        return "warn", f"{type(exc).__name__}: {why}"


async def run(state: AppState, inference: bool = True) -> list[Check]:
    s = state.settings
    out: list[Check] = []
    add = lambda g, n, st, d: out.append(Check(g, n, st, d))  # noqa: E731

    # ---- Python / environment
    v = sys.version_info
    add("Environment", "Python", "ok" if v >= (3, 11) else "fail", f"{v.major}.{v.minor}.{v.micro} ({sys.executable})")
    in_venv = sys.prefix != sys.base_prefix
    add("Environment", "Virtual environment", "ok" if in_venv else "warn",
        sys.prefix if in_venv else "not running in a venv; run setup.ps1 to create .venv (no admin needed)")
    for pkg in PACKAGES:
        try:
            add("Environment", f"package {pkg}", "ok", importlib.metadata.version(pkg))
        except importlib.metadata.PackageNotFoundError:
            add("Environment", f"package {pkg}", "fail", "missing - run: .venv\\Scripts\\python -m pip install -e .")
    try:
        with tempfile.NamedTemporaryFile(dir=s.data_path, delete=True) as fh:
            fh.write(b"x")
        add("Environment", "Data directory writable", "ok", str(s.data_path))
    except OSError as exc:
        add("Environment", "Data directory writable", "fail", f"{s.data_path}: {exc}; set data_dir to a user-writable folder")
    if state.root:
        add("Environment", "Project directory", "ok" if os.access(state.root, os.W_OK) else "warn", f"{state.root}" + ("" if os.access(state.root, os.W_OK) else " (read-only)"))
    else:
        add("Environment", "Project directory", "info", "no project open (/open <folder>)")

    # ---- hardware
    hw = detect(s.data_path)
    state.llm.hardware = hw
    add("Hardware", "CPU", "info", f"{hw.cpu} ({hw.cores_physical} cores / {hw.cores_logical} threads)")
    add("Hardware", "RAM", "ok" if hw.ram_total_gb >= 8 else "warn", f"{hw.ram_total_gb} GB total, {hw.ram_free_gb} GB free")
    if hw.gpu_name:
        add("Hardware", "GPU", "ok", f"{hw.gpu_name}, VRAM {hw.vram_total_gb} GB total / {hw.vram_free_gb} GB free, driver CUDA {hw.cuda_driver or 'unknown'}")
    else:
        add("Hardware", "GPU", "info", "no NVIDIA GPU found via nvidia-smi; inference will use CPU (or Vulkan for other GPUs)")
    add("Hardware", "Disk space", "ok" if hw.disk_free_gb >= 10 else "warn", f"{hw.disk_free_gb} GB free on the data drive" + ("" if hw.disk_free_gb >= 10 else " - only small models fit"))

    # ---- runtime / models
    exe = find_server(s)
    variant, reason = recommend_runtime(hw)
    if exe:
        add("Models", "llama.cpp runtime", "ok", f"{exe} ({runtime_info(s).get('label', 'manual')})")
    else:
        add("Models", "llama.cpp runtime", "fail", f"not installed. Run /download runtime (recommended variant: {variant} - {reason})")
    for role in ("chat", "vision", "embed"):
        mid = state.llm.model_id(role)  # type: ignore[arg-type]
        needed = role == "chat"
        if mid:
            inst = state.llm.registry.installed[mid]
            add("Models", f"{role} model", "ok", f"{mid} ({inst.spec.size_gb} GB, {inst.spec.quant}, sha256 {'verified' if inst.verified else 'not verified'})")
        else:
            rec = state.llm.registry.default_choice(hw, role)  # type: ignore[arg-type]
            add("Models", f"{role} model", "fail" if needed else "warn", f"none installed. Suggested: /download {rec.id if rec else '?'}")
    if inference and exe and state.llm.model_id("chat"):
        try:
            t0 = time.perf_counter()
            res = await state.llm.chat([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=16)
            add("Models", "Inference test", "ok", f"model answered {res.text.strip()[:30]!r} in {time.perf_counter() - t0:.1f}s ({res.tokens_per_s:.0f} tok/s)")
        except Exception as exc:  # noqa: BLE001
            add("Models", "Inference test", "fail", str(exc)[:600])
    elif inference:
        add("Models", "Inference test", "info", "skipped (runtime or chat model missing)")

    # ---- network
    proxies = {k: v for k, v in urllib.request.getproxies().items() if k != "no"}
    env_proxy = {k: v for k, v in os.environ.items() if k.lower() in ("http_proxy", "https_proxy", "all_proxy")}
    if env_proxy:
        add("Network", "Proxy (environment)", "info", ", ".join(f"{k}={_mask(v)}" for k, v in env_proxy.items()))
    elif proxies:
        add("Network", "Proxy (Windows settings)", "info",
            ", ".join(f"{k}={_mask(v)}" for k, v in proxies.items()) + " - set HTTPS_PROXY to this value if downloads fail")
    else:
        add("Network", "Proxy", "info", "no proxy configured (direct connection)")
    add("Network", "NO_PROXY", "ok" if "localhost" in os.environ.get("NO_PROXY", "") else "warn", os.environ.get("NO_PROXY", "(unset)"))
    add("Network", "System trust store", "info", "enabled (corporate root CAs are trusted)" if s.use_system_trust_store else "disabled")
    hf, hf_detail = _http_probe(s.llm.hf_endpoint)
    add("Network", "Hugging Face", hf, f"{s.llm.hf_endpoint}: {hf_detail}")
    gh, gh_detail = _http_probe("https://github.com")
    add("Network", "GitHub (runtime downloads)", gh, gh_detail)
    offline = hf != "ok"
    add("Network", "Offline capability", "info",
        ("External network unreachable. " if offline else "") +
        "Works offline: installed models, chat/agent, indexing, search, editing, memory, local MCP (Maia). "
        "Needs network: /download of models and runtime. Note: Studio Pro's own MCP server needs Studio Pro to have internet access.")
    try:
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        cli = socket.create_connection(srv.getsockname(), timeout=2)
        cli.close()
        srv.close()
        add("Network", "Localhost communication", "ok", "loopback TCP works")
    except OSError as exc:
        add("Network", "Localhost communication", "fail", f"loopback TCP failed: {exc} - a local firewall rule blocks 127.0.0.1")

    # ---- MCP
    for name, conn in state.mcp.connections.items():
        if name == "mendix":
            continue
        if conn.connected:
            try:
                add("MCP", name, "ok", f"{conn.target}: {len(conn.tools)} tools, ping {(await conn.ping()) * 1000:.0f} ms")
            except Exception as exc:  # noqa: BLE001
                add("MCP", name, "warn", f"connected but ping failed: {exc}")
        else:
            add("MCP", name, "warn", f"{conn.target}: not connected ({conn.last_error or 'run /mcp connect ' + name})")
    if not [n for n in state.mcp.connections if n != "mendix"]:
        add("MCP", "servers", "info", "no additional MCP servers configured (config/mcp_servers.yaml)")

    # ---- Maia
    if not state.mendix.connected:
        await state.mendix.connect()
    d = state.mendix.last_diagnosis
    if d:
        for step in d.steps:
            add("Mendix Maia MCP", step.name, "ok" if step.ok else "info" if step.ok is None else "warn", step.detail)
        if d.connected:
            caps = state.mendix.capabilities()
            add("Mendix Maia MCP", "capabilities", "ok", f"{len(caps['read'])} read tools, {len(caps['write'])} modifying tools (see /mendix tools)")
        else:
            add("Mendix Maia MCP", "result", "fail", "not connected. " + " ".join(d.hints[:3]))
    if state.mendix.cfg.url:
        host = urlparse(state.mendix.cfg.url).hostname
        if host not in ("localhost", "127.0.0.1", "::1"):
            add("Mendix Maia MCP", "configured url", "fail", f"{host} is not a loopback address; the Studio Pro MCP server is local-only")
    return out


def _mask(url: str) -> str:
    """Hide credentials embedded in proxy URLs."""
    p = urlparse(url)
    if p.username:
        return url.replace(f"{p.username}:{p.password or ''}@", "***:***@")
    return url
