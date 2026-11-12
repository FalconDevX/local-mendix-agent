"""Hardware detection without admin rights or extra drivers."""
from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil


@dataclass
class Hardware:
    os: str
    python: str
    cpu: str
    cores_physical: int
    cores_logical: int
    ram_total_gb: float
    ram_free_gb: float
    gpu_name: str | None
    vram_total_gb: float
    vram_free_gb: float
    cuda_driver: str | None  # CUDA version supported by the installed NVIDIA driver
    disk_free_gb: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _nvidia() -> tuple[str | None, float, float, str | None]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, 0.0, 0.0, None
    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, creationflags=flags,
        ).stdout.strip().splitlines()
        if not out:
            return None, 0.0, 0.0, None
        name, total, free = [x.strip() for x in out[0].split(",")]
        head = subprocess.run([exe], capture_output=True, text=True, timeout=10, creationflags=flags).stdout
        m = re.search(r"CUDA(?: UMD)? Version:\s*([\d.]+)", head)
        cuda = m.group(1) if m else None
        return name, float(total) / 1024, float(free) / 1024, cuda
    except (OSError, ValueError, subprocess.SubprocessError):
        return None, 0.0, 0.0, None


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def detect(storage: Path) -> Hardware:
    vm = psutil.virtual_memory()
    gpu, vt, vf, cuda = _nvidia()
    return Hardware(
        os=f"{platform.system()} {platform.release()} ({platform.version()})",
        python=platform.python_version(),
        cpu=_cpu_name(),
        cores_physical=psutil.cpu_count(logical=False) or 1,
        cores_logical=psutil.cpu_count() or 1,
        ram_total_gb=round(vm.total / 2**30, 1),
        ram_free_gb=round(vm.available / 2**30, 1),
        gpu_name=gpu,
        vram_total_gb=round(vt, 1),
        vram_free_gb=round(vf, 1),
        cuda_driver=cuda,
        disk_free_gb=round(shutil.disk_usage(storage).free / 2**30, 1),
    )


def recommend_runtime(hw: Hardware) -> tuple[str, str]:
    """Pick a llama.cpp Windows build. Returns (variant, reason)."""
    if hw.gpu_name and hw.cuda_driver:
        try:
            major, minor = (int(x) for x in hw.cuda_driver.split(".")[:2])
        except ValueError:
            major, minor = 0, 0
        if (major, minor) >= (12, 4) and hw.disk_free_gb >= 3:
            return "cuda-12.4", f"{hw.gpu_name}, driver supports CUDA {hw.cuda_driver} (~680 MB download incl. CUDA runtime DLLs)"
        return "vulkan", "NVIDIA GPU present, but CUDA 12.4 build not usable (old driver or low disk); Vulkan build is small and GPU accelerated"
    if hw.gpu_name:
        return "vulkan", "GPU present without CUDA information"
    return "cpu", "no supported GPU detected"
