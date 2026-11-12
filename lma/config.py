"""Settings, paths and secret resolution.

All state lives in user-writable directories. Nothing is written system-wide.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(os.environ.get("LMA_CONFIG_DIR", ROOT / "config"))


class LLMSettings(BaseModel):
    chat_model: str | None = None
    vision_model: str | None = None
    embed_model: str | None = None
    runtime_variant: str | None = None  # cpu | vulkan | cuda-12.4 (None = auto)
    runtime_tag: str | None = None  # llama.cpp release tag, None = latest at install time
    context_size: int = 16384
    gpu_layers: int = -1  # -1 = auto (all layers that fit)
    temperature: float = 0.2
    max_output_tokens: int = 2048
    hf_endpoint: str = "https://huggingface.co"
    github_api: str = "https://api.github.com"
    startup_timeout_s: int = 180


class AgentSettings(BaseModel):
    max_steps: int = 25
    max_retries: int = 2
    tool_result_chars: int = 6000
    tool_timeout_s: int = 120
    plan_first: bool = True
    auto_apply_edits: bool = False
    allow_shell: bool = False


class ProjectSettings(BaseModel):
    ignore_dirs: list[str] = Field(
        default_factory=lambda: [
            ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build",
            ".mendix-cache", "deployment", ".idea", ".vs", ".pytest_cache", ".mypy_cache",
        ]
    )
    max_file_bytes: int = 1_000_000
    chunk_lines: int = 60
    chunk_overlap: int = 10
    watch_interval_s: int = 0  # 0 = watcher off


class Settings(BaseModel):
    data_dir: str = str(ROOT / "data")
    models_dir: str | None = None
    project: str | None = None
    mode: str = "agent"
    use_system_trust_store: bool = True
    llm: LLMSettings = Field(default_factory=LLMSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    index: ProjectSettings = Field(default_factory=ProjectSettings)

    # ---- derived paths -------------------------------------------------
    @property
    def data_path(self) -> Path:
        p = Path(os.environ.get("LMA_DATA_DIR", self.data_dir)).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def models_path(self) -> Path:
        p = Path(self.models_dir).expanduser() if self.models_dir else self.data_path / "models"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def runtime_path(self) -> Path:
        p = self.data_path / "runtime"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_path(self) -> Path:
        return self.data_path / "agent.sqlite"


SETTINGS_FILE = CONFIG_DIR / "settings.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_settings() -> Settings:
    return Settings.model_validate(load_yaml(SETTINGS_FILE))


def save_settings(s: Settings) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = s.model_dump(exclude_defaults=True)
    with SETTINGS_FILE.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def set_setting(s: Settings, dotted: str, raw: str) -> Settings:
    """Set `a.b.c` to a YAML-parsed value, validate, persist, return new Settings."""
    data = s.model_dump()
    node = data
    parts = dotted.split(".")
    for p in parts[:-1]:
        if not isinstance(node.get(p), dict):
            raise KeyError(dotted)
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(dotted)
    node[parts[-1]] = yaml.safe_load(raw)
    new = Settings.model_validate(data)
    save_settings(new)
    return new


# ---- secrets -----------------------------------------------------------
KEYRING_SERVICE = "local-mendix-agent"


def resolve_secret(value: str) -> str:
    """Resolve `env:NAME` / `keyring:NAME` references. Plain values pass through.

    Config files should only ever contain references, never the secret itself.
    """
    if value.startswith("env:"):
        name = value[4:]
        if name not in os.environ:
            raise KeyError(f"environment variable {name} is not set")
        return os.environ[name]
    if value.startswith("keyring:"):
        import keyring

        got = keyring.get_password(KEYRING_SERVICE, value[8:])
        if got is None:
            raise KeyError(f"no secret named {value[8:]!r} in the OS credential store (use /config secret {value[8:]})")
        return got
    return value


def store_secret(name: str, secret: str) -> None:
    import keyring

    keyring.set_password(KEYRING_SERVICE, name, secret)


def prepare_network_env(settings: Settings) -> None:
    """Make sure loopback traffic never goes through a corporate proxy and that the
    OS certificate store (which contains corporate root CAs) is trusted.

    This does not bypass anything: localhost is not routable through a proxy at all,
    and the OS trust store is exactly what the company configured.
    """
    for var in ("NO_PROXY", "no_proxy"):
        cur = [x for x in os.environ.get(var, "").split(",") if x]
        for host in ("localhost", "127.0.0.1", "::1"):
            if host not in cur:
                cur.append(host)
        os.environ[var] = ",".join(cur)
    if settings.use_system_trust_store:
        try:
            import truststore

            truststore.inject_into_ssl()
        except Exception:  # noqa: BLE001 - optional hardening only
            pass
