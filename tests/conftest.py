from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "tests" / "fixtures" / "demo_server.py"
sys.path.insert(0, str(ROOT))

from lma.config import Settings, prepare_network_env  # noqa: E402

prepare_network_env(Settings())


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_demo_http(port: int) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, str(DEMO), "http", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                return proc
        except OSError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("demo MCP server did not start")


@pytest.fixture
def demo_http():
    port = free_port()
    proc = start_demo_http(port)
    yield f"http://127.0.0.1:{port}/mcp", port
    proc.kill()
    proc.wait()


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Isolated settings: all state goes to a temp directory, config files are not touched."""
    monkeypatch.setattr("lma.config.SETTINGS_FILE", tmp_path / "settings.yaml")
    monkeypatch.setattr("lma.mcpx.manager.MCP_CONFIG_FILE", tmp_path / "mcp_servers.yaml")
    return Settings(data_dir=str(tmp_path / "data"))


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    root = tmp_path / "shop"
    files = {
        "README.md": "# Shop\nOrder service. shop.auth handles login tokens, shop.orders computes totals.\n",
        "shop/__init__.py": "",
        "shop/auth/__init__.py": "",
        "shop/auth/tokens.py": (
            '"""Authentication: password hashing and session tokens."""\nimport hashlib\nimport secrets\n\nSESSIONS: dict[str, str] = {}\n\n\n'
            "def hash_password(password: str, salt: str) -> str:\n    return hashlib.sha256((salt + password).encode()).hexdigest()\n\n\n"
            "def login(user: str, password: str, stored_hash: str, salt: str) -> str | None:\n"
            "    if hash_password(password, salt) != stored_hash:\n        return None\n"
            "    token = secrets.token_hex(16)\n    SESSIONS[token] = user\n    return token\n\n\n"
            "def current_user(token: str) -> str | None:\n    return SESSIONS.get(token)\n"),
        "shop/orders/__init__.py": "",
        "shop/orders/pricing.py": (
            '"""Order pricing."""\nfrom shop.auth.tokens import current_user\n\nTAX_RATE = 0.2\n\n\n'
            "def order_total(prices: list[float], token: str) -> float:\n    if current_user(token) is None:\n"
            '        raise PermissionError("not logged in")\n    return round(sum(prices) * (1 + TAX_RATE), 2)\n\n\n'
            "def apply_discount(total: float, percent: float) -> float:\n    return total - total * percent\n"),
        "shop/orders/api.py": (
            "from shop.orders.pricing import apply_discount, order_total\n\n\n"
            "def checkout(prices, token, discount_percent=0):\n    return {\"total\": apply_discount(order_total(prices, token), discount_percent)}\n"),
        "node_modules/junk/index.js": "export const ignored = 1;\n",
        "web/app.ts": "import { helper } from './util';\nexport class Cart {}\nexport function total() { return helper(); }\n",
        "web/util.ts": "export const helper = () => 1;\n",
    }
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    (root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\0\0\0\0")
    return root


def maia_url() -> str | None:
    return os.environ.get("LMA_MAIA_URL")
