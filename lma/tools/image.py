"""Image loading for the vision model: files, clipboard, metadata, downscaling."""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

MAX_SIDE = 1568  # larger screenshots are downscaled; keeps vision token cost bounded
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

SCREENSHOT_PROMPT = (
    "You are looking at a screenshot, most likely of Mendix Studio Pro (domain model, microflow, page editor, "
    "error list or a dialog) or of a running application. Describe precisely what is visible: window/editor type, "
    "all readable names (entities, attributes, associations, activities, widgets), error or validation messages "
    "verbatim, and anything highlighted or selected. Do not guess text you cannot read; say it is unreadable."
)


@dataclass
class LoadedImage:
    label: str
    data: bytes  # PNG or JPEG bytes ready for the model
    width: int
    height: int
    original_size: tuple[int, int]
    fmt: str
    bytes_on_disk: int

    def meta(self) -> str:
        scaled = "" if (self.width, self.height) == self.original_size else f", sent as {self.width}x{self.height}"
        return f"{self.label}: {self.fmt} {self.original_size[0]}x{self.original_size[1]}{scaled}, {self.bytes_on_disk/1024:.0f} KB"


def _prepare(img: Image.Image, label: str, fmt: str, raw_size: int) -> LoadedImage:
    original = img.size
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return LoadedImage(label, buf.getvalue(), img.size[0], img.size[1], original, fmt, raw_size)


def load_image(path: Path) -> LoadedImage:
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    with Image.open(path) as img:
        img.load()
        return _prepare(img, path.name, img.format or path.suffix.lstrip(".").upper(), path.stat().st_size)


def load_bytes(data: bytes, label: str) -> LoadedImage:
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        return _prepare(img, label, img.format or "?", len(data))


def from_clipboard() -> LoadedImage | None:
    """Image currently on the clipboard (e.g. after Win+Shift+S), or None."""
    try:
        from PIL import ImageGrab

        grabbed = ImageGrab.grabclipboard()
    except Exception:  # noqa: BLE001 - clipboard access can be blocked by policy
        return None
    if isinstance(grabbed, Image.Image):
        buf = io.BytesIO()
        grabbed.save(buf, format="PNG")
        return _prepare(grabbed, "clipboard", "PNG", buf.tell())
    if isinstance(grabbed, list):  # copied file(s) in Explorer
        for item in grabbed:
            p = Path(str(item))
            if p.suffix.lower() in IMAGE_SUFFIXES:
                return load_image(p)
    return None
