"""Local inference: llama-server processes (one per role) and an async OpenAI-compatible client.

Everything listens on 127.0.0.1 only. There is no cloud fallback anywhere in this module.
"""
from __future__ import annotations

import asyncio
import atexit
import base64
import json
import re
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from lma.config import Settings, save_settings
from lma.llm import winjob
from lma.llm.hardware import Hardware, detect
from lma.llm.registry import Installed, ModelRegistry, Role
from lma.llm.runtime import MANUAL_HINT, find_server


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""
    parse_error: str | None = None


@dataclass
class ChatResult:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    tokens_per_s: float = 0.0


SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "repeat_penalty", "presence_penalty")


def sampling_params(llm: Any, spec: Any) -> dict[str, Any]:
    """Global settings, then the model card's recommendation, then the user's per-model override."""
    out: dict[str, Any] = {"temperature": llm.temperature}
    for key in ("top_p", "top_k", "min_p"):
        if getattr(llm, key) is not None:
            out[key] = getattr(llm, key)
    if llm.repeat_penalty != 1.0:
        out["repeat_penalty"] = llm.repeat_penalty
    out.update({k: v for k, v in (spec.sampling or {}).items() if k in SAMPLING_KEYS})
    out.update({k: v for k, v in (llm.sampling.get(spec.id) or {}).items() if k in SAMPLING_KEYS})
    if "top_k" in out:
        out["top_k"] = int(out["top_k"])
    return out


class LoopGuard:
    """Detects degenerate output: the last chunk of text already occurred many times just before."""

    def __init__(self, window: int = 4000, probe: int = 48, repeats: int = 6) -> None:
        self.window, self.probe, self.repeats = window, probe, repeats
        self._checked = 0

    def looping(self, text: str) -> bool:
        if len(text) - self._checked < 200 or len(text) < self.probe * self.repeats:
            return False
        self._checked = len(text)
        tail = text[-self.probe:]
        return len(tail.strip()) > 8 and text[-self.window:].count(tail) >= self.repeats

    def trim(self, text: str) -> str:
        first = text.find(text[-self.probe:])
        return text[: first + self.probe] if first >= 0 else text


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 3)


_WRAPPERS = re.compile(r"</?(?:tool_call|tools|tool|function_call)>|```(?:json)?", re.I)
_DECODER = json.JSONDecoder()


def _as_call(obj: Any, idx: int, known: set[str] | None) -> ToolCall | None:
    if isinstance(obj, dict) and isinstance(obj.get("function"), dict):
        obj = obj["function"]
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or "description" in obj:
        return None  # a tool *definition* echoed back by the model is not a call
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, dict) and args.get("type") == "object" and "properties" in args:
        return None  # JSON schema, not arguments
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict) or (known is not None and obj["name"] not in known):
        return None
    return ToolCall(f"text_{idx}", obj["name"], args, json.dumps(args))


def extract_text_tool_calls(text: str, known: set[str] | None = None) -> tuple[str, list[ToolCall]]:
    """Fallback for models that write tool calls as text instead of structured tool_calls.

    Small models use many spellings (<tool_call>{..}</tool_call>, <tools>{..}</tools>, fenced JSON,
    {"type":"function","function":{..}}). Every top-level JSON object in the text is examined and
    accepted only if it names a tool that really exists (`known`).
    """
    calls: list[ToolCall] = []
    spans: list[tuple[int, int]] = []
    i = 0
    while (start := text.find("{", i)) != -1:
        try:
            obj, end = _DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            i = start + 1
            continue
        call = _as_call(obj, len(calls), known)
        if call:
            calls.append(call)
            spans.append((start, end))
        i = end
    if not calls:
        return text, []
    rest = text
    for start, end in reversed(spans):
        rest = rest[:start] + rest[end:]
    return _WRAPPERS.sub("", rest).strip(), calls


class LlamaServer:
    def __init__(self, exe: Path, inst: Installed, role: Role, ctx: int, gpu_layers: int, log_dir: Path, timeout: int) -> None:
        self.exe, self.inst, self.role, self.ctx = exe, inst, role, ctx
        self.gpu_layers, self.timeout = gpu_layers, timeout
        self.port = free_port()
        self.log_file = log_dir / f"llama-{role}.log"
        self.proc: subprocess.Popen[bytes] | None = None
        self.last_used = 0.0

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def command(self) -> list[str]:
        cmd = [str(self.exe), "-m", self.inst.path, "--host", "127.0.0.1", "--port", str(self.port), "--no-webui"]
        if self.role == "embed":
            cmd += ["--embedding", "-ngl", "0", "-c", str(min(self.ctx, self.inst.spec.ctx_max)), "-b", "2048", "-ub", "2048"]
        else:
            cmd += ["-c", str(min(self.ctx, self.inst.spec.ctx_max)), "--jinja"]
            if self.gpu_layers >= 0:
                cmd += ["-ngl", str(self.gpu_layers)]
            if self.inst.mmproj_path:
                cmd += ["--mmproj", self.inst.mmproj_path]
        return cmd

    async def start(self) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_file.open("wb")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            self.proc = subprocess.Popen(self.command(), stdout=log, stderr=subprocess.STDOUT, creationflags=flags,
                                         cwd=str(self.exe.parent))
        except OSError as exc:
            raise LLMError(
                f"cannot start {self.exe}: {exc}. If this is blocked by application control (AppLocker/WDAC), "
                "ask IT to allow llama-server.exe from this folder or point LMA_LLAMA_SERVER at an approved copy.") from exc
        finally:
            log.close()
        winjob.attach(int(getattr(self.proc, "_handle", 0)))  # the server dies with us, even on a crash
        deadline = time.monotonic() + self.timeout
        async with httpx.AsyncClient(timeout=3) as c:
            while time.monotonic() < deadline:
                if not self.alive:
                    raise LLMError(f"llama-server exited with code {self.proc.returncode} while loading "
                                   f"{Path(self.inst.path).name}.\n{self.log_tail()}")
                try:
                    if (await c.get(self.base + "/health")).status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.5)
        self.stop()
        raise LLMError(f"llama-server did not become ready within {self.timeout}s.\n{self.log_tail()}")

    def log_tail(self, lines: int = 12) -> str:
        try:
            text = self.log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        tail = "\n".join("    " + line for line in text[-lines:])
        hint = ""
        joined = "\n".join(text[-60:]).lower()
        if "out of memory" in joined or "failed to allocate" in joined:
            hint = "\n  -> Out of memory: pick a smaller model, lower llm.context_size, or set llm.gpu_layers to a smaller number."
        elif "dll" in joined:
            hint = "\n  -> A DLL is missing: install the `cpu` or `vulkan` runtime variant instead (/download runtime vulkan)."
        return f"  last log lines ({self.log_file}):\n{tail}{hint}"

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


class LLMManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.registry = ModelRegistry(settings)
        self.servers: dict[str, LlamaServer] = {}
        self.hardware: Hardware = detect(settings.data_path)
        self.last_usage: ChatResult | None = None
        self._lock = asyncio.Lock()
        atexit.register(self.shutdown)

    # ---- model selection ----------------------------------------------
    def model_id(self, role: Role) -> str | None:
        configured = {"chat": self.settings.llm.chat_model, "vision": self.settings.llm.vision_model,
                      "embed": self.settings.llm.embed_model}[role]
        if configured and configured in self.registry.installed:
            return configured
        installed = self.registry.by_role(role, installed_only=True)
        if role == "vision":
            installed = [m for m in installed if m.vision]
        elif role == "chat":  # prefer real chat models over vision models that can also chat
            installed.sort(key=lambda m: m.role != "chat")
        return installed[0].id if installed else None

    def set_model(self, role: Role, model_id: str) -> None:
        if model_id not in self.registry.installed:
            raise LLMError(f"{model_id!r} is not installed; run /download {model_id}")
        spec = self.registry.installed[model_id].spec
        if role == "vision" and not spec.vision:
            raise LLMError(f"{model_id!r} has no vision projector and cannot read images")
        if role == "embed" and spec.role != "embed":
            raise LLMError(f"{model_id!r} is not an embedding model")
        setattr(self.settings.llm, f"{role}_model", model_id)
        save_settings(self.settings)
        self.stop(role)

    def chat_supports_vision(self) -> bool:
        mid = self.model_id("chat")
        return bool(mid and self.registry.installed[mid].spec.vision)

    def status(self) -> dict[str, str]:
        out = {}
        for role in ("chat", "vision", "embed"):
            mid = self.model_id(role)  # type: ignore[arg-type]
            srv = self.servers.get(role)
            out[role] = f"{mid or 'none'}{' (loaded)' if srv and srv.alive else ''}"
        return out

    def chat_models(self) -> list[str]:
        """Installed models that can hold a conversation (chat models first)."""
        specs = sorted(self.registry.by_role("chat", installed_only=True), key=lambda m: (m.role != "chat", m.size_gb))
        return [m.id for m in specs]

    # ---- server lifecycle ---------------------------------------------
    async def ensure(self, role: Role, model: str | None = None) -> LlamaServer:
        """Running server for `role`. `model` overrides the configured model for this one call
        (sub-agents may use their own model); such servers live in a slot named `role:model`."""
        async with self._lock:
            default = self.model_id(role)
            if model and model not in self.registry.installed:
                raise LLMError(f"model {model!r} is not installed; run /download {model} (or /agents model <role> default)")
            mid = model or default
            if mid is None:
                raise LLMError(f"no {role} model installed. Run /models to see recommendations and /download <id>.")
            slot = role if mid == default else f"{role}:{mid}"
            srv = self.servers.get(slot)
            if srv and srv.alive and srv.inst.spec.id == mid:
                srv.last_used = time.monotonic()
                return srv
            if srv:
                self.servers.pop(slot, None)
                if srv not in self.servers.values():
                    srv.stop()
            # one loaded model can serve several slots (chat + vision, or a sub-agent using the chat model)
            for other_slot, other in self.servers.items():
                if other.alive and other.inst.spec.id == mid and other.role != "embed" and role != "embed":
                    self.servers[slot] = other
                    other.last_used = time.monotonic()
                    return other
            exe = find_server(self.settings)
            if exe is None:
                raise LLMError("llama.cpp runtime is not installed. Run /download runtime." + MANUAL_HINT)
            inst = self.registry.installed[mid]
            if role in ("chat", "vision"):
                self._make_room(inst)
            srv = LlamaServer(exe, inst, role, self.settings.llm.context_size, self.settings.llm.gpu_layers,
                              self.settings.data_path / "logs", self.settings.llm.startup_timeout_s)
            srv.log_file = srv.log_file.with_name(f"llama-{slot.replace(':', '-')}.log")
            await srv.start()
            srv.last_used = time.monotonic()
            self.servers[slot] = srv
            return srv

    def _make_room(self, inst: Installed) -> None:
        """Several chat/vision models rarely fit in memory together: unload the least recently used first."""
        budget = self.hardware.vram_total_gb or self.hardware.ram_total_gb / 2
        loaded = {id(s): s for s in self.servers.values() if s.alive and s.role != "embed"}
        for victim in sorted(loaded.values(), key=lambda s: s.last_used):
            if sum(s.inst.spec.size_gb for s in loaded.values()) + inst.spec.size_gb + 2 <= budget:
                return
            victim.stop()
            del loaded[id(victim)]
            for slot in [k for k, v in self.servers.items() if v is victim]:
                del self.servers[slot]

    def loaded(self) -> list[str]:
        """Ids of the models that are in memory right now."""
        return sorted({s.inst.spec.id for s in self.servers.values() if s.alive})

    def stop(self, role: str | None = None) -> None:
        slots = [s for s in self.servers if s == role or s.startswith(f"{role}:")] if role else list(self.servers)
        for r in slots:
            srv = self.servers.pop(r, None)
            if srv and srv not in self.servers.values():
                srv.stop()

    def shutdown(self) -> None:
        for srv in list(self.servers.values()):
            srv.stop()
        self.servers.clear()

    # ---- inference -----------------------------------------------------
    async def chat_stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, role: Role = "chat",
        max_tokens: int | None = None, temperature: float | None = None, model: str | None = None,
    ) -> AsyncIterator[tuple[str, Any]]:
        """Yield ("text", str) / ("reasoning", str) deltas and finally ("done", ChatResult)."""
        srv = await self.ensure(role, model)
        body: dict[str, Any] = {
            "messages": messages, "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": max_tokens or self.settings.llm.max_output_tokens,
            **sampling_params(self.settings.llm, srv.inst.spec),
        }
        if temperature is not None:
            body["temperature"] = temperature
        guard = LoopGuard() if self.settings.llm.loop_guard else None
        if tools:
            body["tools"] = tools
        result = ChatResult()
        partial: dict[int, dict[str, str]] = {}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=600)) as client:
                async with client.stream("POST", srv.base + "/v1/chat/completions", json=body) as resp:
                    if resp.status_code != 200:
                        detail = (await resp.aread()).decode("utf-8", "replace")[:600]
                        if "context" in detail.lower() and "exceed" in detail.lower():
                            raise LLMError(f"context window exceeded ({detail})")
                        raise LLMError(f"llama-server returned HTTP {resp.status_code}: {detail}")
                    async for line in resp.aiter_lines():
                        if result.finish_reason == "loop":
                            break  # closing the stream makes llama-server stop generating
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        if usage := chunk.get("usage"):
                            result.prompt_tokens = usage.get("prompt_tokens", 0)
                            result.completion_tokens = usage.get("completion_tokens", 0)
                        if timings := chunk.get("timings"):
                            result.tokens_per_s = float(timings.get("predicted_per_second") or 0)
                        for choice in chunk.get("choices", []):
                            delta = choice.get("delta") or {}
                            if delta.get("reasoning_content"):
                                yield "reasoning", delta["reasoning_content"]
                            if delta.get("content"):
                                result.text += delta["content"]
                                yield "text", delta["content"]
                                if guard and guard.looping(result.text):
                                    result.finish_reason = "loop"
                                    result.text = guard.trim(result.text) + "\n\n[stopped: the model kept repeating itself]"
                                    yield "text", "\n\n[stopped: the model kept repeating itself]"
                                    break
                            for tc in delta.get("tool_calls") or []:
                                slot = partial.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                                slot["id"] = tc.get("id") or slot["id"]
                                fn = tc.get("function") or {}
                                slot["name"] += fn.get("name") or ""
                                slot["args"] += fn.get("arguments") or ""
                            if choice.get("finish_reason"):
                                result.finish_reason = choice["finish_reason"]
        except httpx.HTTPError as exc:
            if not srv.alive:
                raise LLMError(f"llama-server crashed during generation.\n{srv.log_tail()}") from exc
            raise LLMError(f"local inference request failed: {type(exc).__name__}: {exc}") from exc
        for idx in sorted(partial):
            slot = partial[idx]
            try:
                args = json.loads(slot["args"]) if slot["args"].strip() else {}
                err = None if isinstance(args, dict) else "arguments are not a JSON object"
            except json.JSONDecodeError as exc:
                args, err = {}, f"invalid JSON arguments: {exc}"
            result.tool_calls.append(ToolCall(slot["id"] or f"call_{idx}", slot["name"], args if isinstance(args, dict) else {}, slot["args"], err))
        if not result.tool_calls and tools and result.text:
            known = {t["function"]["name"] for t in tools}
            result.text, result.tool_calls = extract_text_tool_calls(result.text, known)
        self.last_usage = result
        yield "done", result

    async def chat(self, messages: list[dict[str, Any]], **kw: Any) -> ChatResult:
        final = ChatResult()
        async for kind, payload in self.chat_stream(messages, **kw):
            if kind == "done":
                final = payload
        return final

    async def describe_images(self, prompt: str, images: list[bytes], max_tokens: int = 900) -> str:
        """Run the vision model on images (PNG/JPEG bytes) and return its text answer."""
        mid = self.model_id("vision")
        if mid is None:
            raise LLMError("no vision model installed. Run /download qwen3-vl-4b (or another vision model from /models).")
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            mime = "image/jpeg" if img[:3] == b"\xff\xd8\xff" else "image/png"
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(img).decode()}"}})
        res = await self.chat([{"role": "user", "content": content}], role="vision", max_tokens=max_tokens, temperature=0.1)
        return res.text.strip()

    async def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        srv = await self.ensure("embed")
        spec = srv.inst.spec
        prefix = spec.query_prefix if is_query else spec.doc_prefix
        limit = spec.ctx_max * 2  # chars; conservative so no chunk exceeds the embedding context
        out: list[list[float]] = []
        async with httpx.AsyncClient(timeout=300) as client:
            for i in range(0, len(texts), 16):
                batch = [prefix + t[:limit] for t in texts[i:i + 16]]
                resp = await client.post(srv.base + "/v1/embeddings", json={"input": batch})
                if resp.status_code != 200:
                    raise LLMError(f"embedding request failed: HTTP {resp.status_code} {resp.text[:300]}")
                data = sorted(resp.json()["data"], key=lambda d: d["index"])
                out.extend(d["embedding"] for d in data)
        return out

    def has(self, role: Role) -> bool:
        return self.model_id(role) is not None and find_server(self.settings) is not None
