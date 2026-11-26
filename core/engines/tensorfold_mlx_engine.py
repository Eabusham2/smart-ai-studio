"""Owned native TensorFold sidecar for a local MLX-format text checkpoint.

Raw completions preserve the desktop/API harness's own chat template. The
native server is private to this process and is never used as a remote API.
Training and unsupported model families are deliberately not advertised.
"""
from __future__ import annotations

from functools import partial
import http.client
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import tempfile
import threading
import time

from core.controller_runtime import UniversalControllerBackend, resolve_local_snapshot
from core.cross_platform_mlx import tensorfold_executable
from core.engines.prism_speculative_engine import _CancellableHTTPResponse
from core.generation_metrics import TokenMeter
from core.tool_boundary_stop import ToolBoundaryStopper


class _Cancellation:
    def __init__(self, event, external=None):
        self.event, self.external = event, external

    def is_set(self):
        return self.event.is_set() or bool(self.external and self.external())


class TensorFoldMLXBackend(UniversalControllerBackend):
    runtime = "tensorfold_mlx"
    supports_learning_modes = False
    supports_entropy = False

    def __init__(self, model_path, model_info):
        super().__init__(model_path, {**model_info, "controller_runtime": self.runtime})
        self.input_modalities = {"text"}
        self.can_train = False
        self.memory_placement = {"inference_only": True}
        self.last_error = ""
        self._process = None
        self._port = None
        self._key = None
        self._model_name = None
        self._log = None
        self._cancel = threading.Event()
        self._runtime_model_lock = threading.RLock()
        self.context_tokens = int(model_info.get("default_context") or 8192)
        self.n_ctx = self.context_tokens

    def _connection(self, timeout=1800, cancelled=None):
        connection = http.client.HTTPConnection("127.0.0.1", self._port, timeout=timeout)
        if cancelled is not None:
            connection.response_class = partial(_CancellableHTTPResponse, cancelled=cancelled)
        return connection

    def _headers(self):
        return {"Authorization": "Bearer " + self._key, "Content-Type": "application/json"}

    def _models(self):
        conn = self._connection(timeout=2)
        try:
            conn.request("GET", "/v1/models", headers=self._headers())
            resp = conn.getresponse()
            if resp.status != 200:
                return []
            return [item.get("id") for item in json.loads(resp.read(1024 * 1024)).get("data", [])]
        finally:
            conn.close()

    def _command(self, executable, source):
        return [executable, "serve", str(source), "--host", "127.0.0.1",
                "--port", str(self._port), "--name", self._model_name,
                "--api-key", self._key, "--parallel", "1",
                "--context", str(self.n_ctx), "--no-drafts"]

    def load_model(self, progress=None, cancelled=None, context_tokens=None):
        with self._runtime_model_lock:
            self.unload_model()
            self.last_error = ""
            self._cancel.clear()
            try:
                executable = tensorfold_executable()
                if not executable:
                    raise RuntimeError("Install TensorFold native or set SMARTAI_TENSORFOLD_EXECUTABLE.")
                source = Path(resolve_local_snapshot(
                    self.model_path, self.model_info.get("repo_revision") or self.model_info.get("source_revision"),
                    required_files=self.model_info.get("download_required_files"),
                ))
                if not source.is_dir() or not (source / "config.json").is_file() or not any(source.glob("*.safetensors")):
                    raise RuntimeError("Download the complete selected MLX snapshot before loading it.")
                if cancelled and cancelled():
                    raise RuntimeError("Model load cancelled.")
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=False)
                if not getattr(tokenizer, "chat_template", None):
                    raise RuntimeError("The selected checkpoint must include its own chat template.")
                self.n_ctx = int(context_tokens or self.context_tokens)
                self.context_tokens = self.n_ctx
                if self.n_ctx < 1:
                    raise ValueError("Context must be positive.")
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    self._port = sock.getsockname()[1]
                self._key = secrets.token_hex(32)
                self._model_name = "smartai-" + secrets.token_hex(8)
                self._log = tempfile.TemporaryFile(mode="w+b")
                self._process = subprocess.Popen(self._command(executable, source.resolve()),
                    stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT,
                    cwd=str(Path(executable).resolve().parent),
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if progress:
                    progress("Starting native TensorFold with the selected MLX checkpoint…")
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    if cancelled and cancelled():
                        raise RuntimeError("Model load cancelled.")
                    if self._process.poll() is not None:
                        raise RuntimeError("TensorFold rejected the checkpoint or GPU; check the native runtime and model compatibility.")
                    try:
                        if self._model_name in self._models():
                            self.tokenizer = tokenizer
                            self.model = object()
                            self.is_loaded = True
                            return True
                    except (OSError, ValueError, http.client.HTTPException):
                        pass
                    time.sleep(.25)
                raise RuntimeError("TensorFold model startup timed out.")
            except Exception as exc:
                self.last_error = str(exc)
                if self._log is not None:
                    try:
                        self._log.seek(0, os.SEEK_END)
                        self._log.seek(max(0, self._log.tell() - 2048))
                        tail = self._log.read().decode("utf-8", "replace").strip()
                        if tail:
                            self.last_error += ": " + tail.replace(self._key or "", "[redacted]")
                    except (OSError, ValueError):
                        pass
                self.unload_model()
                return False

    def api_prompt_count(self, prompt, speedups=None):
        return len(self.tokenizer.encode(str(prompt), add_special_tokens=False))

    def stream_generate_tokens(self, prompt, max_tokens=1536, temperature=.65, top_p=.92,
                               stop_at_tool_boundary=False, cancelled=None):
        with self._runtime_model_lock:
            if not self.is_loaded or self._process is None or self._process.poll() is not None:
                raise RuntimeError("The selected TensorFold model is not loaded.")
            self._cancel.clear()
            cancellation = _Cancellation(self._cancel, cancelled)
            if cancellation.is_set():
                return
            count = self.api_prompt_count(prompt)
            if count >= self.n_ctx:
                raise ValueError("The prompt fills the selected Context; no input was truncated.")
            maximum = min(int(max_tokens), self.n_ctx - count)
            if maximum <= 0:
                return
            meter = TokenMeter(self, "tensorfold-mlx")
            stopper = ToolBoundaryStopper() if stop_at_tool_boundary else None
            conn = self._connection(cancelled=cancellation)
            try:
                body = {"model": self._model_name, "prompt": str(prompt), "max_tokens": maximum,
                        "temperature": float(temperature), "top_p": float(top_p), "stream": True}
                conn.request("POST", "/v1/completions", json.dumps(body), self._headers())
                resp = conn.getresponse()
                if resp.status != 200:
                    raise RuntimeError(f"TensorFold rejected generation (HTTP {resp.status}): " +
                                       resp.read(4096).decode("utf-8", "replace"))
                finished = False
                for raw in resp:
                    if cancellation.is_set():
                        return
                    if not raw.startswith(b"data:"):
                        continue
                    data = raw[5:].strip()
                    if data == b"[DONE]":
                        finished = True
                        break
                    event = json.loads(data)
                    if event.get("error"):
                        raise RuntimeError(str(event["error"]))
                    usage = event.get("usage") or {}
                    if usage:
                        meter.observe(count=usage.get("completion_tokens"),
                                      prompt_tokens=usage.get("prompt_tokens"))
                    for choice in event.get("choices") or []:
                        if choice.get("finish_reason"):
                            finished = True
                        piece = choice.get("text") or ""
                        if not piece:
                            continue
                        meter.observe(text=True)
                        stop_here = False
                        if stopper:
                            offset = len(stopper.text)
                            stop_here = stopper.feed(piece)
                            piece = stopper.text[offset:]
                        if piece:
                            yield piece
                        if stop_here:
                            return
                if not finished and not cancellation.is_set():
                    raise RuntimeError("TensorFold stream ended without a completion event.")
            except (OSError, http.client.HTTPException):
                if not cancellation.is_set():
                    raise
            finally:
                conn.close()
                meter.finish()

    def _generate(self, prompt, max_tokens, temperature, media=None):
        if media:
            raise RuntimeError("This TensorFold adapter is text-only.")
        return "".join(self.stream_generate_tokens(prompt, max_tokens, temperature))

    def generate_branches(self, prompt, branch_count=1, max_tokens=1536, temperature=.65,
                          top_p=.92, progress_callback=None, stop_at_tool_boundary=False, cancelled=None):
        outputs = []
        for index in range(max(1, int(branch_count))):
            if cancelled and cancelled():
                break
            t = temperature[index % len(temperature)] if isinstance(temperature, (list, tuple)) else temperature
            started = time.perf_counter()
            text = "".join(self.stream_generate_tokens(prompt, max_tokens, t, top_p,
                            stop_at_tool_boundary=stop_at_tool_boundary, cancelled=cancelled))
            if cancelled and cancelled():
                break
            outputs.append(text)
            if progress_callback:
                progress_callback(index + 1, branch_count, self.last_generation_tokens, time.perf_counter() - started)
        return outputs

    def cancel_current_request(self):
        self._cancel.set()
        return True

    def owned_process_ids(self):
        return [self._process.pid] if self._process and self._process.poll() is None else []

    def training_ready(self):
        return False

    def supports_media_input(self, kind):
        return False

    def review_media_input(self, path, kind, prompt=""):
        return {"perception_available": False, "reason": "TensorFold integration is text-only."}

    def configure_auto_speed(self, enabled=True, **kwargs):
        return {"status": "unavailable", "message": "TensorFold native speed controls are not qualified for Studio yet."}

    def unload_model(self):
        self._cancel.set()
        with self._runtime_model_lock:
            process, self._process = self._process, None
            self.is_loaded = False
            self.model = self.tokenizer = None
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if self._log is not None:
                self._log.close()
                self._log = None
            self._key = None
            self._port = None
