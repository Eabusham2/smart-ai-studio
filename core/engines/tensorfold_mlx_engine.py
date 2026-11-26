"""Own a private TensorFold process serving the selected, local MLX text checkpoint."""
from __future__ import annotations

from contextlib import contextmanager
import http.client
import json
import math
import os
from pathlib import Path
import secrets
import socket
import subprocess
import tempfile
import threading
import time

from core.controller_runtime import UniversalControllerBackend
from core.cross_platform_mlx import tensorfold_executable
from core.engines.prism_speculative_engine import NativeContextOverflow
from core.engines.tensorfold_protocol import NativeServer, completion_events
from core.generation_metrics import TokenMeter
from core.hf_downloader import resolve_cached_model_path
from core.tool_boundary_stop import ToolBoundaryStopper


class _Cancellation:
    def __init__(self, lifetime, request, external):
        self.lifetime, self.request, self.external = lifetime, request, external

    def is_set(self):
        return self.lifetime.is_set() or self.request.is_set() or bool(self.external and self.external())


def _native_backend(executable, requested):
    """XPU uses the XPU binary's automatic backend. HIP's serve switch is 'rocm'."""
    run = subprocess.run([executable, "capabilities", "--json"], stdin=subprocess.DEVNULL,
                         capture_output=True, timeout=15, check=True)
    info = json.loads(run.stdout)
    if not isinstance(info, dict) or info.get("engine") != "zig":
        raise RuntimeError("Select the TensorFold native executable that provides capabilities --json.")
    choices = info.get("serve", {}).get("flags", {}).get("--backend", {}).get("values", [])
    devices = info.get("backends") or []
    modes = {"auto": ("auto", None), "cuda": ("cuda", "cuda"),
             "rocm": ("rocm", "hip"), "hip": ("rocm", "hip"), "xpu": ("auto", "xpu")}
    if requested not in modes:
        raise ValueError("TensorFold backend must be auto, cuda, rocm, hip or xpu.")
    flag, device = modes[requested]
    if flag not in choices or (device and device not in devices):
        raise RuntimeError(f"TensorFold binary cannot serve on {requested}. "
                           f"Compiled backends: {devices}. Supported --backend values: {choices}.")
    return flag, info


class TensorFoldMLXBackend(UniversalControllerBackend):
    runtime = "tensorfold_mlx"
    supports_learning_modes = False
    supports_entropy = False
    entropy_kind = "unavailable"

    def __init__(self, model_path, model_info):
        super().__init__(model_path, {**model_info, "controller_runtime": self.runtime})
        self.input_modalities = {"text"}
        self.can_train = False
        self.memory_placement = {"inference_only": True}
        self.last_error = ""
        self.last_generation = {}
        self.last_branch_usage = {}
        self._process = self._server = self._log = None
        self.loaded_snapshot = None
        self._lifetime = threading.Event()
        self._active_request = None
        self._runtime_model_lock = threading.RLock()
        self._request_lock = threading.RLock()
        self.context_tokens = int(model_info.get("default_context") or 8192)
        self.n_ctx = self.context_tokens

    def _load_tokenizer(self, source):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=False)
        if not getattr(tokenizer, "chat_template", None):
            raise RuntimeError("The selected checkpoint must include its own chat template.")
        return tokenizer

    def _source(self):
        path = resolve_cached_model_path(self.model_info, explicit_path=self.model_path)
        if path:
            source = Path(path).resolve()
            if (source.is_dir() and (source / "config.json").is_file()
                    and any(p.is_file() and p.stat().st_size for p in source.glob("*.safetensors"))):
                return source
        raise RuntimeError("Download the complete selected MLX safetensors snapshot before loading it.")

    def _command(self, executable, source, server, backend):
        return [executable, "serve", str(source), "--host", "127.0.0.1", "--port", str(server.port),
                "--name", server.model_name, "--parallel", "1",
                "--context", str(self.n_ctx), "--no-drafts", "--drafter", "none", "--backend", backend]

    def load_model(self, progress=None, cancelled=None, context_tokens=None):
        self.cancel_current_request()
        with self._runtime_model_lock:
            self.unload_model()
            self.last_error = ""
            self._lifetime = threading.Event()
            cancel = _Cancellation(self._lifetime, threading.Event(), cancelled)
            self._active_request = cancel.request
            try:
                if cancel.is_set():
                    raise RuntimeError("Model load cancelled.")
                executable = tensorfold_executable()
                if not executable:
                    raise RuntimeError("Install TensorFold native or set SMARTAI_TENSORFOLD_EXECUTABLE.")
                source = self._source()
                requested = self.model_info.get("tensorfold_backend") or os.getenv("SMARTAI_TENSORFOLD_BACKEND", "auto")
                backend, capabilities = _native_backend(executable, str(requested).strip().lower())
                tokenizer = self._load_tokenizer(source)
                self.n_ctx = int(self.context_tokens if context_tokens is None else context_tokens)
                if self.n_ctx < 1:
                    raise ValueError("Context must be positive.")
                if cancel.is_set():
                    raise RuntimeError("Model load cancelled.")
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                server = NativeServer(port, "smartai-" + secrets.token_hex(12), secrets.token_hex(32))
                self._server = server
                self._log = tempfile.TemporaryFile(mode="w+b")
                self._process = subprocess.Popen(self._command(executable, source, server, backend),
                    stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT,
                    cwd=str(Path(executable).resolve().parent),
                    env={**os.environ, "TENSORFOLD_API_KEY": server.key},
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if progress:
                    progress("Starting TensorFold with the selected MLX checkpoint…")
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    if cancel.is_set():
                        raise RuntimeError("Model load cancelled.")
                    if self._process.poll() is not None:
                        raise RuntimeError("TensorFold exited while loading this checkpoint and GPU.")
                    try:
                        card = server.model_card(cancel)
                    except (OSError, http.client.HTTPException):
                        self._lifetime.wait(.15)
                        continue
                    if cancel.is_set():
                        raise RuntimeError("Model load cancelled.")
                    reported = card.get("max_model_len")
                    if reported is not None:
                        if type(reported) is not int or reported < 1:
                            raise RuntimeError("TensorFold reported an invalid context length.")
                        self.n_ctx = min(self.n_ctx, reported)
                    self.context_tokens = self.n_ctx
                    self.tokenizer, self.model = tokenizer, server
                    self.loaded_snapshot = str(source)
                    self.memory_placement = {"inference_only": True, "snapshot": str(source),
                                             "native_version": capabilities.get("version"),
                                             "compiled_backends": capabilities.get("backends", [])}
                    self.is_loaded = True
                    return True
                raise RuntimeError("TensorFold model startup timed out.")
            except Exception as exc:
                self.last_error = str(exc)
                if self._log is not None:
                    self._log.seek(0, os.SEEK_END)
                    self._log.seek(max(0, self._log.tell() - 2048))
                    tail = self._log.read().decode("utf-8", "replace").strip()
                    if tail:
                        self.last_error += ": " + tail.replace(self._server.key, "[redacted]")
                self.unload_model()
                return False
            finally:
                self._active_request = None

    @contextmanager
    def _request_scope(self, cancelled=None):
        # A suspended generator owns its own endpoint and cancellation event.
        # Unload can reap the child even if the consumer never resumes iteration.
        with self._request_lock:
            with self._runtime_model_lock:
                if not self.is_loaded or self._process is None or self._process.poll() is not None:
                    raise RuntimeError("The selected TensorFold model is not loaded.")
                token = _Cancellation(self._lifetime, threading.Event(), cancelled)
                self._active_request = token.request
                server = self._server
                tokenizer = self.tokenizer
                context = self.n_ctx
                budget = getattr(self, "context_budget_tokens", None)
                if type(budget) is int and budget > 0:
                    context = min(budget, context)
            try:
                yield server, token, context, tokenizer
            finally:
                if self._active_request is token.request:
                    self._active_request = None

    @staticmethod
    def _token_ids(server, tokenizer, prompt, cancel):
        ids = server.tokenize(prompt, cancel)
        snapshot_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if ids != list(snapshot_ids):
            raise RuntimeError("TensorFold and the selected checkpoint disagree on the prompt token IDs.")
        return ids

    def api_prompt_count(self, prompt, speedups=None):
        with self._request_scope() as (server, cancel, _context, tokenizer):
            return len(self._token_ids(server, tokenizer, prompt, cancel))

    def count_tokens(self, messages):
        if isinstance(messages, str):
            return self.api_prompt_count(messages)
        return self.api_prompt_count("\n".join(str(m.get("content", "")) for m in messages or []))

    def _stream(self, server, cancel, context, tokenizer, prompt, max_tokens, temperature, top_p, stop_at_tool_boundary):
        self.last_generation = {"status": "failed", "usage": {}}
        meter = None
        try:
            if cancel.is_set():
                return
            if type(max_tokens) is not int or max_tokens < 1:
                raise ValueError("Maximum output tokens must be a positive integer.")
            temperature, top_p = float(temperature), float(top_p)
            if not math.isfinite(temperature) or temperature < 0 or not 0 <= top_p <= 1:
                raise ValueError("TensorFold needs a finite nonnegative temperature and top_p between 0 and 1.")
            ids = self._token_ids(server, tokenizer, prompt, cancel)
            if not ids:
                raise ValueError("TensorFold requires at least one prompt token.")
            if len(ids) >= context:
                raise NativeContextOverflow("The prompt fills the selected context. No input was truncated.")
            if cancel.is_set():
                return
            maximum = min(max_tokens, context - len(ids))
            payload = {"model": server.model_name, "prompt": ids, "max_tokens": maximum,
                       "temperature": temperature, "top_p": top_p, "stream": True}
            meter = TokenMeter(self, "tensorfold-mlx", prompt_tokens=len(ids))
            stopper = ToolBoundaryStopper() if stop_at_tool_boundary else None
            usage = {}
            finish_reason = None
            with server.response("/v1/completions", payload, cancelled=cancel, stream=True, timeout=1800) as resp:
                for piece, counts, reason in completion_events(resp, server.model_name, len(ids)):
                    if cancel.is_set():
                        return
                    if counts is not None:
                        usage = counts
                        meter.observe(count=counts["completion_tokens"])
                    if reason:
                        finish_reason = reason
                    if not piece:
                        continue
                    meter.observe(text=True)
                    stop = False
                    if stopper:
                        offset = len(stopper.text)
                        stop = stopper.feed(piece)
                        piece = stopper.text[offset:]
                    if stop:
                        self.last_generation = {"status": "completed", "stop_type": "tool_boundary", "usage": {}}
                    if piece:
                        yield piece
                    if stop:
                        return
            self.last_generation = {"status": "completed", "stop_type": finish_reason, "usage": usage}
        except (OSError, http.client.HTTPException):
            if not cancel.is_set():
                raise
        finally:
            if cancel.is_set():
                self.last_generation = {"status": "cancelled", "usage": {}}
            if meter:
                meter.finish()

    def stream_generate_tokens(self, prompt, max_tokens=1536, temperature=.65, top_p=.92,
                               stop_at_tool_boundary=False, cancelled=None):
        with self._request_scope(cancelled) as scope:
            yield from self._stream(*scope, prompt, max_tokens, temperature, top_p, stop_at_tool_boundary)

    def _generate(self, prompt, max_tokens, temperature, media=None):
        if media:
            raise RuntimeError("This TensorFold adapter is text-only.")
        return "".join(self.stream_generate_tokens(prompt, max_tokens, temperature))

    def generate_branches(self, prompt, branch_count=1, max_tokens=1536, temperature=.65,
                          top_p=.92, progress_callback=None, stop_at_tool_boundary=False, cancelled=None):
        if type(branch_count) is not int or branch_count < 1:
            raise ValueError("TensorFold branch_count must be a positive integer.")
        temperatures = temperature if isinstance(temperature, (list, tuple)) else [temperature]
        if not temperatures:
            raise ValueError("Supply at least one temperature.")
        outputs, usages = [], []
        with self._request_scope(cancelled) as scope:
            self.last_branch_usage = {}
            for index in range(branch_count):
                started = time.perf_counter()
                t = temperatures[index % len(temperatures)]
                pieces = []
                stream = self._stream(*scope, prompt, max_tokens, t, top_p, stop_at_tool_boundary)
                try:
                    for piece in stream:
                        pieces.append(piece)
                        if progress_callback:
                            progress_callback(index, branch_count, self.last_generation_tokens,
                                              time.perf_counter() - started, finished=False)
                finally:
                    stream.close()
                if scope[1].is_set() or self.last_generation.get("status") != "completed":
                    break
                outputs.append("".join(pieces))
                usages.append(self.last_generation["usage"])
                if progress_callback:
                    progress_callback(index + 1, branch_count, self.last_generation_tokens, time.perf_counter() - started)
            if len(usages) == branch_count and all(usages):
                self.last_branch_usage = {key: sum(row[key] for row in usages) for key in usages[0]}
        return outputs

    def cancel_current_request(self):
        request = self._active_request
        if request is not None:
            request.set()
        return True

    cancel_generation = cancel_current_request

    def owned_process_ids(self):
        process = self._process
        return [process.pid] if process is not None and process.poll() is None else []

    def training_ready(self):
        return False

    def train_mini_batch(self, *args, **kwargs):
        raise RuntimeError("TensorFold is inference-only in Studio.")

    def score_learning_probe(self, *args, **kwargs):
        raise RuntimeError("TensorFold is inference-only in Studio.")

    def calculate_token_entropy(self, prompt):
        raise RuntimeError("TensorFold does not supply full-vocabulary entropy to Studio.")

    def supports_media_input(self, kind):
        return False

    def review_media_input(self, path, kind, prompt=""):
        return {"perception_available": False, "reason": "TensorFold integration is text-only."}

    def configure_auto_speed(self, enabled=True, **kwargs):
        return {"status": "unavailable", "message": "TensorFold speed controls are unavailable in Studio."}

    def unload_model(self):
        self._lifetime.set()
        self.cancel_current_request()
        with self._runtime_model_lock:
            self._lifetime.set()
            self.is_loaded = False
            self.model = self.tokenizer = self.loaded_snapshot = None
            self.memory_placement = {"inference_only": True}
            process = self._process
            # Keep the handle and log until exit is confirmed, including when a
            # Windows driver delays TerminateProcess or the final wait times out.
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            self._process = self._server = None
            if self._log is not None:
                self._log.close()
                self._log = None
