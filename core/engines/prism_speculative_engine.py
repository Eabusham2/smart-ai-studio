"""Owned, isolated llama-server transport for exact Bonsai draft verification.

The user selects both the native executable and the GGUF.  This backend neither
installs a runtime nor downloads/requantizes/replaces a model.  The executable
must carry the build receipt for the pinned, serial-target verifier.  Full-f16
KV, a single slot and a one-row target graph are held fixed in Off and draft
modes.  This conservative exactness path is not a claim of measured speedup.
"""
from __future__ import annotations

import http.client
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from typing import Any, Callable, Dict, Generator, List, Optional

from core.speculative_runtime import (
    EXACTNESS_CONTRACT,
    SpeculativePlan,
    normalize_speculative_mode,
    plan_speculation,
    sha256_file,
    speculative_arguments,
    validate_runtime_receipt,
)


class _NativeModelProxy:
    def __init__(self, backend: "PrismSpeculativeReasoningBackend"):
        self.backend = backend

    def tokenize(self, raw: bytes, **_kwargs) -> List[int]:
        result = self.backend._request("/tokenize", {"content": raw.decode("utf-8", errors="strict"),
                                                    "add_special": True, "parse_special": True})
        tokens = result.get("tokens")
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise RuntimeError("The native tokenizer returned invalid token IDs")
        return tokens

    def encode(self, text: str, *_args, **_kwargs) -> List[int]:
        return self.tokenize(text.encode("utf-8"))

    def n_ctx(self) -> int:
        return self.backend.n_ctx


class PrismSpeculativeReasoningBackend:
    """Text-only native backend; generation and cancellation own one sidecar.

    Initialization is model-free.  ``preflight`` reads local checksums and a
    small receipt; ``load_model`` is the sole method which starts inference.
    Existing Python/MLX backends are never silently switched to this runtime.
    """

    runtime = "prism_speculative"
    supports_entropy = False

    def __init__(
        self, model_path: str, server_path: str, *, mode: str = "auto",
        draft_model_path: Optional[str] = None, runtime_source_dir: Optional[str] = None,
        runtime_receipt_path: Optional[str] = None, n_ctx: int = 32768,
        n_gpu_layers: int = -1, n_threads: Optional[int] = None,
        n_threads_batch: Optional[int] = None, max_draft_tokens: Optional[int] = None,
        mmproj_path: Optional[str] = None, adapter_root: Optional[str] = None,
        sampling_defaults: Optional[Dict[str, Any]] = None, **_kwargs,
    ):
        self.model_path = str(Path(model_path).expanduser().absolute())
        self.server_path = str(Path(server_path).expanduser().absolute())
        self.draft_model_path = str(Path(draft_model_path).expanduser().absolute()) if draft_model_path else None
        self.runtime_source_dir = runtime_source_dir
        self.runtime_receipt_path = runtime_receipt_path
        self.mode = normalize_speculative_mode(mode)
        self.max_draft_tokens = max_draft_tokens
        self.n_ctx = int(n_ctx)
        if self.n_ctx <= 0:
            raise ValueError("Context must be positive; the runtime will not silently resize it")
        # The awake caller can supply its selected total context budget and
        # pass an output upper bound. Only then do we compute the exact remaining
        # output allowance here, using this same server's tokenizer.
        self.context_budget_tokens: Optional[int] = None
        self.n_gpu_layers = int(n_gpu_layers)
        self.n_threads = n_threads
        self.n_threads_batch = n_threads_batch
        self.mmproj_path = mmproj_path
        self.adapter_root = str(adapter_root or Path(self.server_path).parent)
        self.adapter_path = ""
        self.adapters: Dict[str, Any] = {}
        self.model = None
        self.tokenizer = None
        self.chat_handler = None
        self.is_loaded = False
        self.is_gguf_available = True
        self.last_error = ""
        self.memory_placement: Dict[str, Any] = {}
        self.auto_speed_result: Dict[str, Any] = {}
        self._process: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._io_lock = threading.RLock()
        self._runtime_model_lock = self._io_lock
        self._state_lock = threading.Lock()
        self._active_connection: Optional[http.client.HTTPConnection] = None
        self._active_socket: Optional[socket.socket] = None
        self._cancelled = threading.Event()
        self._plan: Optional[SpeculativePlan] = None
        self._runtime_receipt: Dict[str, Any] = {}
        self._last_timings: Dict[str, Any] = {}
        self.last_generation: Dict[str, Any] = {}
        self.last_branch_usage: Optional[Dict[str, int]] = None
        # Match the ordinary llama-cpp-python completion defaults used by the
        # GGUF backend.  The selected temperature/top_p override these per call.
        self.sampling_defaults: Dict[str, Any] = {
            "top_k": 40, "min_p": 0.05, "typical_p": 1.0,
            "repeat_penalty": 1.1, "repeat_last_n": 64,
            "frequency_penalty": 0.0, "presence_penalty": 0.0,
            "samplers": ["penalties", "top_k", "typ_p", "top_p", "min_p", "temperature"],
        }
        if sampling_defaults:
            allowed = set(self.sampling_defaults) | {"seed"}
            unknown = set(sampling_defaults) - allowed
            if unknown:
                raise ValueError("Unsupported sampling defaults: " + ", ".join(sorted(unknown)))
            self.sampling_defaults.update(sampling_defaults)

    def preflight(self, cancelled: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
        """Validate bytes and ownership without loading a model or starting a process."""
        if self.mmproj_path:
            raise ValueError("The exact native speculative path is text-only; retain the standard Prism backend for media")
        receipt = validate_runtime_receipt(self.server_path, self.runtime_receipt_path, cancelled)
        target_hash = sha256_file(self.model_path, cancelled)
        draft_hash = sha256_file(self.draft_model_path, cancelled) if self.draft_model_path else None
        plan = plan_speculation(self.mode, target_sha256=target_hash, draft_sha256=draft_hash,
                                max_draft_tokens=self.max_draft_tokens)
        self._runtime_receipt = receipt
        self._plan = plan
        return {"status": "ready", "plan": plan.as_dict(), "model_loaded": False,
                "message": "Local artifacts and native verifier receipt match; model output and speed have not been measured."}

    def _build_command(self, port: int) -> List[str]:
        if self._plan is None:
            raise RuntimeError("Run preflight before constructing a native command")
        command = [
            self.server_path, "-m", self.model_path,
            "--host", "127.0.0.1", "--port", str(port),
            "-c", str(self.n_ctx), "-np", "1", "-b", "512", "-ub", "512",
            "-ngl", str(self.n_gpu_layers),
            "--cache-type-k", "f16", "--cache-type-v", "f16",
            "--no-context-shift", "--no-warmup",
        ]
        for flag, value in (("-t", self.n_threads), ("-tb", self.n_threads_batch)):
            if value is not None:
                if int(value) < 1:
                    raise ValueError("Native thread counts must be positive")
                command += [flag, str(int(value))]
        command += speculative_arguments(self._plan, self.draft_model_path)
        return command

    def _runtime_env(self) -> Dict[str, str]:
        # llama-server accepts many argument environment variables.  An ambient
        # draft, quantized cache, RoPE override or remote model must not override
        # our selected target and strict verification configuration.
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("LLAMA_ARG_", "SMARTAI_EXACT_"))}
        directory = str(Path(self.server_path).parent)
        library_key = "PATH" if os.name == "nt" else ("DYLD_LIBRARY_PATH" if os.sys.platform == "darwin" else "LD_LIBRARY_PATH")
        env[library_key] = os.pathsep.join([directory, env.get(library_key, "")])
        return env

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    @staticmethod
    def _external_cancelled(cancelled: Optional[Callable[[], bool]]) -> bool:
        try:
            return bool(cancelled and cancelled())
        except Exception:
            return True

    @contextmanager
    def _monitor_cancellation(self, cancelled):
        stopped = threading.Event()
        watcher = None
        if cancelled is not None:
            def watch():
                while not stopped.is_set():
                    if self._external_cancelled(cancelled):
                        self.cancel_generation()
                        return
                    stopped.wait(0.05)
            watcher = threading.Thread(target=watch, name="prism-native-cancel", daemon=True)
            watcher.start()
        try:
            yield
        finally:
            stopped.set()
            if watcher is not None:
                watcher.join(timeout=1.0)

    def load_model(self, cancelled: Optional[Callable[[], bool]] = None) -> bool:
        self.unload_model()
        self.last_error = ""
        if self._external_cancelled(cancelled):
            self.last_error = "Native model loading cancelled"
            return False
        self._cancelled.clear()
        with self._monitor_cancellation(cancelled):
            return self._load_selected_model(cancelled)

    def _load_selected_model(self, cancelled) -> bool:
        try:
            def cancelled_now():
                return self._cancelled.is_set() or self._external_cancelled(cancelled)
            self.preflight(cancelled=cancelled_now)
            if cancelled_now():
                raise RuntimeError("Native model loading cancelled")
            port = self._free_port()
            command = self._build_command(port)
            environment = self._runtime_env()
            if cancelled_now():
                raise RuntimeError("Native model loading cancelled")
            self._process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                             env=environment)
            self._port = port
            deadline = time.monotonic() + 180.0
            while time.monotonic() < deadline and not cancelled_now():
                if self._process.poll() is not None:
                    raise RuntimeError("The native server exited before becoming ready")
                try:
                    result = self._request("/health", timeout=1.0)
                    if result.get("status") != "ok":
                        raise RuntimeError("The native server is still loading")
                    if cancelled_now():
                        raise RuntimeError("Native model loading cancelled")
                    self.model = self.tokenizer = _NativeModelProxy(self)
                    self.is_loaded = True
                    return True
                except (OSError, RuntimeError, http.client.HTTPException):
                    self._cancelled.wait(0.1)
            raise RuntimeError("Native server startup was cancelled or timed out")
        except Exception as exc:
            self.last_error = str(exc)
            self.unload_model()
            return False

    def _connect(self, timeout: float) -> http.client.HTTPConnection:
        if self._port is None or self._cancelled.is_set():
            raise RuntimeError("The native server is stopped or the request was cancelled")
        connection = http.client.HTTPConnection("127.0.0.1", self._port, timeout=timeout)
        with self._state_lock:
            if self._cancelled.is_set():
                raise RuntimeError("Native request cancelled")
            self._active_connection = connection
        return connection

    def _close_connection(self, connection: http.client.HTTPConnection) -> None:
        with self._state_lock:
            if self._active_connection is connection:
                self._active_connection = None
                self._active_socket = None
        connection.close()

    def _remember_socket(self, connection: http.client.HTTPConnection) -> None:
        # HTTPConnection transfers ownership of a Connection: close socket to
        # HTTPResponse, setting connection.sock to None. Retain its reference
        # before getresponse so cancellation can still interrupt response I/O.
        with self._state_lock:
            if self._active_connection is connection:
                self._active_socket = connection.sock
        if self._cancelled.is_set():
            self.cancel_generation()
            raise RuntimeError("Native request cancelled")

    def _request(self, path: str, payload: Optional[Dict[str, Any]] = None,
                 timeout: float = 120.0) -> Dict[str, Any]:
        with self._io_lock:
            connection = self._connect(timeout)
            response = None
            try:
                raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8") if payload is not None else None
                connection.request("POST" if payload is not None else "GET", path, body=raw,
                                   headers={"Content-Type": "application/json"})
                self._remember_socket(connection)
                response = connection.getresponse()
                body = response.read(16 * 1024 * 1024 + 1)
                if len(body) > 16 * 1024 * 1024:
                    raise RuntimeError("Native JSON response exceeds the allowed size")
                if response.status != 200:
                    raise RuntimeError(f"Native server HTTP {response.status}: {body[:1024].decode('utf-8', errors='replace')}")
                data = json.loads(body)
                if not isinstance(data, dict) or "error" in data:
                    raise RuntimeError("Invalid native JSON response: " + str(data)[:1024])
                return data
            finally:
                if response is not None:
                    response.close()
                self._close_connection(connection)

    def _completion_payload(self, prompt: str, max_tokens: int, temperature: float,
                            top_p: float, **overrides) -> Dict[str, Any]:
        if not isinstance(prompt, str):
            raise ValueError("The raw native prompt must be text")
        if not math.isfinite(float(temperature)) or float(temperature) < 0:
            raise ValueError("Temperature must be finite and nonnegative")
        if not math.isfinite(float(top_p)) or not 0 <= float(top_p) <= 1:
            raise ValueError("top_p must lie between 0 and 1")
        if int(max_tokens) <= 0:
            raise ValueError("Maximum output tokens must be positive")
        # Never shrink or truncate the user's request to make a draft fit.
        payload = dict(self.sampling_defaults)
        payload.update({"prompt": prompt, "n_predict": int(max_tokens),
                        "temperature": float(temperature), "top_p": float(top_p),
                        "stream": True, "cache_prompt": False,
                        "stop": ["<|im_end|>", "</s>", "\nuser", "\nHuman:"]})
        allowed = set(self.sampling_defaults) | {"seed", "stop"}
        if set(overrides) - allowed:
            raise ValueError("Unsupported native sampling override")
        payload.update(overrides)
        return payload

    def stream_generate_tokens(self, prompt: str, max_tokens: int = 1536,
                               temperature: float = 0.75, top_p: float = 0.92,
                               image_bytes: Optional[bytes] = None,
                               cancelled: Optional[Callable[[], bool]] = None,
                               **sampling) -> Generator[str, None, None]:
        if image_bytes is not None:
            raise RuntimeError("This native speculative adapter does not accept images")
        if not self.is_loaded:
            raise RuntimeError("Load the selected native runtime before generation")
        with self._io_lock, self._monitor_cancellation(cancelled):
            self._last_timings = {}
            self.last_generation = {}
            if self._external_cancelled(cancelled):
                self.cancel_generation()
                self.last_generation = {"status": "cancelled", "usage": None}
                return
            self._cancelled.clear()
            if self._external_cancelled(cancelled):
                self.cancel_generation()
                self.last_generation = {"status": "cancelled", "usage": None}
                return
            payload = self._completion_payload(prompt, max_tokens, temperature, top_p, **sampling)
            # Ask this exact runtime's tokenizer.  Refuse overflow instead of
            # allowing native context shifting or a shorter requested answer.
            try:
                prompt_tokens = len(self.tokenizer.encode(prompt))
            except Exception:
                if self._cancelled.is_set():
                    self.last_generation = {"status": "cancelled", "usage": None}
                    return
                raise
            if self._cancelled.is_set() or self._external_cancelled(cancelled):
                self.cancel_generation()
                self.last_generation = {"status": "cancelled", "usage": None}
                return
            budget = self.context_budget_tokens
            effective_context = min(self.n_ctx, budget) if type(budget) is int and budget > 0 else self.n_ctx
            if prompt_tokens >= effective_context:
                raise ValueError("Prompt exhausts the selected context; no prompt was truncated")
            if type(budget) is int and budget > 0:
                payload["n_predict"] = min(int(max_tokens), effective_context - prompt_tokens)
            elif prompt_tokens + int(max_tokens) > effective_context:
                raise ValueError("Prompt plus requested output exceeds the selected context; no prompt or output budget was truncated")
            try:
                connection = self._connect(1800.0)
            except RuntimeError:
                if not self._cancelled.is_set():
                    raise
                self.last_generation = {"status": "cancelled", "usage": None}
                return
            completed = False
            response = None
            try:
                connection.request("POST", "/completion",
                                   body=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                                   headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
                self._remember_socket(connection)
                response = connection.getresponse()
                if response.status != 200:
                    raise RuntimeError(f"Native server HTTP {response.status}: {response.read(1024).decode('utf-8', errors='replace')}")
                event_data: List[str] = []
                event_bytes = 0
                while not self._cancelled.is_set():
                    raw = response.readline(1024 * 1024 + 1)
                    if len(raw) > 1024 * 1024:
                        raise RuntimeError("Native stream line exceeds the allowed size")
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="strict").rstrip("\r\n")
                    if line.startswith("data:"):
                        event_data.append(line[5:].lstrip(" "))
                        event_bytes += len(raw)
                        if event_bytes > 4 * 1024 * 1024:
                            raise RuntimeError("Native stream event exceeds the allowed size")
                        continue
                    if line or not event_data:
                        continue
                    body = "\n".join(event_data)
                    event_data.clear()
                    event_bytes = 0
                    if body == "[DONE]":
                        completed = True
                        break
                    item = json.loads(body)
                    if not isinstance(item, dict) or "error" in item:
                        raise RuntimeError("Native generation error: " + str(item)[:1024])
                    content = item.get("content", "")
                    if not isinstance(content, str):
                        raise RuntimeError("Native stream returned invalid text")
                    if content:
                        yield content
                    if isinstance(item.get("timings"), dict):
                        self._last_timings = dict(item["timings"])
                    if item.get("stop") is True:
                        self._record_generation(item)
                        completed = True
                        break
                if not completed and not self._cancelled.is_set():
                    raise RuntimeError("Native stream ended without a completion marker")
            except (OSError, http.client.HTTPException) as exc:
                if not self._cancelled.is_set():
                    raise RuntimeError(f"Native stream transport failed: {exc}") from exc
            except RuntimeError:
                if not self._cancelled.is_set():
                    raise
            finally:
                if not completed and self._cancelled.is_set():
                    self.last_generation = {"status": "cancelled", "usage": None}
                if response is not None:
                    response.close()
                self._close_connection(connection)

    def generate_branches(self, prompt: str, branch_count: int = 1, max_tokens: int = 1536,
                          temperature: Any = 0.75, top_p: float = 0.92,
                          image_bytes: Optional[bytes] = None,
                          cancelled: Optional[Callable[[], bool]] = None, **sampling) -> List[str]:
        output = []
        usage = []
        self.last_branch_usage = None
        self.last_generation = {}
        requested_count = max(1, int(branch_count))
        for index in range(requested_count):
            if self._external_cancelled(cancelled):
                self.cancel_generation()
                self.last_generation = {"status": "cancelled", "usage": None}
                break
            temp = temperature[index % len(temperature)] if isinstance(temperature, (list, tuple)) else temperature
            text = "".join(self.stream_generate_tokens(prompt, max_tokens, float(temp), top_p,
                                                       image_bytes=image_bytes, cancelled=cancelled, **sampling)).strip()
            if self._cancelled.is_set():
                break
            output.append(text)
            usage.append(self.last_generation.get("usage"))
        if len(output) == requested_count and not self._cancelled.is_set() and not self._external_cancelled(cancelled) and usage and all(isinstance(item, dict) and all(type(item.get(key)) is int for key in
                         ("prompt_tokens", "completion_tokens")) for item in usage):
            self.last_branch_usage = {key: sum(item[key] for item in usage)
                                      for key in ("prompt_tokens", "completion_tokens")}
            self.last_branch_usage["total_tokens"] = sum(self.last_branch_usage.values())
        return output

    def _record_generation(self, item: Dict[str, Any]) -> None:
        timings = item.get("timings") if isinstance(item.get("timings"), dict) else {}
        prompt_tokens = item.get("tokens_evaluated", timings.get("prompt_n"))
        completion_tokens = item.get("tokens_predicted", timings.get("predicted_n"))
        usage = None
        if all(type(value) is int and value >= 0 for value in (prompt_tokens, completion_tokens)):
            usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                     "total_tokens": prompt_tokens + completion_tokens}
        self.last_generation = {"status": "completed", "usage": usage, "timings": dict(timings),
                                "stop_type": item.get("stop_type"), "source": "native_runtime"}

    def owned_process_ids(self) -> List[int]:
        process = self._process
        if process is None or process.poll() is not None:
            return []
        pid = getattr(process, "pid", None)
        return [pid] if type(pid) is int and pid > 0 else []

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Native token counting requires the selected server")
        return len(self.tokenizer.encode("\n".join(str(message.get("content", "")) for message in messages or [])))

    def cancel_generation(self) -> None:
        self._cancelled.set()
        with self._state_lock:
            connection = self._active_connection
            active_socket = self._active_socket
        if connection is not None:
            # Interrupt a blocked readline/getresponse before waiting for the
            # request lock.  Closing the generator alone cannot interrupt I/O.
            sock = active_socket or connection.sock
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            connection.close()

    def unload_model(self) -> None:
        self.cancel_generation()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        # If termination fails, preserve the process handle so callers can retry.
        self._process = None
        self._port = None
        self.is_loaded = False
        self.model = None
        self.tokenizer = None

    def configure_speculation(self, mode: str = "none", max_draft_tokens: Optional[int] = None,
                              draft_model_path: Optional[str] = None) -> Dict[str, Any]:
        selected = normalize_speculative_mode(mode)
        if self.is_loaded:
            return {"status": "reload_required", "message": "Reload this native runtime to change its draft configuration."}
        self.mode = selected
        self.max_draft_tokens = max_draft_tokens
        self.draft_model_path = str(Path(draft_model_path).expanduser().absolute()) if draft_model_path else None
        self._plan = None
        return {"status": "configured", "requested_mode": selected, "model_loaded": False}

    def get_speculative_telemetry(self) -> Dict[str, Any]:
        timings = self.last_generation.get("timings", {}) if self.last_generation.get("status") == "completed" else {}
        proposed = timings.get("draft_n")
        accepted = timings.get("draft_n_accepted")
        valid = type(proposed) is int and type(accepted) is int and 0 <= accepted <= proposed
        measured_tps = timings.get("predicted_per_second")
        if type(measured_tps) not in (int, float) or not math.isfinite(measured_tps) or measured_tps <= 0:
            measured_tps = None
        return {
            "mode": self._plan.active_mode if self.is_loaded and self._plan else "none",
            "requested_mode": self.mode, "runtime_contract": EXACTNESS_CONTRACT,
            "status": "active" if self.is_loaded else "not_loaded",
            "draft_tokens_proposed": proposed if valid else None,
            "draft_tokens_accepted": accepted if valid else None,
            "acceptance_rate_percent": (100.0 * accepted / proposed) if valid and proposed else None,
            "speedup": None, "output_equivalence_measured": False,
            "decode_tokens_per_second": measured_tps,
            "message": "One-row target verification preserves target sampling; draft speedup and hardware parity are unmeasured.",
        }

    def configure_auto_speed(self, *args, **kwargs) -> Dict[str, Any]:
        del args, kwargs
        return {"status": "unsupported", "settings": {},
                "message": "The exact native verifier keeps one-row target execution fixed; automatic batch tuning is disabled."}

    def calculate_token_entropy(self, prompt: str) -> float:
        del prompt
        raise RuntimeError("This native runtime does not expose full-vocabulary entropy")

    def supports_media_input(self, kind: str) -> bool:
        del kind
        return False

    def training_ready(self) -> bool:
        return False

    def can_train(self) -> bool:
        return False

    def train_mini_batch(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("The pinned native speculative target is immutable; select a learning-capable backend for training")
