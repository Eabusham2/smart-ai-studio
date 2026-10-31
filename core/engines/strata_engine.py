"""Text inference through Studio's owned, authenticated local Strata service.

Strata owns the native model and its GPU/RAM/SSD placement. This adapter only
transports complete structured conversations and the caller's sampling choices.
It never interprets ChatML literals in user text or substitutes model responses.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import http.client
import ipaddress
import json
import math
import socket
import threading
from typing import Any, Callable, Generator, Optional
from urllib.parse import urlsplit


class StrataError(RuntimeError):
    """The owned Strata service failed or returned an invalid response."""


class StrataCancelled(StrataError):
    """An active local request was cancelled."""


def _chat_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("Strata needs a nonempty structured conversation.")
    snapshot = copy.deepcopy(list(messages))
    for item in snapshot:
        if not isinstance(item, dict) or not isinstance(item.get("role"), str):
            raise ValueError("Each Strata message needs a role.")
        if item["role"] not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError("Unsupported Strata message role.")
        if not isinstance(item.get("content"), str):
            raise ValueError("This Strata backend accepts text message content only.")
    # Check the full payload without dropping extra fields or coercing content.
    json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
    return snapshot


class StrataChatPrompt(str):
    """A string-compatible carrier for a conversation, with no delimiter parsing."""

    def __new__(cls, messages: Any):
        snapshot = _chat_messages(messages)
        value = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        instance = super().__new__(cls, value)
        instance._messages = snapshot
        return instance

    @property
    def messages(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._messages)

    def __reduce__(self):
        return type(self), (self.messages,)


class _ActiveRequest:
    def __init__(self, connection: http.client.HTTPConnection):
        self.connection = connection
        self.cancelled = threading.Event()
        self.socket: Optional[socket.socket] = None

    def cancel(self) -> None:
        self.cancelled.set()
        # HTTPConnection may clear .sock after reading Connection: close headers.
        # Retain the original socket so shutdown interrupts a blocked body read.
        sock = self.socket or self.connection.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class StrataReasoningBackend:
    """Controller contract backed by a separately managed Strata engine."""

    CONNECT_TIMEOUT = 10.0
    # Cold SSD expert reads may spend minutes inside a native prefill chunk.
    # Keep a finite inactivity limit while socket cancellation stays immediate.
    READ_TIMEOUT = 900.0
    MAX_JSON_BYTES = 1024 * 1024
    MAX_EVENT_BYTES = 1024 * 1024
    token_count_kind = "conservative_utf8_estimate"
    entropy_kind = "unavailable"
    can_train = False
    is_mlx_available = False
    input_modalities = {"text"}

    def __init__(self, model_path: str, model_info: Optional[dict[str, Any]] = None):
        self.model_path = str(model_path)
        self.model_info = dict(model_info or {})
        self.model_key = str(self.model_info.get("model_key") or self.model_info.get("strata_model_key")
                             or self.model_info.get("key") or model_path)
        self.runtime = "strata"
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.adapter_path = None
        self.adapters: dict[str, Any] = {}
        self.is_loaded = False
        self.context_budget_tokens = 32768
        self.model_id = ""
        self.health: dict[str, Any] = {}
        self.plan: dict[str, Any] = {}
        self.source_identity: dict[str, Any] = {}
        self.last_usage: dict[str, Any] = {}
        self.last_branch_usage: dict[str, int] = {}
        self.last_generation: dict[str, Any] = {}
        self.last_error = ""
        self._manager = None
        self._endpoint: Optional[tuple[str, int]] = None
        self._api_key = ""
        self._loaded_context = 0
        self._request_lock = threading.RLock()
        self._transport_lock = threading.Lock()
        self._active: Optional[_ActiveRequest] = None

    @staticmethod
    def prepare_chat_prompt(messages: Any) -> StrataChatPrompt:
        return StrataChatPrompt(messages)

    @staticmethod
    def _messages(prompt: Any) -> list[dict[str, Any]]:
        if isinstance(prompt, StrataChatPrompt):
            return prompt.messages
        if isinstance(prompt, (list, tuple)):
            return _chat_messages(prompt)
        if not isinstance(prompt, str):
            raise ValueError("Strata expects text or a structured conversation.")
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def _local_endpoint(base_url: Any) -> tuple[str, int]:
        if not isinstance(base_url, str):
            raise StrataError("The managed Strata service did not return a local address.")
        try:
            parsed = urlsplit(base_url)
            host, port = parsed.hostname, parsed.port
            if (parsed.scheme != "http" or parsed.username is not None or parsed.password is not None
                    or parsed.path not in ("", "/") or parsed.query or parsed.fragment
                    or not host or not port or not 1 <= port <= 65535 or "%" in host):
                raise ValueError()
            # Never use environment proxy settings or resolve a nonlocal hostname.
            host = "127.0.0.1" if host.lower() == "localhost" else host
            address = ipaddress.ip_address(host)
            if not address.is_loopback:
                raise ValueError()
            return str(address), port
        except ValueError:
            raise StrataError("Strata must use an owned HTTP service on a loopback address.") from None

    @staticmethod
    def _cancelled(cancelled: Optional[Callable[[], bool]]) -> bool:
        if cancelled is None:
            return False
        try:
            return bool(cancelled())
        except Exception:
            return True

    def _safe_message(self, value: Any) -> str:
        text = str(value)
        if self._api_key:
            text = text.replace(self._api_key, "[redacted]")
        return " ".join(text.split())[:500]

    def cancel_current_request(self) -> bool:
        """Interrupt headers/prefill/stream reads, including between token events."""
        with self._transport_lock:
            active = self._active
        if active is None:
            return False
        active.cancel()
        return True

    @contextmanager
    def _request(self, method: str, path: str, payload=None, cancelled=None):
        if self._endpoint is None:
            raise StrataError("The managed Strata service is not connected.")
        body = None if payload is None else json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        with self._request_lock:
            connection = http.client.HTTPConnection(*self._endpoint, timeout=self.CONNECT_TIMEOUT)
            active = _ActiveRequest(connection)
            stopped = threading.Event()
            response = None
            watcher = None
            with self._transport_lock:
                self._active = active

            def watch():
                while not stopped.wait(0.05):
                    if self._cancelled(cancelled):
                        active.cancel()
                        return

            try:
                if self._cancelled(cancelled):
                    active.cancel()
                if active.cancelled.is_set():
                    raise StrataCancelled("Strata request cancelled.")
                if cancelled is not None:
                    watcher = threading.Thread(target=watch, name="strata-request-cancel", daemon=True)
                    watcher.start()
                connection.connect()
                active.socket = connection.sock
                if active.cancelled.is_set():
                    active.cancel()
                    raise StrataCancelled("Strata request cancelled.")
                if connection.sock is not None:
                    connection.sock.settimeout(self.READ_TIMEOUT)
                connection.request(method, path, body=body, headers={
                    "Authorization": "Bearer " + self._api_key,
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream" if path == "/v1/chat/completions" else "application/json",
                })
                response = connection.getresponse()
                if active.cancelled.is_set():
                    raise StrataCancelled("Strata request cancelled.")
                yield response, active
            except (OSError, http.client.HTTPException) as exc:
                if active.cancelled.is_set():
                    raise StrataCancelled("Strata request cancelled.") from None
                raise StrataError("Strata connection failed: " + self._safe_message(exc)) from None
            finally:
                stopped.set()
                active.cancel()
                if response is not None:
                    response.close()
                connection.close()
                with self._transport_lock:
                    if self._active is active:
                        self._active = None
                if watcher is not None:
                    watcher.join(timeout=1.0)

    @staticmethod
    def _json(raw: bytes | str) -> Any:
        def nonfinite(_value):
            raise ValueError("Nonfinite JSON number")

        def finite_float(value):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("Nonfinite JSON number")
            return number

        try:
            return json.loads(raw, parse_constant=nonfinite, parse_float=finite_float)
        except (UnicodeError, ValueError):
            raise StrataError("Strata returned malformed JSON.") from None

    def _response_json(self, response) -> Any:
        raw = response.read(self.MAX_JSON_BYTES + 1)
        if len(raw) > self.MAX_JSON_BYTES:
            raise StrataError("Strata returned an oversized JSON response.")
        value = self._json(raw)
        if response.status != 200:
            error = value.get("error", {}) if isinstance(value, dict) else {}
            message = error.get("message", "Request rejected.") if isinstance(error, dict) else error
            raise StrataError(f"Strata request failed (HTTP {response.status}): {self._safe_message(message)}")
        return value

    def load_model(self, progress=None, cancelled=None, context_tokens=None) -> bool:
        """Start only our managed runtime and verify its live identity before use."""
        from core.strata_runtime import StrataRuntime

        self.unload_model()
        self.last_error = ""
        requested = self.context_budget_tokens if context_tokens is None else context_tokens
        if type(requested) is not int or requested <= 0:
            raise ValueError("Strata context size must be a positive integer.")
        self._manager = StrataRuntime(self.model_key, self.model_info)
        self._manager.speed_memory_limit_gb = getattr(self, "speed_memory_limit_gb", None)
        try:
            launched = self._manager.start(requested, progress=progress, cancelled=cancelled)
            if not isinstance(launched, dict):
                raise StrataError("Strata did not return a managed runtime record.")
            source_identity = {}
            if self.model_info.get("source_kind") == "strata_local":
                from config.strata_models import validate_strata_selection
                try:
                    selected = validate_strata_selection(self.model_info)
                except (OSError, ValueError, TypeError) as exc:
                    raise StrataError("The chosen local Strata source changed before loading: " + str(exc)) from None
                expected_source = {
                    "id": selected["strata_source_id"], "path": selected["strata_model_path"],
                    "files": selected["strata_source_files"], "companions": selected["strata_companion_files"],
                }
                source_identity = launched.get("source_identity")
                if (not isinstance(source_identity, dict)
                        or any(source_identity.get(key) != value for key, value in expected_source.items())):
                    raise StrataError("The managed Strata source does not match the chosen local model files.")
            self._endpoint = self._local_endpoint(launched.get("base_url"))
            key = launched.get("api_key")
            if not isinstance(key, str) or not key or len(key) > 4096 or any(not 33 <= ord(c) <= 126 for c in key):
                raise StrataError("The managed Strata service has no valid authentication key.")
            self._api_key = key
            model_id = launched.get("model_id")
            if not isinstance(model_id, str) or not model_id or len(model_id) > 512:
                raise StrataError("The managed Strata service has no verified model identity.")
            expected_model = self.model_info.get("strata_api_model")
            if expected_model is not None and model_id != expected_model:
                raise StrataError("The managed Strata model does not match the selected preset.")
            with self._request("GET", "/health", cancelled=cancelled) as (response, active):
                health = self._response_json(response)
                if active.cancelled.is_set():
                    raise StrataCancelled("Strata load cancelled.")
            if (not isinstance(health, dict) or health.get("service") != "strata"
                    or health.get("status") != "ok" or health.get("loaded") is not True
                    or health.get("api_key") is not True or health.get("model") != model_id
                    or type(health.get("max_context")) is not int or health["max_context"] <= 0):
                raise StrataError("The local service is not an authenticated, loaded Strata model with the expected identity.")
            with self._request("GET", "/v1/models", cancelled=cancelled) as (response, active):
                models = self._response_json(response)
                if active.cancelled.is_set():
                    raise StrataCancelled("Strata load cancelled.")
            rows = models.get("data") if isinstance(models, dict) else None
            if not isinstance(rows, list) or not any(
                isinstance(row, dict) and row.get("id") == model_id
                and isinstance(row.get("status"), dict)
                and row["status"].get("value") == "loaded" for row in rows
            ):
                raise StrataError("Strata's model list does not confirm the expected loaded model.")
            if self._cancelled(cancelled):
                raise StrataCancelled("Strata load cancelled.")
            self.model_id = model_id
            self.health = copy.deepcopy(health)
            self.plan = copy.deepcopy(launched.get("plan") or {})
            self.source_identity = copy.deepcopy(source_identity)
            self._loaded_context = health["max_context"]
            self.context_budget_tokens = self._loaded_context
            self.last_usage = {}
            self.last_branch_usage = {}
            self.last_generation = {}
            self.is_loaded = True
            self.model = self
            return True
        except Exception as exc:
            self.last_error = self._safe_message(exc)
            try:
                self.unload_model()
            except Exception as cleanup:
                self.last_error += "; runtime cleanup failed: " + self._safe_message(cleanup)
            raise

    def _events(self, response, active):
        data: list[str] = []
        size = 0
        while True:
            if active.cancelled.is_set():
                raise StrataCancelled("Strata request cancelled.")
            raw = response.readline(self.MAX_EVENT_BYTES + 1)
            if active.cancelled.is_set():
                raise StrataCancelled("Strata request cancelled.")
            if not raw:
                raise StrataError("Strata stream ended before its completion marker.")
            if len(raw) > self.MAX_EVENT_BYTES:
                raise StrataError("Strata returned an oversized stream event.")
            try:
                line = raw.decode("utf-8").rstrip("\r\n")
            except UnicodeError:
                raise StrataError("Strata returned invalid UTF-8 in its stream.") from None
            if not line:
                if data:
                    value = "\n".join(data)
                    data, size = [], 0
                    if value == "[DONE]":
                        return
                    yield self._json(value)
                continue
            if line.startswith(":"):
                continue
            field, sep, value = line.partition(":")
            if sep and value.startswith(" "):
                value = value[1:]
            if field == "data":
                size += len(raw)
                if size > self.MAX_EVENT_BYTES:
                    raise StrataError("Strata returned an oversized stream event.")
                data.append(value)

    @staticmethod
    def _usage(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise StrataError("Strata returned invalid token usage.")
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if type(value.get(name)) is not int or value[name] < 0:
                raise StrataError("Strata returned invalid token usage.")
        return copy.deepcopy(value)

    def stream_generate_tokens(self, prompt, max_tokens=1536, temperature=0.75,
                               top_p=0.92, image_bytes=None, cancelled=None) -> Generator[str, None, None]:
        if image_bytes is not None:
            raise NotImplementedError("This Strata controller currently accepts text only.")
        if not self.is_loaded:
            raise StrataError("Load the Strata model before generating text.")
        if type(max_tokens) is not int or max_tokens < 0:
            raise ValueError("Strata max_tokens must be a nonnegative integer.")
        if isinstance(temperature, bool) or not math.isfinite(float(temperature)) or float(temperature) < 0:
            raise ValueError("Strata temperature must be finite and nonnegative.")
        if isinstance(top_p, bool) or not math.isfinite(float(top_p)) or not 0 < float(top_p) <= 1:
            raise ValueError("Strata top_p must be in (0, 1].")
        model_id = self.model_id
        payload = {"model": model_id, "messages": self._messages(prompt),
                   "max_tokens": max_tokens, "temperature": float(temperature), "top_p": float(top_p),
                   "stream": True, "stream_options": {"include_usage": True}, "strata_mcp": False}
        with self._request_lock:
            self.last_error = ""
            self.last_usage = {}
            self.last_branch_usage = {}
            self.last_generation = {"status": "streaming", "model": model_id, "usage": {}}
            reasoning = False
            finished = False
            try:
                with self._request("POST", "/v1/chat/completions", payload, cancelled) as (response, active):
                    if response.status != 200:
                        self._response_json(response)
                    if response.getheader("Content-Type", "").split(";", 1)[0].strip().lower() != "text/event-stream":
                        raise StrataError("Strata did not return an event stream.")
                    for chunk in self._events(response, active):
                        if active.cancelled.is_set():
                            raise StrataCancelled("Strata request cancelled.")
                        if not isinstance(chunk, dict):
                            raise StrataError("Strata returned an invalid stream chunk.")
                        if "error" in chunk:
                            error = chunk["error"]
                            message = error.get("message", "Generation failed.") if isinstance(error, dict) else error
                            raise StrataError("Strata generation failed: " + self._safe_message(message))
                        if chunk.get("model") != model_id:
                            raise StrataError("Strata streamed a different model identity.")
                        if chunk.get("usage") is not None:
                            self.last_usage = self._usage(chunk["usage"])
                            self.last_generation["usage"] = copy.deepcopy(self.last_usage)
                        if isinstance(chunk.get("timings"), dict):
                            self.last_generation["timings"] = copy.deepcopy(chunk["timings"])
                        choices = chunk.get("choices")
                        if not isinstance(choices, list) or len(choices) > 1:
                            raise StrataError("Strata returned invalid streaming choices.")
                        for choice in choices:
                            if not isinstance(choice, dict) or choice.get("index") != 0:
                                raise StrataError("Strata returned an unexpected choice index.")
                            delta = choice.get("delta")
                            if not isinstance(delta, dict):
                                raise StrataError("Strata returned an invalid text delta.")
                            if delta.get("tool_calls") or delta.get("function_call"):
                                raise StrataError("Strata returned a tool call to a text-only request.")
                            thought, content = delta.get("reasoning_content"), delta.get("content")
                            if any(value is not None and not isinstance(value, str) for value in (thought, content)):
                                raise StrataError("Strata returned a non-text delta.")
                            if finished and (thought or content):
                                raise StrataError("Strata returned text after its finish event.")
                            if thought:
                                if not reasoning:
                                    reasoning = True
                                    yield "<think>"
                                yield thought
                            if content:
                                if reasoning:
                                    reasoning = False
                                    yield "</think>"
                                yield content
                            reason = choice.get("finish_reason")
                            if reason is not None:
                                if not isinstance(reason, str) or not reason:
                                    raise StrataError("Strata returned an invalid finish reason.")
                                finished = True
                                self.last_generation["finish_reason"] = reason
                    if not finished:
                        raise StrataError("Strata completed its stream without a finish event.")
                    if reasoning:
                        yield "</think>"
                self.last_generation["status"] = "completed"
            except StrataCancelled:
                self.last_generation["status"] = "cancelled"
                if reasoning:
                    yield "</think>"
                return
            except GeneratorExit:
                self.last_generation["status"] = "cancelled"
                raise
            except Exception as exc:
                self.last_error = self._safe_message(exc)
                self.last_generation.update(status="error", error=self.last_error)
                raise

    def generate_branches(self, prompt, branch_count=1, max_tokens=1536, temperature=0.75,
                          top_p=0.92, image_bytes=None, cancelled=None) -> list[str]:
        self.last_branch_usage = {}
        if type(branch_count) is not int or branch_count < 1:
            raise ValueError("Strata branch_count must be a positive integer.")
        ladder = list(temperature) if isinstance(temperature, (list, tuple)) else [temperature]
        if not ladder:
            raise ValueError("Strata needs at least one sampling temperature.")
        out = []
        aggregate = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        usage_complete = True
        for index in range(branch_count):
            if self._cancelled(cancelled):
                self.last_usage = {}
                self.last_generation = {"status": "cancelled", "model": self.model_id}
                break
            text = "".join(self.stream_generate_tokens(
                prompt, max_tokens, ladder[index % len(ladder)], top_p, image_bytes, cancelled))
            if self.last_generation.get("status") == "cancelled":
                break
            out.append(text)
            usage = self.last_generation.get("usage", {})
            if self.last_generation.get("status") != "completed" or any(
                type(usage.get(key)) is not int or usage[key] < 0 for key in aggregate
            ):
                usage_complete = False
            else:
                for key in aggregate:
                    aggregate[key] += usage[key]
        if usage_complete and len(out) == branch_count:
            self.last_branch_usage = aggregate
        return out

    def count_tokens(self, messages) -> int:
        """Conservative UTF-8 estimate, not a measured tokenizer count.

        The OpenAI endpoint has no matching tokenize API. Its actual tokenizer
        checks context and rejects overflow; this estimate never truncates input.
        """
        if not messages:
            return 0
        if isinstance(messages, (list, tuple)) and len(messages) == 1 and isinstance(messages[0], dict):
            nested = messages[0].get("content")
            if isinstance(nested, StrataChatPrompt):
                messages = nested
        packed = json.dumps(self._messages(messages), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return len(packed.encode("utf-8")) + 512

    def n_ctx(self) -> int:
        return self._loaded_context or self.context_budget_tokens

    def calculate_token_entropy(self, prompt: str) -> float:
        raise NotImplementedError("Strata exposes no measured entropy here; use the controller's prompt heuristic.")

    def training_ready(self) -> bool:
        return False

    def owned_process_ids(self) -> list[int]:
        manager = self._manager
        return list(manager.owned_pids()) if manager is not None else []

    def configure_auto_speed(self, enabled=True, force=False, progress=None, cancelled=None) -> dict[str, Any]:
        if self._cancelled(cancelled):
            return {"status": "cancelled", "settings": {}, "message": "Strata configuration check cancelled."}
        if not self.is_loaded:
            return {"status": "deferred", "settings": {}, "message": "Load a Strata model to inspect its native placement."}
        if not enabled:
            return {"status": "disabled", "settings": {}, "message": "Additional calibration is off; Strata uses its loaded native placement."}
        message = "Strata configured its native GPU/RAM/SSD placement; no measured speed gain is claimed."
        if progress is not None:
            progress(message)
        return {"status": "configured", "settings": {"context_tokens": self.n_ctx()}, "message": message,
                "gain_pct": 0.0, "measured": False}

    def train_mini_batch(self, *args, **kwargs):
        raise NotImplementedError("Persistent training is not available through Strata's inference API.")

    def supports_media_input(self, kind: str) -> bool:
        return False

    def review_media_input(self, path: str, kind: str, prompt: str = "") -> dict[str, Any]:
        return {"perception_available": False, "reason": "This Strata controller currently accepts text only."}

    def unload_model(self) -> None:
        self.cancel_current_request()
        manager = self._manager
        self.is_loaded = False
        self.model = None
        if manager is not None:
            try:
                manager.stop()
            except Exception as exc:
                # A failed bounded stop can leave verified owned processes alive.
                # Keep their owner for retry and memory accounting, but prohibit
                # new inference against a potentially partially stopped service.
                self.last_error = self._safe_message(exc)
                raise
        self._manager = None
        self.model_id = ""
        self._endpoint = None
        self._api_key = ""
        self._loaded_context = 0
        self.health = {}
        self.plan = {}
        self.source_identity = {}
