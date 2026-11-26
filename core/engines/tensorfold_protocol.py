"""TensorFold native 1.3.5 loopback protocol (upstream commit 02b324ae9767253d9911d753d5c835a5070eed1b)."""
from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from functools import partial
import http.client
import json

from core.engines.prism_speculative_engine import _CancellableHTTPResponse
from core.generation_metrics import valid_count


@dataclass(frozen=True)
class NativeServer:
    port: int
    model_name: str
    key: str = field(repr=False)

    @contextmanager
    def response(self, path, payload=None, *, cancelled=None, stream=False, timeout=30):
        with closing(http.client.HTTPConnection("127.0.0.1", self.port, timeout=min(5, timeout))) as conn:
            if cancelled is not None:
                conn.response_class = partial(_CancellableHTTPResponse, cancelled=cancelled)
            headers = {"Authorization": "Bearer " + self.key, "Content-Type": "application/json",
                       "Accept": "text/event-stream" if stream else "application/json"}
            body = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
            conn.request("GET" if body is None else "POST", path, body, headers)
            conn.sock.settimeout(timeout)
            with closing(conn.getresponse()) as response:
                if response.status != 200:
                    detail = response.read(4096).decode("utf-8", "replace").replace(self.key, "[redacted]")
                    raise RuntimeError(f"TensorFold {path} returned HTTP {response.status}: {detail}")
                expected = "text/event-stream" if stream else "application/json"
                if response.headers.get_content_type() != expected:
                    raise RuntimeError(f"TensorFold {path} did not return {expected}.")
                yield response

    def read_json(self, path, payload=None, *, cancelled=None, timeout=30):
        with self.response(path, payload, cancelled=cancelled, timeout=timeout) as response:
            raw = response.read(32 * 1024 * 1024 + 1)
        if len(raw) > 32 * 1024 * 1024:
            raise RuntimeError(f"TensorFold {path} response exceeded 32 MiB.")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise RuntimeError(f"TensorFold {path} returned an invalid JSON object.")
        return data

    def model_card(self, cancelled):
        data = self.read_json("/v1/models", cancelled=cancelled, timeout=2)
        for card in data.get("data") or []:
            if (isinstance(card, dict) and card.get("id") == self.model_name
                    and card.get("owned_by") == "tensorfold"):
                return card
        raise RuntimeError("TensorFold served a different model identity.")

    def tokenize(self, prompt, cancelled):
        if not isinstance(prompt, str):
            raise TypeError("TensorFold requires a rendered text prompt.")
        # The harness already rendered the checkpoint's own special tokens.
        data = self.read_json("/v1/tokenize", {"prompt": prompt, "add_special_tokens": False},
                              cancelled=cancelled)
        tokens = data.get("tokens")
        if (not isinstance(tokens, list) or not valid_count(data.get("count"))
                or data["count"] != len(tokens) or not all(valid_count(token) for token in tokens)):
            raise RuntimeError("TensorFold returned invalid prompt token IDs.")
        return tokens


def _frames(response):
    lines = []
    size = 0
    while True:
        raw = response.readline(1024 * 1024 + 1)
        if not raw:
            raise RuntimeError("TensorFold stream ended before [DONE].")
        if len(raw) > 1024 * 1024:
            raise RuntimeError("TensorFold stream line exceeded 1 MiB.")
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            size += len(raw)
            if size > 4 * 1024 * 1024:
                raise RuntimeError("TensorFold stream event exceeded 4 MiB.")
            lines.append(line[5:].removeprefix(" "))
        elif not line and lines:
            yield "\n".join(lines)
            lines.clear()
            size = 0


def completion_events(response, model_name, prompt_count):
    """Yield (text, verified usage, finish reason) and require both terminal markers."""
    finished = False
    seen_usage = False
    for frame in _frames(response):
        if frame == "[DONE]":
            if not finished:
                raise RuntimeError("TensorFold sent [DONE] before a finish reason.")
            return
        event = json.loads(frame)
        if not isinstance(event, dict):
            raise RuntimeError("TensorFold returned an invalid completion event.")
        if "error" in event:
            raise RuntimeError("TensorFold generation error: " + str(event["error"])[:1024])
        if event.get("model") != model_name or event.get("object") != "text_completion":
            raise RuntimeError("TensorFold stream returned a different model or completion type.")
        choices, usage = event.get("choices"), event.get("usage")
        if usage is not None:
            if (seen_usage or not isinstance(usage, dict) or usage.get("prompt_tokens") != prompt_count
                    or not all(valid_count(usage.get(key)) for key in
                               ("prompt_tokens", "completion_tokens", "total_tokens"))
                    or usage["total_tokens"] != prompt_count + usage["completion_tokens"]):
                raise RuntimeError("TensorFold returned inconsistent completion usage.")
            usage = {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
            seen_usage = True
        if choices == [] and finished and usage is not None:
            yield "", usage, None
            continue
        if (finished or not isinstance(choices, list) or len(choices) != 1
                or not isinstance(choices[0], dict)):
            raise RuntimeError("TensorFold returned invalid completion choices.")
        choice = choices[0]
        text = choice.get("text")
        reason = choice.get("finish_reason")
        if (type(choice.get("index")) is not int or choice["index"] != 0
                or not isinstance(text, str) or "finish_reason" not in choice
                or (reason is not None and reason not in ("stop", "length"))):
            raise RuntimeError(f"TensorFold returned an invalid or unsuccessful completion: {reason!r}")
        finished = reason is not None
        yield text, usage, reason
