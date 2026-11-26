"""TensorFold native protocol and process tests. The loopback fixture never opens GPU weights."""
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from core.engines import tensorfold_mlx_engine as engine
from core.engines.tensorfold_protocol import completion_events


# A real child exercises Windows-compatible termination, Connection: close,
# partial socket reads and generator close without requiring a GPU or model download.
SERVER = r'''
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
args = sys.argv[1:]
source = Path(args[1])
assert "--api-key" not in args and os.environ["TENSORFOLD_API_KEY"] != "inherited-shared-key"
def flag(name):
    return args[args.index(name) + 1]
model = flag("--name")
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def authorized(self):
        if self.headers["Authorization"] == "Bearer " + os.environ["TENSORFOLD_API_KEY"]:
            return True
        self.send_error(401)
    def reply(self, data):
        raw = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
    def do_GET(self):
        if self.authorized() and self.path == "/v1/models":
            self.reply({"data": [{"id": model, "owned_by": "tensorfold", "max_model_len": 9}]})
    def do_POST(self):
        if not self.authorized():
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/tokenize":
            assert body["add_special_tokens"] is False
            ids = list(body["prompt"].encode("utf-8"))
            self.reply({"tokens": ids, "count": len(ids), "max_model_len": 9})
            return
        assert self.path == "/v1/completions" and body["model"] == model
        with (source / "requests.jsonl").open("a", encoding="utf-8") as file:
            print(json.dumps(body), file=file)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        def frame(piece, reason, usage=None):
            event = {"object": "text_completion", "model": model,
                     "choices": [{"index": 0, "text": piece, "finish_reason": reason}]}
            if usage:
                event["usage"] = usage
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.flush()
        if body["prompt"] in [list(b"hang"), list(b"pause")]:
            if body["prompt"] == list(b"pause"):
                frame("part", None)
            self.wfile.write(b"data: {")
            self.wfile.flush()
            (source / "stalled").touch()
            time.sleep(15)
            return
        frame("A", None)
        count = len(body["prompt"])
        frame("B", "length", {"prompt_tokens": count, "completion_tokens": 2, "total_tokens": count + 2})
        self.wfile.write(b"data: [DONE]\n\n")
ThreadingHTTPServer(("127.0.0.1", int(flag("--port"))), Handler).serve_forever()
'''


@pytest.fixture
def sidecar(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_5"}')
    (tmp_path / "model.safetensors").write_bytes(b"protocol fixture only")
    monkeypatch.setenv("TENSORFOLD_API_KEY", "inherited-shared-key")
    tok = SimpleNamespace(encode=lambda text, *, add_special_tokens: list(text.encode()))
    backend = engine.TensorFoldMLXBackend(str(tmp_path), {"default_context": 16})
    monkeypatch.setattr(backend, "_load_tokenizer", lambda source: tok)
    monkeypatch.setattr(engine, "tensorfold_executable", lambda: sys.executable)
    monkeypatch.setattr(engine, "_native_backend", lambda *_: ("auto", {"backends": ["fixture"]}))
    command = backend._command
    monkeypatch.setattr(backend, "_command", lambda *args: [sys.executable, "-u", "-c", SERVER, *command(*args)[1:]])
    try:
        assert backend.load_model(), backend.last_error
        yield backend, tmp_path
    finally:
        backend.unload_model()


def test_native_prompt_ids_context_and_branch_usage(sidecar):
    backend, path = sidecar
    process = backend._process
    assert backend.owned_process_ids() == [process.pid]
    assert backend.loaded_snapshot == str(path.resolve())
    assert backend.n_ctx == 9 and backend.api_prompt_count("é") == 2
    assert backend.generate_branches("é", 2, 100, [0, .4]) == ["AB", "AB"]
    calls = [json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines()]
    assert [row["temperature"] for row in calls] == [0, .4]
    assert all(row["prompt"] == [195, 169] and row["max_tokens"] == 7 for row in calls)
    assert backend.last_generation["stop_type"] == "length"
    assert backend.last_branch_usage == {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8}
    assert backend.last_generation_tokens == 2
    with pytest.raises(engine.NativeContextOverflow):
        list(backend.stream_generate_tokens("123456789"))
    backend.unload_model()
    assert process.poll() is not None and backend.owned_process_ids() == []


def test_token_mismatch_stops_before_generation(sidecar):
    backend, source = sidecar
    backend.tokenizer.encode = lambda *args, **kwargs: [999]
    with pytest.raises(RuntimeError, match="disagree"):
        backend.api_prompt_count("é")
    with pytest.raises(RuntimeError, match="disagree"):
        list(backend.stream_generate_tokens("é"))
    assert not (source / "requests.jsonl").exists()


@pytest.mark.parametrize("external", [True, False])
def test_cancel_blocked_partial_sse_without_publishing_branch(sidecar, external):
    import time
    backend, source = sidecar
    cancelled = threading.Event()
    result = []
    worker = threading.Thread(target=lambda: result.append(backend.generate_branches(
        "hang", 2, cancelled=cancelled.is_set if external else None)), daemon=True)
    worker.start()
    deadline = time.monotonic() + 4
    while not (source / "stalled").exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert (source / "stalled").exists()
    if external:
        cancelled.set()
    else:
        backend.cancel_generation()
    worker.join(2)
    assert not worker.is_alive() and result == [[]]
    assert backend.last_generation["status"] == "cancelled" and not backend.last_branch_usage
    assert list(backend.stream_generate_tokens("x")) == ["A", "B"]


def test_unload_can_reap_child_with_suspended_generator(sidecar):
    backend, _source = sidecar
    stream = backend.stream_generate_tokens("pause")
    process = backend._process
    assert next(stream) == "part"
    worker = threading.Thread(target=backend.unload_model, daemon=True)
    worker.start()
    worker.join(2)
    assert not worker.is_alive() and process.poll() is not None
    assert list(stream) == [] and backend.last_generation["status"] == "cancelled"


def test_cleanup_failure_retains_child_log_and_allows_retry(monkeypatch):
    backend = engine.TensorFoldMLXBackend("fixture", {})
    class Child:
        pid = 321
        status = None
        def poll(self):
            return self.status
        def terminate(self):
            pass
        def kill(self):
            pass
        def wait(self, timeout):
            raise subprocess.TimeoutExpired("fixture", timeout)
    child, log = Child(), io.BytesIO()
    backend._process, backend._log = child, log
    with pytest.raises(subprocess.TimeoutExpired):
        backend.unload_model()
    assert backend.owned_process_ids() == [321] and not log.closed and not backend.is_loaded
    with pytest.raises(subprocess.TimeoutExpired):
        backend.load_model()
    assert backend._process is child
    child.status = 0
    backend.unload_model()
    assert backend._process is None and log.closed


@pytest.mark.parametrize("requested,devices,switches,expected", [
    ("cuda", ["cuda"], ["auto", "cuda"], "cuda"),
    ("hip", ["hip"], ["auto", "rocm"], "rocm"),
    ("rocm", ["hip"], ["auto", "rocm"], "rocm"),
    ("xpu", ["xpu"], ["auto", "cuda"], "auto"),
    ("xpu", ["cuda"], ["auto", "cuda"], None),
    ("cuda", ["hip"], ["auto", "rocm"], None),
])
def test_backend_is_a_capability_of_selected_binary(monkeypatch, requested, devices, switches, expected):
    caps = {"engine": "zig", "backends": devices, "serve": {"flags": {"--backend": {"values": switches}}}}
    def run(command, **kwargs):
        assert command == ["native", "capabilities", "--json"]
        return SimpleNamespace(stdout=json.dumps(caps).encode())
    monkeypatch.setattr(subprocess, "run", run)
    if expected is None:
        with pytest.raises(RuntimeError, match="cannot serve"):
            engine._native_backend("native", requested)
    else:
        assert engine._native_backend("native", requested)[0] == expected


def _event(text="", reason=None, **overrides):
    return {"model": "fixture", "object": "text_completion",
            "choices": [{"index": 0, "text": text, "finish_reason": reason}], **overrides}


def _sse(*events):
    wire = b": keepalive\r\n\r\n"
    for e in events:
        data = e if isinstance(e, str) else json.dumps(e)
        wire += ("data: " + data + "\r\n\r\n").encode()
    return io.BytesIO(wire)


def test_usage_only_event_after_finish_and_multiline_sse():
    usage = {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
    raw = json.dumps(_event("é", "stop"), ensure_ascii=False).replace(' "object"', '\r\ndata: "object"')
    received = list(completion_events(_sse(raw, _event(choices=[], usage=usage), "[DONE]"), "fixture", 2))
    assert received == [("é", None, "stop"), ("", usage, None)]


@pytest.mark.parametrize("events", [
    ["[DONE]"], [_event("text")], [_event(reason="stop")],
    [_event(reason="stop"), _event("after"), "[DONE]"],
    [_event("text", model="other"), "[DONE]"], [_event(choices=[None]), "[DONE]"],
    [_event("text", reason="deadline"), "[DONE]"], [{"error": "engine failed"}],
    [_event(reason="stop", usage={"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}), "[DONE]"],
    [_event("text", choices=[{"index": 0, "text": {"invalid": 1}, "finish_reason": None}])],
])
def test_invalid_or_incomplete_stream_is_not_a_completed_generation(events):
    with pytest.raises(RuntimeError):
        list(completion_events(_sse(*events), "fixture", 2))


def test_size_limit_and_invalid_utf8():
    with pytest.raises(RuntimeError, match="line exceeded"):
        list(completion_events(io.BytesIO(b"data: " + b"a" * 1024**2), "fixture", 2))
    with pytest.raises(UnicodeDecodeError):
        list(completion_events(io.BytesIO(b"data: \xff\n\n"), "fixture", 2))


def test_source_requires_complete_exact_snapshot_and_disables_training(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model-1.safetensors").write_bytes(b"present")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"layer1": "model-1.safetensors", "layer2": "model-2.safetensors"}}))
    backend = engine.TensorFoldMLXBackend(str(tmp_path), {})
    with pytest.raises(RuntimeError, match="complete selected"):
        backend._source()
    assert not backend.training_ready() and not backend.supports_entropy
    with pytest.raises(RuntimeError, match="entropy"):
        backend.calculate_token_entropy("abc")
    with pytest.raises(RuntimeError, match="inference-only"):
        backend.train_mini_batch([], [])
