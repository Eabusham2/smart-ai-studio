"""Model-free native integration tests: byte contracts and a synthetic HTTP server.

No test loads weights, starts llama-server, executes a model, or treats these
fixtures as numerical-parity or performance evidence.
"""
import hashlib
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import threading
from types import SimpleNamespace

import pytest

from core.speculative_runtime import (
    DFLASH2_SHA256, EXACTNESS_CONTRACT, MTP_SHA256, SOURCE_ARCHIVE_SHA256,
    TARGET_SHA256, normalize_speculative_mode, plan_speculation,
    sha256_file, speculative_arguments, validate_runtime_receipt,
)
from core.engines.prism_speculative_engine import PrismSpeculativeReasoningBackend, _NativeModelProxy


@pytest.mark.parametrize("value,expected", [("Off", "none"), ("PLD", "lookup"), ("AUTO", "auto"), ("DFlash2", "dflash2")])
def test_modes(value, expected):
    assert normalize_speculative_mode(value) == expected


def test_unknown_modes_and_artifacts_are_refused():
    with pytest.raises(ValueError):
        normalize_speculative_mode("jacobi")
    with pytest.raises(ValueError, match="pinned original"):
        plan_speculation("auto", target_sha256="renamed-foreign-file")


def test_auto_does_not_replace_target_or_stack_heads():
    assert plan_speculation("auto", target_sha256=TARGET_SHA256).active_mode == "lookup"
    assert plan_speculation("auto", target_sha256=MTP_SHA256).active_mode == "mtp"
    assert plan_speculation("auto", target_sha256=TARGET_SHA256, draft_sha256=DFLASH2_SHA256).active_mode == "dflash2"
    with pytest.raises(ValueError, match="combined"):
        plan_speculation("mtp", target_sha256=TARGET_SHA256)
    with pytest.raises(ValueError, match="external"):
        plan_speculation("auto", target_sha256=MTP_SHA256, draft_sha256=DFLASH2_SHA256)
    with pytest.raises(ValueError, match="matching"):
        plan_speculation("dflash2", target_sha256=TARGET_SHA256, draft_sha256="wrong-head")


@pytest.mark.parametrize("count", [0, -1, 65, True, 2.5, "2"])
def test_draft_counts_are_bounded_integers(count):
    with pytest.raises(ValueError, match="integer"):
        plan_speculation("lookup", target_sha256=TARGET_SHA256, max_draft_tokens=count)


def test_native_flags_use_actual_head_protocol():
    mtp = plan_speculation("mtp", target_sha256=MTP_SHA256)
    assert speculative_arguments(mtp) == ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "--spec-draft-p-min", "0"]
    dflash = plan_speculation("dflash2", target_sha256=TARGET_SHA256, draft_sha256=DFLASH2_SHA256)
    args = speculative_arguments(dflash, "/local path/head.gguf")
    assert args[1] == "draft-dflash"
    assert args[-2:] == ["--model-draft", "/local path/head.gguf"]
    assert args[3] == "3"
    assert not any(flag in args for flag in ("--temp", "--top-p", "--cache-type-k", "--ctx-size"))
    assert speculative_arguments(plan_speculation("off", target_sha256=MTP_SHA256)) == ["--spec-type", "none"]


def _receipt(tmp_path):
    server = tmp_path / "llama-server"
    server.write_bytes(b"synthetic executable fixture; never executed")
    data = {"schema_version": 1, "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "exactness_contract": EXACTNESS_CONTRACT,
            "patcher_sha256": sha256_file(str(Path(__file__).resolve().parents[2] / "core" / "native_exactness.py")),
            "binary_sha256": hashlib.sha256(server.read_bytes()).hexdigest()}
    receipt = tmp_path / "llama-server.smartai.json"
    receipt.write_text(json.dumps(data))
    return server, receipt, data


def test_receipt_binds_executable_and_exact_runtime_contract(tmp_path):
    server, receipt, data = _receipt(tmp_path)
    assert validate_runtime_receipt(str(server)) == data
    assert sha256_file(str(server)) == data["binary_sha256"]
    server.write_bytes(b"different executable")
    with pytest.raises(ValueError, match="changed"):
        validate_runtime_receipt(str(server))
    data["binary_sha256"] = sha256_file(str(server))
    data["exactness_contract"] = "stock-procreations"
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="serial target"):
        validate_runtime_receipt(str(server))
    data["exactness_contract"] = EXACTNESS_CONTRACT
    data["patcher_sha256"] = "stale-patch"
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="different exactness patch"):
        validate_runtime_receipt(str(server))


def test_native_command_isolated_exactness_and_context(tmp_path, monkeypatch):
    model_path = str(tmp_path / "model path" / "original.gguf")
    server_path = str(tmp_path / "runtime path" / "llama-server")
    backend = PrismSpeculativeReasoningBackend(model_path, server_path,
                                               mode="lookup", n_ctx=8912, n_gpu_layers=15)
    backend._plan = plan_speculation("lookup", target_sha256=TARGET_SHA256)
    args = backend._build_command(9001)
    assert args[args.index("-m") + 1] == model_path
    assert args[0] == server_path
    assert args[args.index("-c") + 1] == "8912"
    assert args[args.index("-np") + 1] == "1"
    assert args[args.index("-ub") + 1] == "512"  # prefill remains batched
    assert "--smartai-exact-target" not in args  # the patched binary enables it unconditionally
    assert args[args.index("--cache-type-k") + 1] == "f16"
    assert args[args.index("--cache-type-v") + 1] == "f16"
    assert "--no-context-shift" in args
    monkeypatch.setenv("LLAMA_ARG_MODEL", "unwanted-remote-model")
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_K", "q4_0")
    assert "LLAMA_ARG_MODEL" not in backend._runtime_env()
    assert "LLAMA_ARG_CACHE_TYPE_K" not in backend._runtime_env()
    assert backend.get_speculative_telemetry()["output_equivalence_measured"] is False


def test_preflight_never_starts_inference_and_rejects_missing_receipt(tmp_path, monkeypatch):
    server = tmp_path / "llama-server"
    server.write_bytes(b"fixture")
    backend = PrismSpeculativeReasoningBackend(str(tmp_path / "selected.gguf"), str(server))
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: pytest.fail("No model may start in preflight"))
    with pytest.raises(FileNotFoundError):
        backend.preflight()
    assert backend.model is None


@pytest.fixture
def http_backend():
    received = []
    mode = {"response": "normal"}
    blocked = threading.Event()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_GET(self):
            self._json({"status": "ok"})

        def _json(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, data))
            if self.path == "/tokenize":
                if mode["response"] == "blocked-tokenizer":
                    blocked.set()
                    release.wait(5)
                    return
                self._json({"tokens": [1, 2, 3]})
                return
            if mode["response"] == "error":
                self._json({"error": "context exceeded"}, status=400)
                return
            if mode["response"] == "blocked-headers":
                blocked.set()
                release.wait(5)
                return
            if mode["response"] in {"blocked", "blocked-close", "blocked-partial"}:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                if mode["response"] == "blocked-close":
                    self.send_header("Connection", "close")
                self.end_headers()
                if mode["response"] == "blocked-partial":
                    self.wfile.write(b'data: {"content":"unfinished')
                self.wfile.flush()
                blocked.set()
                release.wait(5)
                return
            if mode["response"] == "truncated":
                body = b'data: {"content":"prefix"}\n\n'
            elif mode["response"] == "malformed":
                body = b'data: invalid json\n\n'
            else:
                body = (': heartbeat\r\n\r\ndata: {"content":"hello "}\r\n\r\n'
                        'data: {"content":"世界",\n'
                        'data: "stop":false}\n\n'
                        'data: {"content":"", "stop":true, "tokens_evaluated":3, "tokens_predicted":2, '
                        '"timings":{"draft_n":8,"draft_n_accepted":5,"predicted_per_second":4.25}}\n\n').encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            if mode["response"] == "chunked":
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for start in range(0, len(body), 7):
                    chunk = body[start:start + 7]
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
                return
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    backend = PrismSpeculativeReasoningBackend("unused.gguf", "unused-server", mode="mtp")
    backend._port = server.server_port
    backend.is_loaded = True
    backend._plan = plan_speculation("mtp", target_sha256=MTP_SHA256)
    backend.model = backend.tokenizer = _NativeModelProxy(backend)
    try:
        yield backend, received, mode, blocked
    finally:
        release.set()
        backend.cancel_generation()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_real_http_stream_and_tokenize_keep_request_sampling(http_backend):
    backend, received, _, _ = http_backend
    assert backend.count_tokens([{"content": "literal <|im_start|>"}]) == 3
    chunks = list(backend.stream_generate_tokens("verbatim\n prompt", 17, 0.0, 0.0, seed=123))
    assert chunks == ["hello ", "世界"]
    payload = received[-1][1]
    assert payload["prompt"] == "verbatim\n prompt"
    assert payload["n_predict"] == 17
    assert payload["temperature"] == payload["top_p"] == 0.0
    assert payload["seed"] == 123
    assert payload["repeat_penalty"] == 1.1
    assert payload["min_p"] == 0.05
    assert payload["cache_prompt"] is False
    assert backend.get_speculative_telemetry()["draft_tokens_accepted"] == 5
    assert backend.get_speculative_telemetry()["speedup"] is None
    assert backend.get_speculative_telemetry()["decode_tokens_per_second"] == 4.25
    assert backend.last_generation["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert backend._active_connection is None


def test_context_overflow_refuses_before_completion(http_backend):
    backend, received, _, _ = http_backend
    backend.n_ctx = 5
    with pytest.raises(ValueError, match="no prompt or output budget was truncated"):
        list(backend.stream_generate_tokens("full prompt", max_tokens=3))
    assert [path for path, _ in received] == ["/tokenize"]
    assert received[0][1]["parse_special"] is True


def test_native_lifecycle_starts_only_explicit_server_and_releases_owner(monkeypatch):
    backend = PrismSpeculativeReasoningBackend("selected.gguf", "selected-server", mode="off")
    backend._plan = plan_speculation("off", target_sha256=TARGET_SHA256)
    calls = []

    class Process:
        stopped = False

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.stopped = True

        def wait(self, timeout=None):
            assert self.stopped
            return 0

    process = Process()
    monkeypatch.setattr(backend, "preflight", lambda **kwargs: {"status": "ready"})
    monkeypatch.setattr(backend, "_free_port", lambda: 9191)
    monkeypatch.setattr(backend, "_request", lambda *args, **kwargs: {"status": "ok"})

    def start(command, **kwargs):
        calls.append(command)
        assert kwargs["stdin"] is not None
        return process

    monkeypatch.setattr("subprocess.Popen", start)
    assert backend.load_model()
    assert backend.model is not None
    assert calls[0][0] == backend.server_path
    assert calls[0][calls[0].index("-m") + 1] == backend.model_path
    assert calls[0][-2:] == ["--spec-type", "none"]
    backend.unload_model()
    assert process.stopped
    assert backend._process is None and backend.model is None
    assert backend.is_loaded is False


def test_native_startup_exit_is_not_reported_as_loaded(monkeypatch):
    backend = PrismSpeculativeReasoningBackend("selected.gguf", "selected-server")
    backend._plan = plan_speculation("lookup", target_sha256=TARGET_SHA256)
    monkeypatch.setattr(backend, "preflight", lambda **kwargs: {"status": "ready"})
    monkeypatch.setattr(backend, "_free_port", lambda: 9191)
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: SimpleNamespace(poll=lambda: 7))
    assert not backend.load_model()
    assert "exited" in backend.last_error
    assert backend.model is None and backend._process is None


@pytest.mark.parametrize("response", ["error", "truncated", "malformed"])
def test_http_errors_never_become_silent_success(http_backend, response):
    backend, _, mode, _ = http_backend
    mode["response"] = response
    with pytest.raises((RuntimeError, ValueError)):
        list(backend.stream_generate_tokens("test"))
    assert backend._active_connection is None


@pytest.mark.parametrize("response", ["blocked", "blocked-close", "blocked-headers", "blocked-tokenizer", "blocked-partial"])
def test_cancel_interrupts_a_blocked_socket_read(http_backend, response, monkeypatch):
    backend, _, mode, blocked = http_backend
    mode["response"] = response
    # Reproduce platforms where cross-thread shutdown cannot wake the reader.
    # Cancellation must depend on the event below buffering, not OS wakeups.
    monkeypatch.setattr(socket.socket, "shutdown", lambda *_args: None)
    failures = []

    def run():
        try:
            list(backend.stream_generate_tokens("test"))
        except Exception as exc:
            failures.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    assert blocked.wait(2)
    backend.cancel_generation()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert failures == []


def test_unload_preserves_process_owner_if_termination_fails():
    backend = PrismSpeculativeReasoningBackend("unused", "unused")
    owner = SimpleNamespace(poll=lambda: None,
                            terminate=lambda: (_ for _ in ()).throw(OSError("cannot terminate")))
    backend._process = owner
    with pytest.raises(OSError):
        backend.unload_model()
    assert backend._process is owner


def test_no_permissive_training_or_media_or_fake_entropy():
    backend = PrismSpeculativeReasoningBackend("unused", "unused")
    assert backend.runtime == "prism_speculative"
    assert backend._runtime_model_lock is backend._io_lock
    assert not backend.training_ready()
    assert not backend.supports_media_input("image")
    with pytest.raises(RuntimeError, match="immutable"):
        backend.train_mini_batch()
    with pytest.raises(RuntimeError, match="entropy"):
        backend.calculate_token_entropy("prompt")


def test_branch_usage_is_summed_from_native_counters_only(http_backend):
    backend, _, _, _ = http_backend
    assert backend.generate_branches("test", branch_count=2) == ["hello 世界", "hello 世界"]
    assert backend.last_branch_usage == {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10}
    backend._record_generation({"stop": True})
    assert backend.last_generation["usage"] is None
    backend._last_timings = {"predicted_per_second": float("nan")}
    assert backend.get_speculative_telemetry()["decode_tokens_per_second"] is None


def test_pre_cancelled_operations_do_not_start_or_clear_user_cancel(http_backend, monkeypatch):
    backend, received, _, _ = http_backend
    stop = threading.Event()
    stop.set()
    backend.last_generation = {"usage": {"completion_tokens": 99}}
    assert list(backend.stream_generate_tokens("test", cancelled=stop.is_set)) == []
    assert received == []
    assert backend.last_generation == {"status": "cancelled", "usage": None}
    assert backend.generate_branches("test", branch_count=3, cancelled=stop.is_set) == []
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: pytest.fail("Cancelled load started a process"))
    assert not backend.load_model(cancelled=stop.is_set)
    assert "cancelled" in backend.last_error


def test_callback_cancels_blocked_native_request_and_stops_remaining_branches(http_backend, monkeypatch):
    backend, received, mode, blocked = http_backend
    mode["response"] = "blocked-close"
    monkeypatch.setattr(socket.socket, "shutdown", lambda *_args: None)
    stop = threading.Event()
    failures = []

    def run():
        try:
            backend.generate_branches("test", branch_count=4, cancelled=stop.is_set)
        except Exception as exc:
            failures.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    assert blocked.wait(2)
    stop.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert failures == []
    assert sum(path == "/completion" for path, _ in received) == 1
    assert backend.last_branch_usage is None


def test_cancellable_reader_preserves_chunk_framing_and_fragmented_utf8(http_backend):
    backend, _, mode, _ = http_backend
    mode["response"] = "chunked"
    assert list(backend.stream_generate_tokens("test")) == ["hello ", "世界"]
    assert backend.last_generation["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def test_owned_process_ids_never_scans_unrelated_processes():
    backend = PrismSpeculativeReasoningBackend("unused", "unused")
    assert backend.owned_process_ids() == []
    backend._process = SimpleNamespace(pid=12345, poll=lambda: None)
    assert backend.owned_process_ids() == [12345]
    backend._process = SimpleNamespace(pid=12345, poll=lambda: 0)
    assert backend.owned_process_ids() == []


def test_hash_verification_is_cancellable_without_deserializing(tmp_path):
    artifact = tmp_path / "bytes"
    artifact.write_bytes(b"small synthetic bytes")
    with pytest.raises(RuntimeError, match="cancelled"):
        sha256_file(str(artifact), cancelled=lambda: True)


def test_cancel_during_preflight_never_spawns_native_server(monkeypatch):
    backend = PrismSpeculativeReasoningBackend("unused", "unused")
    stop = threading.Event()

    def preflight(**kwargs):
        stop.set()
        return {"status": "ready"}

    monkeypatch.setattr(backend, "preflight", preflight)
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: pytest.fail("Cancelled load started a process"))
    assert not backend.load_model(cancelled=stop.is_set)
    assert "cancelled" in backend.last_error


def test_cancel_during_tokenization_cannot_start_completion(http_backend):
    backend, received, _, _ = http_backend
    stop = threading.Event()

    def tokenize(prompt):
        stop.set()
        return [1, 2, 3]

    backend.tokenizer = SimpleNamespace(encode=tokenize)
    assert backend.generate_branches("test", branch_count=4, cancelled=stop.is_set) == []
    assert received == []
    assert backend.last_generation["status"] == "cancelled"
    assert backend.last_branch_usage is None


def test_new_request_after_cancel_restarts_with_exact_selected_context_budget(http_backend):
    backend, received, _, _ = http_backend
    backend.cancel_generation()
    backend.context_budget_tokens = 10
    assert "".join(backend.stream_generate_tokens("test", max_tokens=100, cancelled=lambda: False)) == "hello 世界"
    assert [path for path, _ in received] == ["/tokenize", "/completion"]
    assert received[-1][1]["n_predict"] == 7  # Exact tokenizer returns three IDs.
    assert received[-1][1]["prompt"] == "test"
    backend.context_budget_tokens = 3
    with pytest.raises(ValueError, match="Prompt exhausts"):
        list(backend.stream_generate_tokens("test", max_tokens=100))
    assert sum(path == "/completion" for path, _ in received) == 1
