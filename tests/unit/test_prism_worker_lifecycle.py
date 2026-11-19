"""A surviving ordinary Prism worker remains owned until its confirmed exit."""
import io
import subprocess
import sys

import pytest

from core.engines.prism_gguf_engine import PrismGGUFReasoningBackend
from tests.unit.test_strata_integration import engine_for, pro_module


class Worker:
    def __init__(self, exits_on="graceful", stdin=True):
        self.pid = 12345
        self.stdin = io.StringIO() if stdin else None
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.exits_on = exits_on
        self.phase = "graceful"
        self.returncode = None
        self.events = []

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.events.append(("wait", timeout))
        if self.phase == self.exits_on:
            self.returncode = 0
            return 0
        raise subprocess.TimeoutExpired("prism", timeout)

    def terminate(self):
        self.events.append("terminate")
        self.phase = "terminate"

    def kill(self):
        self.events.append("kill")
        self.phase = "kill"


def backend_for(worker):
    backend = PrismGGUFReasoningBackend.__new__(PrismGGUFReasoningBackend)
    backend._worker = worker
    backend.model = object()
    backend.chat_handler = object()
    backend._speed_tuner = object()
    return backend


@pytest.mark.parametrize("exits_on", ["graceful", "terminate", "kill"])
def test_shutdown_escalates_only_until_confirmed_exit_and_closes_pipes(exits_on):
    worker = Worker(exits_on)
    backend = backend_for(worker)
    backend.unload_model()
    assert worker.poll() == 0
    assert backend._worker is None and backend.owned_process_ids() == []
    assert backend.model is None and backend._speed_tuner is None
    assert backend.chat_handler is None
    assert all(getattr(worker, name).closed for name in ("stdin", "stdout", "stderr"))
    assert ("terminate" in worker.events) == (exits_on != "graceful")
    assert ("kill" in worker.events) == (exits_on == "kill")
    backend.unload_model()  # Repeated cleanup is safe.


def test_worker_with_no_shutdown_pipe_is_still_stopped():
    worker = Worker("terminate", stdin=False)
    backend = backend_for(worker)
    backend.unload_model()
    assert worker.poll() == 0 and backend._worker is None
    assert "terminate" in worker.events and "kill" not in worker.events


def test_signal_errors_do_not_drop_a_live_owned_worker(monkeypatch):
    worker = Worker("never")
    backend = backend_for(worker)

    def denied():
        raise PermissionError("OS rejected process signal")

    monkeypatch.setattr(worker, "terminate", denied)
    monkeypatch.setattr(worker, "kill", denied)
    with pytest.raises(RuntimeError, match="still running") as error:
        backend.unload_model()
    assert isinstance(error.value.__cause__, PermissionError)
    assert backend._worker is worker and backend.owned_process_ids() == [12345]
    assert not worker.stdin.closed and backend.model is not None


def test_failed_stop_retains_every_owner_and_blocks_replacement(tmp_path, monkeypatch):
    worker = Worker("never")
    backend = backend_for(worker)
    owners = (backend.model, backend.chat_handler, backend._speed_tuner)
    path = tmp_path / "selected.gguf"
    path.write_bytes(b"unused: a replacement must not be inspected or launched")
    backend.model_path = str(path)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("replacement worker launched"))

    with pytest.raises(RuntimeError, match="still running.*Retry unloading"):
        backend.unload_model()
    assert backend._worker is worker and backend.owned_process_ids() == [12345]
    assert (backend.model, backend.chat_handler, backend._speed_tuner) == owners
    assert not worker.stdin.closed and not worker.stdout.closed
    assert worker.events[-1] == ("wait", 2.0)  # kill must also be followed by wait.
    with pytest.raises(RuntimeError, match="still running"):
        backend.load_model()
    assert backend._worker is worker

    worker.exits_on = "graceful"
    worker.phase = "graceful"
    backend.unload_model()
    assert backend._worker is None and backend.owned_process_ids() == []
    assert worker.stdin.closed and worker.stdout.closed


def test_pro_model_switch_preserves_surviving_prism_owner(pro_module):
    worker = Worker("never")
    backend = backend_for(worker)
    engine = engine_for(pro_module)
    engine.gguf_backend = backend
    engine.active_backend = "gguf"
    engine.active_model_name = "selected-prism"
    engine.active_model_path = "selected.gguf"
    assert engine.unload_model()["status"] == "error"
    result = engine.load_model("replacement")
    assert result["status"] == "error" and result["requested_model"] == "replacement"
    assert engine.active_model_name == "selected-prism"
    assert engine.active_model_path == "selected.gguf"
    assert engine.gguf_backend is backend and backend.owned_process_ids() == [12345]


def test_repeated_real_children_are_reaped_and_pipe_handles_closed():
    # Real OS subprocess cleanup without downloading models or native libraries.
    script = "import json, sys\nfor line in sys.stdin:\n if json.loads(line).get('op') == 'shutdown': break\n"
    for _ in range(4):
        worker = subprocess.Popen([sys.executable, "-u", "-c", script],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True)
        backend = backend_for(worker)
        try:
            backend.unload_model()
            assert worker.poll() == 0
            assert backend._worker is None and backend.owned_process_ids() == []
            assert worker.stdin.closed and worker.stdout.closed and worker.stderr.closed
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait(timeout=3)
            for pipe in (worker.stdin, worker.stdout, worker.stderr):
                pipe.close()
