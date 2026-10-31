"""Lifecycle and concurrency checks independent of any real model download."""
import threading
from types import SimpleNamespace

from core.native_speed import install_native_runtime_lock
from core.speed_control import install_speed_control


class FakeBackend:
    def __init__(self):
        self.model = object()
        self._runtime_model_lock = threading.RLock()
        self.calls = []

    def configure_auto_speed(self, **kwargs):
        self.calls.append(kwargs)
        kwargs["progress"]("measured")
        return {"status": "baseline", "message": "Previous settings retained."}


class FakeEngine:
    def __init__(self):
        self.settings = SimpleNamespace(use_mock=False)
        self._model_lock = threading.Lock()
        self.active_backend = None
        self.mlx_backend = None

    def load_model(self, model="test"):
        with self._model_lock:
            self.unload_model()
            self.active_backend = "mlx"
            self.mlx_backend = FakeBackend()
            return {"status": "loaded", "model": model}

    def unload_model(self):
        self.mlx_backend = None
        self.active_backend = None
        return {"status": "unloaded"}


install_speed_control(FakeEngine)


def test_successful_load_configures_once_inside_reentrant_lifecycle():
    engine = FakeEngine()
    engine._context_budget_tokens = 4096
    engine.speed_memory_limit_gb = 12
    result = engine.load_model()
    assert result["status"] == "loaded"
    assert result["speed"]["status"] == "baseline"
    assert len(engine.mlx_backend.calls) == 1
    assert engine.mlx_backend.context_budget_tokens == 4096
    assert engine.mlx_backend.speed_memory_limit_gb == 12
    assert engine.speed_tuning_active is False


def test_saved_off_is_forwarded_and_mock_load_does_not_calibrate():
    engine = FakeEngine()
    engine.auto_speed_enabled = False
    engine.load_model()
    assert engine.mlx_backend.calls[0]["enabled"] is False
    engine.settings.use_mock = True
    engine.load_model()
    assert engine.mlx_backend.calls == []


def test_native_failure_does_not_turn_successful_load_into_failure(monkeypatch):
    def fail(self, **kwargs):
        raise RuntimeError("unsupported native API")
    monkeypatch.setattr(FakeBackend, "configure_auto_speed", fail)
    engine = FakeEngine()
    result = engine.load_model()
    assert result["status"] == "loaded"
    assert result["speed"]["status"] == "error"
    assert engine.mlx_backend.model is not None
    assert not engine.speed_tuning_active


def test_busy_backend_does_not_flip_saved_mode_or_enqueue_a_tune():
    engine = FakeEngine()
    engine.load_model()
    entered, release = threading.Event(), threading.Event()

    def hold_runtime():
        with engine.mlx_backend._runtime_model_lock:
            entered.set()
            release.wait(2)

    worker = threading.Thread(target=hold_runtime)
    worker.start()
    assert entered.wait(1)
    try:
        assert engine.configure_auto_speed(enabled=False)["status"] == "busy"
        assert engine.auto_speed_enabled is True
        assert len(engine.mlx_backend.calls) == 1
    finally:
        release.set()
        worker.join(2)


def test_failed_disable_keeps_mode_consistent_with_working_settings():
    engine = FakeEngine()
    engine.load_model()
    engine.mlx_backend.configure_auto_speed = lambda **kw: {"status": "error", "message": "setter unavailable"}
    assert engine.configure_auto_speed(enabled=False)["status"] == "error"
    assert engine.auto_speed_enabled is True


def test_cancelled_enable_preserves_saved_off_mode():
    engine = FakeEngine()
    engine.auto_speed_enabled = False
    engine.load_model()
    engine.mlx_backend.configure_auto_speed = lambda **kw: {"status": "cancelled", "message": "stopped"}
    assert engine.configure_auto_speed(enabled=True)["status"] == "cancelled"
    assert engine.auto_speed_enabled is False


def test_unload_cancels_tune_before_waiting_for_model_lock():
    engine = FakeEngine()
    engine.load_model()
    backend = engine.mlx_backend
    entered = threading.Event()
    release = threading.Event()

    def tune(**kwargs):
        entered.set()
        assert release.wait(2)
        assert kwargs["cancelled"]()
        assert backend.model is not None
        return {"status": "cancelled", "message": "Stopped"}

    backend.configure_auto_speed = tune
    tuning = threading.Thread(target=engine.configure_auto_speed)
    tuning.start()
    assert entered.wait(1)
    unloading = threading.Thread(target=engine.unload_model)
    unloading.start()
    assert engine._speed_cancel_event.wait(1)
    release.set()
    tuning.join(2)
    unloading.join(2)
    assert not tuning.is_alive() and not unloading.is_alive()
    assert engine.active_backend is None
    assert not engine.speed_tuning_active


def test_direct_native_stream_owns_runtime_until_generator_closes():
    class Native:
        def stream_generate_tokens(self):
            yield "one"
            yield "two"

        def train_mini_batch(self):
            return "trained"

    install_native_runtime_lock(Native)
    backend = Native()
    stream = backend.stream_generate_tokens()
    assert next(stream) == "one"
    entered = threading.Event()

    def train():
        backend.train_mini_batch()
        entered.set()

    worker = threading.Thread(target=train)
    worker.start()
    assert not entered.wait(0.03)
    stream.close()
    assert entered.wait(1)
    worker.join(1)
