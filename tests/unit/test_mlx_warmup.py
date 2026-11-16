"""Warmup lifecycle tests with deterministic primitive fakes and no real MLX."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("mlx_warmup_test", ROOT / "core/mlx_warmup.py")
warmup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(warmup)


class Tensor:
    def __init__(self, data):
        self.data = data
    def __add__(self, other):
        return Tensor([a + b for a, b in zip(self.data, other.data)])


class FakeMX:
    gpu = "fake-gpu"
    float32 = "float32"
    def __init__(self):
        self.calls = []
        self.random = SimpleNamespace(state=["untouched RNG"])
        self.metal = SimpleNamespace(is_available=lambda: True)
    @contextmanager
    def stream(self, device):
        self.calls.append(("enter", device))
        yield
        self.calls.append(("exit", device))
    def array(self, data, dtype):
        self.calls.append(("array", data, dtype))
        return Tensor(data)
    def eval(self, result):
        self.calls.append(("eval", result.data))


def test_default_disabled_has_no_runtime_calls():
    runtime = FakeMX()
    helper = warmup.SafeMLXWarmup()
    assert not helper.prepare(mx=runtime)
    assert not runtime.calls


def test_constant_kernel_only_preserves_rng_and_restores_stream():
    runtime = FakeMX()
    helper = warmup.SafeMLXWarmup(enabled=True)
    before_threads = set(threading.enumerate())
    assert helper.prepare(mx=runtime, runtime_lock=threading.RLock())
    assert runtime.calls == [("enter", "fake-gpu"), ("array", [0.0, 1.0, 2.0, 3.0], "float32"),
                             ("eval", [0.0, 2.0, 4.0, 6.0]), ("exit", "fake-gpu")]
    assert runtime.random.state == ["untouched RNG"]
    assert set(threading.enumerate()) == before_threads
    assert not any(value is runtime for value in vars(helper).values())


def test_idle_lifecycle_touch_close_and_no_post_unload_calls():
    now = [10.0]
    runtime = FakeMX()
    helper = warmup.SafeMLXWarmup(enabled=True, idle_seconds=30, clock=lambda: now[0])
    assert helper.prepare(mx=runtime)
    now[0] = 20
    assert not helper.prepare(mx=runtime)
    helper.touch()
    now[0] = 45
    assert not helper.prepare(mx=runtime)
    now[0] = 51
    assert helper.prepare(mx=runtime)
    helper.close()
    now[0] = 999
    previous = list(runtime.calls)
    assert not helper.prepare(mx=runtime)
    helper.touch()
    assert helper._last_activity is None
    assert runtime.calls == previous
    assert helper.telemetry()["kernel_warmup_runs"] == 2
    assert not helper.telemetry()["kernel_warmup_background"]


def test_busy_runtime_is_skipped_without_waiting_or_running_kernel():
    runtime = FakeMX()
    lock = threading.Lock()
    lock.acquire()
    try:
        helper = warmup.SafeMLXWarmup(enabled=True)
        assert not helper.prepare(mx=runtime, runtime_lock=lock)
        assert lock.locked()
        assert not runtime.calls
    finally:
        lock.release()


def test_pressure_missing_gpu_and_failures_are_safe_and_release_lock():
    helper = warmup.SafeMLXWarmup(enabled=True)
    runtime = FakeMX()
    lock = threading.Lock()
    assert not helper.prepare(mx=runtime, runtime_lock=lock, memory_ok=lambda: False)
    assert not lock.locked()
    assert not runtime.calls
    runtime.metal.is_available = lambda: False
    assert not helper.prepare(mx=runtime, runtime_lock=lock)
    assert not lock.locked()
    runtime.metal.is_available = lambda: True
    with patch.object(runtime, "eval", side_effect=RuntimeError("GPU unavailable")):
        assert not helper.prepare(mx=runtime, runtime_lock=lock)
    assert not lock.locked()
    assert helper.telemetry()["kernel_warmup_failures"] == 1
    assert runtime.random.state == ["untouched RNG"]
