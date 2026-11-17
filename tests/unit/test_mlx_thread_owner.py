"""A real background OS thread owns all MLX work; no model weights required."""
from __future__ import annotations

import sys
import threading
import time
from types import ModuleType

import pytest

from core.mlx_thread_owner import MLXThreadOwner, install_mlx_thread_ownership


@pytest.fixture
def fake_stream_packages(monkeypatch):
    """Reproduce a module-global MLX-LM stream first initialized on Tk's thread."""
    main = threading.get_ident()
    mlx = ModuleType("mlx")
    mlx.__path__ = []
    core = ModuleType("mlx.core")
    core.default_device = lambda: "gpu"
    core.new_thread_local_stream = lambda device: (device, threading.get_ident())
    mlx.core = core
    lm = ModuleType("mlx_lm")
    lm.__path__ = []
    generate = ModuleType("mlx_lm.generate")
    generate.generation_stream = ("gpu", main)
    lm.generate = generate
    vlm = ModuleType("mlx_vlm")
    vlm.__path__ = []
    vlm_generate = ModuleType("mlx_vlm.generate")
    vlm_generate.generation_stream = ("gpu", main)
    vlm.generate = vlm_generate
    for name, module in (("mlx", mlx), ("mlx.core", core),
                         ("mlx_lm", lm), ("mlx_lm.generate", generate),
                         ("mlx_vlm", vlm), ("mlx_vlm.generate", vlm_generate)):
        monkeypatch.setitem(sys.modules, name, module)
    yield generate, vlm_generate


def test_import_time_mlx_generation_streams_rebound_on_owner(fake_stream_packages):
    lm, vlm = fake_stream_packages
    main_thread = threading.get_ident()
    owner = MLXThreadOwner()
    try:
        owned = owner.call(threading.get_ident)
        assert owned != main_thread
        assert lm.generation_stream == ("gpu", owned)
        assert vlm.generation_stream == ("gpu", owned)
        assert owner.call(lambda: lm.generation_stream[1] == threading.get_ident())
    finally:
        owner.shutdown()


def test_load_stream_pro_tune_unload_on_same_thread_and_stream_close_on_owner(fake_stream_packages):
    lm, vlm = fake_stream_packages

    class FakeBackend:
        def __init__(self):
            self._mlx_owner = None
            self._mlx_owner_init_lock = threading.Lock()
            self.owner_thread = None
            self.closed_thread = None
            self.unloaded_thread = None
            self.generated = []

        def load_model(self):
            self.owner_thread = threading.get_ident()
            assert lm.generation_stream[1] == self.owner_thread
            return True

        def calculate_token_entropy(self, prompt):
            assert threading.get_ident() == self.owner_thread
            return 0.55

        def configure_auto_speed(self):
            assert threading.get_ident() == self.owner_thread
            return {"status": "baseline"}

        def generate_branches(self, prompt):
            assert threading.get_ident() == self.owner_thread
            self.generated.append(prompt)
            return ["candidate A", "candidate B"]

        def stream_generate_tokens(self, prompt):
            assert threading.get_ident() == self.owner_thread
            try:
                for number in range(100):
                    yield str(number)
                    time.sleep(0.002)
            finally:
                self.closed_thread = threading.get_ident()

        def unload_model(self):
            assert threading.get_ident() == self.owner_thread
            self.unloaded_thread = threading.get_ident()
            return {"status": "unloaded"}

    install_mlx_thread_ownership(FakeBackend)
    backend = FakeBackend()
    main_thread = threading.get_ident()
    assert backend.load_model() is True
    assert backend.owner_thread != main_thread
    assert backend.calculate_token_entropy("prompt") == 0.55
    assert backend.generate_branches("prompt") == ["candidate A", "candidate B"]
    assert backend.configure_auto_speed()["status"] == "baseline"
    started = backend.stream_generate_tokens("test")
    assert next(started) == "0"
    assert next(started) == "1"
    started.close()
    # Generator close must run on the MLX thread, not on the chat thread.
    assert backend.closed_thread == backend.owner_thread
    assert backend.unload_model()["status"] == "unloaded"
    assert backend.unloaded_thread == backend.owner_thread
    assert backend._mlx_owner is None
    # Reloading gives a new valid owner without reusing a dead stream.
    assert backend.load_model() is True
    assert lm.generation_stream[1] == backend.owner_thread
    assert backend.unload_model()["status"] == "unloaded"


def test_cross_thread_generator_exception_reaches_caller_with_clean_shutdown(fake_stream_packages):
    class Broken:
        def __init__(self):
            self._mlx_owner = None
            self._mlx_owner_init_lock = threading.Lock()
            self.finally_thread = None

        def stream_generate_tokens(self, prompt):
            try:
                yield "good"
                raise RuntimeError("native generation failed")
            finally:
                self.finally_thread = threading.get_ident()

        def unload_model(self):
            return {"status": "unloaded"}

    install_mlx_thread_ownership(Broken)
    obj = Broken()
    stream = obj.stream_generate_tokens("prompt")
    assert next(stream) == "good"
    with pytest.raises(RuntimeError, match="native generation failed"):
        next(stream)
    assert obj.finally_thread != threading.get_ident()
    obj.unload_model()
    assert obj._mlx_owner is None


def test_cancellation_while_producer_backpressured_does_not_deadlock(fake_stream_packages):
    owner = MLXThreadOwner()
    closed = threading.Event()

    def fast_stream():
        try:
            for i in range(2000):
                yield str(i)
        finally:
            closed.set()

    iterator = owner.stream(fast_stream)
    assert next(iterator) == "0"
    time.sleep(0.04)  # force its bounded queue to fill
    iterator.close()
    assert closed.wait(2)
    owner.shutdown()


def test_no_model_code_is_executed_on_tk_main_thread(fake_stream_packages):
    """Actual MLX backend method wrappers are installed, not only test dummies."""
    from core.mlx_engine import MLXReasoningBackend
    import inspect
    assert MLXReasoningBackend._mlx_thread_ownership_installed
    assert inspect.isgeneratorfunction(MLXReasoningBackend.stream_generate_tokens)
    assert inspect.isgeneratorfunction(MLXReasoningBackend.generate_branches) is False
    # The owner MUST be the OUTERMOST wrapper, installed after runtime locks,
    # raw LoRA restoration, generation caps and native-speed extensions.
    for name in ("load_model", "unload_model", "generate_branches",
                 "stream_generate_tokens", "train_mini_batch", "configure_auto_speed"):
        assert MLXReasoningBackend.__dict__[name].__code__.co_filename.endswith(
            "mlx_thread_owner.py"), name


def test_token_observer_context_and_first_token_tps_survive_thread_handoff(fake_stream_packages):
    from core.generation_metrics import TokenMeter, observe_generations
    owner = MLXThreadOwner()
    snapshots = []

    def native_tokens():
        meter = TokenMeter(source="mlx-vlm")
        try:
            for n in (1, 2, 3):
                meter.observe(count=n, text=True)
                yield "token" + str(n)
        finally:
            meter.finish()

    try:
        with observe_generations(snapshots.append):
            assert list(owner.stream(native_tokens)) == ["token1", "token2", "token3"]
        assert snapshots
        assert any(p["tokens"] == 1 for p in snapshots)
        assert snapshots[-1]["tokens"] == 3
        assert snapshots[-1]["finished"] is True
    finally:
        owner.shutdown()
