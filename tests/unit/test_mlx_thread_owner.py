"""A real background OS thread owns all MLX work; no model weights required."""
from __future__ import annotations

import sys
import threading
import time
from types import ModuleType, SimpleNamespace

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


def test_split_vlm_target_and_draft_stream_aliases_rebound_on_reload(fake_stream_packages, monkeypatch):
    _, public = fake_stream_packages
    modules = {}
    target_stream, draft_stream = public.generation_stream, object()
    for name in ("mlx_vlm.generate.common", "mlx_vlm.generate.ar", "mlx_vlm.generate.dispatch",
                 "mlx_vlm.speculative.common", "mlx_vlm.speculative.mtp"):
        module = ModuleType(name)
        module.generation_stream = draft_stream if "speculative" in name else target_stream
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys.modules["mlx.core"], "new_thread_local_stream",
                        lambda device: SimpleNamespace(device=device, thread=threading.get_ident()))
    for _ in range(2):
        owner = MLXThreadOwner()
        try:
            thread = owner.call(threading.get_ident)
            target = modules["mlx_vlm.generate.common"].generation_stream
            draft = modules["mlx_vlm.speculative.common"].generation_stream
            assert public.generation_stream is target
            assert modules["mlx_vlm.generate.ar"].generation_stream is target
            assert modules["mlx_vlm.generate.dispatch"].generation_stream is target
            assert modules["mlx_vlm.speculative.mtp"].generation_stream is draft
            assert target is not draft
            assert target.thread == draft.thread == thread
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


def test_stream_does_not_prefetch_past_visible_token_or_mark_cancelled_request_complete(fake_stream_packages):
    """Cancellation after the first displayed token must not complete prefetch."""
    owner = MLXThreadOwner()
    advanced = []
    finished = []

    def native_stream():
        completed = False
        try:
            advanced.append("first")
            yield "one"
            advanced.append("second")
            yield "two"
            completed = True
        finally:
            finished.append((threading.get_ident(), completed))

    try:
        stream = owner.stream(native_stream)
        assert next(stream) == "one"
        time.sleep(0.06)  # The old queue could drain all native tokens here.
        assert advanced == ["first"], "MLX generated output the UI had not requested"
        stream.close()
        assert finished == [(owner.call(threading.get_ident), False)]
    finally:
        owner.shutdown()


def test_desktop_load_tune_speedups_reload_unload_never_cross_thread_lock(fake_stream_packages):
    """Reproduce the macOS hang at 'Comparing native execution settings'.

    Fake model weights only. MLX owns an RLock and its outer wrapper dispatches
    work to a different OS thread, exactly as the production desktop does.
    """
    from core.native_speed import install_native_runtime_lock
    from core.speed_control import install_speed_control

    class OwnedBackend:
        def __init__(self):
            self.model = object()
            self._runtime_model_lock = threading.RLock()
            self.events = []
            self._mlx_owner = None
            self._mlx_owner_init_lock = threading.Lock()

        def run_on_mlx_owner(self, function, *args, **kwargs):
            return self._mlx_owner.call(function, *args, **kwargs)

        def load_model(self):
            with self._runtime_model_lock:
                self.events.append(("load", threading.get_ident()))
            return True

        def configure_auto_speed(self, **kwargs):
            with self._runtime_model_lock:
                self.events.append(("tune", threading.get_ident()))
                return {"status": "baseline", "message": "Original settings retained."}

        def configure_lossless_speedups(self, **kwargs):
            with self._runtime_model_lock:
                self.events.append(("speedups", threading.get_ident()))
                return dict(kwargs)

        def unload_model(self):
            with self._runtime_model_lock:
                self.events.append(("unload", threading.get_ident()))
                self.model = None
            return {"status": "unloaded"}

    # Same wrapper ordering as core/__init__.py: native lock, then MLX owner.
    install_native_runtime_lock(OwnedBackend)
    install_mlx_thread_ownership(OwnedBackend)

    class DesktopEngine:
        def __init__(self):
            self.settings = SimpleNamespace(use_mock=False)
            self.active_backend = None
            self.mlx_backend = None

        def load_model(self):
            result = self.unload_model()
            assert result["status"] == "unloaded"
            self.mlx_backend = OwnedBackend()
            self.active_backend = "mlx"
            assert self.mlx_backend.load_model() is True
            return {"status": "loaded"}

        def unload_model(self):
            if self.mlx_backend is not None:
                assert self.mlx_backend.unload_model()["status"] == "unloaded"
            self.mlx_backend = None
            self.active_backend = None
            return {"status": "unloaded"}

    install_speed_control(DesktopEngine)
    engine = DesktopEngine()
    main = threading.get_ident()

    def bounded(operation):
        outcomes = []
        done = threading.Event()

        def worker():
            try:
                outcomes.append((True, operation()))
            except BaseException as exc:
                outcomes.append((False, exc))
            finally:
                done.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        assert done.wait(3), "Deadlock: caller holds MLX RLock while waiting for its owner"
        thread.join(1)
        ok, value = outcomes[0]
        if not ok:
            raise value
        return value

    first = bounded(engine.load_model)
    assert first["status"] == "loaded"
    assert first["speed"]["status"] == "baseline"
    backend = engine.mlx_backend
    assert [k for k, _ in backend.events] == ["load", "speedups", "tune"]
    assert len({thread_id for _, thread_id in backend.events}) == 1
    assert backend.events[0][1] != main

    assert bounded(lambda: engine.configure_auto_speed(force=True))["status"] == "baseline"
    assert bounded(lambda: engine.configure_lossless_speedups(
        {"prompt_reuse": False}))["status"] == "configured"
    assert backend.events[-2][0] == "tune"
    assert backend.events[-1][0] == "speedups"

    second = bounded(engine.load_model)
    assert second["status"] == "loaded" and second["speed"]["status"] == "baseline"
    assert backend.events[-1][0] == "unload"
    assert len({thread_id for _, thread_id in backend.events}) == 1
    replacement = engine.mlx_backend
    assert bounded(engine.unload_model)["status"] == "unloaded"
    assert replacement.events[-1][0] == "unload"


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
