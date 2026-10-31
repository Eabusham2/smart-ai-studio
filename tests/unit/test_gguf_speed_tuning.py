"""Real adapter behavior using an instrumented llama.cpp stand-in, without weights."""
from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch
from tests.unit.gguf_fixtures import write_gguf


ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


speed = load("_gguf_speed_core_tests", "core/speed_tuning.py")
native = types.ModuleType("core.native_speed")
native.BENCHMARK_VERSION = "test-workload-v1"
native.BENCHMARK_PROMPT = "Explain the result of this fixed deterministic workload. " * 80
native.BENCHMARK_TOKENS = 24
native.tune_budget_seconds = lambda: 90.0
native.tuning_memory_ok = lambda _limit=None: True
core = types.ModuleType("core")
core.__path__ = [str(ROOT / "core")]
engines = types.ModuleType("core.engines")
engines.__path__ = [str(ROOT / "core/engines")]
config = types.ModuleType("config")
config.__path__ = [str(ROOT / "config")]
paths = types.ModuleType("config.paths")
paths.get_portable_data_dir = tempfile.gettempdir
MODULES = {"core": core, "core.engines": engines, "core.speed_tuning": speed,
           "core.native_speed": native, "config": config, "config.paths": paths}
with patch.dict(sys.modules, MODULES):
    adapter = load("_gguf_speed_adapter_tests", "core/gguf_speed_tuning.py")
    MODULES["core.gguf_speed_tuning"] = adapter
    gguf = load("_gguf_speed_engine_tests", "core/engines/gguf_engine.py")
    MODULES["core.engines.gguf_engine"] = gguf
    prism = load("_gguf_speed_prism_tests", "core/engines/prism_gguf_engine.py")
    worker = load("_gguf_speed_worker_tests", "core/engines/prism_gguf_worker.py")


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class FakeSampler:
    def __init__(self):
        self.sampler = object()
        self.closed = False
        self.close_count = 0
        self.rng_state = 123

    def close(self):
        self.close_count += 1
        self.closed = True
        self.sampler = None


class FakeLlama:
    def __init__(self, clock, **_kwargs):
        self.clock = clock
        self.n_threads = 4
        self.n_threads_batch = 8
        self.n_tokens = 0
        self._seed = 12345
        self._sampler = FakeSampler()
        self._mirostat_mu = object()
        self.generated_samplers = []
        self.context_params = types.SimpleNamespace(type_k=1, type_v=1, n_batch=512)
        self.calls = []
        self.resets = 0
        self.closed = 0
        self.fail = False
        self.change_output = False
        self.on_token = None
        self.fast_factor = 0.8
        self.batch_factors = {}
        self._ctx = types.SimpleNamespace(set_n_threads=self.set_n_threads)

    def set_n_threads(self, threads, batch):
        self.calls.append((threads, batch))

    def n_ctx(self):
        return 4096

    def reset(self):
        self.resets += 1
        self.n_tokens = 0

    def set_seed(self, value):
        self._seed = value

    def tokenize(self, _text):
        return list(range(100))

    def generate(self, tokens, **kwargs):
        assert self.n_tokens == 0, "prefix cache leaked into a benchmark sample"
        assert kwargs["temp"] == 0.0 and kwargs["reset"] is True
        assert self._seed == 0
        # Simulate a binding/fork that eagerly closes its attached chain before
        # creating a new one. Retaining a Python reference alone is insufficient.
        if self._sampler is not None:
            self._sampler.close()
        self._sampler = FakeSampler()
        self.generated_samplers.append(self._sampler)
        self._mirostat_mu = object()
        factor = self.fast_factor if self.n_threads == 2 else 1.0
        factor *= self.batch_factors.get(self.n_threads_batch, 1.0)
        try:
            for index in range(24):
                self._sampler.rng_state += 1
                self.clock.value += (0.1 if index == 0 else 0.02) * factor
                self.n_tokens = len(tokens) + index + 1
                if self.on_token:
                    self.on_token(index)
                if self.fail and index == 5:
                    raise RuntimeError("native failure")
                yield index + (1000 if self.change_output and self.n_threads == 2 else 10)
        finally:
            self.closed += 1


class TestLlamaSpeedAdapter(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, MODULES)
        self.modules.start()
        self.cores = patch.object(adapter, "physical_cpu_count", return_value=4)
        self.cores.start()
        self.directory = tempfile.TemporaryDirectory()
        self.model_path = Path(self.directory.name) / "model.gguf"
        self.model_path.write_bytes(b"fake-model-weights")
        self.clock = Clock()
        self.model = FakeLlama(self.clock)
        self.store = speed.SpeedProfileStore(str(Path(self.directory.name) / "profiles.json"))
        self.tuner = adapter.LlamaSpeedTuner(
            self.model, backend="gguf", model_path=str(self.model_path),
            context=4096, store=self.store, clock=self.clock,
        )

    def tearDown(self):
        self.directory.cleanup()
        self.cores.stop()
        self.modules.stop()

    def test_selects_live_measured_threads_and_restores_seed_and_fresh_cache(self):
        sampler = self.model._sampler
        mirostat = self.model._mirostat_mu
        result = self.tuner.configure()
        self.assertEqual(result["status"], "improved")
        self.assertEqual(result["settings"], {"n_threads": 2, "n_threads_batch": 8})
        self.assertEqual((self.model.n_threads, self.model.n_threads_batch), (2, 8))
        self.assertEqual(self.model._seed, 12345)
        self.assertIs(self.model._sampler, sampler)
        self.assertFalse(sampler.closed)
        self.assertEqual(sampler.close_count, 0)
        self.assertEqual(sampler.rng_state, 123)
        self.assertIs(self.model._mirostat_mu, mirostat)
        self.assertTrue(all(s.closed and s.close_count == 1 for s in self.model.generated_samplers))
        self.assertEqual(self.model.n_tokens, 0)
        self.assertGreater(self.model.closed, 4)
        self.assertEqual(self.model.resets, self.model.closed * 2)
        self.assertTrue(result["profile_saved"])
        self.assertEqual(self.model.context_params.type_k, 1)
        self.assertEqual(self.model.context_params.n_batch, 512)

    def test_disable_restores_original_constructor_threads(self):
        self.tuner.configure()
        result = self.tuner.configure(False)
        self.assertEqual(result["status"], "disabled")
        self.assertEqual((self.model.n_threads, self.model.n_threads_batch), (4, 8))

    def test_cancel_mid_token_restores_working_selection_and_does_not_cache(self):
        event = threading.Event()
        self.model.n_threads = 3  # the pre-run working selection must survive
        self.model.on_token = lambda index: event.set() if index == 4 else None
        result = self.tuner.configure(force=True, cancelled=event.is_set)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(self.model.n_threads, 3)
        self.assertEqual(self.model.n_tokens, 0)
        self.assertEqual(self.model._seed, 12345)
        self.assertEqual(self.model.closed, 1)
        self.assertIsNone(self.store.get(self.tuner.fingerprint))

    def test_native_failure_rolls_back_threads_and_sample_state(self):
        self.model.fail = True
        result = self.tuner.configure()
        self.assertEqual(result["status"], "error")
        self.assertEqual((self.model.n_threads, self.model.n_threads_batch), (4, 8))
        self.assertEqual(self.model._seed, 12345)
        self.assertEqual(self.model.n_tokens, 0)
        self.assertIsNone(self.store.get(self.tuner.fingerprint))

    def test_failed_retune_keeps_previously_saved_and_running_winner(self):
        self.tuner.configure()
        saved = self.store.get(self.tuner.fingerprint)
        self.model.fail = True
        result = self.tuner.configure(force=True)
        self.assertEqual(result["status"], "error")
        self.assertEqual(self.model.n_threads, 2)
        self.assertEqual(self.store.get(self.tuner.fingerprint), saved)

    def test_retune_uses_current_winner_as_baseline_when_no_new_gain_is_proven(self):
        self.tuner.configure()
        self.assertEqual(self.model.n_threads, 2)
        # Equal rates cannot justify going back to the original four threads.
        self.model.fast_factor = 1.0
        result = self.tuner.configure(force=True)
        self.assertEqual(result["status"], "baseline")
        self.assertEqual(result["settings"], {"n_threads": 2, "n_threads_batch": 8})
        self.assertEqual(self.model.n_threads, 2)

    def test_successive_retune_changes_one_knob_and_combined_winner_is_cacheable(self):
        self.tuner.configure()
        previous = {"n_threads": 2, "n_threads_batch": 8}
        self.model.batch_factors[4] = 0.7
        result = self.tuner.configure(force=True)
        self.assertEqual(result["status"], "improved")
        self.assertEqual(result["settings"], {"n_threads": 2, "n_threads_batch": 4})
        self.assertTrue(all(sum(trial["settings"][key] != previous[key] for key in previous) == 1
                            for trial in result["trials"]))
        self.tuner.configure(False)
        generated = self.model.closed
        restored = self.tuner.configure()
        self.assertTrue(restored["cached"])
        self.assertEqual(self.model.closed, generated)
        self.assertEqual((self.model.n_threads, self.model.n_threads_batch), (2, 4))

    def test_changed_greedy_tokens_reject_speed_gain(self):
        self.model.change_output = True
        result = self.tuner.configure()
        self.assertEqual(result["status"], "baseline")
        self.assertEqual(self.model.n_threads, 4)
        self.assertTrue(any(trial["reason"] == "output changed" for trial in result["trials"]))

    def test_cached_profile_applies_without_generating_or_clearing_context(self):
        self.tuner.configure()
        self.tuner.configure(False)
        previous = self.model.closed
        self.model.n_tokens = 1456
        result = self.tuner.configure()
        self.assertTrue(result["cached"])
        self.assertEqual(self.model.closed, previous)
        self.assertEqual(self.model.n_tokens, 1456)
        self.assertEqual(self.model.n_threads, 2)

    def test_cancellation_during_profile_lookup_prevents_cached_application(self):
        self.tuner.configure()
        saved = self.store.get(self.tuner.fingerprint)
        self.tuner.configure(False)
        calls = list(self.model.calls)
        event = threading.Event()

        def lookup(_fingerprint):
            event.set()
            return saved

        with patch.object(self.store, "get", side_effect=lookup):
            result = self.tuner.configure(cancelled=event.is_set)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(self.model.calls, calls)
        self.assertEqual(self.model.n_threads, 4)

    def test_memory_drop_during_profile_lookup_prevents_cached_application(self):
        self.tuner.configure()
        self.tuner.configure(False)
        calls = list(self.model.calls)
        with patch.object(native, "tuning_memory_ok", side_effect=[True, False]):
            result = self.tuner.configure()
        self.assertEqual(result["status"], "memory_pressure")
        self.assertEqual(self.model.calls, calls)
        self.assertEqual(self.model.n_threads, 4)

    def test_cancel_closes_temporary_sampler_without_closing_saved_sampler(self):
        original = self.model._sampler
        event = threading.Event()
        self.model.on_token = lambda index: event.set() if index == 4 else None
        result = self.tuner.configure(cancelled=event.is_set)
        self.assertEqual(result["status"], "cancelled")
        self.assertIs(self.model._sampler, original)
        self.assertFalse(original.closed)
        self.assertEqual(original.rng_state, 123)
        self.assertEqual(len(self.model.generated_samplers), 1)
        self.assertTrue(self.model.generated_samplers[0].closed)

    def test_previously_closed_sampler_is_never_resurrected(self):
        original = self.model._sampler
        original.close()
        result = self.tuner.configure()
        self.assertEqual(result["status"], "improved")
        self.assertIsNone(self.model._sampler)
        self.assertEqual(original.close_count, 1)

    def test_inadmissible_cached_settings_cannot_reach_native_setter(self):
        bad = speed.SpeedResult("improved", {"n_threads": 1024, "n_threads_batch": 1024},
                                "bad prior hardware", gain_pct=20)
        self.store.put(self.tuner.fingerprint, bad)
        result = self.tuner.configure()
        self.assertFalse(result["cached"])
        self.assertNotIn((1024, 1024), self.model.calls)

    def test_shared_adapter_defers_nonempty_cache_unless_studio_permits_replay(self):
        self.model.n_tokens = 2000
        result = self.tuner.configure(force=True)
        self.assertEqual(result["status"], "deferred")
        self.assertEqual(self.model.n_tokens, 2000)
        self.assertEqual(self.model.resets, 0)
        conversation = [{"role": "user", "content": "keep my history"}]
        snapshot = json.dumps(conversation)
        result = self.tuner.configure(force=True, allow_cache_reset=True)
        self.assertEqual(result["status"], "improved")
        self.assertEqual(self.model.n_tokens, 0)
        self.assertEqual(json.dumps(conversation), snapshot)
        self.assertEqual(self.model_path.read_bytes(), b"fake-model-weights")

    def test_budget_and_memory_pressure_are_checked_at_token_boundaries(self):
        with patch.object(native, "tune_budget_seconds", return_value=0.15):
            result = self.tuner.configure(force=True)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(self.model.n_tokens, 0)
        self.assertEqual(self.model.n_threads, 4)
        self.assertIsNone(self.store.get(self.tuner.fingerprint))
        with patch.object(native, "tuning_memory_ok", return_value=False):
            result = self.tuner.configure(force=True)
        self.assertEqual(result["status"], "memory_pressure")
        self.assertEqual(self.model.n_threads, 4)

    def test_no_native_setter_reports_unsupported_without_mutation(self):
        self.tuner.setter = None
        result = self.tuner.configure()
        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(self.model.calls, [])

    def test_backend_delegates_live_context_with_studio_replay_and_memory_limit(self):
        backend = gguf.GGUFReasoningBackend.__new__(gguf.GGUFReasoningBackend)
        backend.model = self.model
        backend._speed_tuner = self.tuner
        backend.speed_memory_limit_gb = 8
        self.model.n_tokens = 500
        seen = []
        with patch.object(native, "tuning_memory_ok", side_effect=lambda limit=None: seen.append(limit) or True):
            result = backend.configure_auto_speed(force=True)
        self.assertEqual(result["status"], "improved")
        self.assertTrue(seen and all(limit == 8 for limit in seen))

    def test_worker_memory_guard_uses_application_owner_at_every_boundary(self):
        seen = []

        def guard(limit=None, *, owner_pid=None):
            seen.append((limit, owner_pid))
            return True

        with patch.object(native, "tuning_memory_ok", side_effect=guard):
            result = self.tuner.configure(max_process_gb=10, memory_owner_pid=12345)
        self.assertEqual(result["status"], "improved")
        self.assertGreater(len(seen), 24)
        self.assertTrue(all(pair == (10, 12345) for pair in seen))

    def test_owner_memory_cap_can_stop_worker_before_any_native_operation(self):
        def guard(limit=None, *, owner_pid=None):
            # Worker alone fits; the application's owner plus workers does not.
            return owner_pid is None

        with patch.object(native, "tuning_memory_ok", side_effect=guard):
            result = self.tuner.configure(max_process_gb=10, memory_owner_pid=12345)
        self.assertEqual(result["status"], "memory_pressure")
        self.assertEqual(self.model.calls, [])

    def test_optional_tuner_failure_does_not_break_model_load_or_manual_threads(self):
        write_gguf(self.model_path)
        llama = types.ModuleType("llama_cpp")
        received = []
        llama.Llama = lambda **kwargs: received.append(kwargs) or self.model
        with patch.dict(sys.modules, {"llama_cpp": llama}), \
             patch.object(adapter, "LlamaSpeedTuner", side_effect=ImportError("optional helper missing")):
            backend = gguf.GGUFReasoningBackend(
                str(self.model_path), adapter_root=self.directory.name,
                n_threads=3, n_threads_batch=6, auto_memory_placement=False,
            )
            self.assertTrue(backend.load_model())
            self.assertIs(backend.model, self.model)
            result = backend.configure_auto_speed()
        self.assertEqual(result["status"], "unsupported")
        self.assertIn("optional helper missing", result["message"])
        self.assertEqual(received[0]["n_threads"], 3)
        self.assertEqual(received[0]["n_threads_batch"], 6)


class TestPrismTuningTransport(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, MODULES)
        self.modules.start()

    def tearDown(self):
        self.modules.stop()

    def test_progress_events_are_consumed_before_final_response(self):
        backend = prism.PrismGGUFReasoningBackend.__new__(prism.PrismGGUFReasoningBackend)
        backend._io_lock = threading.RLock()
        backend._worker = types.SimpleNamespace(
            poll=lambda: None, stdin=io.StringIO(),
            stdout=io.StringIO(json.dumps({"event": "speed_progress", "message": "Measuring"}) + "\n" +
                               json.dumps({"status": "ok", "result": {"status": "baseline"}}) + "\n"),
        )
        progress = []
        result = backend._call({"op": "tune_speed"}, progress=progress.append)
        self.assertEqual(progress, ["Measuring"])
        self.assertEqual(result["result"]["status"], "baseline")

    def test_proxy_sends_cancellation_and_cleans_private_marker(self):
        backend = prism.PrismGGUFReasoningBackend.__new__(prism.PrismGGUFReasoningBackend)
        backend.model = object()
        backend.speed_memory_limit_gb = 10
        captured = []

        def call(payload, *, progress=None):
            captured.append(payload)
            # Synchronize with the real cancellation watcher, not a fixed sleep.
            deadline = __import__("time").monotonic() + 1
            while not Path(payload["cancel_path"]).exists() and __import__("time").monotonic() < deadline:
                threading.Event().wait(0.01)
            self.assertTrue(Path(payload["cancel_path"]).exists())
            return {"status": "ok", "result": {"status": "cancelled", "settings": {"n_threads": 4}}}

        backend._call = call
        result = backend.configure_auto_speed(force=True, cancelled=lambda: True)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(captured[0]["max_process_gb"], 10)
        self.assertEqual(captured[0]["memory_owner_pid"], os.getpid())
        self.assertTrue(captured[0]["force"])
        self.assertFalse(Path(captured[0]["cancel_path"]).parent.exists())

    def test_existing_cancellation_reaches_worker_before_watcher_is_scheduled(self):
        backend = prism.PrismGGUFReasoningBackend.__new__(prism.PrismGGUFReasoningBackend)
        backend.model = object()
        captured = []

        def call(payload, *, progress=None):
            captured.append(payload)
            # A saved profile can apply before the OS schedules a new thread.
            self.assertTrue(Path(payload["cancel_path"]).exists())
            return {"status": "ok", "result": {"status": "cancelled", "settings": {"n_threads": 4}}}

        backend._call = call
        # Keep the asynchronous watcher unscheduled to make the race deterministic.
        with patch.object(prism.threading, "Thread"):
            result = backend.configure_auto_speed(cancelled=lambda: True)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(len(captured), 1)
        self.assertFalse(Path(captured[0]["cancel_path"]).parent.exists())

    def test_worker_tunes_actual_model_and_forwards_force_cancel_and_progress(self):
        clock = Clock()
        model = FakeLlama(clock)
        calls = []

        class Tuner:
            def __init__(self, loaded, **identity):
                self.loaded = loaded
                calls.append(identity)

            def configure(self, **kwargs):
                self.assert_actual()
                self.kwargs = kwargs
                calls.append(kwargs)
                kwargs["progress"]("Checking baseline")
                return {"status": "cancelled" if kwargs["cancelled"]() else "baseline", "settings": {}}

            def assert_actual(self):
                assert self.loaded is model

        with tempfile.TemporaryDirectory() as directory:
            cfg = Path(directory) / "worker.json"
            model_path = Path(directory) / "model.gguf"
            write_gguf(model_path)
            cfg.write_text(json.dumps({"model_path": str(model_path), "n_ctx": 4096,
                                       "auto_memory_placement": False}))
            marker = Path(directory) / "cancel"
            marker.touch()
            requests = "\n".join(json.dumps(value) for value in (
                {"op": "tune_speed", "force": True, "cancel_path": str(marker), "max_process_gb": 7,
                 "memory_owner_pid": 12345},
                {"op": "shutdown"},
            )) + "\n"
            output = io.StringIO()
            llama = types.ModuleType("llama_cpp")
            llama.Llama = lambda **kwargs: model
            with patch.dict(sys.modules, {"llama_cpp": llama}), patch.object(adapter, "LlamaSpeedTuner", Tuner), \
                 patch.object(sys, "argv", ["worker", str(cfg)]), patch.object(sys, "stdin", io.StringIO(requests)), \
                 patch.object(sys, "stdout", output):
                self.assertEqual(worker.main(), 0)
            lines = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(lines[0]["event"], "speed_progress")
            self.assertEqual(lines[1]["result"]["status"], "cancelled")
            self.assertEqual(calls[0]["backend"], "prism_gguf")
            self.assertTrue(calls[1]["force"])
            self.assertTrue(calls[1]["allow_cache_reset"])
            self.assertEqual(calls[1]["max_process_gb"], 7)
            self.assertEqual(calls[1]["memory_owner_pid"], 12345)


if __name__ == "__main__":
    unittest.main()
