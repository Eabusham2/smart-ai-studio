"""Exercise the real MLX speed adapter with deterministic native-stream fakes."""
from contextlib import contextmanager
import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def fake_runtime(folder):
    state = types.SimpleNamespace(now=0.0, calls=[], closed=0, outputs=0, arenas=[],
                                  safe_chunk=256, memory_ok=True, changed_output=False,
                                  failure_chunk=None, single_token=False, on_fingerprint=None,
                                  repeat_flush=False, wrong_counts=False, early_eos=False)
    core = types.ModuleType("core")
    core.__path__ = [str(ROOT / "core")]
    mx = types.ModuleType("mlx.core")
    mx.random = types.SimpleNamespace(state=["original RNG"])
    mx.eval = lambda *args: None
    mx.synchronize = lambda: None
    mx.device_info = lambda: {"device_name": "Fake Apple GPU", "memory_size": 16 * 1024 ** 3}
    mlx = types.ModuleType("mlx")
    mlx.__path__ = []
    mlx.core = mx
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.__path__ = []
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **kwargs: dict(kwargs)

    class Arena:
        def __init__(self, model, max_tokens):
            self.model = model
            self.max_tokens = max_tokens
            self.cache = [object()]
            state.arenas.append(self)

        def get_cache(self):
            return self.cache

    cache_module = types.ModuleType("core.kv_cache_manager")
    cache_module.SmartKVCacheManager = Arena
    engine_module = types.ModuleType("core.mlx_engine")
    engine_module._adaptive_prefill_step_size = lambda: state.safe_chunk
    engine_module._model_context_limit = lambda *args: 4096

    def generate(model, tokenizer, **kwargs):
        state.calls.append(kwargs)
        mx.random.state[:] = ["changed by native generation"]
        try:
            count = 1 if state.single_token else 6 if state.early_eos else kwargs["max_tokens"]
            for index in range(count):
                state.now += (2.0 if kwargs["prefill_step_size"] == 128 else 4.0) if index == 0 else 0.02
                state.outputs += 1
                if state.failure_chunk == kwargs["prefill_step_size"] and index == 2:
                    raise RuntimeError("out of memory")
                token = index + (1 if state.changed_output and kwargs["prefill_step_size"] == 128 else 0)
                if state.early_eos and index == count - 1:
                    token = 999
                yield types.SimpleNamespace(token=token, generation_tokens=index + (2 if state.wrong_counts else 1),
                                            prompt_tokens=len(kwargs["prompt"]),
                                            finish_reason=("stop" if state.early_eos else "length") if index == count - 1 else None)
            if state.repeat_flush:
                yield types.SimpleNamespace(token=token, generation_tokens=count, prompt_tokens=len(kwargs["prompt"]),
                                            finish_reason="length")
        finally:
            state.closed += 1

    mlx_lm.stream_generate = generate
    modules = {"core": core, "mlx": mlx, "mlx.core": mx, "mlx_lm": mlx_lm,
               "mlx_lm.sample_utils": sample_utils, "core.mlx_engine": engine_module,
               "core.kv_cache_manager": cache_module}
    with patch.dict(sys.modules, modules):
        tuning = _load("core.speed_tuning", "core/speed_tuning.py")
        _load("core.native_speed", "core/native_speed.py")
        adapter = _load("core.mlx_speed_tuning", "core/mlx_speed_tuning.py")
        adapter.time = types.SimpleNamespace(monotonic=lambda: state.now, perf_counter=lambda: state.now)
        adapter.tune_budget_seconds = lambda: 90.0
        adapter.tuning_memory_ok = lambda limit=None: state.memory_ok
        real_tune = tuning.tune_speed
        adapter.tune_speed = lambda *args, **kwargs: real_tune(*args, **dict(kwargs, clock=lambda: state.now))

        def fingerprint(*args, **kwargs):
            if state.on_fingerprint:
                state.on_fingerprint()
            return "fixed-test-fingerprint"

        adapter.speed_fingerprint = fingerprint
        store = tuning.SpeedProfileStore(str(Path(folder) / "profiles.json"))
        adapter.SpeedProfileStore = lambda: store
        original_cache = [object()]
        backend = types.SimpleNamespace(
            _speed_prefill_step_size=None, is_mlx_available=True, model=object(),
            _vlm_runtime=None, model_path="test/model", model_info={}, adapter_path=None,
            context_budget_tokens=1024, max_stateful_kv_tokens=2048,
            tokenizer=types.SimpleNamespace(encode=lambda prompt: list(range(512))),
            kv_cache_manager=types.SimpleNamespace(get_cache=lambda: original_cache),
        )
        yield types.SimpleNamespace(adapter=adapter, state=state, backend=backend, store=store,
                                    tuning=tuning, mx=mx, original_cache=original_cache)


@contextmanager
def fake_vlm_runtime(folder, *, runtime="mlx_repo_vlm", default_chunk=256):
    with fake_runtime(folder) as env:
        package = types.ModuleType("mlx_vlm")
        package.__path__ = []
        generation = types.ModuleType("mlx_vlm.generate")

        def step(*args, prefill_step_size=default_chunk, **kwargs):
            pass

        generation.generate_step = step
        env.backend._vlm_runtime = runtime
        env.backend._speed_profile_context = None
        vision_tower = object()
        env.backend.model = types.SimpleNamespace(
            language_model=types.SimpleNamespace(_rope_deltas="original delta"), vision_tower=vision_tower)
        env.backend.processor = types.SimpleNamespace(tokenizer=env.backend.tokenizer)
        env.backend.tokenizer.thinking_budget_criteria = "original thinking"
        env.backend._vlm_loader = types.SimpleNamespace(config={"model_type": "bonsai", "vision": True})
        env.backend._prepare_vlm_prompt = lambda prompt, images: "formatted::" + prompt
        env.state.corrupt_final = False

        def stream(model, processor, **kwargs):
            env.state.calls.append(kwargs)
            env.mx.random.state[:] = ["native VLM changed RNG"]
            env.backend.tokenizer.thinking_budget_criteria = "synthetic criterion"
            model.language_model._rope_deltas = "synthetic delta"
            model._position_ids = "synthetic positions"
            chunk = kwargs.get("prefill_step_size", default_chunk)
            tokens = []
            try:
                count = 1 if env.state.single_token else kwargs["max_tokens"]
                for index in range(count):
                    env.state.now += (2.0 if chunk == 128 else 4.0) if index == 0 else 0.02
                    env.state.outputs += 1
                    if env.state.failure_chunk == chunk and index == 2:
                        raise RuntimeError("out of memory")
                    token = index + (1 if env.state.changed_output and chunk == 128 else 0)
                    tokens.append(token)
                    yield types.SimpleNamespace(token=token, text=str(token), prompt_tokens=512,
                                                generation_tokens=index + 1, token_ids=None)
                # Native mlx-vlm repeats its final token during text flushing.
                final_ids = tokens if not env.state.corrupt_final else tokens + [999]
                yield types.SimpleNamespace(token=tokens[-1], text="", prompt_tokens=512,
                                            generation_tokens=count, token_ids=final_ids)
            finally:
                env.state.closed += 1

        generation.stream_generate = stream
        with patch.dict(sys.modules, {"mlx_vlm": package, "mlx_vlm.generate": generation}), \
             patch.object(sys.modules["mlx_lm"], "stream_generate", side_effect=AssertionError("MLX-LM must not run for VLM")), \
             patch.object(sys.modules["mlx_lm.sample_utils"], "make_sampler", side_effect=AssertionError("MLX-LM sampler must not run for VLM")), \
             patch.object(sys.modules["core.kv_cache_manager"], "SmartKVCacheManager", side_effect=AssertionError("Generic KV arena must not be injected into VLM")):
            env.generation = generation
            env.vision_tower = vision_tower
            yield env


def engine_method(name, state):
    """Execute the actual small engine method without importing unrelated weights."""
    tree = ast.parse((ROOT / "core/mlx_engine.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MLXReasoningBackend")
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"Dict": dict, "time": types.SimpleNamespace(perf_counter=lambda: state.now),
                 "_adaptive_prefill_step_size": lambda: state.safe_chunk,
                 "_reclaim_if_needed": lambda *args, **kwargs: None}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "core/mlx_engine.py", "exec"), namespace)
    return namespace[name]


class TestMLXSpeedAdapter(unittest.TestCase):
    def test_native_final_token_and_duplicate_flush_count_each_generated_index_once(self):
        for early_eos, repeat_flush in ((False, False), (True, False), (False, True)):
            with self.subTest(early_eos=early_eos, repeat_flush=repeat_flush), tempfile.TemporaryDirectory() as folder, fake_runtime(folder) as env:
                env.state.early_eos = early_eos
                env.state.repeat_flush = repeat_flush
                result = env.adapter.configure_mlx_speed(env.backend)
                self.assertEqual(result["status"], "improved")
                self.assertAlmostEqual(result["baseline"]["decode_tokens_per_second"], 50.0)
                self.assertAlmostEqual(result["selected"]["decode_tokens_per_second"], 50.0)

    def test_nonconsecutive_native_token_counts_fail_without_saving(self):
        with tempfile.TemporaryDirectory() as folder, fake_runtime(folder) as env:
            env.state.wrong_counts = True
            result = env.adapter.configure_mlx_speed(env.backend)
            self.assertEqual(result["status"], "error")
            self.assertIsNone(env.store.get("fixed-test-fingerprint"))
            self.assertIsNone(env.backend._speed_prefill_step_size)

    def test_real_tuner_selects_chunk_without_touching_chat_cache_or_rng(self):
        with tempfile.TemporaryDirectory() as folder, fake_runtime(folder) as env:
            result = env.adapter.configure_mlx_speed(env.backend)
            self.assertEqual(result["status"], "improved")
            self.assertEqual(env.backend._speed_prefill_step_size, 128)
            self.assertEqual(env.backend._speed_profile_context, 1024)
            self.assertEqual(env.mx.random.state, ["original RNG"])
            self.assertIs(env.backend.kv_cache_manager.get_cache(), env.original_cache)
            self.assertTrue(all(arena.cache is None for arena in env.state.arenas))
            self.assertEqual(env.state.closed, len(env.state.calls))
            self.assertEqual(env.store.get("fixed-test-fingerprint")["settings"], {"prefill_step_size": 128})
            for call in env.state.calls:
                self.assertEqual(call["prompt"], list(range(512)))
                self.assertEqual(call["sampler"], {"temp": 0.0, "top_p": 1.0})
                self.assertEqual(call["max_tokens"], 24)
                self.assertIsNot(call["prompt_cache"], env.original_cache)

    def test_cancel_mid_stream_restores_previous_chunk_rng_and_releases_arena(self):
        with tempfile.TemporaryDirectory() as folder, fake_runtime(folder) as env:
            env.backend._speed_prefill_step_size = 256
            result = env.adapter.configure_mlx_speed(env.backend, cancelled=lambda: env.state.outputs >= 5)
            self.assertEqual(result["status"], "cancelled")
            self.assertEqual(env.backend._speed_prefill_step_size, 256)
            self.assertEqual(env.mx.random.state, ["original RNG"])
            self.assertEqual(env.state.closed, 1)
            self.assertIsNone(env.state.arenas[0].cache)
            self.assertIsNone(env.store.get("fixed-test-fingerprint"))

    def test_candidate_oom_keeps_previous_profile_and_never_caches_partial_result(self):
        with tempfile.TemporaryDirectory() as folder, fake_runtime(folder) as env:
            env.backend._speed_prefill_step_size = 256
            env.state.failure_chunk = 128
            result = env.adapter.configure_mlx_speed(env.backend, force=True)
            self.assertEqual(result["status"], "error")
            self.assertEqual(env.backend._speed_prefill_step_size, 256)
            self.assertEqual(env.mx.random.state, ["original RNG"])
            self.assertTrue(all(arena.cache is None for arena in env.state.arenas))
            self.assertIsNone(env.store.get("fixed-test-fingerprint"))

    def test_changed_output_retains_baseline_with_honest_status(self):
        with tempfile.TemporaryDirectory() as folder, fake_runtime(folder) as env:
            env.state.changed_output = True
            result = env.adapter.configure_mlx_speed(env.backend)
            self.assertEqual(result["status"], "baseline")
            self.assertEqual(env.backend._speed_prefill_step_size, 256)
            self.assertIn("quality-preserving", result["message"])
            self.assertEqual(result["gain_pct"], 0)

    def test_one_token_eos_does_not_produce_a_fake_throughput_win(self):
        with tempfile.TemporaryDirectory() as folder, fake_runtime(folder) as env:
            env.state.single_token = True
            result = env.adapter.configure_mlx_speed(env.backend)
            self.assertEqual(result["status"], "error")
            self.assertIsNone(env.backend._speed_prefill_step_size)
            self.assertIsNone(env.store.get("fixed-test-fingerprint"))

    def test_saved_setting_is_reused_without_synthetic_generation(self):
        with tempfile.TemporaryDirectory() as folder, fake_runtime(folder) as env:
            first = env.adapter.configure_mlx_speed(env.backend)
            self.assertEqual(first["status"], "improved")
            call_count = len(env.state.calls)
            env.backend._speed_prefill_step_size = None
            second = env.adapter.configure_mlx_speed(env.backend)
            self.assertTrue(second["cached"])
            self.assertEqual(len(env.state.calls), call_count)
            self.assertEqual(env.backend._speed_prefill_step_size, 128)

    def test_out_of_range_saved_setting_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder, fake_runtime(folder) as env:
            poisoned = env.tuning.SpeedResult("improved", {"prefill_step_size": 8192}, "stale", 10)
            self.assertTrue(env.store.put("fixed-test-fingerprint", poisoned))
            result = env.adapter.configure_mlx_speed(env.backend)
            self.assertEqual(result["status"], "improved")
            self.assertGreater(len(env.state.calls), 0)
            self.assertEqual(env.backend._speed_prefill_step_size, 128)

    def test_cancellation_during_cache_lookup_prevents_saved_profile_application(self):
        with tempfile.TemporaryDirectory() as folder, fake_runtime(folder) as env:
            env.adapter.configure_mlx_speed(env.backend)
            env.backend._speed_prefill_step_size = 256
            stopped = [False]
            env.state.on_fingerprint = lambda: stopped.__setitem__(0, True)
            result = env.adapter.configure_mlx_speed(env.backend, cancelled=lambda: stopped[0])
            self.assertEqual(result["status"], "cancelled")
            self.assertEqual(env.backend._speed_prefill_step_size, 256)

    def test_memory_drop_during_cache_lookup_keeps_previous_setting(self):
        with tempfile.TemporaryDirectory() as folder, fake_runtime(folder) as env:
            env.adapter.configure_mlx_speed(env.backend)
            env.backend._speed_prefill_step_size = 256
            env.state.on_fingerprint = lambda: setattr(env.state, "memory_ok", False)
            result = env.adapter.configure_mlx_speed(env.backend)
            self.assertEqual(result["status"], "memory_pressure")
            self.assertEqual(env.backend._speed_prefill_step_size, 256)

    def test_disabling_restores_live_adaptive_policy_without_running_model(self):
        with tempfile.TemporaryDirectory() as folder, fake_runtime(folder) as env:
            env.backend._speed_prefill_step_size = 128
            result = env.adapter.configure_mlx_speed(env.backend, enabled=False)
            self.assertEqual(result["status"], "disabled")
            self.assertIsNone(env.backend._speed_prefill_step_size)
            self.assertEqual(env.state.calls, [])


class TestVLMNativeSpeedAdapter(unittest.TestCase):
    def test_bonsai_and_jang_use_actual_native_vlm_default_and_fresh_cache_path(self):
        for runtime in ("mlx_repo_vlm", "jang_vlm"):
            with self.subTest(runtime=runtime), tempfile.TemporaryDirectory() as folder, fake_vlm_runtime(folder, runtime=runtime) as env:
                result = env.adapter.configure_mlx_speed(env.backend)
                self.assertEqual(result["status"], "improved")
                self.assertEqual(env.backend._speed_prefill_step_size, 128)
                self.assertEqual(env.backend._speed_profile_context, 1024)
                self.assertEqual(env.mx.random.state, ["original RNG"])
                self.assertEqual(env.backend.tokenizer.thinking_budget_criteria, "original thinking")
                self.assertEqual(env.backend.model.language_model._rope_deltas, "original delta")
                self.assertFalse(hasattr(env.backend.model, "_position_ids"))
                self.assertIs(env.backend.model.vision_tower, env.vision_tower)
                self.assertEqual(env.state.arenas, [])
                self.assertEqual(env.state.closed, len(env.state.calls))
                self.assertNotIn("prefill_step_size", env.state.calls[0])
                self.assertTrue(any(call.get("prefill_step_size") == 128 for call in env.state.calls))
                for call in env.state.calls:
                    self.assertTrue(call["prompt"].startswith("formatted::"))
                    self.assertEqual(call["image"], [])
                    self.assertEqual(call["temperature"], 0.0)
                    self.assertEqual(call["top_p"], 1.0)
                    self.assertNotIn("prompt_cache", call)
                    self.assertNotIn("sampler", call)
                    self.assertNotIn("kv_bits", call)
                # A final repeated flush token was not counted as a 25th token.
                self.assertAlmostEqual(result["baseline"]["decode_tokens_per_second"], 50.0)

    def test_unverified_or_non_chunking_old_runtime_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder, fake_vlm_runtime(folder, default_chunk=None) as env:
            result = env.adapter.configure_mlx_speed(env.backend)
            self.assertEqual(result["status"], "unsupported")
            self.assertEqual(env.state.calls, [])
        with tempfile.TemporaryDirectory() as folder, fake_vlm_runtime(folder) as env:
            env.backend.model.language_model.no_chunked_prefill = True
            result = env.adapter.configure_mlx_speed(env.backend)
            self.assertEqual(result["status"], "unsupported")
            self.assertEqual(env.state.calls, [])

    def test_unknown_default_is_not_replaced_by_an_arbitrary_adaptive_baseline(self):
        with tempfile.TemporaryDirectory() as folder, fake_vlm_runtime(folder, default_chunk=2048) as env:
            self.assertEqual(env.state.safe_chunk, 256)
            result = env.adapter.configure_mlx_speed(env.backend)
            self.assertEqual(result["status"], "memory_pressure")
            self.assertIsNone(env.backend._speed_prefill_step_size)
            self.assertEqual(env.state.calls, [])

    def test_vlm_cancel_restores_model_tokenizer_rng_and_previous_override(self):
        with tempfile.TemporaryDirectory() as folder, fake_vlm_runtime(folder) as env:
            env.backend._speed_prefill_step_size = 128
            env.backend._speed_profile_context = 2048  # stale context, inactive at 1024
            result = env.adapter.configure_mlx_speed(env.backend, cancelled=lambda: env.state.outputs >= 5)
            self.assertEqual(result["status"], "cancelled")
            self.assertEqual(env.backend._speed_prefill_step_size, 128)
            self.assertEqual(env.backend._speed_profile_context, 2048)
            self.assertEqual(env.mx.random.state, ["original RNG"])
            self.assertEqual(env.backend.model.language_model._rope_deltas, "original delta")
            self.assertEqual(env.backend.tokenizer.thinking_budget_criteria, "original thinking")
            self.assertFalse(hasattr(env.backend.model, "_position_ids"))
            self.assertIsNone(env.store.get("fixed-test-fingerprint"))

    def test_vlm_output_change_retains_library_default_without_injecting_override(self):
        with tempfile.TemporaryDirectory() as folder, fake_vlm_runtime(folder) as env:
            env.state.changed_output = True
            result = env.adapter.configure_mlx_speed(env.backend)
            self.assertEqual(result["status"], "baseline")
            self.assertEqual(result["settings"], {"prefill_step_size": 256})
            self.assertIsNone(env.backend._speed_prefill_step_size)
            self.assertIsNone(env.backend._speed_profile_context)

    def test_vlm_corrupt_final_token_list_never_wins(self):
        with tempfile.TemporaryDirectory() as folder, fake_vlm_runtime(folder) as env:
            env.state.corrupt_final = True
            result = env.adapter.configure_mlx_speed(env.backend)
            self.assertEqual(result["status"], "error")
            self.assertIsNone(env.store.get("fixed-test-fingerprint"))
            self.assertIsNone(env.backend._speed_prefill_step_size)

    def test_native_vlm_cached_profile_reuses_settings_and_still_honors_cancel(self):
        with tempfile.TemporaryDirectory() as folder, fake_vlm_runtime(folder) as env:
            env.adapter.configure_mlx_speed(env.backend)
            count = len(env.state.calls)
            env.backend._speed_prefill_step_size = None
            env.backend._speed_profile_context = None
            result = env.adapter.configure_mlx_speed(env.backend)
            self.assertTrue(result["cached"])
            self.assertEqual(len(env.state.calls), count)
            self.assertEqual(env.backend._speed_prefill_step_size, 128)
            env.backend._speed_prefill_step_size = None
            env.backend._speed_profile_context = None
            cancelled = [False]
            env.state.on_fingerprint = lambda: cancelled.__setitem__(0, True)
            result = env.adapter.configure_mlx_speed(env.backend, cancelled=lambda: cancelled[0])
            self.assertEqual(result["status"], "cancelled")
            self.assertIsNone(env.backend._speed_prefill_step_size)

    def test_existing_vlm_chat_calls_keep_sampling_and_add_only_valid_context_override(self):
        with tempfile.TemporaryDirectory() as folder, fake_vlm_runtime(folder) as env:
            env.backend._effective_generation_cap = lambda *args, **kwargs: 24
            env.backend._speed_vlm_prefill_kwargs = types.MethodType(engine_method("_speed_vlm_prefill_kwargs", env.state), env.backend)
            branches = engine_method("generate_branches", env.state)
            streaming = engine_method("stream_generate_tokens", env.state)
            branches(env.backend, "user text", branch_count=1, temperature=0.7, top_p=0.85)
            self.assertNotIn("prefill_step_size", env.state.calls[-1])
            self.assertEqual(env.state.calls[-1]["temperature"], 0.7)
            self.assertEqual(env.state.calls[-1]["top_p"], 0.85)
            env.backend._speed_prefill_step_size = 128
            env.backend._speed_profile_context = 1024
            list(streaming(env.backend, "user text", temperature=0.8, top_p=0.91))
            self.assertEqual(env.state.calls[-1]["prefill_step_size"], 128)
            self.assertEqual(env.state.calls[-1]["temperature"], 0.8)
            self.assertEqual(env.state.calls[-1]["top_p"], 0.91)
            env.backend.context_budget_tokens = 2048
            list(streaming(env.backend, "user text", temperature=0.8, top_p=0.91))
            self.assertNotIn("prefill_step_size", env.state.calls[-1])

    def test_native_lm_context_change_uses_original_adaptive_chunk(self):
        with tempfile.TemporaryDirectory() as folder, fake_runtime(folder) as env:
            choose = engine_method("_speed_prefill_size", env.state)
            env.backend._speed_prefill_step_size = 128
            env.backend._speed_profile_context = 1024
            self.assertEqual(choose(env.backend), 128)
            env.backend.context_budget_tokens = 2048
            self.assertEqual(choose(env.backend), 256)


if __name__ == "__main__":
    unittest.main()
