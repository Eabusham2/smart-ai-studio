"""Request-memory regressions, with tiny native arrays for traceback cleanup.

No test here allocates a 27B model or claims that every conversation fits RAM.
"""
import gc
import sys
import types
import weakref

import pytest

from core.mlx_memory_guard import GIB, MemoryHeadroom, memory_headroom, trim_allocator
from core.prompt_reuse import ExactPrefixStore
from tests.unit.test_prompt_reuse import populated, runtime


def headroom(usable):
    return MemoryHeadroom(8 * GIB, 8 * GIB, 8 * GIB, 0, 12 * GIB, GIB, usable)


def test_overlapping_host_process_and_metal_counters_are_not_added(monkeypatch):
    import psutil
    import core.training_memory as training

    monkeypatch.setattr(psutil, "virtual_memory", lambda: types.SimpleNamespace(
        available=10 * GIB, total=16 * GIB))
    monkeypatch.setattr(training, "process_memory_bytes", lambda: 9 * GIB)
    mx = types.SimpleNamespace(
        get_active_memory=lambda: 8 * GIB, get_cache_memory=lambda: GIB,
        metal=types.SimpleNamespace(is_available=lambda: True),
        device_info=lambda: {"max_recommended_working_set_size": 12 * GIB})
    result = memory_headroom(mx=mx, max_process_gb=14)
    assert result.usable_bytes == 3 * GIB - result.reserve_bytes
    assert result.prefill_tokens == 256
    assert not result.can_snapshot(256 * 1024 ** 2)


@pytest.mark.parametrize("metal", [False, True])
def test_mac_headroom_keeps_metal_accounting_without_cuda_device_probe(monkeypatch, metal):
    import psutil
    from unittest.mock import Mock
    import core.training_memory as training
    import core.mlx_memory_guard as guard

    monkeypatch.setattr(guard, "platform", types.SimpleNamespace(system=lambda: "Darwin"))
    monkeypatch.setattr(psutil, "virtual_memory", lambda: types.SimpleNamespace(available=10 * GIB, total=16 * GIB))
    monkeypatch.setattr(training, "process_memory_bytes", lambda: 9 * GIB)
    mx = types.SimpleNamespace(
        get_active_memory=lambda: 8 * GIB, get_cache_memory=lambda: GIB,
        metal=types.SimpleNamespace(is_available=lambda: metal),
        device_info=lambda: {"max_recommended_working_set_size": 12 * GIB},
        default_device=Mock(side_effect=AssertionError("Mac must not use the CUDA probe")), gpu=object(),
    )
    result = memory_headroom(mx=mx, max_process_gb=14)
    assert result.usable_bytes == (3 if metal else 5) * GIB - result.reserve_bytes
    assert result.device_limit_bytes == (12 * GIB if metal else None)
    mx.default_device.assert_not_called()


def test_allocator_trims_cached_memory_without_releasing_model_arrays(monkeypatch):
    import core.mlx_memory_guard as guard

    limits, cleared = [], []
    current = headroom(GIB // 4)
    current = MemoryHeadroom(**{**vars(current), "cache_bytes": 3 * GIB})
    monkeypatch.setattr(guard, "memory_headroom", lambda **kwargs: current)
    mx = types.SimpleNamespace(set_cache_limit=limits.append, clear_cache=lambda: cleared.append(True))
    assert trim_allocator(mx) is current
    assert limits == [0] and cleared == [True]


def test_watchdog_cannot_raise_request_cache_cap_when_host_memory_recovers(monkeypatch):
    import psutil
    import core.mlx_memory_guard as guard
    from core.memory_watchdog import SystemMemoryWatchdog

    mlx, mx = types.ModuleType("mlx"), types.ModuleType("mlx.core")
    mlx.core = mx
    cap, writes = [2 * GIB], []

    def set_limit(value):
        previous = cap[0]
        cap[0] = value
        writes.append(value)
        return previous

    mx.set_cache_limit = set_limit
    mx.clear_cache = lambda: None
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: types.SimpleNamespace(available=8 * GIB))
    current = [headroom(GIB // 4)]
    monkeypatch.setattr(guard, "memory_headroom", lambda **kwargs: current[0])

    trim_allocator(mx)
    assert cap[0] == 0
    # The watchdog sees abundant host RAM but does not know about the active
    # request's tighter Metal/process bound. It must preserve that request cap.
    SystemMemoryWatchdog.adjust_dynamic_metal_headroom()
    assert cap[0] == 0 and writes == [0]

    # A subsequent request can re-evaluate its own safe allowance. The monitor
    # then preserves it instead of restoring the old multi-GiB allocator cache.
    current[0] = headroom(8 * GIB)
    trim_allocator(mx)
    assert cap[0] == guard.MAX_ALLOCATOR_CACHE_BYTES
    SystemMemoryWatchdog.adjust_dynamic_metal_headroom()
    assert cap[0] == guard.MAX_ALLOCATOR_CACHE_BYTES
    # No repeated zero probe: a monitor tick must not churn a healthy cache.
    assert writes == [0, guard.MAX_ALLOCATOR_CACHE_BYTES]


def test_unknown_headroom_never_admits_a_snapshot():
    unknown = headroom(None)
    assert unknown.pressure and unknown.prefill_tokens == 128
    assert not unknown.can_snapshot(1)


def test_last_logits_control_requires_a_supported_native_contract():
    from core.mlx_memory_guard import native_last_logits_kwargs

    class Explicit:
        def __call__(self, ids, *, logits_to_keep=0):
            return ids

    class Alternate:
        def __call__(self, ids, *, num_logits_to_keep=0):
            return ids

    class Legacy:
        def __call__(self, ids):
            return ids

    class Unknown:
        def __call__(self, ids, **kwargs):
            return ids

    assert native_last_logits_kwargs(Explicit()) == {"logits_to_keep": 1}
    assert native_last_logits_kwargs(Alternate()) == {"num_logits_to_keep": 1}
    assert native_last_logits_kwargs(Legacy()) == {}
    assert native_last_logits_kwargs(Unknown()) == {}

    class Vmlx:
        def __call__(self, ids, **kwargs):
            return ids

    Vmlx.__call__._vmlx_prefill_last_logits_only = True

    class PackedVmlx(Vmlx):
        pass  # JANG replaces projections while retaining the native callable.

    class UnmarkedOverride(Vmlx):
        def __call__(self, ids, **kwargs):
            return ids

    assert native_last_logits_kwargs(PackedVmlx()) == {"prefill_last_logits_only": True}
    assert native_last_logits_kwargs(UnmarkedOverride()) == {}

    # The released mlx-vlm implementation uses **kwargs, not a named parameter.
    # Bytecode metadata survives packaging even when inspect.getsource cannot.
    namespace = {"__name__": "mlx_vlm.models.qwen3_5.language"}
    exec("def project_logits(value, projection, count):\n"
         "    return projection(value[:, -count:] if count else value)\n"
         "class Native:\n"
         "    def __call__(self, ids, **kwargs):\n"
         "        return project_logits(ids, lambda x: x, kwargs.get('logits_to_keep'))\n", namespace)
    namespace["project_logits"].__module__ = "mlx_vlm.models.base"
    native = namespace["Native"]()
    assert native_last_logits_kwargs(native) == {"logits_to_keep": 1}
    namespace["project_logits"].__module__ = "unverified_projection"
    assert native_last_logits_kwargs(native) == {}


def test_entropy_preserves_long_prompt_but_projects_only_last_token(runtime, monkeypatch):
    import numpy as np

    mx = sys.modules["mlx.core"]
    monkeypatch.setattr(mx, "array", np.asarray)
    monkeypatch.setattr(mx, "softmax", lambda value, axis: np.exp(value) / np.exp(value).sum(axis, keepdims=True), raising=False)
    monkeypatch.setattr(mx, "log", np.log, raising=False)
    monkeypatch.setattr(mx, "sum", np.sum, raising=False)
    calls = []

    class Language:
        def __call__(self, ids, *, logits_to_keep=0):
            calls.append((ids.copy(), logits_to_keep))
            assert logits_to_keep == 1, "full sequence vocabulary allocation was requested"
            return types.SimpleNamespace(logits=np.array([[[1., 2., 3.]]]))

    backend = runtime.backend
    language = Language()
    backend.model = types.SimpleNamespace(language_model=language)
    prompt = "system instructions\n" + "r" * 3000 + "\nsecond message"
    actual = backend.calculate_token_entropy(prompt)
    probs = np.exp([1., 2., 3.]) / np.exp([1., 2., 3.]).sum()
    assert actual == pytest.approx(-np.sum(probs * np.log(probs + 1e-12)))
    assert len(calls) == 1 and calls[0][0].tolist() == [[ord(c) for c in prompt]]
    assert backend.model.language_model is language


def test_entropy_oom_frees_forward_frames_before_recovery(runtime, monkeypatch):
    owners, recoveries = [], []
    backend = runtime.backend
    recover = backend._recover_generation_memory

    class Temporary:
        pass

    class Language:
        def __call__(self, ids):
            temporary = Temporary()
            owners.append(weakref.ref(temporary))
            raise RuntimeError("[METAL] Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)")

    def recovered(mx):
        gc.collect()
        assert all(owner() is None for owner in owners)
        recoveries.append(True)
        recover(mx)

    backend.model = types.SimpleNamespace(language_model=Language())
    monkeypatch.setattr(backend, "_recover_generation_memory", recovered)
    assert backend.calculate_token_entropy("full routing prompt") == .45
    assert recoveries == [True]
    assert backend._oom_prefill_step_size == 1
    assert backend.last_memory_recovery


@pytest.mark.parametrize("contract", ["explicit", "vmlx"])
def test_real_mlx_last_projection_entropy_matches_original_forward(contract):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    qwen = pytest.importorskip("mlx_lm.models.qwen3_5")
    from mlx.utils import tree_flatten
    from core.entropy_router import EntropyRouter
    from core.mlx_engine import MLXReasoningBackend

    lengths = []

    class NativeLastLogits(qwen.TextModel):
        def __call__(self, inputs, *, logits_to_keep=0):
            hidden = self.model(inputs)
            if logits_to_keep:
                hidden = hidden[:, -logits_to_keep:, :]
            lengths.append((int(inputs.shape[1]), int(hidden.shape[1])))
            return self.lm_head(hidden)

    class VmlxLastLogits(NativeLastLogits):
        def __call__(self, inputs, **kwargs):
            count = 1 if kwargs.pop("prefill_last_logits_only", False) else 0
            return super().__call__(inputs, logits_to_keep=count)

    VmlxLastLogits.__call__._vmlx_prefill_last_logits_only = True

    mx.random.seed(907)
    args = qwen.TextModelArgs(hidden_size=128, intermediate_size=128, vocab_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
            full_attention_interval=2, linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=64, linear_value_head_dim=64)
    language = (VmlxLastLogits if contract == "vmlx" else NativeLastLogits)(args)
    language.lm_head = nn.QuantizedLinear.from_linear(language.lm_head, group_size=64, bits=2)
    language.eval()
    tokens = [i % 128 for i in range(257)]
    original = qwen.TextModel.__call__(language, mx.array([tokens]))
    probs = mx.softmax(original[:, -1, :], axis=-1)
    expected = -mx.sum(probs * mx.log(probs + 1e-12), axis=-1).item()
    del original, probs
    parameters = dict(tree_flatten(language.parameters()))
    backend = MLXReasoningBackend("fixture/entropy-only")
    backend.model = types.SimpleNamespace(language_model=language)
    backend.tokenizer = types.SimpleNamespace(encode=lambda text: tokens)
    backend.is_mlx_available = True
    measured = backend.calculate_token_entropy("complete untruncated prompt")
    assert measured == pytest.approx(expected, abs=2e-5)
    assert lengths == [(257, 1)]
    assert EntropyRouter().route(measured) == EntropyRouter().route(expected)
    assert backend.model.language_model is language
    assert all(value is parameters[key] for key, value in tree_flatten(language.parameters()))


def test_replacing_prefix_drops_the_old_copy_before_allocating_the_new_one(runtime, monkeypatch):
    import core.prompt_reuse as reuse

    store = ExactPrefixStore(max_entries=1)
    assert store.remember([1, 2, 3, 4], populated(), 2)
    old = weakref.ref(next(iter(store.entries.values())).cache[0].keys)
    clone = reuse.clone_cache

    def tracked(*args):
        assert old() is None
        return clone(*args)

    monkeypatch.setattr(reuse, "clone_cache", tracked)
    assert store.remember([1, 2, 3, 4], populated(), 2)
    assert len(store.entries) == 1


def test_pressure_does_not_allocate_a_prefix_copy(runtime, monkeypatch):
    import core.prompt_reuse as reuse

    store = ExactPrefixStore(memory_admission=lambda size: False)
    monkeypatch.setattr(reuse, "clone_cache", lambda *args: pytest.fail("unadmitted cache allocation"))
    assert not store.remember([1, 2, 3, 4], populated(), 2)
    assert not store.entries and not store.nbytes


def install_vlm(runtime, monkeypatch, produce):
    backend = runtime.backend
    backend._vlm_runtime = "jang_vlm"
    backend.processor = types.SimpleNamespace(tokenizer=backend.tokenizer)
    backend.context_budget_tokens = 32768
    generation = types.ModuleType("mlx_vlm.generate")
    generation.generate_step = lambda *, prefill_step_size=2048: None
    generation.stream_generate = produce
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", generation)
    monkeypatch.setattr("core.vlm_prompt_tokens.prepare_vlm_inputs", lambda model, processor, prompt: ({}, None))
    return backend


def test_second_vlm_message_rechecks_headroom_without_tuning(runtime, monkeypatch):
    import core.mlx_memory_guard as guard

    current = [headroom(8 * GIB)]
    monkeypatch.setattr(guard, "memory_headroom", lambda **kwargs: current[0])
    monkeypatch.setattr(runtime.module, "_adaptive_prefill_step_size", lambda **kwargs: current[0].prefill_tokens)
    calls = []

    def stream(model, processor, **kwargs):
        calls.append((model, processor, kwargs))
        yield types.SimpleNamespace(text="reply", token=1, generation_tokens=1, prompt_tokens=len(kwargs["prompt"]))

    backend = install_vlm(runtime, monkeypatch, stream)
    selected_model = backend.model
    assert list(backend.stream_generate_tokens("first question", max_tokens=3000)) == ["reply"]
    assert "prefill_step_size" not in calls[0][2]
    assert backend._prompt_prefixes.remember([1, 2, 3, 4], populated(), 2)
    # The second complete request now includes the preceding long response.
    current[0] = headroom(GIB // 4)
    full_history = "system instructions\nfirst question\n" + "r" * 3000 + "\nsecond question"
    assert list(backend.stream_generate_tokens(full_history, max_tokens=3000)) == ["reply"]
    assert calls[1][2]["prefill_step_size"] == 128
    assert calls[1][2]["prompt"] == full_history
    assert calls[1][0] is selected_model is backend.model
    assert not backend._prompt_prefixes.entries
    assert backend._lossless_speedups["tokenizer_memo"]


@pytest.mark.parametrize("branches", [False, True])
def test_vlm_oom_retries_only_before_output_and_keeps_full_request(runtime, monkeypatch, branches):
    import core.mlx_memory_guard as guard

    monkeypatch.setattr(guard, "memory_headroom", lambda **kwargs: headroom(8 * GIB))
    monkeypatch.setattr(runtime.module, "_adaptive_prefill_step_size", lambda **kwargs: 128 if kwargs.get("aggressive") else 2048)
    calls, closed, owners = [], [], []

    class Temporary:
        pass

    def stream(model, processor, **kwargs):
        calls.append(kwargs)
        temporary = Temporary()
        owners.append(weakref.ref(temporary))
        try:
            if len(calls) == 1:
                raise RuntimeError("[METAL] Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)")
            assert owners[0]() is None
            yield types.SimpleNamespace(text="once", token=1, generation_tokens=1, prompt_tokens=6)
        finally:
            closed.append(len(calls))

    backend = install_vlm(runtime, monkeypatch, stream)
    if branches:
        output = backend.generate_branches("whole history", branch_count=1, max_tokens=3000,
                                           temperature=0.37, top_p=0.82)
    else:
        output = list(backend.stream_generate_tokens("whole history", max_tokens=3000,
                                                    temperature=0.37, top_p=0.82))
    assert output == ["once"] and len(calls) == len(closed) == 2
    assert calls[0].get("prefill_step_size", 2048) == 2048
    assert calls[1]["prefill_step_size"] == 128
    assert all(call["prompt"] == "whole history" and call["temperature"] == 0.37
               and call["top_p"] == 0.82
               and call["max_tokens"] == 32768 - len("whole history") for call in calls)
    assert all(owner() is None for owner in owners)
    assert backend.last_memory_recovery and not backend._prompt_prefixes.entries


@pytest.mark.parametrize("vlm", [False, True])
def test_oom_after_output_cleans_up_and_never_replays_visible_tokens(runtime, monkeypatch, vlm):
    calls, closed = [], []

    def stream(*args, **kwargs):
        calls.append(kwargs)
        try:
            yield types.SimpleNamespace(text="already visible", token=1, generation_tokens=1, prompt_tokens=3)
            raise RuntimeError("Metal out of memory")
        finally:
            closed.append(True)

    backend = install_vlm(runtime, monkeypatch, stream) if vlm else runtime.backend
    if not vlm:
        monkeypatch.setattr(sys.modules["mlx_lm"], "stream_generate", stream)
    iterator = backend.stream_generate_tokens("abc", max_tokens=3)
    assert next(iterator) == "already visible"
    with pytest.raises(RuntimeError, match="out of memory"):
        next(iterator)
    assert len(calls) == len(closed) == 1
    assert backend.last_memory_recovery
    assert not backend._prompt_prefixes.entries
    assert backend.kv_cache_manager is None or backend.kv_cache_manager.cache is None


def test_failed_retry_is_bounded_and_non_memory_errors_are_not_retried(runtime):
    backend = runtime.backend
    attempts = []

    def failing(message):
        attempts.append(message)
        raise RuntimeError(message)
        yield

    with pytest.raises(RuntimeError, match="out of memory"):
        list(backend._stream_with_memory_retry(sys.modules["mlx.core"], lambda: failing("out of memory")))
    assert len(attempts) == 2
    attempts.clear()
    with pytest.raises(RuntimeError, match="shape mismatch"):
        list(backend._stream_with_memory_retry(sys.modules["mlx.core"], lambda: failing("shape mismatch")))
    assert attempts == ["shape mismatch"]


def test_real_mlx_failed_prefill_tensor_is_freed_before_retry(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from core.mlx_engine import MLXReasoningBackend

    backend = MLXReasoningBackend("fixture/never-load-weights")
    samples, allocations = [], []
    original_recover = backend._recover_generation_memory

    def recovered(module):
        original_recover(module)
        gc.collect()
        module.synchronize()
        assert all(owner() is None for owner in allocations)
        samples.append(module.get_active_memory())

    monkeypatch.setattr(backend, "_recover_generation_memory", recovered)
    for _ in range(8):
        attempts = []

        def native_prefill():
            attempts.append(True)
            temporary = mx.ones((512, 512), dtype=mx.float32)
            mx.eval(temporary)
            allocations.append(weakref.ref(temporary))
            if len(attempts) == 1:
                raise RuntimeError("Metal out of memory")
            yield "reply"

        assert list(backend._stream_with_memory_retry(mx, native_prefill)) == ["reply"]
        assert len(attempts) == 2
    mx.synchronize()
    gc.collect()
    assert all(owner() is None for owner in allocations)
    assert max(samples[2:]) - min(samples[2:]) <= 64 * 1024
