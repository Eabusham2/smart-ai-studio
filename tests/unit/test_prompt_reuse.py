"""Model-free cache contracts. These tests make no real-model exactness claim."""
import copy
import sys
import types

import numpy as np
import pytest

from core.prompt_reuse import ExactPrefixStore, TokenizerMemo, clone_cache


class Array:
    def __init__(self, values):
        self.data = np.asarray(values)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def nbytes(self):
        return self.data.nbytes


class KVCache:
    def __init__(self):
        self.keys = self.values = None
        self.offset = 0

    @property
    def state(self):
        return self.keys, self.values, self.offset


class QuantizedKVCache(KVCache):
    def __init__(self):
        super().__init__()
        self.group_size = 64
        self.bits = 8


class ArraysCache:
    def __init__(self):
        self.cache = [None, None]
        self.left_padding = self.lengths = None

    @property
    def state(self):
        return self.cache, self.left_padding, self.lengths


class Tokenizer:
    def __init__(self):
        self.calls = 0
        self.added = {}

    def encode(self, prompt):
        self.calls += 1
        return [ord(c) for c in prompt]

    def get_added_vocab(self):
        return self.added


@pytest.fixture
def runtime(monkeypatch):
    import core.kv_cache_manager as kv
    import core.mlx_engine as engine
    import core.mlx_memory_guard as memory_guard
    # A previous real-VLM test can leave thread-local generation streams in
    # sys.modules. They cannot be rebound to this fixture's synthetic mx.
    for name in list(sys.modules):
        if name == "mlx_vlm.generate" or name.startswith(("mlx_vlm.generate.", "mlx_vlm.speculative.")):
            monkeypatch.delitem(sys.modules, name)
    mx = types.ModuleType("mlx.core")
    mx.array = Array
    mx.eval = lambda *args: None
    mx.clear_cache = lambda: None
    mlx = types.ModuleType("mlx")
    mlx.core = mx
    mlx.__path__ = []
    lm = types.ModuleType("mlx_lm")
    lm.__path__ = []
    models = types.ModuleType("mlx_lm.models")
    cache = types.ModuleType("mlx_lm.models.cache")
    cache.KVCache = KVCache
    cache.QuantizedKVCache = QuantizedKVCache
    cache.ArraysCache = ArraysCache
    models.cache = cache
    generation = types.ModuleType("mlx_lm.generate")
    generation.generate_step = lambda *, prompt_cache=None, prefill_step_size=2048: None
    sample = types.ModuleType("mlx_lm.sample_utils")
    sample.make_sampler = lambda **kwargs: kwargs
    modules = {"mlx": mlx, "mlx.core": mx, "mlx_lm": lm,
               "mlx_lm.models": models, "mlx_lm.models.cache": cache,
               "mlx_lm.generate": generation, "mlx_lm.sample_utils": sample}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(kv, "MLX_AVAILABLE", True)
    monkeypatch.setattr(kv, "make_prompt_cache", lambda model: [KVCache()])
    monkeypatch.setattr(engine, "_memory_pressure", lambda: False)
    monkeypatch.setattr(memory_guard, "memory_headroom", lambda **kwargs: memory_guard.MemoryHeadroom(
        8 * memory_guard.GIB, 2 * memory_guard.GIB, 2 * memory_guard.GIB, 0,
        12 * memory_guard.GIB, memory_guard.GIB, 8 * memory_guard.GIB))
    monkeypatch.setattr(engine, "_adaptive_prefill_step_size", lambda **kwargs: 1 if kwargs.get("aggressive") else 2)

    class Model:
        def __init__(self):
            self.calls = []

        def __call__(self, tokens, cache):
            ids = tokens.data.reshape(-1).tolist()
            self.calls.append(tuple(ids))
            for layer in cache:
                if isinstance(layer, ArraysCache):
                    layer.cache[0] = Array([[sum(ids)]])
                    continue
                old = [] if layer.keys is None else layer.keys.data.reshape(-1).tolist()
                layer.keys = Array(np.array(old + ids).reshape(1, 1, -1, 1))
                layer.values = copy.deepcopy(layer.keys)
                layer.offset += len(ids)
            return Array([0])

    state = types.SimpleNamespace(streams=[], closed=0, fail_before=0, fail_after=False)

    def stream_generate(model, tokenizer, *, prompt, prompt_cache, prefill_step_size, max_tokens, sampler):
        state.streams.append({"prompt": tuple(prompt), "cache": prompt_cache,
                              "sampler": sampler, "max_tokens": max_tokens,
                              "step": prefill_step_size})
        try:
            if state.fail_before:
                state.fail_before -= 1
                raise RuntimeError("Metal out of memory")
            pending = list(prompt)
            while len(pending) > 1:
                n = min(prefill_step_size, len(pending) - 1)
                model(Array([pending[:n]]), cache=prompt_cache)
                pending = pending[n:]
            model(Array([pending]), cache=prompt_cache)
            for index in range(max_tokens):
                value = int(prompt_cache[0].keys.data.sum()) % 31
                yield types.SimpleNamespace(text=str(value), token=value, generation_tps=0.0)
                if state.fail_after:
                    raise RuntimeError("decode failed")
                model(Array([[value]]), cache=prompt_cache)
        finally:
            state.closed += 1

    lm.stream_generate = stream_generate
    backend = engine.MLXReasoningBackend(model_path="never-load-this-fixture")
    backend.model = Model()
    backend.tokenizer = Tokenizer()
    backend.is_mlx_available = True
    backend.context_budget_tokens = 7
    return types.SimpleNamespace(backend=backend, state=state, module=engine, kv=kv, cache=cache)


def populated(n=4, cls=KVCache):
    layer = cls()
    layer.offset = n
    layer.keys = Array(np.arange(n).reshape(1, 1, n, 1))
    layer.values = copy.deepcopy(layer.keys)
    return [layer]


def test_tokenizer_memo_returns_isolated_ids_and_invalidates_public_mutation():
    tok = Tokenizer()
    memo = TokenizerMemo()
    first = memo.encode(tok, "abc")
    first.clear()
    assert memo.encode(tok, "abc") == [97, 98, 99]
    assert tok.calls == 1
    tok.added["custom"] = 999
    memo.encode(tok, "abc")
    assert tok.calls == 2
    tok.chat_template = "changed"
    memo.encode(tok, "abc")
    assert tok.calls == 3
    memo.encode(Tokenizer(), "abc")
    assert memo.hits == 1


def test_tokenizer_memory_and_disabled_mode_are_bounded():
    tok = Tokenizer()
    memo = TokenizerMemo(max_entries=2, max_tokens=4, max_bytes=1024)
    for prompt in ("a", "bb", "cc", "d", "x" * 5000):
        assert memo.encode(tok, prompt) == [ord(c) for c in prompt]
    assert len(memo.entries) <= 2 and memo.tokens <= 4 and memo.nbytes <= 1024
    memo.clear()
    memo.encode(tok, "a", enabled=False)
    memo.encode(tok, "a", enabled=False)
    assert not memo.entries


def test_exact_prefix_clones_are_isolated_and_leave_last_token(runtime):
    store = ExactPrefixStore()
    cache = populated()
    assert store.remember([1, 2, 3, 4], cache, 2)
    cache[0].keys.data[:] = 99
    first, offset = store.lookup([1, 2, 3, 4, 5], 2)
    assert offset == 4
    first[0].keys.data[:] = 77
    second, _ = store.lookup([1, 2, 3, 4, 5], 2)
    assert second[0].keys.data.reshape(-1).tolist() == [0, 1, 2, 3]
    assert store.lookup([1, 2, 3, 4], 2) is None
    assert store.lookup([1, 2, 0, 4, 5], 2) is None
    assert store.lookup([1, 2, 3, 4, 5], 3) is None


def test_chunk_boundary_guard_for_longer_context(runtime):
    store = ExactPrefixStore()
    assert store.remember([1, 2, 3], populated(3), 2)
    assert store.lookup([1, 2, 3, 4], 2)
    assert store.lookup([1, 2, 3, 4, 5], 2) is None
    assert store.remember([1, 2, 3, 4], populated(4), 2)
    assert store.lookup([1, 2, 3, 4, 5, 6, 7], 2)[1] == 4


def test_unknown_cache_corrupt_offsets_and_aliasing_fail_closed(runtime, monkeypatch):
    store = ExactPrefixStore()
    class CustomCache(KVCache):
        pass
    assert not store.remember([1, 2], populated(2, CustomCache), 2)
    assert not store.remember([1, 2], populated(1), 2)
    cache = populated(2)
    cache[0].external_state = []
    assert not store.remember([1, 2], cache, 2)
    monkeypatch.setattr(KVCache, "__deepcopy__", lambda self, memo: self, raising=False)
    assert not store.remember([1, 2], populated(2), 2)


def test_hybrid_whole_state_and_native_quantized_metadata_survive_clone(runtime):
    recurrent = ArraysCache()
    recurrent.cache = [Array([[4, 6]]), Array([[7, 8]])]
    cache = populated(4, QuantizedKVCache) + [recurrent]
    cloned = clone_cache(cache, 4)
    cloned[1].cache[0].data[:] = 9
    assert recurrent.cache[0].data.tolist() == [[4, 6]]
    assert cloned[0].bits == 8 and cloned[0].group_size == 64
    recurrent.left_padding = Array([0])
    with pytest.raises(ValueError, match="batched"):
        clone_cache(cache, 4)


def test_prefix_store_has_entry_byte_and_token_limits(runtime):
    store = ExactPrefixStore(max_entries=1, max_tokens=4, max_bytes=1024)
    assert store.remember([1, 2], populated(2), 2)
    assert store.remember([3, 4], populated(2), 2)
    assert len(store.entries) == 1
    assert not store.remember([1, 2, 3, 4, 5], populated(5), 2)
    store.max_bytes = 1
    assert not store.remember([1, 2], populated(2), 2)
    store.clear()
    assert store.nbytes == 0


def test_production_branches_share_one_prefill_keep_sampling_and_context(runtime):
    backend, state = runtime.backend, runtime.state
    outputs = backend.generate_branches("abcde", branch_count=3, max_tokens=2,
                                         temperature=[0.2, 0.7, 1.1], top_p=0.91)
    assert len(outputs) == 3
    assert backend.tokenizer.calls == 1
    assert backend.model.calls.count((97, 98)) == 1
    assert backend.model.calls.count((99, 100)) == 1
    assert [request["prompt"] for request in state.streams] == [(101,)] * 3
    assert [request["sampler"]["temp"] for request in state.streams] == [0.2, 0.7, 1.1]
    assert all(request["sampler"]["top_p"] == 0.91 and request["max_tokens"] == 2 for request in state.streams)
    assert len({id(request["cache"]) for request in state.streams}) == 3
    assert backend.kv_cache_manager.cache is None
    backend.configure_lossless_speedups(False)
    reference = backend.generate_branches("abcde", branch_count=3, max_tokens=2,
                                           temperature=[0.2, 0.7, 1.1], top_p=0.91)
    assert outputs == reference
    assert state.streams[-1]["prompt"] == tuple(map(ord, "abcde"))


def test_production_repeat_and_extension_reuse_exact_tokens(runtime):
    backend = runtime.backend
    expected = list(backend.stream_generate_tokens("abcde", max_tokens=2))
    assert list(backend.stream_generate_tokens("abcde", max_tokens=2)) == expected
    assert backend.tokenizer.calls == 1
    assert backend.last_prompt_reused_tokens == 4
    backend.context_budget_tokens = 9
    list(backend.stream_generate_tokens("abcdefg", max_tokens=2))
    assert backend.last_prompt_reused_tokens == 4
    assert backend.model.calls.count((97, 98)) == 1
    list(backend.stream_generate_tokens("abXde", max_tokens=2))
    assert backend.last_prompt_reused_tokens == 0


def test_cancel_and_decode_failure_never_publish_mutable_decode_state(runtime):
    backend, state = runtime.backend, runtime.state
    stream = backend.stream_generate_tokens("abcde", max_tokens=3)
    first = next(stream)
    stream.close()
    assert state.closed == 1 and backend.kv_cache_manager.cache is None
    backend.context_budget_tokens = 6
    assert list(backend.stream_generate_tokens("abcde", max_tokens=1)) == [first]
    state.fail_after = True
    with pytest.raises(RuntimeError, match="decode failed"):
        list(backend.stream_generate_tokens("abcde", max_tokens=3))
    assert backend.kv_cache_manager.cache is None
    assert all(snapshot.cache[0].offset == 4 for snapshot in backend._prompt_prefixes.entries.values())


def test_oom_retry_preserves_policy_and_never_replays_visible_tokens(runtime):
    backend, state = runtime.backend, runtime.state
    state.fail_before = 1
    list(backend.stream_generate_tokens("abcde", max_tokens=2, temperature=0.37, top_p=0.82))
    assert len(state.streams) == 2
    assert [call["step"] for call in state.streams] == [2, 1]
    assert all(call["sampler"] == {"temp": 0.37, "top_p": 0.82} for call in state.streams)
    assert all(call["max_tokens"] == 2 for call in state.streams)


def test_model_tokenizer_training_mutations_and_pressure_invalidate(runtime, monkeypatch):
    backend = runtime.backend
    list(backend.stream_generate_tokens("abcde", max_tokens=1))
    backend.tokenizer.added["changed"] = 999
    list(backend.stream_generate_tokens("abcde", max_tokens=1))
    assert backend.last_prompt_reused_tokens == 0
    old = backend.model
    backend.model = type(old)()
    list(backend.stream_generate_tokens("abcde", max_tokens=1))
    assert backend.last_prompt_reused_tokens == 0
    def partial_failure(self, *args, **kwargs):
        raise RuntimeError("partial weight update")
    monkeypatch.setattr(runtime.module._base.MLXReasoningBackend, "train_mini_batch", partial_failure)
    with pytest.raises(RuntimeError, match="partial weight"):
        backend.train_mini_batch({}, [])
    assert not backend._prompt_prefixes.entries and not backend._tokenizer_memo.entries
    list(backend.stream_generate_tokens("abcde", max_tokens=1))
    monkeypatch.setattr(runtime.module, "_memory_pressure", lambda: True)
    list(backend.stream_generate_tokens("abcde", max_tokens=1))
    assert not backend._prompt_prefixes.entries
    assert runtime.state.streams[-1]["prompt"] == tuple(map(ord, "abcde"))


def test_unsupported_cache_recomputes_with_original_policy(runtime, monkeypatch):
    class TurboQuantFixture(KVCache):
        pass
    monkeypatch.setattr(runtime.kv, "make_prompt_cache", lambda model: [TurboQuantFixture()])
    backend = runtime.backend
    list(backend.stream_generate_tokens("abcde", max_tokens=1))
    assert runtime.state.streams[0]["prompt"] == tuple(map(ord, "abcde"))
    assert type(runtime.state.streams[0]["cache"][0]) is TurboQuantFixture
    assert not backend._prompt_prefixes.entries


def test_fresh_arena_does_not_reset_again_for_large_complete_prompt(runtime, monkeypatch):
    calls = []
    monkeypatch.setattr(runtime.kv, "make_prompt_cache", lambda model: calls.append(1) or [KVCache()])
    arena = runtime.kv.SmartKVCacheManager(object(), max_tokens=2)
    arena.ensure_fresh()
    assert not arena.auto_compact_if_needed(50000)
    assert len(calls) == 1
    arena.release()
    assert arena.cache is None


def test_cache_precision_changes_and_training_mode_bypass_saved_state(runtime, monkeypatch):
    backend = runtime.backend
    list(backend.stream_generate_tokens("abcde", max_tokens=1))
    monkeypatch.setattr(runtime.kv, "make_prompt_cache", lambda model: [QuantizedKVCache()])
    list(backend.stream_generate_tokens("abcde", max_tokens=1))
    assert backend.last_prompt_reused_tokens == 0
    assert type(runtime.state.streams[-1]["cache"][0]) is QuantizedKVCache
    backend.model.training = True
    list(backend.stream_generate_tokens("abcde", max_tokens=1))
    assert backend.last_prompt_reused_tokens == 0
    assert runtime.state.streams[-1]["prompt"] == tuple(map(ord, "abcde"))


def test_supplied_context_count_avoids_tokenizer_and_preserves_total_budget(runtime, monkeypatch):
    backend = runtime.backend
    monkeypatch.setattr(backend.tokenizer, "encode", lambda text: pytest.fail("redundant tokenization"))
    assert backend._effective_generation_cap("abcde", 999, prompt_token_count=5) == 2
    with pytest.raises(RuntimeError, match="Context full"):
        backend._effective_generation_cap("abcde", 999, prompt_token_count=7)


def test_vlm_production_keeps_processor_prompt_and_sampling_and_clears_on_cancel(runtime, monkeypatch):
    backend = runtime.backend
    backend._vlm_runtime = "mlx_vlm"
    backend.processor = object()
    class Checkpoints:
        def __init__(self):
            self.requests = []
            self.cleared = 0
            self.last_reason = "fixture native checkpoint"
            self.max_bytes = 256 * 1024 ** 2
            self.manager = object()
        def request_kwargs(self, *args, **kwargs):
            self.requests.append((args, kwargs))
            return {"apc_manager": self.manager, "apc_tenant": "fixture-tenant"}
        def clear(self):
            self.cleared += 1
    checkpoints = Checkpoints()
    backend._vlm_prefixes = checkpoints
    generation = types.ModuleType("mlx_vlm.generate")
    calls = []
    def stream(model, processor, **kwargs):
        calls.append((model, processor, kwargs))
        yield types.SimpleNamespace(text="a", cached_tokens=4)
        yield types.SimpleNamespace(text="b", cached_tokens=4)
    generation.stream_generate = stream
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", generation)
    assert backend.generate_branches("abcde", branch_count=3, temperature=[0.1, 0.3, 0.8], top_p=0.89) == ["ab"] * 3
    assert all(model is backend.model and processor is backend.processor for model, processor, _ in calls)
    assert all(kwargs["prompt"] == "abcde" and kwargs["image"] == [] for _, _, kwargs in calls)
    assert [kwargs["temperature"] for _, _, kwargs in calls] == [0.1, 0.3, 0.8]
    assert all(kwargs["top_p"] == 0.89 and kwargs["apc_manager"] is checkpoints.manager for _, _, kwargs in calls)
    assert all(request[1]["enabled"] for request in checkpoints.requests)
    assert backend.last_prompt_reused_tokens == 4
    cleared = checkpoints.cleared
    active = backend.stream_generate_tokens("abcde")
    assert next(active) == "a"
    active.close()
    assert checkpoints.cleared == cleared + 1
    assert not backend.model.calls


def test_local_artifact_stamp_detects_changed_loader_bytes(tmp_path):
    from core.mlx_engine import _local_artifact_stamp
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"initial fixture bytes, not real weights")
    before = _local_artifact_stamp(path)
    path.write_bytes(b"different fixture bytes")
    assert _local_artifact_stamp(path) != before
    assert _local_artifact_stamp(tmp_path / "missing") is None
