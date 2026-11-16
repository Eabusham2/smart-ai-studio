"""Native VLM APC adapter tests: fake API contracts only, no models imported."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("vlm_prompt_reuse_test", ROOT / "core/vlm_prompt_reuse.py")
reuse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reuse)


@contextmanager
def native_api():
    state = SimpleNamespace(created=[], checkpoint=True, custom=False)
    generation = ModuleType("mlx_vlm.generate")
    def stream_generate(*args, **kwargs):
        apc_manager = kwargs.get("apc_manager")
        apc_tenant = kwargs.get("apc_tenant")
        exact_checkpoint_lengths = []
        raise AssertionError("The adapter must never execute generation")
    def generate_step(*args, prefill_step_size=2048, prompt_cache_checkpoint=None,
                      prompt_cache_checkpoint_lengths=None, **kwargs):
        raise AssertionError("The adapter must never execute prefill")
    generation.stream_generate = stream_generate
    generation.generate_step = generate_step
    apc = ModuleType("mlx_vlm.apc")
    class APCManager:
        def __init__(self, num_blocks, disk, *, overrides):
            self.num_blocks = num_blocks
            self.disk = disk
            self._exact_cache_max = overrides["checkpoint_entries"]
            self.checkpoint_interval_tokens = overrides["checkpoint_interval_tokens"]
            self.exact_cache_guard_tokens = overrides["checkpoint_guard_tokens"]
            self.memory_max_bytes = int(overrides["memory_max_gb"] * 1024 ** 3)
            self.cleared = self.closed = 0
            state.created.append(self)
        def clear(self):
            self.cleared += 1
        def close(self):
            self.closed += 1
    apc.APCManager = APCManager
    native_cache = ModuleType("mlx_vlm.models.cache")
    class KVCache:
        pass
    class ArraysCache:
        pass
    native_cache.KVCache = KVCache
    native_cache.ArraysCache = ArraysCache
    adapters = ModuleType("mlx_vlm.apc_adapters")
    adapters.ADAPTER_SCHEMA_VERSION = 4
    state.adapters = adapters
    coordinator_module = ModuleType("mlx_vlm.apc_coordinator")
    class APCCoordinator:
        def __init__(self, manager, model):
            self.is_checkpoint = state.checkpoint
        def fresh_cache(self):
            return [object()] if state.custom else [KVCache(), ArraysCache()]
        def checkpoint_lengths(self):
            pass
        def store_checkpoint(self):
            pass
    coordinator_module.APCCoordinator = APCCoordinator
    modules = {"mlx_vlm.generate": generation, "mlx_vlm.apc": apc,
               "mlx_vlm.apc_coordinator": coordinator_module, "mlx_vlm.models.cache": native_cache,
               "mlx_vlm.apc_adapters": adapters}
    with patch.dict(sys.modules, modules):
        yield state, generation


def inputs():
    class NoForward:
        def __call__(self, *args, **kwargs):
            raise AssertionError("No real or fake forward is permitted here")
    model = NoForward()
    model.language_model = NoForward()
    return model, SimpleNamespace(tokenizer=object())


def test_native_checkpoint_is_bounded_final_only_and_reused_for_identical_prompt():
    with native_api() as (state, _):
        helper = reuse.NativeVLMCacheReuse(max_bytes=1024 * 1024)
        model, processor = inputs()
        first = helper.request_kwargs(model, processor, "unchanged prepared prompt")
        second = helper.request_kwargs(model, processor, "unchanged prepared prompt")
        assert first == second
        assert len(state.created) == 1
        manager = state.created[0]
        assert manager.num_blocks == 0 and manager.disk is None
        assert manager._exact_cache_max == 1
        assert manager.checkpoint_interval_tokens == 0
        assert manager.exact_cache_guard_tokens == 1
        assert manager.memory_max_bytes == 1024 * 1024
        assert all(value is not model and value is not processor for value in vars(helper).values())


def test_changed_prompt_chunk_or_revision_releases_previous_snapshot():
    with native_api() as (state, _):
        helper = reuse.NativeVLMCacheReuse()
        model, processor = inputs()
        tenants = []
        for text, chunk, revision in (("prompt", 128, 0), ("prompt extended", 128, 0),
                                      ("prompt extended", 256, 0), ("prompt extended", 256, 1)):
            tenants.append(helper.request_kwargs(model, processor, text,
                                                  prefill_step_size=chunk, revision=revision)["apc_tenant"])
        assert len(set(tenants)) == 4
        assert all(manager.cleared == manager.closed == 1 for manager in state.created[:-1])
        helper.clear()
        assert all(manager.cleared == manager.closed == 1 for manager in state.created)
        assert not helper.telemetry()["vlm_checkpoint_enabled"]


def test_unknown_dense_layout_and_disabled_chunking_fail_open():
    with native_api() as (state, _):
        helper = reuse.NativeVLMCacheReuse()
        model, processor = inputs()
        state.checkpoint = False
        assert helper.request_kwargs(model, processor, "prompt") == {}
        assert state.created[0].cleared == state.created[0].closed == 1
        model.no_chunked_prefill = True
        assert helper.request_kwargs(model, processor, "prompt") == {}
        assert len(state.created) == 1


def test_old_runtime_never_receives_unverified_apc_kwargs():
    with native_api() as (state, generation):
        helper = reuse.NativeVLMCacheReuse()
        model, processor = inputs()
        generation.stream_generate = lambda *args, **kwargs: None
        assert helper.request_kwargs(model, processor, "prompt") == {}
        assert not state.created


def test_pressure_disable_and_model_switch_release_state():
    with native_api() as (state, _):
        helper = reuse.NativeVLMCacheReuse()
        model, processor = inputs()
        assert helper.request_kwargs(model, processor, "prompt")
        assert helper.request_kwargs(model, processor, "prompt", memory_ok=False) == {}
        assert state.created[-1].cleared == 1
        assert helper.request_kwargs(model, processor, "prompt")
        assert helper.request_kwargs(model, processor, "prompt", enabled=False) == {}
        assert state.created[-1].cleared == 1
        assert helper.request_kwargs(model, processor, "prompt")
        other_model, _ = inputs()
        assert helper.request_kwargs(other_model, processor, "prompt")
        assert state.created[-2].cleared == 1


def test_custom_layout_and_unreviewed_native_snapshot_schema_fail_open():
    with native_api() as (state, _):
        helper = reuse.NativeVLMCacheReuse()
        model, processor = inputs()
        state.custom = True
        assert helper.request_kwargs(model, processor, "prompt") == {}
        assert state.created[-1].cleared == state.created[-1].closed == 1
        state.custom = False
        state.adapters.ADAPTER_SCHEMA_VERSION = 5
        assert helper.request_kwargs(model, processor, "prompt") == {}
        assert "schema" in helper.last_reason
