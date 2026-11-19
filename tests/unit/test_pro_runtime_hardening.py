from types import SimpleNamespace

import pytest

from core.hf_downloader import (
    is_model_registered_loaded,
    unregister_loaded_model,
)
from core.pro_runtime_hardening import (
    DEFAULT_MLX_ADAPTER_PATH,
    _normalize_adapter_path,
    install_pro_runtime_hardening,
)


class DummyPro:
    def __init__(self, *, use_mock=False, loaded=True, lora_adapter_path=None):
        self.settings = SimpleNamespace(use_mock=use_mock)
        self._loaded = loaded
        self.active_backend = "mlx" if loaded else None
        self.mlx_backend = None
        self.tokenizer = None
        self.lora_adapter_path = lora_adapter_path

    @property
    def is_model_loaded(self):
        return self._loaded

    def load_model(self, model_name, model_path=None, backend=None):
        self._loaded = True
        self.active_backend = backend or "mlx"
        return {
            "status": "loaded",
            "model": model_name,
            "path": model_path or "owner/model",
            "backend": self.active_backend,
        }

    def unload_model(self):
        if getattr(self, "unload_error", False):
            return {"status": "error", "unloaded": False}
        self._loaded = False
        self.active_backend = None
        return {"status": "unloaded"}

    def _generate_fallback_branches(self, prompt, branch_count):
        return ["synthetic fallback"] * branch_count

    def solve(self, prompt, **kwargs):
        return "one two three four", {"tok_speed": 10.9, "backend": "fake", "memory_rss_mb": 1536.0}


install_pro_runtime_hardening(DummyPro)


def setup_function():
    unregister_loaded_model()


def teardown_function():
    unregister_loaded_model()


def test_default_and_legacy_directory_adapter_paths_become_safetensors_files(tmp_path):
    assert DEFAULT_MLX_ADAPTER_PATH.endswith("adapters.safetensors")
    assert _normalize_adapter_path(None) == DEFAULT_MLX_ADAPTER_PATH
    legacy_dir = tmp_path / "legacy_adapter"
    legacy_dir.mkdir()
    assert _normalize_adapter_path(str(legacy_dir)) == str(legacy_dir / "adapters.safetensors")


def test_live_engine_uses_model_bound_state_unless_a_legacy_adapter_is_explicit(tmp_path):
    engine = DummyPro(use_mock=False, loaded=False)
    assert engine.lora_adapter_path is None
    assert engine._explicit_adapter_path is False

    legacy_dir = tmp_path / "legacy_adapter"
    legacy_dir.mkdir()
    engine2 = DummyPro(use_mock=False, loaded=False, lora_adapter_path=str(legacy_dir))
    assert engine2.lora_adapter_path == str(legacy_dir / "adapters.safetensors")
    assert engine2._explicit_adapter_path is True


def test_live_loaded_model_never_falls_back_to_synthetic_output():
    engine = DummyPro(use_mock=False, loaded=True)
    with pytest.raises(RuntimeError, match="Refusing to substitute mock/synthetic output"):
        engine._generate_fallback_branches("hello", 2)


def test_explicit_mock_mode_keeps_test_fallback():
    engine = DummyPro(use_mock=True, loaded=True)
    assert engine._generate_fallback_branches("hello", 2) == ["synthetic fallback", "synthetic fallback"]


def test_unloaded_live_model_keeps_user_load_guidance_path():
    engine = DummyPro(use_mock=False, loaded=False)
    assert engine._generate_fallback_branches("hello", 1) == ["synthetic fallback"]


def test_solve_overwrites_hardcoded_telemetry_with_measured_metadata():
    engine = DummyPro(use_mock=False, loaded=True)
    response, meta = engine.solve("hello")
    assert response == "one two three four"
    assert "tokens_generated" not in meta  # Words are not model token counts.
    assert meta["token_usage_available"] is False
    assert meta["tok_speed"] == 0.0
    assert meta["tok_speed"] != 10.9
    assert meta["memory_rss_mb"] >= 0.0
    assert meta["memory_rss_mb"] != 1536.0
    assert meta["backend"] == "mlx"
    assert meta["measured_wall_time_s"] > 0.0


def test_successful_live_load_registers_model_and_unload_clears_it():
    engine = DummyPro(use_mock=False, loaded=False)
    result = engine.load_model("Friendly Name", model_path="owner/model")
    assert result["status"] == "loaded"
    assert is_model_registered_loaded("owner/model") is True
    assert is_model_registered_loaded("Friendly Name") is True

    engine.unload_model()
    assert is_model_registered_loaded("owner/model") is False
    assert is_model_registered_loaded("Friendly Name") is False


def test_mock_load_never_marks_real_cache_ready():
    engine = DummyPro(use_mock=True, loaded=False)
    engine.load_model("Mock Model", model_path="owner/mock")
    assert is_model_registered_loaded("owner/mock") is False


def test_failed_strata_shutdown_keeps_registration_until_successful_retry():
    engine = DummyPro(use_mock=False, loaded=False)
    engine.load_model("Strata model", model_path="strata://gyro-s", backend="controller")
    engine.controller_backend = SimpleNamespace(runtime="strata")
    engine.unload_error = True
    assert engine.unload_model()["status"] == "error"
    assert is_model_registered_loaded("strata://gyro-s")
    engine.unload_error = False
    assert engine.unload_model()["status"] == "unloaded"
    assert not is_model_registered_loaded("strata://gyro-s")


@pytest.mark.parametrize("runtime", ["prism_speculative", "prism_gguf", "gguf"])
def test_failed_native_shutdown_keeps_registered_files_and_owner_until_retry(runtime):
    engine = DummyPro(use_mock=False, loaded=False)
    engine.load_model("Pinned native", model_path="local/pinned.gguf", backend="gguf")
    native = SimpleNamespace(runtime=runtime, model=object())
    engine.gguf_backend = native
    engine.unload_error = True
    assert engine.unload_model()["status"] == "error"
    assert engine.gguf_backend is native
    assert is_model_registered_loaded("local/pinned.gguf")
    assert is_model_registered_loaded("Pinned native")
    engine.unload_error = False
    assert engine.unload_model()["status"] == "unloaded"
    assert not is_model_registered_loaded("local/pinned.gguf")
    assert not is_model_registered_loaded("Pinned native")


@pytest.mark.parametrize("usage", [{}, {"completion_tokens": 83, "prompt_tokens": 120, "total_tokens": 203}])
def test_strata_telemetry_uses_native_usage_and_owned_process_memory(monkeypatch, usage):
    from core.memory_watchdog import SystemMemoryWatchdog

    observed = []

    def memory(owned_pids=None):
        observed.append(owned_pids)
        return {"total_allocated_gb": 4.5}

    monkeypatch.setattr(SystemMemoryWatchdog, "get_detailed_memory_breakdown", memory)
    engine = DummyPro(use_mock=False, loaded=True)
    engine.active_backend = "controller"
    engine.controller_backend = SimpleNamespace(
        runtime="strata", last_branch_usage=usage, owned_process_ids=lambda: [123, 456],
    )
    engine.mlx_backend = SimpleNamespace(last_tok_per_sec=999.0, last_generation_cap_reason="Old MLX request")
    response, meta = engine.solve("hello")
    assert response == "one two three four"
    assert meta["backend"] == "strata"
    assert meta["decode_tok_speed"] == 0.0
    assert meta["generation_cap_reason"] == ""
    assert meta["usage"] == usage
    assert meta["memory_rss_mb"] == 4.5 * 1024
    assert observed == [[123, 456]]
    if usage:
        assert meta["tokens_generated"] == 83
        assert meta["token_usage_available"] is True
        assert meta["tok_speed"] * meta["measured_wall_time_s"] == pytest.approx(83)
    else:
        assert "tokens_generated" not in meta
        assert meta["token_usage_available"] is False
        assert meta["tok_speed"] == 0.0


@pytest.mark.parametrize("usage", [None, {}, {"completion_tokens": 4, "prompt_tokens": 10, "total_tokens": 14}])
def test_exact_native_telemetry_requires_actual_counts_and_excludes_stale_mlx(monkeypatch, usage):
    from core.memory_watchdog import SystemMemoryWatchdog
    observed = []
    def memory(owned_pids=None):
        observed.append(owned_pids)
        return {"total_allocated_gb": 3.0}
    monkeypatch.setattr(SystemMemoryWatchdog, "get_detailed_memory_breakdown", memory)
    engine = DummyPro(use_mock=False, loaded=True)
    engine.active_backend = "gguf"
    engine.gguf_backend = SimpleNamespace(
        runtime="prism_speculative", last_branch_usage=usage,
        owned_process_ids=lambda: [42],
        get_speculative_telemetry=lambda: {"mode": "lookup", "decode_tokens_per_second": 12.5},
    )
    engine.mlx_backend = SimpleNamespace(last_tok_per_sec=999.0, last_generation_cap_reason="stale")
    response, meta = engine.solve("hello")
    assert meta["backend"] == "prism_speculative"
    assert meta["decode_tok_speed"] == 12.5 and meta["generation_cap_reason"] == ""
    assert meta["memory_rss_mb"] == 3 * 1024 and observed == [[42]]
    assert meta["speculative"]["mode"] == "lookup"
    if usage:
        assert meta["tokens_generated"] == 4 and meta["token_usage_available"]
        assert meta["tok_speed"] * meta["measured_wall_time_s"] == pytest.approx(4)
    else:
        assert "tokens_generated" not in meta and not meta["token_usage_available"]
        assert meta["tok_speed"] == 0.0


def test_metadata_and_outer_gui_observer_include_every_real_branch():
    from core.generation_metrics import TokenMeter, observe_generations

    class ObservedPro(DummyPro):
        _pro_runtime_hardening_installed = False
        def solve(self, prompt, **kwargs):
            for count in (5, 7, 3):
                meter = TokenMeter()
                meter.observe(delta=1)
                meter.observe(count=count)
                meter.finish()
            return "winner of arbitrary word length", {"branch_count": 3}

    install_pro_runtime_hardening(ObservedPro)
    samples = {}
    with observe_generations(lambda sample: samples.update({sample['id']: sample})):
        response, meta = ObservedPro().solve("question")
    assert meta['tokens_generated'] == 15
    assert meta['token_usage_available'] is True
    assert meta['generation_invocations'] == 3
    assert meta['tok_speed'] * meta['measured_wall_time_s'] == pytest.approx(15)
    assert len(samples) == 3 and sum(sample['tokens'] for sample in samples.values()) == 15
    assert response == "winner of arbitrary word length"
