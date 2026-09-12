from types import SimpleNamespace

import pytest

from core.hf_downloader import (
    is_model_registered_loaded,
    unregister_loaded_model,
)
from core.pro_runtime_hardening import install_pro_runtime_hardening


class DummyPro:
    def __init__(self, *, use_mock=False, loaded=True):
        self.settings = SimpleNamespace(use_mock=use_mock)
        self._loaded = loaded
        self.active_backend = "mlx" if loaded else None
        self.mlx_backend = None
        self.tokenizer = None

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
    assert meta["tokens_generated"] == 4
    assert meta["tok_speed"] >= 0.0
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
