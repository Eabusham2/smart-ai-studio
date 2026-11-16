"""Preferences and lifecycle contracts with no inference runtime or model load."""
import json
import threading
from types import SimpleNamespace

import pytest

from core.lossless_speed_control import (
    DEFAULT_OPTIONS, apply_loaded_options, configure_speedups, initial_options,
    load_preferences, normalize_options, save_preferences,
)
from core.speed_control import install_speed_control


def engine_with(backend=None, active="mlx"):
    return SimpleNamespace(
        _model_lock=threading.RLock(), active_backend=active,
        mlx_backend=backend if active == "mlx" else None,
        gguf_backend=backend if active == "gguf" else None,
        lossless_speed_options=dict(DEFAULT_OPTIONS),
    )


def test_options_preserve_explicit_paths_and_drafting_is_opt_in():
    selected = normalize_options({"speculative_mode": "off", "cache_dir": " /tmp/prompts ",
                                  "speculative_server_path": "./chosen/runtime"})
    assert selected["speculative_mode"] == "none"
    assert selected["cache_dir"] == "/tmp/prompts"
    assert selected["speculative_server_path"] == "./chosen/runtime"
    assert not selected["persistent_cache"] and not selected["metal_warmup"]
    assert selected["draft_model_path"] is None
    assert selected["native_branch_count"] == 1
    assert normalize_options({"speculative_mode": "PLD"})["speculative_mode"] == "lookup"
    assert normalize_options({"speculative_mode": "dflash"})["speculative_mode"] == "dflash2"


@pytest.mark.parametrize("options", [
    {"enabled": "false"}, {"persistent_cache": 1}, {"unknown": True},
    {"max_draft_tokens": True}, {"max_draft_tokens": 0}, {"max_draft_tokens": 65},
    {"max_draft_tokens": 3.0}, {"cache_dir": 42}, {"cache_dir": "path\0evil"},
    {"draft_model_path": "one\ntwo"}, {"speculative_mode": "invented"},
    {"native_branch_count": 0}, {"native_branch_count": 65},
    {"native_branch_count": True}, {"native_branch_count": "8"},
])
def test_invalid_configuration_never_reaches_backend(options):
    calls = []
    backend = SimpleNamespace(model=object(), configure_lossless_speedups=lambda **kwargs: calls.append(kwargs))
    engine = engine_with(backend)
    before = dict(engine.lossless_speed_options)
    with pytest.raises(ValueError):
        configure_speedups(engine, options)
    assert not calls and engine.lossless_speed_options == before


def test_initial_settings_import_only_explicit_speed_preferences():
    settings = SimpleNamespace(speedup_enabled=False, speedup_branch_prefill=False,
                               speculative_mode="lookup", draft_model_path="local/draft.gguf",
                               unrelated="preserved")
    values = initial_options(settings)
    assert not values["enabled"] and not values["branch_prefill"]
    assert values["speculative_mode"] == "lookup"
    assert values["draft_model_path"] == "local/draft.gguf"
    assert "unrelated" not in values


def test_prompt_options_apply_without_loading_or_changing_native_selection():
    calls = []
    backend = SimpleNamespace(model=object(), configure_lossless_speedups=lambda **kwargs: calls.append(kwargs) or kwargs)
    engine = engine_with(backend)
    result = configure_speedups(engine, {"prompt_reuse": False, "branch_prefill": True,
                                        "cache_dir": "/chosen/cache"})
    assert result["status"] == "configured"
    assert len(calls) == 1
    assert calls[0]["enabled"] is True and calls[0]["prompt_reuse"] is False
    assert calls[0]["branch_prefill"] is True and calls[0]["cache_dir"] == "/chosen/cache"
    assert "speculative_mode" not in calls[0] and "draft_model_path" not in calls[0]
    assert engine.lossless_speed_options["speculative_mode"] == "none"


def test_disk_default_location_is_resolved_only_when_explicitly_enabled(monkeypatch, tmp_path):
    import config.paths
    calls = []
    monkeypatch.setattr(config.paths, "get_portable_data_dir", lambda: str(tmp_path))
    backend = SimpleNamespace(configure_lossless_speedups=lambda **kwargs: calls.append(kwargs) or kwargs)
    engine = engine_with(backend)
    apply_loaded_options(engine)
    assert calls[-1]["cache_dir"] is None
    apply_loaded_options(engine, {"persistent_cache": True})
    assert calls[-1]["cache_dir"] == str(tmp_path / "prompt_state")


def test_native_draft_change_is_deferred_without_mutating_live_target():
    target = object()
    backend = SimpleNamespace(model=target, runtime="prism_speculative",
                              get_speculative_telemetry=lambda: {"mode": "none"})
    engine = engine_with(backend, "gguf")
    result = configure_speedups(engine, {"speculative_mode": "lookup", "max_draft_tokens": 4})
    assert result["status"] == "reload_required"
    assert backend.model is target
    assert engine.lossless_speed_options["speculative_mode"] == "lookup"


@pytest.mark.parametrize("lock_name", ["_model_lock", "_runtime_model_lock"])
def test_busy_lifecycle_or_inference_preserves_saved_configuration(lock_name):
    calls = []
    backend = SimpleNamespace(model=object(), _runtime_model_lock=threading.RLock(),
                              configure_lossless_speedups=lambda **kwargs: calls.append(kwargs))
    engine = engine_with(backend)
    lock = getattr(engine if lock_name == "_model_lock" else backend, lock_name)
    entered, release = threading.Event(), threading.Event()
    def hold():
        with lock:
            entered.set()
            release.wait(2)
    worker = threading.Thread(target=hold)
    worker.start()
    assert entered.wait(1)
    try:
        assert configure_speedups(engine, {"enabled": False})["status"] == "busy"
        assert engine.lossless_speed_options["enabled"] is True
        assert not calls
    finally:
        release.set()
        worker.join(2)
    assert configure_speedups(engine, {"enabled": False})["status"] == "configured"


def test_explicit_native_branch_count_updates_without_reloading_or_entropy_probe():
    backend = SimpleNamespace(model=object(), runtime="prism_speculative",
                              get_speculative_telemetry=lambda: {"mode": "none"},
                              calculate_token_entropy=lambda prompt: pytest.fail("No implicit entropy probe"))
    engine = engine_with(backend, "gguf")
    assert configure_speedups(engine, {"native_branch_count": 8})["status"] == "configured"
    assert engine.lossless_speed_options["native_branch_count"] == 8


def test_preference_save_preserves_other_sections_and_corruption_keeps_defaults(tmp_path):
    path = tmp_path / "preferences.json"
    original = {"memory_limit": {"enabled": True, "limit_gb": 12.5},
                "speed_tuning": {"enabled": False}, "custom": [1, 2]}
    path.write_text(json.dumps(original))
    save_preferences(path, {"tokenizer_memo": False, "speculative_mode": "lookup"})
    written = json.loads(path.read_text())
    assert all(written[key] == value for key, value in original.items())
    assert not load_preferences(path)["tokenizer_memo"]
    assert load_preferences(path)["speculative_mode"] == "lookup"
    assert not list(tmp_path.glob("*.tmp"))
    path.write_text("invalid json")
    fallback = load_preferences(path, defaults={"prompt_reuse": False})
    assert fallback["prompt_reuse"] is False and fallback["speculative_mode"] == "none"


def test_owned_native_unload_cancels_before_waiting_and_retains_failed_owner():
    cancelled = threading.Event()
    class Engine:
        def __init__(self):
            self.settings = SimpleNamespace(use_mock=False)
            self.active_backend = "gguf"
            self.gguf_backend = SimpleNamespace(runtime="prism_speculative", model=object(),
                                                cancel_generation=cancelled.set)
        def load_model(self):
            raise AssertionError("No model may load in this lifecycle test")
        def unload_model(self):
            return {"status": "error", "unloaded": False, "error": "owned process still active"}
    install_speed_control(Engine)
    engine = Engine()
    owner = engine.gguf_backend
    engine._model_lock.acquire()
    results = []
    worker = threading.Thread(target=lambda: results.append(engine.unload_model()))
    worker.start()
    try:
        assert cancelled.wait(1)
        assert not results
        assert engine.gguf_backend is owner
    finally:
        engine._model_lock.release()
        worker.join(2)
    assert results[0]["status"] == "error"
    assert engine.gguf_backend is owner and owner.model is not None
    assert engine.speed_status["status"] == "error"
