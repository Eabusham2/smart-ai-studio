"""Execution-only speedups remain configured when measurement is impossible."""
from types import SimpleNamespace

import pytest

from core.lossless_speed_control import initial_options, load_preferences
from core.speculative_runtime import TARGET_SHA256, MTP_SHA256, DFLASH2_SHA256, plan_speculation
from core.speed_control import install_speed_control


def test_no_headroom_keeps_prompt_speedups_without_running_a_benchmark():
    class Backend:
        model = object()

        def configure_lossless_speedups(self, **options):
            self.options = options
            return {key: value for key, value in options.items() if type(value) is bool}

        def configure_auto_speed(self, **options):
            assert self.options["prompt_reuse"] and self.options["tokenizer_memo"]
            return {"status": "memory_pressure", "message": "Tuning deferred: not enough headroom."}

    class Engine:
        def __init__(self):
            self.settings = SimpleNamespace(use_mock=False)

        def load_model(self):
            self.active_backend = "mlx"
            self.mlx_backend = Backend()
            return {"status": "loaded"}

        def unload_model(self):
            pass

    install_speed_control(Engine)
    engine = Engine()
    result = engine.load_model()
    assert result["status"] == "loaded"
    assert result["speed"]["status"] == "memory_pressure"
    assert set(result["speed"]["configured_speedups"]) == {"prompt_reuse", "branch_prefill", "tokenizer_memo"}
    assert "Speedups enabled without tuning" in result["speed"]["message"]
    assert not result["speedups"]["benchmark_required"]
    assert not engine.mlx_backend.options["metal_warmup"]
    assert not engine.mlx_backend.options["persistent_cache"]


@pytest.mark.parametrize("target,draft,expected", [
    (TARGET_SHA256, None, "lookup"), (MTP_SHA256, None, "mtp"),
    (TARGET_SHA256, DFLASH2_SHA256, "dflash2"),
])
def test_native_auto_selects_installed_compatible_strategy_without_tuning(target, draft, expected):
    # No inference backend, model allocation, or tuner exists in this selection.
    mode = initial_options(SimpleNamespace())["speculative_mode"]
    plan = plan_speculation(mode, target_sha256=target, draft_sha256=draft)
    assert plan.active_mode == expected
    assert plan.max_draft_tokens > 0


def test_explicit_native_off_survives_default_auto(tmp_path):
    path = tmp_path / "preferences.json"
    path.write_text('{"lossless_speedups":{"speculative_mode":"none"}}')
    assert initial_options(SimpleNamespace(speculative_mode="none"))["speculative_mode"] == "none"
    assert load_preferences(path)["speculative_mode"] == "none"
    with pytest.raises(ValueError, match="pinned original"):
        plan_speculation("auto", target_sha256="unverified model")


def test_auto_native_plan_reaches_real_command_without_benchmark(monkeypatch, tmp_path):
    from core.engines import prism_speculative_engine as native

    monkeypatch.setattr(native, "validate_runtime_receipt", lambda *a, **k: {"verified": True})
    monkeypatch.setattr(native, "sha256_file", lambda *a, **k: TARGET_SHA256)
    backend = native.PrismSpeculativeReasoningBackend(
        str(tmp_path / "target.gguf"), str(tmp_path / "verified-server"),
        mode=initial_options(SimpleNamespace())["speculative_mode"])
    assert backend.preflight()["plan"]["active_mode"] == "lookup"
    command = backend._build_command(12345)
    assert command[command.index("--spec-type") + 1] == "ngram-simple"
    assert command[command.index("-m") + 1] == str(tmp_path / "target.gguf")
    assert backend._process is None
    # The native verifier never needs or runs a speculative batch benchmark.
    assert backend.configure_auto_speed()["status"] == "unsupported"
    assert backend._plan.active_mode == "lookup"
