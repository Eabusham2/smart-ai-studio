from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_app_keeps_turboquant_exception_but_safe_speedups_remain():
    mlx = _src("core/mlx_engine.py")
    kv = _src("core/kv_cache_manager.py")
    assert "make_turboquant_prompt_cache" in kv
    assert "_adaptive_prefill_step_size" in mlx
    assert "prompt_ids = self.tokenizer.encode(prompt)" in mlx
    assert "mlx_lm.stream_generate" in mlx
    assert "if not force and not _memory_pressure()" in mlx


def test_eval_uses_full_precision_kv_not_turboquant():
    paths = [
        "eval/master_4000_runtime.py",
        "eval/_master_4000_base.py",
        "eval/lossless_baseline_speedup.py",
        "eval/phase4_pro_rsi.py",
        "eval/rsi_generation_memory_hardening.py",
        "eval/live_generation_stream.py",
        "run_studio_complete.py",
    ]
    for path in paths:
        src = _src(path)
        assert "make_turboquant_prompt_cache" not in src
        assert "from core.turboquant_cache" not in src

    for path in (
        "eval/master_4000_runtime.py",
        "eval/_master_4000_base.py",
        "eval/phase4_pro_rsi.py",
        "eval/rsi_generation_memory_hardening.py",
        "eval/live_generation_stream.py",
        "run_studio_complete.py",
    ):
        assert "make_prompt_cache" in _src(path)


def test_final_phase1_path_has_same_safe_prompt_prefill_wins_as_app():
    src = _src("eval/lossless_baseline_speedup.py")
    assert "prompt_ids = tok.encode(prompt)" in src
    assert "prompt=prompt_ids" in src
    assert "prefill_step_size=_adaptive_prefill_step_size()" in src
    assert "prompt_cache=prompt_cache" in src
    assert "mlx_lm.stream_generate" in src
    assert "if not force and not _memory_pressure()" in src


def test_phase4_and_rsi_share_app_adaptive_prefill_policy():
    phase4 = _src("eval/phase4_pro_rsi.py")
    rsi = _src("eval/rsi_generation_memory_hardening.py")
    assert "_adaptive_prefill_step_size()" in phase4
    assert "_adaptive_prefill_step_size(aggressive=aggressive)" in rsi


def test_lossy_shortcuts_remain_disabled_outside_app_turboquant_exception():
    settings = _src("config/settings.py")
    speculative = _src("core/speculative_engine.py")
    core_init = _src("core/__init__.py")
    full_context = _src("core/full_context_hardening.py")
    compat = _src("run_studio_complete.py")

    assert 'speculative_mode: str = Field(default="none"' in settings
    assert 'os.getenv("SPECULATIVE_MODE", "none")' in settings
    assert 'self.mode = "none"' in speculative
    assert "return []" in speculative
    assert "root_prefix_cache" not in core_init
    assert "no longer drops legitimate history" in full_context
    # Legacy H2O compatibility object is neutralized: it keeps every token index.
    assert "return list(range(max(0,int(n))))" in compat


def test_gguf_safe_speedup_is_shared_core_behavior():
    gguf = _src("core/engines/gguf_engine.py")
    pro = _src("core/pro_engine.py")
    assert "no_perf=True" in gguf
    assert 'getattr(self.gguf_backend, "model", None) is not None' in pro
    assert 'getattr(self.gguf_backend, "llm", None) is not None' not in pro
