"""Regression contracts for memory-safe RSI/Phase-4 branch streaming."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_rsi_keeps_full_16k_generation_ceiling_while_bounding_kv_memory():
    source = _src("eval/rsi_memory_hardening.py")
    assert "RSI_GENERATION_TOKEN_CEILING = 16_384" in source
    assert "max_tokens=max(1, int(max_tokens))" in source
    assert "max_kv_size=kv_tokens" in source
    assert "prefill_step_size=prefill_step" in source
    assert "kv_bits=RSI_KV_BITS" in source
    assert "RSI_KV_BITS = 4" in source
    assert "quantized_kv_start=0" in source
    assert "KV fallback never lowers the 16,384-token reasoning ceiling" in source


def test_metal_oom_retries_smaller_kv_without_reducing_generation_limit():
    source = _src("eval/rsi_memory_hardening.py")
    for kv in ("16_384", "8_192", "4_096", "1_024"):
        assert kv in source
    assert "_is_metal_oom" in source
    assert "Metal OOM at KV=" in source
    assert "retrying smaller KV with the same generation limit" in source
    assert "RSI_CACHE_LIMIT_BYTES = 512 * 1024 * 1024" in source
    assert "mx.set_cache_limit(RSI_CACHE_LIMIT_BYTES)" in source
    assert "mx.set_cache_limit(int(old_cache_limit))" in source


def test_old_sequential_cleanup_behavior_is_restored_around_every_branch():
    source = _src("eval/rsi_memory_hardening.py")
    assert "for branch_idx, temp in enumerate(temperatures, 1):" in source
    assert "_clear_branch_memory(mx)" in source
    assert 'hasattr(iterator, "close")' in source
    assert "pieces.clear()" in source
    assert "with metal_lock:" in source


def test_launcher_orders_live_memory_feedback_then_miss_only_resume_wrappers():
    source = _src("master_4000_eval_suite.py")
    live = source.index("install_phase4_stream(phase4_pro_rsi)")
    memory = source.index("install_rsi_memory_hardening(phase4_pro_rsi)")
    feedback = source.index("install_rsi_feedback_hardening(phase4_pro_rsi)")
    capture = source.index("capture_before_historical_merge(phase4_pro_rsi)")
    resume = source.index("install_rsi_resume_hardening(phase4_pro_rsi)")
    assert live < memory < feedback < capture < resume
