"""Regression contracts for memory-safe RSI/Phase-4 branch streaming."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_rsi_keeps_full_16k_generation_ceiling_while_bounding_kv_memory():
    source = _src("eval/rsi_memory_hardening.py")
    assert "RSI_MAX_KV_TOKENS = 16_384" in source
    assert "max_tokens=max(1, int(max_tokens))" in source
    assert "max_kv_size=RSI_MAX_KV_TOKENS" in source
    assert "prefill_step_size=RSI_PREFILL_STEP_SIZE" in source
    assert "kv_bits=RSI_KV_BITS" in source
    assert "RSI_KV_BITS = 4" in source
    assert "quantized_kv_start=0" in source
    assert "min(int(max_tokens)" not in source


def test_old_sequential_cleanup_behavior_is_restored_around_every_branch():
    source = _src("eval/rsi_memory_hardening.py")
    assert "for branch_idx, temp in enumerate(temperatures, 1):" in source
    assert "_clear_branch_memory(mx)" in source
    assert 'hasattr(iterator, "close")' in source
    assert "pieces.clear()" in source
    assert "with metal_lock:" in source


def test_launcher_orders_live_stream_then_memory_safety_before_rsi_wrappers():
    source = _src("master_4000_eval_suite.py")
    live = source.index("install_phase4_stream(phase4_pro_rsi)")
    memory = source.index("install_rsi_memory_hardening(phase4_pro_rsi)")
    capture = source.index("capture_before_historical_merge(phase4_pro_rsi)")
    resume = source.index("install_rsi_resume_hardening(phase4_pro_rsi)")
    assert live < memory < capture < resume
