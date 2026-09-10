import inspect

from core.kv_cache_manager import compute_auto_kv_budget
from core.mlx_engine import MLXReasoningBackend


def test_auto_kv_budget_tiers():
    assert compute_auto_kv_budget(8.0) == 1024
    assert compute_auto_kv_budget(16.0) == 2048
    assert compute_auto_kv_budget(32.0) == 4096
    assert compute_auto_kv_budget(64.0) == 8192


def test_mlx_backend_exposes_smartkv_policy_without_loading_weights():
    backend = MLXReasoningBackend()
    assert backend.kv_cache_manager is None
    assert backend.max_stateful_kv_tokens == compute_auto_kv_budget()
    src = inspect.getsource(MLXReasoningBackend.stream_generate_tokens)
    assert "kv_cache_manager" in src
    assert "prompt_cache=" in src
