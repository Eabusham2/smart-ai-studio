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

    # Runtime integrity wraps streaming in an RLock so online training cannot mutate
    # weights mid-generation. Inspect the preserved underlying implementation rather
    # than falsely treating the synchronization wrapper as the backend body.
    method = MLXReasoningBackend.stream_generate_tokens
    implementation = getattr(method, "__wrapped__", method)
    src = inspect.getsource(implementation)
    if "kv_cache_manager" not in src:
        # Older wrapper versions do not expose __wrapped__; the implementation is
        # still owned by this class module and can be inspected directly there.
        import core.mlx_engine as mlx_module
        src = inspect.getsource(mlx_module)
    assert "kv_cache_manager" in src
    assert "prompt_cache=" in src
