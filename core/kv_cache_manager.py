"""Stateful prompt/KV cache management for Apple Silicon MLX chat runtimes."""
from typing import Any, List, Optional
import psutil

MLX_AVAILABLE = False
try:
    import mlx.core as mx
    from core.turboquant_cache import make_turboquant_prompt_cache as make_prompt_cache
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    make_prompt_cache = None


def compute_auto_kv_budget(total_ram_gb: Optional[float] = None) -> int:
    """Scale when the temporary prompt-cache arena is rebuilt; never truncate context."""
    if total_ram_gb is None:
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    if total_ram_gb <= 8.0:
        return 1024
    if total_ram_gb <= 16.0:
        return 2048
    if total_ram_gb <= 32.0:
        return 4096
    return 8192


class SmartKVCacheManager:
    """Fresh TurboQuant-preferred MLX prompt-cache arena; it never removes prompt tokens."""
    def __init__(self, model: Any, max_tokens: Optional[int] = None):
        if not MLX_AVAILABLE or make_prompt_cache is None:
            raise RuntimeError("SmartKVCacheManager requires MLX/MLX-LM")
        self.model = model
        self.max_tokens = int(max_tokens or compute_auto_kv_budget())
        self.cache = make_prompt_cache(model)
        self.current_length = 0

    def get_cache(self) -> List[Any]:
        return self.cache

    def reset(self, purge_allocator: bool = False) -> None:
        """Rebuild logical KV state; only purge Metal's allocator on real pressure/OOM."""
        self.cache = None
        if purge_allocator:
            try:
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()
            except Exception:
                pass
        self.cache = make_prompt_cache(self.model)
        self.current_length = 0

    def auto_compact_if_needed(self, incoming_tokens_len: int) -> bool:
        # "Compact" here means throw away stale KV tensors and recompute from the
        # complete packed prompt. No token IDs or conversation content are removed.
        if self.current_length + int(incoming_tokens_len) > self.max_tokens:
            self.reset(purge_allocator=False)
            return True
        return False

    def note_tokens(self, token_count: int) -> None:
        self.current_length += max(0, int(token_count))
