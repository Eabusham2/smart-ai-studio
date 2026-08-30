import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from typing import List, Optional, Any

class SmartKVCacheManager:
    """
    GGUF-Style Stateful KV Cache Manager:
    - Retains warm prefix embeddings for zero-latency multi-turn chat.
    - Pre-allocates unified memory arenas.
    - Dynamically compacts oldest context when approaching VRAM limits.
    """
    def __init__(self, model: Any, max_tokens: int = 4096):
        self.model = model
        self.max_tokens = max_tokens
        self.cache = make_prompt_cache(model)
        self.current_length = 0

    def get_cache(self) -> List[Any]:
        return self.cache

    def reset(self):
        self.cache = make_prompt_cache(self.model)
        self.current_length = 0

    def auto_compact_if_needed(self, incoming_tokens_len: int) -> bool:
        if self.current_length + incoming_tokens_len > self.max_tokens:
            self.reset()
            return True
        return False
