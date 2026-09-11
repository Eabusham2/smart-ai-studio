"""Hardware-scaled MLX KV cache arenas with pressure-aware reuse."""
from __future__ import annotations
from collections import deque
from typing import Any,Deque,List,Optional
import psutil
MLX_AVAILABLE=False
try:
 import mlx.core as mx
 from mlx_lm.models.cache import make_prompt_cache
 MLX_AVAILABLE=True
except ImportError:mx=None;make_prompt_cache=None
def compute_auto_kv_budget(total_ram_gb:Optional[float]=None)->int:
 if total_ram_gb is None:total_ram_gb=psutil.virtual_memory().total/(1024**3)
 if total_ram_gb<=8:return 1024
 if total_ram_gb<=16:return 2048
 if total_ram_gb<=32:return 4096
 return 8192
class PromptCacheArenaPool:
 def __init__(self,model,arena_count=4):
  if not MLX_AVAILABLE:raise RuntimeError("MLX/MLX-LM required")
  self.model=model;self.arena_count=max(1,int(arena_count));self._pool=deque();self.refill()
 def refill(self):
  while len(self._pool)<self.arena_count:self._pool.append(make_prompt_cache(self.model))
 def acquire(self):return self._pool.popleft() if self._pool else make_prompt_cache(self.model)
 def release_empty(self,cache):
  # Never recycle a mutated cache; replenish with a fresh arena instead.
  if len(self._pool)<self.arena_count:self._pool.append(make_prompt_cache(self.model))
 def __len__(self):return len(self._pool)
class SmartKVCacheManager:
 def __init__(self,model,max_tokens:Optional[int]=None,arena_count=4):
  if not MLX_AVAILABLE:raise RuntimeError("SmartKVCacheManager requires MLX/MLX-LM")
  self.model=model;self.max_tokens=int(max_tokens or compute_auto_kv_budget());self.current_length=0;self.arenas=PromptCacheArenaPool(model,arena_count);self.cache=self.arenas.acquire()
 def get_cache(self)->List[Any]:return self.cache
 def reset(self):
  old=self.cache;self.cache=self.arenas.acquire();self.arenas.release_empty(old);self.current_length=0
 def auto_compact_if_needed(self,incoming_tokens_len):
  if self.current_length+int(incoming_tokens_len)>self.max_tokens:self.reset();return True
  return False
 def note_tokens(self,token_count):self.current_length+=max(0,int(token_count))
 def telemetry(self):return {"max_kv_tokens":self.max_tokens,"current_length":self.current_length,"preallocated_empty_arenas":len(self.arenas)}
