"""Safe immutable-root prompt cache reuse for repeated benchmark prompt prefixes."""
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Any,Dict,Optional,Sequence,Tuple
try:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    MLX_AVAILABLE=True
except ImportError:
    mx=None;make_prompt_cache=None;MLX_AVAILABLE=False
@dataclass
class RootCacheEntry:
    prefix_text:str;prefix_tokens:Tuple[int,...];cache:Any
class ImmutableRootPrefixCache:
    def __init__(self,model,tokenizer):self.model=model;self.tokenizer=tokenizer;self.entries={};self.hits=0;self.misses=0
    def register(self,name,prefix_text):
        if not MLX_AVAILABLE or not prefix_text:return False
        toks=tuple(int(x) for x in self.tokenizer.encode(prefix_text))
        if not toks:return False
        cache=make_prompt_cache(self.model);logits=self.model(mx.array([list(toks)]),cache=cache);mx.eval(logits);self.entries[name]=RootCacheEntry(prefix_text,toks,cache);return True
    def clone_if_prefix_matches(self,prompt_tokens:Sequence[int]):
        seq=tuple(int(x) for x in prompt_tokens);best=None
        for e in self.entries.values():
            if len(e.prefix_tokens)<=len(seq) and seq[:len(e.prefix_tokens)]==e.prefix_tokens and (best is None or len(e.prefix_tokens)>len(best.prefix_tokens)):best=e
        if best is None:self.misses+=1;return make_prompt_cache(self.model),0
        try:cloned=copy.deepcopy(best.cache)
        except Exception:self.misses+=1;return make_prompt_cache(self.model),0
        self.hits+=1;return cloned,len(best.prefix_tokens)
    def telemetry(self):
        total=self.hits+self.misses;return {"root_prefix_hits":self.hits,"root_prefix_misses":self.misses,"root_prefix_hit_rate":100*self.hits/max(1,total)}
