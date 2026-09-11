"""Merged MLX backend: rich app behavior + SmartKV + honest Metal telemetry."""
from __future__ import annotations
import logging, threading, time, psutil
import core._mlx_engine_base as _base
from core._mlx_engine_base import *
from core.kv_cache_manager import SmartKVCacheManager, compute_auto_kv_budget
from core.lif_gating import topk_shannon_from_logits

_base.logger=logging.getLogger(__name__)
METAL_STREAM_LOCK=threading.RLock()

class MLXReasoningBackend(_base.MLXReasoningBackend):
    def __init__(self,model_path="orcarouter/Qwen3.8-27B-Uncensored-MLX",adapter_path=None):
        super().__init__(model_path=model_path,adapter_path=adapter_path)
        self.kv_cache_manager=None;self.max_stateful_kv_tokens=compute_auto_kv_budget();self.stream_lock=METAL_STREAM_LOCK
        self.last_tok_per_sec=0.0;self.last_entropy_bits=0.0;self.last_entropy_normalized=0.0;self._generation_count=0

    def load_model(self)->bool:
        ok=super().load_model()
        if ok and self.model is not None:
            self.kv_cache_manager=SmartKVCacheManager(self.model,max_tokens=self.max_stateful_kv_tokens)
        return ok

    def calculate_token_entropy(self,prompt:str)->float:
        if not self.is_mlx_available or self.model is None or self.tokenizer is None:return 0.45
        try:
            import mlx.core as mx
            ids=self.tokenizer.encode(prompt)
            if not ids:return 0.0
            with self.stream_lock:
                result=self.model(mx.array([ids[-min(len(ids),512):]]))
                logits=getattr(result,"logits",result)
                raw,norm=topk_shannon_from_logits(logits[0,-1],40)
            self.last_entropy_bits=raw;self.last_entropy_normalized=norm;return norm
        except Exception:return 0.45

    def stream_generate_tokens(self,prompt:str,max_tokens:int=512,temperature:float=.75,top_p:float=.92):
        if not self.is_mlx_available or self.model is None:return
        import mlx.core as mx, mlx_lm
        prompt_len=0
        try:prompt_len=len(self.tokenizer.encode(prompt))
        except Exception:pass
        if self.kv_cache_manager is None:self.kv_cache_manager=SmartKVCacheManager(self.model,max_tokens=self.max_stateful_kv_tokens)
        else:self.kv_cache_manager.reset()
        self.kv_cache_manager.auto_compact_if_needed(prompt_len)
        sampler=None
        try:
            from mlx_lm.sample_utils import make_sampler
            sampler=make_sampler(temp=temperature,top_p=top_p)
        except Exception:pass
        kwargs={"max_tokens":min(int(max_tokens),4096)}
        if sampler is not None:kwargs["sampler"]=sampler
        else:kwargs.update(temp=temperature,top_p=top_p)
        generated=0;t0=time.perf_counter()
        try:
            with self.stream_lock:
                try:iterator=mlx_lm.stream_generate(self.model,self.tokenizer,prompt=prompt,prompt_cache=self.kv_cache_manager.get_cache(),**kwargs)
                except TypeError:iterator=mlx_lm.stream_generate(self.model,self.tokenizer,prompt=prompt,**kwargs)
                for response in iterator:
                    generated+=1
                    if hasattr(response,"text"):yield response.text
                    elif isinstance(response,str):yield response
                    elif hasattr(response,"token"):yield self.tokenizer.decode([response.token])
            self.last_tok_per_sec=generated/max(.001,time.perf_counter()-t0)
        except Exception as exc:
            _base.logger.error("MLX stream_generate error: %s",exc);self.last_tok_per_sec=0.0
            try:
                with self.stream_lock:ans=mlx_lm.generate(self.model,self.tokenizer,prompt=prompt,max_tokens=min(max_tokens,512),verbose=False)
                if ans:yield ans
            except Exception:return
        finally:
            self.kv_cache_manager.current_length=min(self.max_stateful_kv_tokens,max(0,prompt_len+generated))
            self._generation_count += 1
            try:
                low_ram=psutil.virtual_memory().available/(1024**3)<2.0
                if low_ram or self._generation_count % 8 == 0:
                    if hasattr(mx,"clear_cache"):mx.clear_cache()
                    elif hasattr(mx,"metal") and hasattr(mx.metal,"clear_cache"):mx.metal.clear_cache()
            except Exception:pass


def _native_generate_branches(self,prompt:str,branch_count:int=16,max_tokens:int=512,temperature=.75,top_p:float=.92):
    """Native MLX-LM branch loop: no Python token streaming and no per-branch cache flush."""
    if not self.is_mlx_available or self.model is None or self.tokenizer is None:return []
    import mlx_lm
    try:
        from mlx_lm.sample_utils import make_sampler
    except Exception:
        make_sampler=None
    count=max(1,min(int(branch_count),16)); branches=[];generated_tokens=0;t0=time.perf_counter()
    with self.stream_lock:
        for idx in range(count):
            t=float(temperature[idx % len(temperature)]) if isinstance(temperature,(list,tuple)) else float(temperature)
            kwargs={"max_tokens":min(int(max_tokens),4096),"verbose":False}
            if make_sampler is not None:
                try:kwargs["sampler"]=make_sampler(temp=t,top_p=top_p)
                except Exception:kwargs.update(temp=t,top_p=top_p)
            else:kwargs.update(temp=t,top_p=top_p)
            try:out=mlx_lm.generate(self.model,self.tokenizer,prompt=prompt,**kwargs)
            except TypeError:
                kwargs.pop("top_p",None);out=mlx_lm.generate(self.model,self.tokenizer,prompt=prompt,**kwargs)
            branches.append(out)
            try:generated_tokens+=len(self.tokenizer.encode(out))
            except Exception:pass
    self.last_tok_per_sec=generated_tokens/max(.001,time.perf_counter()-t0) if generated_tokens else 0.0
    self._generation_count+=1
    try:
        import mlx.core as mx
        low_ram=psutil.virtual_memory().available/(1024**3)<2.0
        if low_ram or self._generation_count%8==0:
            if hasattr(mx,"clear_cache"):mx.clear_cache()
            elif hasattr(mx,"metal") and hasattr(mx.metal,"clear_cache"):mx.metal.clear_cache()
    except Exception:pass
    return branches

MLXReasoningBackend.generate_branches=_native_generate_branches
