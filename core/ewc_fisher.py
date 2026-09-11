"""Empirical Fisher diagonal and Fisher-weighted EWC for MLX LoRA parameters."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, Optional

MLX_AVAILABLE=False
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.utils
    MLX_AVAILABLE=True
except ImportError:
    mx=nn=None

class EmpiricalFisherEWC:
    def __init__(self,ewc_lambda:float=400.0):
        self.ewc_lambda=float(ewc_lambda);self.anchor={};self.fisher={};self.samples=0
    @staticmethod
    def _flat_trainable(model):
        return dict(mlx.utils.tree_flatten(model.trainable_parameters())) if MLX_AVAILABLE else {}
    def capture_anchor(self,model):
        if not MLX_AVAILABLE:return {}
        params=self._flat_trainable(model);self.anchor={k:mx.array(v).astype(mx.float32) for k,v in params.items()}
        if self.anchor:mx.eval(*self.anchor.values())
        self.fisher={k:mx.zeros_like(v).astype(mx.float32) for k,v in self.anchor.items()};self.samples=0;return self.anchor
    def accumulate_gradients(self,grads):
        if not MLX_AVAILABLE:return 0
        flat=dict(mlx.utils.tree_flatten(grads));changed=0
        for k,g in flat.items():
            if k in self.fisher:
                gg=g.astype(mx.float32);self.fisher[k]=self.fisher[k]+gg*gg;changed+=1
        if changed:self.samples+=1
        return changed
    def finalize(self):
        if MLX_AVAILABLE and self.samples>0:
            n=float(self.samples)
            for k in self.fisher:self.fisher[k]=self.fisher[k]/n
            if self.fisher:mx.eval(*self.fisher.values())
        return self.fisher
    def compute_from_batches(self,model,batches:Iterable[Any],loss_fn:Callable,max_batches:int=4):
        if not MLX_AVAILABLE:return {}
        if not self.anchor:self.capture_anchor(model)
        grad_fn=nn.value_and_grad(model,loss_fn)
        for idx,batch in enumerate(batches):
            if idx>=max(1,int(max_batches)):break
            _,grads=grad_fn(model,batch);self.accumulate_gradients(grads)
        return self.finalize()
    def penalty(self,model,ewc_lambda:Optional[float]=None):
        if not MLX_AVAILABLE:return 0.0
        lam=float(self.ewc_lambda if ewc_lambda is None else ewc_lambda);params=self._flat_trainable(model);total=mx.array(0.0,dtype=mx.float32)
        for k,w in params.items():
            if k in self.anchor and k in self.fisher:
                d=w.astype(mx.float32)-self.anchor[k];total=total+mx.sum(self.fisher[k]*d*d)
        return 0.5*lam*total
    def parameter_shift_norm(self,model)->float:
        if not MLX_AVAILABLE or not self.anchor:return 0.0
        params=self._flat_trainable(model);total=mx.array(0.0,dtype=mx.float32)
        for k,w in params.items():
            if k in self.anchor:
                d=w.astype(mx.float32)-self.anchor[k];total=total+mx.sum(d*d)
        norm=mx.sqrt(total);mx.eval(norm);return float(norm.item())
    def stats(self):
        return {"ewc_lambda":self.ewc_lambda,"samples":float(self.samples),"tracked_parameters":float(len(self.fisher))}
