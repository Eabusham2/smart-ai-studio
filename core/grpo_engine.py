"""Micro-batch Group Relative Policy Optimization for verifier-scored rollouts."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

MLX_AVAILABLE=False
try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE=True
except ImportError:
    mx=nn=None

@dataclass
class RolloutRecord:
    prompt:str;completion:str;reward:float;temperature:float;advantage:float=0.0;old_logprob:Optional[float]=None

def group_relative_advantages(rewards:Sequence[float],eps:float=1e-8)->List[float]:
    vals=[float(r) for r in rewards]
    if not vals:return []
    mean=sum(vals)/len(vals)
    if len(vals)==1:return [0.0]
    var=sum((r-mean)**2 for r in vals)/len(vals);std=math.sqrt(var)
    return [0.0 for _ in vals] if std<eps else [(r-mean)/(std+eps) for r in vals]

def clipped_surrogate_scalar(new_logprob:float,old_logprob:float,advantage:float,clip_eps:float=.2)->float:
    ratio=math.exp(float(new_logprob)-float(old_logprob));clipped=min(1+clip_eps,max(1-clip_eps,ratio));return -min(ratio*float(advantage),clipped*float(advantage))

class GroupRelativePolicyOptimizer:
    def __init__(self,group_size:int=4,clip_eps:float=.2):self.group_size=max(2,int(group_size));self.clip_eps=float(clip_eps)
    def compute_group_advantages(self,rewards):return group_relative_advantages(rewards)
    def collect_rollouts(self,prompt:str,generate_fn:Callable[[str,float],str],reward_fn:Callable[[str],float],temperatures:Optional[Sequence[float]]=None):
        temps=list(temperatures or (.20,.40,.65,.88)) or [.20];records=[]
        for i in range(self.group_size):
            t=float(temps[i%len(temps)]);completion=generate_fn(prompt,t);records.append(RolloutRecord(prompt,completion,float(reward_fn(completion)),t))
        for r,a in zip(records,self.compute_group_advantages([x.reward for x in records])):r.advantage=float(a)
        return records
    @staticmethod
    def _model_logits(model,token_ids):
        out=model(mx.array([list(map(int,token_ids))]));return getattr(out,"logits",out)
    @staticmethod
    def mean_completion_logprob(model,prompt_ids,completion_ids):
        if not MLX_AVAILABLE:raise RuntimeError("MLX required")
        p=list(map(int,prompt_ids));c=list(map(int,completion_ids))
        if not p or not c:return mx.array(0.0,dtype=mx.float32)
        logits=GroupRelativePolicyOptimizer._model_logits(model,p+c).astype(mx.float32);start=len(p)-1;pred=logits[:,start:start+len(c),:];targets=mx.array([c]);lp=mx.log_softmax(pred,axis=-1);return mx.mean(mx.take_along_axis(lp,targets[...,None],axis=-1).squeeze(-1))
    def attach_old_logprobs(self,model,tokenizer,records,max_completion_tokens:int=64):
        if not MLX_AVAILABLE:return
        for r in records:
            p=tokenizer.encode(r.prompt);c=tokenizer.encode(r.completion)[:max(1,int(max_completion_tokens))]
            if not p or not c:r.old_logprob=0.0;continue
            v=self.mean_completion_logprob(model,p,c);mx.eval(v);r.old_logprob=float(v.item())
    def loss_and_grads(self,model,tokenizer,records,max_completion_tokens:int=64)->Tuple[Any,Any]:
        if not MLX_AVAILABLE:raise RuntimeError("MLX required")
        prepared=[]
        for r in records:
            p=tokenizer.encode(r.prompt);c=tokenizer.encode(r.completion)[:max(1,int(max_completion_tokens))]
            if p and c:prepared.append((p,c,float(r.old_logprob or 0.0),float(r.advantage)))
        if not prepared:raise ValueError("no valid GRPO rollout tokens")
        eps=self.clip_eps
        def objective(active_model):
            terms=[]
            for p,c,old,adv in prepared:
                new=self.mean_completion_logprob(active_model,p,c);ratio=mx.exp(new-old);clip=mx.clip(ratio,1-eps,1+eps);a=mx.array(adv,dtype=mx.float32);terms.append(-mx.minimum(ratio*a,clip*a))
            return mx.mean(mx.stack(terms))
        return nn.value_and_grad(model,objective)(model)
