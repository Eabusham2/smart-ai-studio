"""Hierarchical domain LoRA banks with active/shadow buffers and Fisher-weighted distillation."""
from __future__ import annotations
import json,threading
from pathlib import Path
from typing import Any,Dict,Optional
from consolidation.dual_buffer import DualAdapterBufferManager,_clone_state
try:
 import mlx.core as mx
 MLX_AVAILABLE=True
except ImportError:mx=None;MLX_AVAILABLE=False
class DomainRouter:
 DOMAINS=("code","math","lore","system")
 def route(self,text):
  t=(text or "").lower()
  if any(x in t for x in ("def ","python","code","git","patch","function","class ","swe")):return "code"
  if any(x in t for x in ("math","aime","equation","calculate","solve","\\boxed","theorem","matrix")):return "math"
  if any(x in t for x in ("remember","recall","history","fact","dialogue","session","lore")):return "lore"
  return "system"
class HierarchicalMoELoRAManager:
 def __init__(self,model,controller,bank_dir="adapter_banks"):
  self.model=model;self.controller=controller;self.bank_dir=Path(bank_dir);self.bank_dir.mkdir(parents=True,exist_ok=True);self.router=DomainRouter();self.lock=threading.RLock();self.active_domain="system";initial=controller.adapter_state() if controller is not None else {};self.experts={d:_clone_state(initial) for d in self.router.DOMAINS};self.buffers={d:DualAdapterBufferManager(model,controller,self.lock) for d in self.router.DOMAINS};self.fisher_scores={d:0.0 for d in self.router.DOMAINS};self.master_adapter=_clone_state(initial)
 def route_and_activate(self,prompt):d=self.router.route(prompt);self.activate(d);return d
 def activate(self,domain):
  domain=domain if domain in self.experts else "system"
  with self.lock:
   current=self.controller.adapter_state();old=self.active_domain;self.experts[old]=_clone_state(current);self.buffers[old].adapters_active=_clone_state(current);target=self.buffers[domain];target.activate_active();self.experts[domain]=_clone_state(target.adapters_active);self.active_domain=domain
 def capture_active(self):
  with self.lock:
   cur=_clone_state(self.controller.adapter_state());self.experts[self.active_domain]=cur;self.buffers[self.active_domain].adapters_active=_clone_state(cur)
 def shadow_buffer(self,domain=None):
  d=domain or self.active_domain;return self.buffers[d if d in self.buffers else "system"]
 def note_fisher_score(self,domain,score):
  d=domain if domain in self.experts else self.active_domain;x=max(0.0,float(score));old=float(self.fisher_scores.get(d,0));self.fisher_scores[d]=x if old<=0 else .8*old+.2*x
 def distill(self,weights=None):
  weights=weights or self.fisher_scores
  if sum(max(0,float(weights.get(d,0))) for d in self.experts)<=0:weights={d:1.0 for d in self.experts}
  denom=sum(max(0,float(weights.get(d,0))) for d in self.experts) or 1;keys=set().union(*(s.keys() for s in self.experts.values()));merged={}
  for key in keys:
   vals=[(max(0,float(weights.get(d,0)))/denom,s[key]) for d,s in self.experts.items() if key in s]
   if not vals:continue
   if MLX_AVAILABLE and hasattr(vals[0][1],"shape"):
    acc=mx.zeros_like(vals[0][1]).astype(mx.float32)
    for w,v in vals:acc=acc+w*v.astype(mx.float32)
    mx.eval(acc);merged[key]=acc
   else:
    try:merged[key]=sum(w*v for w,v in vals)
    except Exception:merged[key]=vals[0][1]
  self.master_adapter=_clone_state(merged);return merged
 def distill_from_fisher(self):return self.distill(self.fisher_scores)
 def save_state(self,name,state):
  path=self.bank_dir/f"{name}.safetensors"
  if MLX_AVAILABLE and state:
   try:mx.save_safetensors(str(path),state);return str(path)
   except Exception:pass
  meta=self.bank_dir/f"{name}.json";meta.write_text(json.dumps({"name":name,"tensor_keys":sorted(state),"fisher_scores":self.fisher_scores},indent=2),encoding="utf-8");return str(meta)
 def save_expert(self,domain):
  d=domain if domain in self.experts else self.active_domain;return self.save_state(d,self.experts[d])
 def save_master(self):return self.save_state("master",self.master_adapter)
