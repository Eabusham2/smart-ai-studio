"""Interleaved round-robin LoRA plasticity for large MLX transformer models."""
from __future__ import annotations
import gc
from dataclasses import dataclass
from typing import Any,Dict,Iterator,List,Optional,Tuple
MLX_AVAILABLE=False
try:
 import mlx.core as mx
 import mlx.utils
 from mlx_lm.tuner.lora import LoRALinear
 MLX_AVAILABLE=True
except ImportError:mx=None;LoRALinear=None
@dataclass(frozen=True)
class LoRATarget:layer_index:int;path:str;module:Any
class RoundRobinLoRAController:
 def __init__(self,model,rank=4,alpha=8.0,chunk_size=6):self.model=model;self.rank=max(1,int(rank));self.alpha=float(alpha);self.chunk_size=max(1,int(chunk_size));self.targets=[];self.active_chunk=0
 def _layers(self):
  layers=getattr(self.model,"layers",None)
  if layers is None:layers=getattr(getattr(self.model,"model",None),"layers",None)
  return list(layers or [])
 @staticmethod
 def _get_attr_path(root,dotted):
  parts=dotted.split(".");parent=root
  for part in parts[:-1]:
   if not hasattr(parent,part):return None
   parent=getattr(parent,part)
  attr=parts[-1];return (parent,attr,getattr(parent,attr)) if hasattr(parent,attr) else None
 def attach_all_layers(self):
  if not MLX_AVAILABLE or self.model is None:return []
  try:self.model.freeze()
  except Exception:pass
  paths=("self_attn.q_proj","self_attn.v_proj","mlp.down_proj","linear_attn.out_proj");attached=[];self.targets=[]
  for idx,layer in enumerate(self._layers()):
   for rel in paths:
    found=self._get_attr_path(layer,rel)
    if found is None:continue
    parent,attr,module=found
    if LoRALinear is not None and not isinstance(module,LoRALinear):
     try:module=LoRALinear.from_base(module,r=self.rank,scale=self.alpha/self.rank);setattr(parent,attr,module)
     except Exception:continue
    if LoRALinear is not None and isinstance(module,LoRALinear):
     try:module.freeze()
     except Exception:pass
     path=f"layers.{idx}.{rel}";self.targets.append(LoRATarget(idx,path,module));attached.append(path)
  if self.targets:self.activate_chunk(0)
  return attached
 @property
 def layer_count(self):return len({t.layer_index for t in self.targets})
 @property
 def chunk_count(self):
  layers=self._layers();return (len(layers)+self.chunk_size-1)//self.chunk_size if layers else 0
 def iter_chunks(self):yield from range(self.chunk_count)
 def activate_chunk(self,chunk_index):
  if not self.targets:return []
  idx=int(chunk_index)%max(1,self.chunk_count);start=idx*self.chunk_size;end=start+self.chunk_size;active=[]
  for t in self.targets:
   try:t.module.freeze()
   except Exception:pass
   if start<=t.layer_index<end:
    try:t.module.unfreeze();active.append(t.path)
    except Exception:pass
  self.active_chunk=idx;return active
 def adapter_state(self):
  if not MLX_AVAILABLE:return {}
  state={}
  for t in self.targets:
   try:
    for name,value in mlx.utils.tree_flatten(t.module.parameters()):
     if "lora" in name.lower() or name.lower() in {"a","b"}:state[f"{t.path}::{name}"]=mx.array(value)
   except Exception:pass
  if state:
   try:mx.eval(*state.values())
   except Exception:pass
  return state
 def load_adapter_state(self,state):
  if not MLX_AVAILABLE or not state:return
  by={}
  for key,value in state.items():
   if "::" in key:
    path,name=key.split("::",1);by.setdefault(path,[]).append((name,value))
  targets={t.path:t.module for t in self.targets}
  for path,items in by.items():
   m=targets.get(path)
   if m is not None:
    try:m.update(mlx.utils.tree_unflatten(items))
    except Exception:pass
  try:mx.eval(self.model.parameters())
  except Exception:pass
 def trainable_parameter_count(self):
  if not MLX_AVAILABLE:return 0
  try:return sum(int(getattr(v,"size",0)) for _,v in mlx.utils.tree_flatten(self.model.trainable_parameters()))
  except Exception:return 0
 @staticmethod
 def release_transient_graphs():
  gc.collect()
  if MLX_AVAILABLE:
   try:
    if hasattr(mx,"clear_cache"):mx.clear_cache()
    elif hasattr(mx,"metal") and hasattr(mx.metal,"clear_cache"):mx.metal.clear_cache()
   except Exception:pass
