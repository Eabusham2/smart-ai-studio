"""Dual LoRA adapter buffers with shadow-only updates and atomic promotion."""
from __future__ import annotations
import threading
from dataclasses import dataclass
from typing import Any,Callable,Dict,Optional
MLX_AVAILABLE=False
try:
 import mlx.core as mx
 MLX_AVAILABLE=True
except ImportError:mx=None

def _clone_state(state):
 out={}
 for k,v in state.items():
  if MLX_AVAILABLE:
   try:out[k]=mx.array(v);continue
   except Exception:pass
  try:out[k]=v.copy()
  except Exception:out[k]=v
 if MLX_AVAILABLE and out:
  try:mx.eval(*[v for v in out.values() if hasattr(v,"shape")])
  except Exception:pass
 return out
@dataclass
class ShadowUpdateStatus:running:bool=False;completed:bool=False;error:Optional[str]=None
class DualAdapterBufferManager:
 def __init__(self,model,controller,lock=None):
  self.model=model;self.controller=controller;self.lock=lock or threading.RLock();initial=controller.adapter_state() if controller is not None else {};self.adapters_active=_clone_state(initial);self.adapters_shadow=_clone_state(initial);self.previous_active=_clone_state(initial);self._m={};self._v={};self._step=0;self.status=ShadowUpdateStatus();self._worker=None
 def refresh_from_model(self):
  with self.lock:
   current=self.controller.adapter_state();self.adapters_active=_clone_state(current);self.adapters_shadow=_clone_state(current);self.previous_active=_clone_state(current);self._m.clear();self._v.clear();self._step=0
 def activate_active(self):
  with self.lock:self.controller.load_adapter_state(self.adapters_active)
 def activate_shadow(self):
  with self.lock:self.controller.load_adapter_state(self.adapters_shadow)
 def capture_shadow_from_model(self):
  with self.lock:self.adapters_shadow=_clone_state(self.controller.adapter_state())
 def rollback_active(self):
  with self.lock:self.controller.load_adapter_state(self.adapters_active)
 def atomic_commit_shadow(self):
  with self.lock:
   self.previous_active=_clone_state(self.adapters_active);self.adapters_active=_clone_state(self.adapters_shadow);self.adapters_shadow=_clone_state(self.adapters_active);self.controller.load_adapter_state(self.adapters_active)
 def rollback_previous(self):
  with self.lock:self.adapters_active=_clone_state(self.previous_active);self.adapters_shadow=_clone_state(self.previous_active);self.controller.load_adapter_state(self.adapters_active)
 def apply_shadow_adamw(self,gradients:Dict[str,Any],learning_rate:float,weight_decay=.01,beta1=.9,beta2=.999,eps=1e-8)->int:
  if not MLX_AVAILABLE:return 0
  def norm(n):return str(n).replace("::",".").replace("model.","")
  grad_by={norm(k):v for k,v in gradients.items()}
  with self.lock:
   self._step+=1;changed=0
   for key in list(self.adapters_shadow):
    nk=norm(key);grad=grad_by.get(nk)
    if grad is None:
     for gk,gv in grad_by.items():
      if gk.endswith(nk) or nk.endswith(gk):grad=gv;break
    if grad is None:continue
    try:
     g=grad.astype(mx.float32);p=self.adapters_shadow[key].astype(mx.float32);m0=self._m.get(key,mx.zeros_like(g));v0=self._v.get(key,mx.zeros_like(g));m=beta1*m0+(1-beta1)*g;v=beta2*v0+(1-beta2)*(g*g);mh=m/(1-beta1**self._step);vh=v/(1-beta2**self._step);updated=p-float(learning_rate)*(mh/(mx.sqrt(vh)+eps)+weight_decay*p);self.adapters_shadow[key]=updated.astype(getattr(self.adapters_shadow[key],"dtype",mx.float32));self._m[key]=m;self._v[key]=v;changed+=1
    except Exception:pass
   if changed:
    vals=[v for v in self.adapters_shadow.values() if hasattr(v,"shape")]
    if vals:mx.eval(*vals)
   return changed
 def start_background_update(self,update_fn:Callable[[Dict[str,Any]],Dict[str,Any]]):
  with self.lock:
   if self._worker and self._worker.is_alive():return False
   snapshot=_clone_state(self.adapters_shadow);self.status=ShadowUpdateStatus(running=True)
  def worker():
   try:
    updated=update_fn(snapshot)
    if not isinstance(updated,dict):raise TypeError("shadow update function must return an adapter-state dict")
    with self.lock:self.adapters_shadow=_clone_state(updated);self.status=ShadowUpdateStatus(completed=True)
   except Exception as exc:
    with self.lock:self.status=ShadowUpdateStatus(error=f"{type(exc).__name__}: {exc}")
  self._worker=threading.Thread(target=worker,name="smartai-shadow-adapter-update",daemon=True);self._worker.start();return True
 def wait_for_shadow_update(self,timeout=None):
  if self._worker is not None:self._worker.join(timeout=timeout)
  return self.status
