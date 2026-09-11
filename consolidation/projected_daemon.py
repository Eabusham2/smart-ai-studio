"""Non-stop projected-gradient continual learning with inference-safe scheduling."""
from __future__ import annotations
import math,threading,time
from typing import Any,Dict,List,Optional,Tuple
MLX_AVAILABLE=False
try:
 import mlx.core as mx
 import mlx.nn as nn
 import mlx.utils
 MLX_AVAILABLE=True
except ImportError:mx=nn=None
class GramSchmidtOGPProjector:
 def __init__(self,tolerance=1e-5,max_basis_vectors=8):self.tolerance=float(tolerance);self.max_basis_vectors=max(1,int(max_basis_vectors));self.anchor_basis_vectors=[]
 @staticmethod
 def flatten_gradients(grads_dict):
  if not MLX_AVAILABLE:return None,[]
  flat=[];shapes=[]
  for key in sorted(grads_dict):
   value=grads_dict[key]
   if hasattr(value,"shape"):shapes.append((key,tuple(value.shape)));flat.append(value.reshape(-1).astype(mx.float32))
  return (mx.concatenate(flat),shapes) if flat else (None,shapes)
 @staticmethod
 def unflatten_gradients(flat_vec,shapes):
  if not MLX_AVAILABLE or flat_vec is None:return {}
  out={};off=0
  for key,shape in shapes:
   n=1
   for d in shape:n*=int(d)
   out[key]=flat_vec[off:off+n].reshape(shape);off+=n
  return out
 def register_anchor_gradient(self,flat_anchor_grad):
  if not MLX_AVAILABLE or flat_anchor_grad is None:return False
  v=flat_anchor_grad.astype(mx.float32)
  for b in self.anchor_basis_vectors:v=v-(mx.sum(v*b)/(mx.sum(b*b)+1e-12))*b
  norm=mx.sqrt(mx.sum(v*v));mx.eval(norm);value=float(norm.item())
  if not math.isfinite(value) or value<=self.tolerance:return False
  b=v/value;mx.eval(b);self.anchor_basis_vectors.append(b)
  if len(self.anchor_basis_vectors)>self.max_basis_vectors:self.anchor_basis_vectors.pop(0)
  return True
 def project_gradient(self,g):
  if not MLX_AVAILABLE or g is None:return g
  out=g.astype(mx.float32)
  for b in self.anchor_basis_vectors:out=out-(mx.sum(out*b)/(mx.sum(b*b)+1e-12))*b
  mx.eval(out);return out
 def verify_orthogonality(self,g):
  if not MLX_AVAILABLE or g is None or not self.anchor_basis_vectors:return 0.0
  vals=[mx.abs(mx.sum(g*b)) for b in self.anchor_basis_vectors];mx.eval(*vals);r=max(float(v.item()) for v in vals);return r if math.isfinite(r) else float("inf")
class ProjectedSleepConsolidationDaemon(threading.Thread):
 def __init__(self,moe_manager,ogp_projector,kg,tokenizer,settings,stream_lock=None,round_robin_controller=None,dual_buffer=None,anchor_texts=None):
  super().__init__(daemon=True,name="OGP-Sleep-Daemon");self.moe_manager=moe_manager;self.ogp_projector=ogp_projector;self.kg=kg;self.tokenizer=tokenizer;self.settings=settings;self.stream_lock=stream_lock or threading.Lock();self.round_robin_controller=round_robin_controller;self.dual_buffer=dual_buffer;self.anchor_texts=list(anchor_texts or []);self.running=True;self.total_consolidations=0;self.last_loss=0.;self.last_ortho_overlap=0.;self.queue_length=0;self.last_error=None;self.last_update_time=None;self.last_poll_time=0.;self.anchor_error=None;self.anchor_count=0;self._anchor_bases_by_chunk={}
 def stop(self):self.running=False
 def status(self):return {"running":self.running,"queue_length":self.queue_length,"total_consolidations":self.total_consolidations,"last_loss":self.last_loss,"orthogonal_overlap":self.last_ortho_overlap,"last_error":self.last_error,"last_update_time":self.last_update_time,"anchor_count":self.anchor_count,"anchor_error":self.anchor_error}
 def _compute_anchor_basis_locked(self,chunk_index=0,domain=None):
  model=getattr(self.moe_manager,"model",None)
  if not MLX_AVAILABLE or model is None or self.tokenizer is None:self.anchor_error="MLX model/tokenizer unavailable";return 0
  if not self.anchor_texts:self.anchor_error="no baseline anchor texts supplied";return 0
  chunk_index=int(chunk_index);domain=str(domain or getattr(self.moe_manager,"active_domain","system"));key=(domain,chunk_index)
  if key in self._anchor_bases_by_chunk:self.ogp_projector.anchor_basis_vectors=list(self._anchor_bases_by_chunk[key]);self.anchor_count=len(self.ogp_projector.anchor_basis_vectors);return self.anchor_count
  rr=self.round_robin_controller
  if rr is not None and getattr(rr,"targets",None):rr.activate_chunk(chunk_index)
  self.ogp_projector.anchor_basis_vectors=[];max_samples=max(1,int(getattr(self.settings,"ogp_anchor_samples",8)))
  for text in self.anchor_texts[:max_samples]:
   ids=self.tokenizer.encode(str(text))[:96]
   if len(ids)<2:continue
   arr=mx.array([ids])
   def loss(m):
    out=m(arr);logits=getattr(out,"logits",out)[:,:-1,:].astype(mx.float32);return mx.mean(nn.losses.cross_entropy(logits,arr[:,1:]))
   _,grads=nn.value_and_grad(model,loss)(model);flat,_=self.ogp_projector.flatten_gradients(dict(mlx.utils.tree_flatten(grads)))
   if flat is not None:self.ogp_projector.register_anchor_gradient(flat)
  self._anchor_bases_by_chunk[key]=[mx.array(v) for v in self.ogp_projector.anchor_basis_vectors]
  if self._anchor_bases_by_chunk[key]:mx.eval(*self._anchor_bases_by_chunk[key])
  self.anchor_count=len(self.ogp_projector.anchor_basis_vectors);self.anchor_error=None if self.anchor_count else "baseline gradients produced no OGP basis";return self.anchor_count
 def initialize_anchor_basis(self,anchor_texts=None,chunk_index=0):
  if anchor_texts is not None:self.anchor_texts=list(anchor_texts);self._anchor_bases_by_chunk.clear()
  acquired=False
  try:
   acquired=self.stream_lock.acquire(blocking=False)
   if not acquired:self.anchor_error="deferred: inference active";return 0
   return self._compute_anchor_basis_locked(chunk_index,getattr(self.moe_manager,"active_domain","system"))
  except Exception as exc:self.anchor_error=f"{type(exc).__name__}: {exc}";return 0
  finally:
   if acquired:
    try:self.stream_lock.release()
    except Exception:pass
 def run(self):
  cadence=max(1.,min(5.,float(getattr(self.settings,"daemon_queue_check_seconds",2.))));timed=max(cadence,float(getattr(self.settings,"polling_interval_seconds",300.)));need=max(1,int(getattr(self.settings,"min_batch_queue_size",5)))
  while self.running:
   time.sleep(cadence)
   try:
    items=self.kg.fetch_unconsolidated_high_surprise(getattr(self.settings,"min_surprise_threshold",.85),32);self.queue_length=len(items);now=time.time();due=(now-self.last_poll_time)>=timed
    if items and (len(items)>=need or due):self._consolidate_batch(items[:need]);self.last_poll_time=now
   except Exception as exc:self.last_error=f"{type(exc).__name__}: {exc}"
 def _consolidate_batch(self,items):
  model=getattr(self.moe_manager,"model",None)
  if not MLX_AVAILABLE or model is None or not items:return False
  acquired=self.stream_lock.acquire(blocking=False)
  if not acquired:return False
  processed=[];buf=None
  try:
   rr=self.round_robin_controller;chunks=list(rr.iter_chunks()) if rr is not None and getattr(rr,"targets",None) else [0]
   for i,item in enumerate(items):
    chunk=chunks[i%len(chunks)] if chunks else 0
    try:domain=self.moe_manager.route_and_activate(item.get("prompt",""))
    except Exception:domain=getattr(self.moe_manager,"active_domain",None)
    if self._compute_anchor_basis_locked(chunk,domain)<=0:raise RuntimeError(self.anchor_error or "OGP anchor basis unavailable")
    try:buf=self.moe_manager.shadow_buffer(domain)
    except Exception:buf=self.dual_buffer
    if buf is None:raise RuntimeError("no shadow adapter buffer available")
    buf.activate_shadow()
    if rr is not None and getattr(rr,"targets",None):rr.activate_chunk(chunk)
    text=f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n{item['completion']}<|im_end|>";ids=self.tokenizer.encode(text)[:96]
    if len(ids)<2:buf.rollback_active();continue
    arr=mx.array([ids])
    def loss_fn(m):
     out=m(arr);logits=getattr(out,"logits",out)[:,:-1,:].astype(mx.float32);return mx.mean(nn.losses.cross_entropy(logits,arr[:,1:]))
    loss,grads=nn.value_and_grad(model,loss_fn)(model);flat,shapes=self.ogp_projector.flatten_gradients(dict(mlx.utils.tree_flatten(grads)))
    if flat is None:buf.rollback_active();continue
    var=mx.var(flat);energy=mx.mean(flat*flat);mx.eval(var,energy);lr=float(getattr(self.settings,"base_learning_rate",1e-4))/math.sqrt(1+float(var.item()));proj=self.ogp_projector.project_gradient(flat);overlap=self.ogp_projector.verify_orthogonality(proj)
    if overlap>max(self.ogp_projector.tolerance*10,1e-4):buf.rollback_active();raise RuntimeError(f"OGP orthogonality check failed: {overlap:.3e}")
    changed=buf.apply_shadow_adamw(self.ogp_projector.unflatten_gradients(proj,shapes),learning_rate=lr)
    if not changed:buf.rollback_active();raise RuntimeError("shadow adapter update matched zero trainable tensors")
    buf.atomic_commit_shadow();self.moe_manager.note_fisher_score(domain,float(energy.item()));self.moe_manager.capture_active();self.last_ortho_overlap=overlap;self.last_loss=float(loss.item());processed.append(item.get("id"));self.total_consolidations+=1
   try:self.moe_manager.distill_from_fisher();self.moe_manager.save_master()
   except Exception:pass
   if processed:self.kg.mark_consolidated([x for x in processed if x is not None])
   self.last_update_time=time.time();self.last_error=None;return bool(processed)
  except Exception as exc:
   if buf is not None:
    try:buf.rollback_active()
    except Exception:pass
   self.last_error=f"{type(exc).__name__}: {exc}";return False
  finally:
   try:self.stream_lock.release()
   except Exception:pass
