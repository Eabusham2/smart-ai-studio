"""Live Shannon-entropy routing with stateful Leaky-Integrate-and-Fire gating."""
from __future__ import annotations
import math
from dataclasses import dataclass,field
from typing import Any,List,Sequence,Tuple
MLX_AVAILABLE=False
try:
 import mlx.core as mx
 MLX_AVAILABLE=True
except ImportError:mx=None

def convex_temperature_ladder(num_branches:int,t_min=.20,t_max=.88)->List[float]:
 n=max(1,int(num_branches))
 if n==1:return [0.0]
 if t_min<=0 or t_max<=0:raise ValueError("temperature bounds must be positive")
 ratio=t_max/t_min;return [float(round(t_min*(ratio**(i/(n-1))),4)) for i in range(n)]
def normalized_shannon_from_probabilities(probabilities:Sequence[float])->Tuple[float,float]:
 p=[max(0,float(x)) for x in probabilities];total=sum(p)
 if total<=0:return 0.,0.
 p=[x/total for x in p if x>0];raw=-sum(x*math.log2(max(x,1e-12)) for x in p);den=math.log2(max(2,len(p)));return raw,max(0.,min(1.,raw/den))
def topk_shannon_from_logits(logits:Any,top_k:int=40)->Tuple[float,float]:
 if MLX_AVAILABLE:
  try:
   flat=logits.reshape(-1).astype(mx.float32);k=max(2,min(int(top_k),int(flat.shape[0])));idx=mx.argpartition(-flat,kth=k-1)[:k];probs=mx.softmax(flat[idx],axis=-1);raw_arr=-mx.sum(probs*mx.log2(mx.clip(probs,1e-12,1.)));mx.eval(raw_arr);raw=float(raw_arr.item());return raw,max(0.,min(1.,raw/math.log2(k)))
  except Exception:pass
 try:vals=[float(x) for x in logits]
 except Exception:return 0.,0.
 vals=sorted(vals,reverse=True)[:max(2,min(int(top_k),len(vals)))]
 if not vals:return 0.,0.
 vmax=max(vals);exps=[math.exp(v-vmax) for v in vals];total=sum(exps);return normalized_shannon_from_probabilities([e/total for e in exps])
@dataclass
class LIFNeuronState:
 v_mem:float=0.;v_thresh:float=.55;v_rest:float=0.;beta:float=.85;spike_history:List[int]=field(default_factory=list)
 def step(self,entropy_current):
  x=max(0.,min(1.,float(entropy_current)));self.v_mem=self.beta*self.v_mem+(1-self.beta)*x;spike=int(self.v_mem>=self.v_thresh)
  if spike:self.v_mem=self.v_rest
  self.spike_history.append(spike);return spike,self.v_mem
 def determine_branch_budget(self,entropy,t_min=.20,t_max=.88):
  x=max(0.,min(1.,float(entropy)));spike,_=self.step(x);branches=4 if spike or x>=.65 else 2 if x>=.30 else 1;return branches,convex_temperature_ladder(branches,t_min,t_max),spike
 @property
 def spike_count(self):return int(sum(self.spike_history))
