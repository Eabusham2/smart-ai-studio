"""Bounded Heavy-Hitter / attention-sink retention policy for long KV histories."""
from __future__ import annotations
from dataclasses import dataclass,field
from typing import Dict,Iterable,List,Sequence

@dataclass
class H2OAttentionSinkPolicy:
    sink_tokens:int=4;heavy_tokens:int=1024;recent_tokens:int=32;max_budget:int=2048;scores:List[float]=field(default_factory=list)
    def register_step_attention(self,weights:Iterable[float])->None:
        vals=[float(x) for x in weights]
        if len(self.scores)<len(vals):self.scores.extend([0.0]*(len(vals)-len(self.scores)))
        for i,v in enumerate(vals):self.scores[i]+=max(0.0,v)
    def note_position_scores(self,mapping:Dict[int,float])->None:
        if not mapping:return
        mx=max(int(i) for i in mapping)
        if len(self.scores)<=mx:self.scores.extend([0.0]*(mx+1-len(self.scores)))
        for i,v in mapping.items():self.scores[int(i)]+=max(0.0,float(v))
    def compute_compacted_indices(self,sequence_length:int)->List[int]:
        n=max(0,int(sequence_length));budget=max(1,int(self.max_budget))
        if n<=budget:return list(range(n))
        sink_end=min(n,max(0,int(self.sink_tokens)));recent_start=max(sink_end,n-max(0,int(self.recent_tokens)));keep=set(range(sink_end));keep.update(range(recent_start,n));c=[]
        for i in range(sink_end,recent_start):c.append((self.scores[i] if i<len(self.scores) else 0.0,i))
        c.sort(key=lambda x:(x[0],x[1]),reverse=True);room=max(0,budget-len(keep));keep.update(i for _,i in c[:min(max(0,int(self.heavy_tokens)),room)])
        if len(keep)<budget:
            for i in range(recent_start-1,sink_end-1,-1):
                keep.add(i)
                if len(keep)>=budget:break
        return sorted(keep)
    def compact_token_ids(self,token_ids:Sequence[int])->List[int]:
        ids=list(token_ids);return [ids[i] for i in self.compute_compacted_indices(len(ids))]
    def reset(self):self.scores.clear()
class H2OKVCacheArena(H2OAttentionSinkPolicy):
    def __init__(self,sink_size=4,heavy_size=64,max_budget=128):super().__init__(sink_tokens=sink_size,heavy_tokens=heavy_size,recent_tokens=4,max_budget=max_budget)
