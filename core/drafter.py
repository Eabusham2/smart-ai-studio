"""Speculative drafting: prompt n-grams plus a grammar-guided AST/token trie."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any,Iterable,List,Optional,Sequence
DEFAULT_SCAFFOLDS=("def ","\n    return ","\n    if ","\n    for ","\n    while ","\n    else:","\n    elif ","\n    try:","\n    except ","assert ","```python\n","\\boxed{"," = "," == "," in ")
@dataclass
class DraftTelemetry:
 attempts:int=0;hits:int=0;proposed_tokens:int=0;accepted_tokens:int=0
 @property
 def hit_rate(self):return 100*self.hits/max(1,self.attempts)
 @property
 def acceptance_rate(self):return 100*self.accepted_tokens/max(1,self.proposed_tokens)
class GrammarGuidedASTTrieDrafter:
 def __init__(self,n_gram=3,max_draft=3,scaffolds:Optional[Iterable[str]]=None,tokenizer=None):self.n_gram=max(1,int(n_gram));self.max_draft=max(1,int(max_draft));self.scaffolds=tuple(scaffolds or DEFAULT_SCAFFOLDS);self.tokenizer=None;self._sequences=[];self.telemetry=DraftTelemetry();self.bind_tokenizer(tokenizer) if tokenizer is not None else None
 def bind_tokenizer(self,tokenizer):
  self.tokenizer=tokenizer;seqs=[]
  for text in self.scaffolds:
   try:
    try:ids=tokenizer.encode(text,add_special_tokens=False)
    except TypeError:ids=tokenizer.encode(text)
    ids=[int(x) for x in ids]
    if ids:seqs.append(ids)
   except Exception:pass
  seqs.sort(key=len,reverse=True);self._sequences=seqs
 def _ngram_draft(self,history:Sequence[int]):
  n=self.n_gram
  if len(history)<n*2:return []
  target=list(history[-n:]);current_start=len(history)-n
  for idx in range(current_start-1,-1,-1):
   if list(history[idx:idx+n])==target:
    start=idx+n;end=min(start+self.max_draft,current_start);draft=list(history[start:end])
    if draft:return [int(x) for x in draft]
  return []
 def _grammar_draft(self,history:Sequence[int]):
  hist=list(history)
  for seq in self._sequences:
   for plen in range(min(len(seq)-1,len(hist)),0,-1):
    if hist[-plen:]==seq[:plen]:
     r=seq[plen:plen+self.max_draft]
     if r:return [int(x) for x in r]
  return []
 def find_draft_tokens(self,token_history,tokenizer=None,max_draft=None):
  if tokenizer is not None and tokenizer is not self.tokenizer:self.bind_tokenizer(tokenizer)
  self.telemetry.attempts+=1;old=self.max_draft
  if max_draft is not None:self.max_draft=max(1,int(max_draft))
  try:draft=self._ngram_draft(token_history) or self._grammar_draft(token_history)
  finally:self.max_draft=old
  if draft:self.telemetry.hits+=1;self.telemetry.proposed_tokens+=len(draft)
  return draft
 def note_acceptance(self,accepted,proposed=None):
  self.telemetry.accepted_tokens+=max(0,int(accepted))
  if proposed is not None and proposed>0 and self.telemetry.proposed_tokens<proposed:self.telemetry.proposed_tokens+=int(proposed)
 def get_telemetry(self):return {"attempts":self.telemetry.attempts,"hits":self.telemetry.hits,"hit_rate":self.telemetry.hit_rate,"proposed_tokens":self.telemetry.proposed_tokens,"accepted_tokens":self.telemetry.accepted_tokens,"acceptance_rate":self.telemetry.acceptance_rate}
ASTPrefixTrieDrafter=GrammarGuidedASTTrieDrafter
