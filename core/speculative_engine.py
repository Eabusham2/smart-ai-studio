"""Lossless speculative decoding façade over the legacy draft engines.

Draft proposals are allowed without a target model; acceptance is never faked.
When a real target verifier is unavailable the proposed draft is rejected cleanly.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from core._speculative_engine_base import *
from core import _speculative_engine_base as _base
from core.drafter import GrammarGuidedASTTrieDrafter


class PromptLookupDrafter(GrammarGuidedASTTrieDrafter):
    """Backward-compatible PLD constructor with grammar guidance."""

    def __init__(
        self,
        min_ngram: int = 3,
        max_ngram: int = 5,
        max_draft_tokens: int = 4,
        **kwargs,
    ):
        self.min_ngram = max(1, int(min_ngram))
        self.max_ngram = max(self.min_ngram, int(max_ngram))
        self.max_draft_tokens = max(1, int(max_draft_tokens))
        super().__init__(n_gram=self.min_ngram, max_draft=self.max_draft_tokens)


class SpeculativeEngine(_base.SpeculativeEngine):
    """Same public engine, but never simulates target-model acceptance."""

    def __init__(self, target_model=None, tokenizer=None, mode="pld", max_draft_tokens=4, **kwargs):
        super().__init__(
            target_model=target_model,
            tokenizer=tokenizer,
            mode=mode,
            max_draft_tokens=max_draft_tokens,
            **kwargs,
        )
        self.pld_drafter = PromptLookupDrafter(
            min_ngram=3,
            max_ngram=5,
            max_draft_tokens=max_draft_tokens,
        )

    def verify_draft_tokens_rejection_sampling(self,context_tokens:List[int],draft_tokens:List[int])->Tuple[List[int],Optional[int]]:
        if not draft_tokens:return [],None
        stats=getattr(self,"stats",None)
        if stats is not None:
            stats.speculative_cycles+=1;stats.draft_tokens_proposed+=len(draft_tokens)
        model=getattr(self,"target_model",None);tok=getattr(self,"tokenizer",None)
        if model is None or tok is None:return [],None

        try:
            import mlx.core as mx
            seq=list(map(int,context_tokens))+list(map(int,draft_tokens))
            out=model(mx.array([seq])); logits=getattr(out,"logits",out)[0]
            accepted=[]
            for i,draft in enumerate(draft_tokens):
                pos=len(context_tokens)-1+i
                pred=mx.argmax(logits[pos]);mx.eval(pred)
                if int(pred.item())!=int(draft):break
                accepted.append(int(draft))
            bonus=None
            bonus_pos=len(context_tokens)-1+len(accepted)
            if bonus_pos<logits.shape[0]:
                b=mx.argmax(logits[bonus_pos]);mx.eval(b);bonus=int(b.item())
            if stats is not None:
                stats.draft_tokens_accepted+=len(accepted);self._update_metrics()
            return accepted,bonus
        except Exception:
            pass

        if hasattr(model,"eval"):
            try:return super().verify_draft_tokens_rejection_sampling(context_tokens,draft_tokens)
            except Exception:pass
        return [],None
