"""Speculative acceleration compatibility layer with strict no-quality-loss policy.

The previous implementation called itself lossless but accepted non-argmax draft
symbols above a probability threshold and even fabricated/randomly accepted tokens
when no target verifier was available. That can change model output, so those paths
are deliberately removed.

MLX-LM currently has an upstream speculative-decoding divergence report even for
zero-temperature greedy generation. Until exact output equivalence can be proven on
this model/backend, speculative generation is disabled. The lightweight drafter
classes remain only for API/import compatibility and diagnostics; they never alter
production generation. Normal MLX generation, prompt caching, full-precision KV,
cache cleanup, sequential branches, and prefill chunking remain available elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SpeculativeStats:
    draft_tokens_proposed: int = 0
    draft_tokens_accepted: int = 0
    speculative_cycles: int = 0
    total_tokens_generated: int = 0
    acceptance_rate: float = 0.0
    estimated_speedup: float = 1.0
    vram_overhead_mb: float = 0.0
    mode: str = "NONE"


class PromptLookupDrafter:
    """Harmless n-gram proposal helper; proposals are never auto-accepted."""

    def __init__(self, min_ngram: int = 3, max_ngram: int = 5, max_draft_tokens: int = 4):
        self.min_ngram = int(min_ngram)
        self.max_ngram = int(max_ngram)
        self.max_draft = int(max_draft_tokens)

    def find_draft_tokens(self, token_ids: List[int]) -> List[int]:
        total_len = len(token_ids)
        if total_len < self.min_ngram + 1:
            return []
        for ngram_size in range(min(self.max_ngram, total_len - 1), self.min_ngram - 1, -1):
            target_ngram = token_ids[-ngram_size:]
            for start_idx in range(total_len - ngram_size - 1, -1, -1):
                if token_ids[start_idx:start_idx + ngram_size] == target_ngram:
                    draft_start = start_idx + ngram_size
                    draft_end = min(draft_start + self.max_draft, total_len)
                    if draft_start < draft_end:
                        return list(token_ids[draft_start:draft_end])
        return []


class LookaheadJacobiDrafter:
    """Compatibility-only proposal cache; never participates in generation."""

    def __init__(self, max_draft_tokens: int = 4):
        self.max_draft = int(max_draft_tokens)
        self.ngram_cache: Dict[Tuple[int, ...], List[int]] = {}

    def update_cache(self, token_ids: List[int]):
        for i in range(max(0, len(token_ids) - 3)):
            key = (int(token_ids[i]), int(token_ids[i + 1]))
            self.ngram_cache[key] = list(token_ids[i + 2:i + 2 + self.max_draft])

    def find_draft_tokens(self, token_ids: List[int]) -> List[int]:
        if len(token_ids) < 2:
            return []
        return list(self.ngram_cache.get((int(token_ids[-2]), int(token_ids[-1])), []))[:self.max_draft]


class BlockDrafter:
    """Compatibility-only holder; neural draft heads are disabled in production."""

    def __init__(self, draft_model=None, max_draft_tokens: int = 4):
        self.draft_model = draft_model
        self.max_draft = int(max_draft_tokens)

    def find_draft_tokens(self, token_ids: List[int]) -> List[int]:
        return []


class SpeculativeEngine:
    """Fail-closed shim: never changes target-model generation."""

    def __init__(
        self,
        target_model=None,
        tokenizer=None,
        mode: str = "none",
        max_draft_tokens: int = 4,
        draft_model=None,
    ):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.requested_mode = str(mode or "none").lower()
        self.mode = "none"
        self.max_draft_tokens = int(max_draft_tokens)
        self.draft_model = draft_model
        self.pld_drafter = PromptLookupDrafter(max_draft_tokens=max_draft_tokens)
        self.jacobi_drafter = LookaheadJacobiDrafter(max_draft_tokens=max_draft_tokens)
        self.block_drafter = BlockDrafter(draft_model=draft_model, max_draft_tokens=max_draft_tokens)
        self.stats = SpeculativeStats(mode="NONE")

    def propose_draft_tokens(self, token_ids: List[int]) -> List[int]:
        # Production generation must remain exactly the target decoder's output.
        return []

    def verify_draft_tokens_rejection_sampling(
        self,
        context_tokens: List[int],
        draft_tokens: List[int],
    ) -> Tuple[List[int], Optional[int]]:
        raise RuntimeError(
            "Speculative decoding is disabled: exact output equivalence is not proven "
            "for the active MLX backend/model. Use normal target-model decoding."
        )

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "mode": "DISABLED_LOSSLESS_ONLY",
            "requested_mode": self.requested_mode.upper(),
            "draft_tokens_proposed": 0,
            "draft_tokens_accepted": 0,
            "acceptance_rate_percent": 0.0,
            "estimated_speedup": "1.0x",
            "vram_overhead_mb": 0.0,
            "speculative_cycles": 0,
            "reason": "Exact MLX speculative equivalence is not currently guaranteed",
        }
