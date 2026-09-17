"""Lossless compatibility API for the former H2O KV compaction policy.

The historical implementation discarded low-scoring context positions to reduce KV
memory. That is incompatible with the current no-quality-loss invariant. Keep the
same public classes/methods so old imports remain valid, but never remove, reorder,
or approximate context tokens.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence


@dataclass
class H2OAttentionSinkPolicy:
    sink_tokens: int = 4
    heavy_tokens: int = 1024
    recent_tokens: int = 32
    max_budget: int = 2048
    scores: List[float] = field(default_factory=list)

    def register_step_attention(self, weights: Iterable[float]) -> None:
        vals = [float(x) for x in weights]
        if len(self.scores) < len(vals):
            self.scores.extend([0.0] * (len(vals) - len(self.scores)))
        for i, value in enumerate(vals):
            self.scores[i] += max(0.0, value)

    def note_position_scores(self, mapping: Dict[int, float]) -> None:
        if not mapping:
            return
        max_index = max(int(i) for i in mapping)
        if len(self.scores) <= max_index:
            self.scores.extend([0.0] * (max_index + 1 - len(self.scores)))
        for i, value in mapping.items():
            self.scores[int(i)] += max(0.0, float(value))

    def compute_compacted_indices(self, sequence_length: int) -> List[int]:
        """Lossless policy: retain every context position in original order."""
        return list(range(max(0, int(sequence_length))))

    def compact_token_ids(self, token_ids: Sequence[int]) -> List[int]:
        """Lossless policy: return the complete token sequence unchanged."""
        return list(token_ids)

    def reset(self) -> None:
        self.scores.clear()


class H2OKVCacheArena(H2OAttentionSinkPolicy):
    def __init__(self, sink_size=4, heavy_size=64, max_budget=128):
        super().__init__(
            sink_tokens=sink_size,
            heavy_tokens=heavy_size,
            recent_tokens=4,
            max_budget=max_budget,
        )
