"""Preserve the existing Phase-4 Pro item context for real benchmark wrappers.

The real-data adapters wrap _evaluate_single_item after phase4_pro_rsi installs.
This tiny outer wrapper restores the same current-item metadata pro_eval normally
sets, so answer-blind branch verification and Phase-4 metadata work unchanged.
"""
from __future__ import annotations


def install(phase4_module, cls) -> None:
    original_eval = cls._evaluate_single_item

    def evaluate(self, split, item):
        if not item.get("_real_kind") or not str(
            getattr(self, "_current_phase", "")
        ).startswith("Phase 4"):
            return original_eval(self, split, item)

        self._phase4_current_item = item
        self._last_phase4_pro_meta = None
        result = original_eval(self, split, item)
        phase4_module._append_pro_metadata(self)
        return result

    cls._evaluate_single_item = evaluate
