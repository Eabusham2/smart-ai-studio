"""Count opt-in DeepSWE verified RSI traces in the existing Phase-3 fetch budget."""
from __future__ import annotations


def install(phase4_module) -> None:
    original = phase4_module._run_rsi_self_improvement

    def run(self, splits, cache):
        total_seeded = int(original(self, splits, cache) or 0)
        # The wrapped DeepSWE RSI path returns normal + DeepSWE verified traces.
        # Keep the existing Phase-3 memory query budget aligned with that actual total.
        self._phase1_rsi_seed_count = total_seeded
        return total_seeded

    phase4_module._run_rsi_self_improvement = run
