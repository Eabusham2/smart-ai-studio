"""Adjust suite ETA for prompt-aware Pro synthesis generation work only.

The base ETA layer estimates branch-generation work. Hierarchical synthesis adds
N-1 real model generations after N branches. This overlay updates only the stored
post-Phase-1 estimates; it does not alter evaluation, RSI, Pro routing, or training.
"""
from __future__ import annotations


def install(phase4_module, cls) -> None:
    if getattr(cls, "_pro_synthesis_eta_adjust_installed", False):
        return

    original = cls._evaluate_all_splits

    def evaluate_all(self, splits, cache, phase, start, total):
        result = original(self, splits, cache, phase, start, total)
        if not str(phase).startswith("Phase 1"):
            return result

        state = getattr(self, "_eta_progress_state", None)
        if not isinstance(state, dict):
            return result

        avg = float(state.get("normal_phase1_avg", 0.0) or 0.0)
        if avg <= 0:
            return result

        # RSI: each round is 4 sampled branches + 3 hierarchical synthesis calls.
        normal_rsi_items = int(state.get("normal_rsi_planned_rounds", 0) or 0) // 2
        deep_rsi_items = int(state.get("deep_rsi_planned_rounds", 0) or 0) // 2
        state["normal_rsi_est"] = normal_rsi_items * avg * 7.0 * 2.0

        deep_avg = float(state.get("deep_baseline_avg", 0.0) or avg)
        state["deep_rsi_est"] = deep_rsi_items * deep_avg * 7.0 * 2.0

        # Phase 4: N branch generations + (N-1) synthesis calls = 2N-1.
        normal_units = 0.0
        for name, items in splits.items():
            misses = sum(
                cache.get(f"Phase 1: Baseline_{item['id']}") is False
                for item in items
            )
            if not misses:
                continue
            try:
                n = 16 if phase4_module._has_answer_blind_verifier(name) else 8
            except Exception:
                n = 8
            normal_units += misses * (2 * n - 1)
        state["normal_phase4_est"] = normal_units * avg

        deep_misses = int(state.get("deep_phase4_planned", 0) or 0)
        state["deep_phase4_est"] = deep_misses * deep_avg * 31.0  # N=16 -> 16 + 15 merges.
        state["phase4_est"] = (
            float(state.get("normal_phase4_est", 0.0))
            + float(state.get("deep_phase4_est", 0.0))
        )

        print(
            "[ETA] Pro synthesis accounted: RSI=7 generation units/round; "
            "Phase-4=N+(N-1) generation units/item.",
            flush=True,
        )
        return result

    cls._evaluate_all_splits = evaluate_all
    cls._pro_synthesis_eta_adjust_installed = True
