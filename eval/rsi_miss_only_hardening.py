"""Keep RSI strictly limited to Phase-1 benchmark misses.

Historical-good-merge still restores LearningFacts, retention, and real parameter
shift telemetry. Its extra autonomous/environmental RLVR tasks are intentionally
not executed in the master benchmark RSI stage: RSI is Phase-1 misses only.
"""
from __future__ import annotations


def capture_before_historical_merge(p4) -> None:
    """Capture the already-routed miss-only RSI function before historical merge wraps it."""
    if not hasattr(p4, "_phase1_miss_only_rsi"):
        p4._phase1_miss_only_rsi = p4._run_rsi_self_improvement


def enforce_after_historical_merge(p4) -> None:
    """Restore the captured Phase-1-miss-only RSI while preserving other historical layers."""
    base = getattr(p4, "_phase1_miss_only_rsi", None)
    if base is None:
        raise RuntimeError("RSI miss-only base was not captured before historical merge")
    p4._run_rsi_self_improvement = base
    p4._rsi_phase1_miss_only_enforced = True
