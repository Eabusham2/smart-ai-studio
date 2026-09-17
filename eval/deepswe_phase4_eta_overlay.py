"""Include optional DeepSWE work in the normal Phase-4 total ETA display."""
from __future__ import annotations


def install(runtime_module, eta_module) -> None:
    original = runtime_module._fmt_status

    def fmt_status(phase, split_name, overall, total, speed, eta):
        line = original(phase, split_name, overall, total, speed, eta)
        if not str(phase).startswith("Phase 4"):
            return line
        state = getattr(eta_module, "_ACTIVE", None)
        if not state or not eta_module._deep_enabled(state):
            return line

        deep_state = eta_module._deep_state(state)
        baseline = deep_state.get("baseline") or {}
        final = deep_state.get("final") or {}
        misses = [task_id for task_id, passed in baseline.items() if passed is False]
        done = sum(task_id in final for task_id in misses)
        left = max(0, len(misses) - done)
        state["deep_phase4_planned"] = len(misses)
        state["deep_phase4_done"] = max(state.get("deep_phase4_done", 0), done)

        unit = float(state.get("deep_phase4_avg", 0.0))
        if unit <= 0:
            unit = float(state.get("deep_baseline_avg", 0.0)) * 16.0
        if unit <= 0:
            unit = float(state.get("normal_phase1_avg", 0.0)) * 16.0

        deep_remaining = left * unit
        total_eta = eta_module._secs(eta) + deep_remaining
        total_pct = eta_module._smooth_total_pct(state, total_eta)
        # Replace only ETA overlay text; preserve the evaluator's original status line.
        base = line.split(" | Total:", 1)[0]
        return (
            f"{base} | DeepSWE Pro left: {left}/{len(misses)} | "
            f"Total: {total_pct:6.2f}% | Total ETA: {eta_module._td(total_eta)}"
        )

    runtime_module._fmt_status = fmt_status
