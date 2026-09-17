"""Detailed Phase-2 Learn/MCTS progress overlay; no MCTS behavior changes."""
from __future__ import annotations

import time
from datetime import timedelta


def _td(seconds: float) -> str:
    return str(timedelta(seconds=max(0, int(seconds))))


def install(cls) -> None:
    original_run = cls.run_full_suite

    def run(self):
        mcts = getattr(getattr(self, "engine", None), "mcts", None)
        original_search = getattr(mcts, "search_best_invariant", None)
        if not callable(original_search):
            return original_run(self)

        done = 0
        avg = 0.0
        planned = 30

        def search(*args, **kwargs):
            nonlocal done, avg
            t0 = time.perf_counter()
            result = original_search(*args, **kwargs)
            duration = max(0.001, time.perf_counter() - t0)
            state = getattr(self, "_eta_progress_state", None)
            if state is not None and state.get("stage") == "learn":
                done += 1
                avg = duration if avg <= 0 else avg * 0.75 + duration * 0.25
                learn_left = max(0, planned - done) * avg
                state["learn_est"] = learn_left
                remaining = (
                    learn_left
                    + state.get("normal_rsi_est", 0.0)
                    + state.get("deep_rsi_est", 0.0)
                    + state.get("phase3_est", 0.0)
                    + state.get("learning_test_est", 0.0)
                    + state.get("phase4_est", 0.0)
                )
                elapsed = max(0.0, time.perf_counter() - state["run_started"])
                raw_total = 100.0 * elapsed / max(0.001, elapsed + remaining)
                total_pct = max(float(state.get("last_total_pct", 0.0)), raw_total)
                state["last_total_pct"] = min(99.99, total_pct)
                print(
                    f"[ETA] Learn/MCTS {done}/{planned} ({100.0*done/planned:.2f}%) | "
                    f"Learn ETA {_td(learn_left)} | Total {state['last_total_pct']:.2f}% | "
                    f"Total ETA {_td(remaining)}",
                    flush=True,
                )
            return result

        mcts.search_best_invariant = search
        try:
            return original_run(self)
        finally:
            mcts.search_best_invariant = original_search

    cls.run_full_suite = run
