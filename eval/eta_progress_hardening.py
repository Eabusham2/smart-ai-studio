"""Stage-aware ETA/total-progress overlay without rewriting the evaluator loops.

Phase-1 keeps the user's requested rough whole-run estimate: current Phase-1 ETA
plus four 30% Phase-1 allowances. Once Phase 1 is complete, the plan is rebuilt
from the measured per-item time and actual miss count. RSI explicitly accounts for
4 branches x 2 rounds; Phase 4 estimates branch-scaled miss work and then naturally
converges to the evaluator's measured rolling ETA as retests run.
"""
from __future__ import annotations

import time
from datetime import timedelta
from typing import Any, Dict

_ACTIVE = None


def _secs(text: str) -> float:
    text = str(text or "").strip()
    if not text or text == "calculating":
        return 0.0
    try:
        days = 0
        if "day" in text:
            left, text = text.split(",", 1)
            days = int(left.split()[0])
            text = text.strip()
        parts = [int(float(x)) for x in text.split(":")]
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h, m, s = 0, *parts
        else:
            return float(parts[0])
        return days * 86400 + h * 3600 + m * 60 + s
    except Exception:
        return 0.0


def _td(seconds: float) -> str:
    return str(timedelta(seconds=max(0, int(seconds))))


def _total_pct(state: Dict[str, Any], remaining: float) -> float:
    elapsed = max(0.0, time.perf_counter() - state["run_started"])
    return max(0.0, min(100.0, 100.0 * elapsed / max(0.001, elapsed + max(0.0, remaining))))


def install(runtime_module, phase4_module, cls) -> None:
    global _ACTIVE

    original_fmt = runtime_module._fmt_status
    original_eval_all = cls._evaluate_all_splits
    original_rsi = phase4_module._run_rsi_self_improvement
    original_rsi_log = phase4_module._append_rsi_log
    original_phase3 = phase4_module._run_phase3_consolidation
    original_learning_test = phase4_module._run_learning_retention_test
    original_run = cls.run_full_suite

    def run(self):
        global _ACTIVE
        state = {
            "run_started": time.perf_counter(),
            "stage": "startup",
            "phase1_total": 0,
            "phase1_end": None,
            "baseline_elapsed": 0.0,
            "avg_item": 0.0,
            "misses": 0,
            "learn_est": 0.0,
            "rsi_est": 0.0,
            "phase3_est": 0.0,
            "learning_test_est": 0.0,
            "phase4_est": 0.0,
            "rsi_planned_rounds": 0,
            "rsi_done_rounds": 0,
            "rsi_round_started": None,
            "rsi_round_durations": [],
        }
        self._eta_progress_state = state
        _ACTIVE = state
        try:
            return original_run(self)
        finally:
            _ACTIVE = None

    def fmt_status(phase, split_name, overall, total, speed, eta):
        base = original_fmt(phase, split_name, overall, total, speed, eta)
        state = _ACTIVE
        if not state:
            return base
        phase_eta = _secs(eta)
        if str(phase).startswith("Phase 1"):
            # Requested first-pass estimate: Phase-1 ETA + (30% of Phase-1 ETA * 4).
            total_eta = phase_eta * (1.0 + 0.30 * 4.0)
            frac = max(0.0, min(1.0, float(overall) / max(1, int(total))))
            total_pct = 100.0 * frac / (1.0 + 0.30 * 4.0)
        elif str(phase).startswith("Phase 4"):
            # At this point everything but final retakes is complete; the rolling
            # measured Phase-4 ETA is more accurate than the pre-stage estimate.
            total_eta = phase_eta
            total_pct = _total_pct(state, total_eta)
        else:
            total_eta = phase_eta
            total_pct = _total_pct(state, total_eta)
        return f"{base} | Total: {total_pct:6.2f}% | Total ETA: {_td(total_eta)}"

    def eval_all(self, splits, cache, phase, start, total):
        state = self._eta_progress_state
        is_p1 = str(phase).startswith("Phase 1")
        is_p4 = str(phase).startswith("Phase 4")
        before = sum(
            1 for items in splits.values() for item in items
            if f"{phase}_{item['id']}" in cache
        )
        t0 = time.perf_counter()
        if is_p1:
            state["stage"] = "phase1"
            state["phase1_total"] = int(total)
        elif is_p4:
            state["stage"] = "phase4"
            print(
                f"[ETA] Phase 4 Pro retake estimate: {_td(state.get('phase4_est', 0.0))} | "
                f"Total elapsed: {_td(time.perf_counter() - state['run_started'])}",
                flush=True,
            )

        result = original_eval_all(self, splits, cache, phase, start, total)
        elapsed = max(0.001, time.perf_counter() - t0)

        if is_p1:
            after = sum(
                1 for items in splits.values() for item in items
                if f"{phase}_{item['id']}" in cache
            )
            newly = max(0, after - before)
            avg = elapsed / max(1, newly or int(total))
            misses_by_split: Dict[str, int] = {}
            for name, items in splits.items():
                count = sum(cache.get(f"Phase 1: Baseline_{item['id']}") is False for item in items)
                if count:
                    misses_by_split[name] = count
            misses = sum(misses_by_split.values())
            rsi_misses = sum(
                count for name, count in misses_by_split.items() if "DialogueRecall" not in name
            )
            # Existing RSI is capped at 64 misses and generates four branches for
            # each of two rounds. This is an upper-bound estimate; early passes lower it.
            rsi_items = min(64, rsi_misses)
            rsi_est = rsi_items * avg * 4.0 * 2.0

            # Before Phase 4 entropy is known, estimate each miss at 8 branches;
            # test-backed families the existing router forces to 16 are weighted at 16.
            phase4_units = 0.0
            for name, count in misses_by_split.items():
                try:
                    forced = bool(phase4_module._has_answer_blind_verifier(name))
                except Exception:
                    forced = False
                phase4_units += count * (16.0 if forced else 8.0)
            phase4_est = phase4_units * avg

            state.update({
                "baseline_elapsed": elapsed,
                "avg_item": avg,
                "misses": misses,
                "learn_est": elapsed * 0.30,
                "rsi_est": rsi_est,
                "phase3_est": elapsed * 0.30,
                "learning_test_est": 5.0 * avg,
                "phase4_est": phase4_est,
                "rsi_planned_rounds": rsi_items * 2,
                "phase1_end": time.perf_counter(),
                "stage": "learn",
            })
            remaining = (
                state["learn_est"] + state["rsi_est"] + state["phase3_est"]
                + state["learning_test_est"] + state["phase4_est"]
            )
            print(
                "[ETA] Phase-1 measured plan rebuilt from actual misses:\n"
                f"      Learn / memory + MCTS est: {_td(state['learn_est'])}\n"
                f"      RSI est ({rsi_items} misses × 4 branches × 2 rounds): {_td(state['rsi_est'])}\n"
                f"      Phase-3 consolidation est: {_td(state['phase3_est'])}\n"
                f"      Learning retention test est: {_td(state['learning_test_est'])}\n"
                f"      Phase-4 Pro miss retake est: {_td(state['phase4_est'])}\n"
                f"      Total remaining est: {_td(remaining)} | Total: {_total_pct(state, remaining):.2f}%",
                flush=True,
            )
        elif is_p4:
            state["stage"] = "done"
        return result

    def rsi(self, splits, cache):
        state = self._eta_progress_state
        if state.get("phase1_end"):
            learn_actual = max(0.0, time.perf_counter() - state["phase1_end"])
            state["learn_actual"] = learn_actual
        state["stage"] = "rsi"
        state["rsi_started"] = time.perf_counter()
        state["rsi_round_started"] = time.perf_counter()
        print(
            f"[ETA] RSI start: up to {state.get('rsi_planned_rounds', 0)} branch-round selections | "
            f"est {_td(state.get('rsi_est', 0.0))}",
            flush=True,
        )
        result = original_rsi(self, splits, cache)
        state["rsi_actual"] = max(0.0, time.perf_counter() - state["rsi_started"])
        state["stage"] = "phase3"
        return result

    def rsi_log(self, split, item_id, round_idx, candidate, passed, selection):
        result = original_rsi_log(self, split, item_id, round_idx, candidate, passed, selection)
        state = getattr(self, "_eta_progress_state", None)
        if not state or state.get("stage") != "rsi":
            return result
        now = time.perf_counter()
        last = state.get("rsi_round_started") or now
        duration = max(0.001, now - last)
        state["rsi_round_started"] = now
        state["rsi_round_durations"].append(duration)
        state["rsi_done_rounds"] += 1
        if passed and int(round_idx) == 1:
            state["rsi_planned_rounds"] = max(
                state["rsi_done_rounds"], state["rsi_planned_rounds"] - 1
            )
        done = state["rsi_done_rounds"]
        planned = max(done, state["rsi_planned_rounds"])
        avg_round = sum(state["rsi_round_durations"][-16:]) / len(state["rsi_round_durations"][-16:])
        rsi_left = max(0, planned - done) * avg_round
        remaining = rsi_left + state.get("phase3_est", 0.0) + state.get("learning_test_est", 0.0) + state.get("phase4_est", 0.0)
        print(
            f"[ETA] RSI rounds {done}/{planned} ({100.0*done/max(1,planned):.2f}%) | "
            f"RSI ETA {_td(rsi_left)} | Total {_total_pct(state, remaining):.2f}% | "
            f"Total ETA {_td(remaining)}",
            flush=True,
        )
        return result

    def phase3(self):
        state = self._eta_progress_state
        state["stage"] = "phase3"
        t0 = time.perf_counter()
        print(f"[ETA] Phase-3 consolidation est: {_td(state.get('phase3_est', 0.0))}", flush=True)
        result = original_phase3(self)
        state["phase3_actual"] = max(0.0, time.perf_counter() - t0)
        state["stage"] = "learning_test"
        return result

    def learning_test(self, model_identity):
        state = self._eta_progress_state
        state["stage"] = "learning_test"
        t0 = time.perf_counter()
        print(f"[ETA] Learning retention test est: {_td(state.get('learning_test_est', 0.0))}", flush=True)
        result = original_learning_test(self, model_identity)
        state["learning_test_actual"] = max(0.0, time.perf_counter() - t0)
        state["stage"] = "phase4"
        return result

    cls.run_full_suite = run
    cls._evaluate_all_splits = eval_all
    runtime_module._fmt_status = fmt_status
    phase4_module._run_rsi_self_improvement = rsi
    phase4_module._append_rsi_log = rsi_log
    phase4_module._run_phase3_consolidation = phase3
    phase4_module._run_learning_retention_test = learning_test
