"""Suite-wide stage-aware ETA and smooth total-progress overlay.

This wraps reporting only; it does not change benchmark generation, scoring, RSI,
training, or Pro branching. Phase-1 uses the requested rough whole-run estimate:
remaining Phase-1 ETA + four 30% Phase-1 allowances. After Phase 1 it rebuilds the
plan from measured speed and actual misses. Optional flagship DeepSWE contributes
to the same counters/ETA plan when enabled and contributes exactly zero when skipped.
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


def _ema(old: float, new: float, alpha: float = 0.25) -> float:
    if new <= 0:
        return old
    return new if old <= 0 else old * (1.0 - alpha) + new * alpha


def _raw_total_pct(state: Dict[str, Any], remaining: float) -> float:
    elapsed = max(0.0, time.perf_counter() - state["run_started"])
    return max(0.0, min(100.0, 100.0 * elapsed / max(0.001, elapsed + max(0.0, remaining))))


def _smooth_total_pct(state: Dict[str, Any], remaining: float, *, raw: float | None = None) -> float:
    value = _raw_total_pct(state, remaining) if raw is None else float(raw)
    # Never move the displayed total backwards when estimates are refined.
    value = max(float(state.get("last_total_pct", 0.0)), value)
    if state.get("stage") != "done":
        value = min(value, 99.99)
    state["last_total_pct"] = value
    return value


def _deep_enabled(state: Dict[str, Any]) -> bool:
    owner = state.get("owner")
    return bool(owner is not None and getattr(owner, "_deepswe_enabled", False))


def _deep_state(state: Dict[str, Any]) -> Dict[str, Any]:
    owner = state.get("owner")
    value = getattr(owner, "_deepswe_state", {}) if owner is not None else {}
    return value if isinstance(value, dict) else {}


def _remaining_after_rsi(state: Dict[str, Any]) -> float:
    return (
        float(state.get("phase3_est", 0.0))
        + float(state.get("learning_test_est", 0.0))
        + float(state.get("phase4_est", 0.0))
    )


def install(runtime_module, phase4_module, cls) -> None:
    global _ACTIVE

    original_fmt = runtime_module._fmt_status
    original_eval_all = cls._evaluate_all_splits
    original_rsi = phase4_module._run_rsi_self_improvement
    original_rsi_log = phase4_module._append_rsi_log
    original_phase3 = phase4_module._run_phase3_consolidation
    original_learning_test = phase4_module._run_learning_retention_test
    original_run = cls.run_full_suite

    # Observe optional DeepSWE work without changing its benchmark logic.
    try:
        from eval import deepswe_optional_flagship as deep_module
        original_deep_generate = deep_module._generate_patch
        original_deep_verify = deep_module._verify_selected_patch
    except Exception:
        deep_module = None
        original_deep_generate = None
        original_deep_verify = None

    def run(self):
        global _ACTIVE
        state = {
            "owner": self,
            "run_started": time.perf_counter(),
            "stage": "startup",
            "last_total_pct": 0.0,
            "normal_phase1_total": 0,
            "normal_phase1_avg": 0.0,
            "normal_phase1_elapsed": 0.0,
            "deep_total": int(getattr(deep_module, "DEEPSWE_TASK_COUNT", 0) or 0),
            "deep_baseline_done": 0,
            "deep_baseline_misses": 0,
            "deep_baseline_avg": 0.0,
            "deep_baseline_elapsed": 0.0,
            "deep_rsi_planned_rounds": 0,
            "deep_rsi_done_rounds": 0,
            "deep_rsi_avg_round": 0.0,
            "deep_phase4_planned": 0,
            "deep_phase4_done": 0,
            "deep_phase4_avg": 0.0,
            "deep_active_starts": {},
            "learn_est": 0.0,
            "normal_rsi_est": 0.0,
            "deep_rsi_est": 0.0,
            "phase3_est": 0.0,
            "learning_test_est": 0.0,
            "normal_phase4_est": 0.0,
            "deep_phase4_est": 0.0,
            "phase4_est": 0.0,
            "normal_rsi_planned_rounds": 0,
            "normal_rsi_done_rounds": 0,
            "normal_rsi_avg_round": 0.0,
            "rsi_round_started": None,
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
            state["normal_phase1_total"] = max(int(total), int(state.get("normal_phase1_total", 0)))
            deep_total = state["deep_total"] if _deep_enabled(state) else 0
            deep_cached = len((_deep_state(state).get("baseline") or {})) if deep_total else 0
            state["deep_baseline_done"] = max(state["deep_baseline_done"], deep_cached)
            phase1_total = int(total) + deep_total
            phase1_done = min(int(total), int(overall)) + state["deep_baseline_done"]

            normal_left = max(0, int(total) - int(overall))
            if normal_left > 0 and phase_eta > 0:
                state["normal_phase1_avg"] = _ema(
                    state["normal_phase1_avg"], phase_eta / normal_left
                )
            fallback_avg = state["normal_phase1_avg"] or 1.0
            deep_avg = state["deep_baseline_avg"] or fallback_avg
            deep_left_eta = max(0, deep_total - state["deep_baseline_done"]) * deep_avg
            phase1_remaining = phase_eta + deep_left_eta

            # User-requested rough whole-run formula while baseline is still underway.
            total_eta = phase1_remaining * (1.0 + 0.30 * 4.0)
            phase1_fraction = phase1_done / max(1, phase1_total)
            rough_total_pct = 100.0 * phase1_fraction / (1.0 + 0.30 * 4.0)
            total_pct = _smooth_total_pct(state, total_eta, raw=rough_total_pct)
            return (
                f"{base} | Phase1 total: {phase1_done}/{phase1_total} "
                f"({100.0*phase1_fraction:5.2f}%) | Total: {total_pct:6.2f}% | "
                f"Total ETA: {_td(total_eta)}"
            )

        total_eta = phase_eta
        total_pct = _smooth_total_pct(state, total_eta)
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
            state["normal_phase1_total"] = int(total)
        elif is_p4:
            state["stage"] = "phase4"
            print(
                f"[ETA] Phase 4 Pro retake estimate (all enabled benchmarks): "
                f"{_td(state.get('phase4_est', 0.0))} | Total "
                f"{_smooth_total_pct(state, state.get('phase4_est', 0.0)):.2f}%",
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
            # DeepSWE is inside original_eval_all when enabled; remove its measured
            # time before deriving the normal-suite per-item speed.
            normal_elapsed = max(0.001, elapsed - float(state.get("deep_baseline_elapsed", 0.0)))
            normal_avg = normal_elapsed / max(1, newly or int(total))
            state["normal_phase1_elapsed"] = normal_elapsed
            state["normal_phase1_avg"] = _ema(state["normal_phase1_avg"], normal_avg, 0.50)

            misses_by_split: Dict[str, int] = {}
            for name, items in splits.items():
                count = sum(cache.get(f"Phase 1: Baseline_{item['id']}") is False for item in items)
                if count:
                    misses_by_split[name] = count
            normal_misses = sum(misses_by_split.values())
            normal_rsi_misses = sum(
                count for name, count in misses_by_split.items() if "DialogueRecall" not in name
            )

            deep_misses = 0
            deep_total = 0
            if _deep_enabled(state):
                baseline = _deep_state(state).get("baseline") or {}
                deep_total = state["deep_total"]
                deep_misses = sum(value is False for value in baseline.values())
                state["deep_baseline_done"] = len(baseline)
                state["deep_baseline_misses"] = deep_misses

            # Normal RSI keeps the existing 64-item cap and 4 branches x up to 2 rounds.
            normal_rsi_items = min(64, normal_rsi_misses)
            normal_rsi_est = normal_rsi_items * state["normal_phase1_avg"] * 4.0 * 2.0
            # DeepSWE opt-in RSI intentionally processes every DeepSWE miss, also 4 x 2.
            deep_avg = state["deep_baseline_avg"] or state["normal_phase1_avg"]
            deep_rsi_est = deep_misses * deep_avg * 4.0 * 2.0

            normal_phase4_units = 0.0
            for name, count in misses_by_split.items():
                try:
                    forced = bool(phase4_module._has_answer_blind_verifier(name))
                except Exception:
                    forced = False
                normal_phase4_units += count * (16.0 if forced else 8.0)
            normal_phase4_est = normal_phase4_units * state["normal_phase1_avg"]
            # DeepSWE is test-backed, so the unchanged Pro router forces N=16.
            deep_phase4_est = deep_misses * deep_avg * 16.0

            state.update({
                "misses": normal_misses + deep_misses,
                "learn_est": normal_elapsed * 0.30,
                "normal_rsi_est": normal_rsi_est,
                "deep_rsi_est": deep_rsi_est,
                "phase3_est": normal_elapsed * 0.30,
                "learning_test_est": 5.0 * state["normal_phase1_avg"],
                "normal_phase4_est": normal_phase4_est,
                "deep_phase4_est": deep_phase4_est,
                "phase4_est": normal_phase4_est + deep_phase4_est,
                "normal_rsi_planned_rounds": normal_rsi_items * 2,
                "deep_rsi_planned_rounds": deep_misses * 2,
                "deep_phase4_planned": deep_misses,
                "phase1_end": time.perf_counter(),
                "stage": "learn",
            })
            remaining = (
                state["learn_est"] + normal_rsi_est + deep_rsi_est
                + state["phase3_est"] + state["learning_test_est"]
                + state["phase4_est"]
            )
            total_pct = _smooth_total_pct(state, remaining)
            print(
                "[ETA] Phase-1 measured plan rebuilt from actual misses (suite-wide):\n"
                f"      Normal misses: {normal_misses} | DeepSWE misses: {deep_misses}/{deep_total if deep_total else 0}\n"
                f"      Learn / memory + MCTS est: {_td(state['learn_est'])}\n"
                f"      RSI normal est ({normal_rsi_items} × 4 branches × 2 rounds): {_td(normal_rsi_est)}\n"
                f"      RSI DeepSWE est ({deep_misses} × 4 agent branches × 2 rounds): {_td(deep_rsi_est)}\n"
                f"      Phase-3 consolidation est: {_td(state['phase3_est'])}\n"
                f"      Learning retention test est: {_td(state['learning_test_est'])}\n"
                f"      Phase-4 normal Pro est: {_td(normal_phase4_est)}\n"
                f"      Phase-4 DeepSWE Pro N=16 est: {_td(deep_phase4_est)}\n"
                f"      Total remaining est: {_td(remaining)} | Total: {total_pct:.2f}%",
                flush=True,
            )
        elif is_p4:
            state["stage"] = "done"
            state["last_total_pct"] = 100.0
            print("[ETA] Evaluation lifecycle complete | Total: 100.00% | Total ETA: 0:00:00", flush=True)
        return result

    def rsi(self, splits, cache):
        state = self._eta_progress_state
        if state.get("phase1_end"):
            state["learn_actual"] = max(0.0, time.perf_counter() - state["phase1_end"])
        state["stage"] = "rsi"
        state["rsi_started"] = time.perf_counter()
        state["rsi_round_started"] = time.perf_counter()
        remaining = state.get("normal_rsi_est", 0.0) + state.get("deep_rsi_est", 0.0) + _remaining_after_rsi(state)
        print(
            f"[ETA] RSI start: normal rounds≤{state.get('normal_rsi_planned_rounds', 0)}, "
            f"DeepSWE rounds≤{state.get('deep_rsi_planned_rounds', 0)} | "
            f"RSI est {_td(state.get('normal_rsi_est',0)+state.get('deep_rsi_est',0))} | "
            f"Total {_smooth_total_pct(state, remaining):.2f}% | Total ETA {_td(remaining)}",
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
        state["normal_rsi_avg_round"] = _ema(state["normal_rsi_avg_round"], duration)
        state["normal_rsi_done_rounds"] += 1
        if passed and int(round_idx) == 1:
            state["normal_rsi_planned_rounds"] = max(
                state["normal_rsi_done_rounds"], state["normal_rsi_planned_rounds"] - 1
            )
        done = state["normal_rsi_done_rounds"]
        planned = max(done, state["normal_rsi_planned_rounds"])
        normal_left = max(0, planned - done) * state["normal_rsi_avg_round"]
        deep_left = max(
            0, state["deep_rsi_planned_rounds"] - state["deep_rsi_done_rounds"]
        ) * (state["deep_rsi_avg_round"] or state["deep_baseline_avg"] * 4.0)
        remaining = normal_left + deep_left + _remaining_after_rsi(state)
        print(
            f"[ETA] RSI normal {done}/{planned} ({100.0*done/max(1,planned):.2f}%) | "
            f"DeepSWE {state['deep_rsi_done_rounds']}/{max(1,state['deep_rsi_planned_rounds'])} | "
            f"RSI ETA {_td(normal_left+deep_left)} | Total {_smooth_total_pct(state, remaining):.2f}% | "
            f"Total ETA {_td(remaining)}",
            flush=True,
        )
        return result

    def phase3(self):
        state = self._eta_progress_state
        state["stage"] = "phase3"
        t0 = time.perf_counter()
        remaining = state.get("phase3_est", 0.0) + state.get("learning_test_est", 0.0) + state.get("phase4_est", 0.0)
        print(
            f"[ETA] Phase-3 consolidation est: {_td(state.get('phase3_est', 0.0))} | "
            f"Total {_smooth_total_pct(state, remaining):.2f}% | Total ETA {_td(remaining)}",
            flush=True,
        )
        result = original_phase3(self)
        state["phase3_actual"] = max(0.0, time.perf_counter() - t0)
        state["stage"] = "learning_test"
        return result

    def learning_test(self, model_identity):
        state = self._eta_progress_state
        state["stage"] = "learning_test"
        t0 = time.perf_counter()
        remaining = state.get("learning_test_est", 0.0) + state.get("phase4_est", 0.0)
        print(
            f"[ETA] Learning retention test est: {_td(state.get('learning_test_est', 0.0))} | "
            f"Total {_smooth_total_pct(state, remaining):.2f}% | Total ETA {_td(remaining)}",
            flush=True,
        )
        result = original_learning_test(self, model_identity)
        state["learning_test_actual"] = max(0.0, time.perf_counter() - t0)
        state["stage"] = "phase4"
        return result

    # DeepSWE progress hooks. They wrap only timing/reporting around the official
    # generation and selected-patch verification functions.
    if deep_module is not None and original_deep_generate and original_deep_verify:
        def deep_category(stage: str) -> str:
            if stage == "baseline":
                return "baseline"
            if stage.startswith("rsi-r"):
                return "rsi-r" + stage.split("rsi-r", 1)[1].split("-", 1)[0]
            if stage.startswith("phase4"):
                return "phase4"
            return stage

        def deep_generate(pier, task_dir, base_url, universal_rule, stage, temperature, prior_patch=""):
            state = _ACTIVE
            if state:
                key = (deep_category(stage), task_dir.name)
                state["deep_active_starts"].setdefault(key, time.perf_counter())
            return original_deep_generate(
                pier, task_dir, base_url, universal_rule, stage, temperature, prior_patch
            )

        def deep_verify(pier, task_dir, patch, stage):
            state = _ACTIVE
            category = deep_category(stage)
            result = original_deep_verify(pier, task_dir, patch, stage)
            if not state:
                return result
            key = (category, task_dir.name)
            started = state["deep_active_starts"].pop(key, time.perf_counter())
            duration = max(0.001, time.perf_counter() - started)

            if category == "baseline":
                state["deep_baseline_done"] += 1
                state["deep_baseline_elapsed"] += duration
                state["deep_baseline_avg"] = _ema(state["deep_baseline_avg"], duration)
                if not result:
                    state["deep_baseline_misses"] += 1
                normal_total = state.get("normal_phase1_total", 0)
                phase_total = normal_total + state["deep_total"]
                phase_done = normal_total + state["deep_baseline_done"]
                deep_left = max(0, state["deep_total"] - state["deep_baseline_done"])
                phase1_eta = deep_left * state["deep_baseline_avg"]
                total_eta = phase1_eta * (1.0 + 0.30 * 4.0)
                rough = 100.0 * (phase_done / max(1, phase_total)) / (1.0 + 0.30 * 4.0)
                pct = _smooth_total_pct(state, total_eta, raw=rough)
                print(
                    f"[DeepSWE Phase 1] {state['deep_baseline_done']}/{state['deep_total']} | "
                    f"Phase1 total {phase_done}/{phase_total} ({100.0*phase_done/max(1,phase_total):.2f}%) | "
                    f"Total {pct:.2f}% | Total ETA {_td(total_eta)}",
                    flush=True,
                )

            elif category.startswith("rsi-r"):
                state["deep_rsi_done_rounds"] += 1
                state["deep_rsi_avg_round"] = _ema(state["deep_rsi_avg_round"], duration)
                round_idx = int(category.rsplit("r", 1)[-1])
                if result and round_idx == 1:
                    state["deep_rsi_planned_rounds"] = max(
                        state["deep_rsi_done_rounds"], state["deep_rsi_planned_rounds"] - 1
                    )
                deep_left = max(
                    0, state["deep_rsi_planned_rounds"] - state["deep_rsi_done_rounds"]
                ) * state["deep_rsi_avg_round"]
                remaining = deep_left + _remaining_after_rsi(state)
                print(
                    f"[DeepSWE RSI] rounds {state['deep_rsi_done_rounds']}/{max(1,state['deep_rsi_planned_rounds'])} | "
                    f"DeepSWE RSI ETA {_td(deep_left)} | Total {_smooth_total_pct(state, remaining):.2f}% | "
                    f"Total ETA {_td(remaining)}",
                    flush=True,
                )

            elif category == "phase4":
                state["deep_phase4_done"] += 1
                state["deep_phase4_avg"] = _ema(state["deep_phase4_avg"], duration)
                left = max(0, state["deep_phase4_planned"] - state["deep_phase4_done"])
                remaining = left * state["deep_phase4_avg"]
                print(
                    f"[DeepSWE Phase 4 Pro] {state['deep_phase4_done']}/{max(1,state['deep_phase4_planned'])} | "
                    f"DeepSWE ETA {_td(remaining)} | Total {_smooth_total_pct(state, remaining):.2f}% | "
                    f"Total ETA {_td(remaining)}",
                    flush=True,
                )
            return result

        deep_module._generate_patch = deep_generate
        deep_module._verify_selected_patch = deep_verify

    cls.run_full_suite = run
    cls._evaluate_all_splits = eval_all
    runtime_module._fmt_status = fmt_status
    phase4_module._run_rsi_self_improvement = rsi
    phase4_module._append_rsi_log = rsi_log
    phase4_module._run_phase3_consolidation = phase3
    phase4_module._run_learning_retention_test = learning_test
