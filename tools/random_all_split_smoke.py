#!/usr/bin/env python3
"""Read-only real-model smoke: one random item from every benchmark split.

The evaluator checkpoint is READ ONLY. Every split gets one NEW-policy sample.
For any family whose task prompt changed on this branch, missing historical
PASS/FAIL is replaced by an actual OLD-prompt run on the same item. That old
output is rescored with the current strict reader/verifier when available, so
`NO_HISTORY` is never confused with model output such as `unknown`.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import master_4000_eval_suite as suite
import eval.live_generation_stream as live_stream
import eval.master_4000_runtime as rt
import eval.phase4_pro_rsi as p4
import eval.scoring_hardening as scoring


VERIFIED_GEMINI_PROMPT = (
    "You are a fast symbolic computing engine. "
    "Keep your internal scratchpad (<think>) strictly minimal: write only concise intermediate formulas or numbers. "
    "No conversational monologue, no self-reflection, and no verification loops. "
    "Close </think> immediately once calculated and output the answer."
)

CHANGED_SPLITS = (
    "HumanEval", "LiveCodeBench", "DeepSWE", "AIME", "GPQA", "MMLU-Pro",
    "HLE", "BFCL", "TensorGraphDSL", "AutonomousEvolution", "DialogueRecall",
)
LEGACY_MARKERS = ("__v2done__:", "__eyad_v3_done__:", "__eyad_v5_done__:")

SMOKE_CAPS = {
    "HumanEval-164": 1024,
    "LiveCodeBench-Hard": 1024,
    "GSM8K-500": 768,
    "MATH-500": 768,
    "AIME-150": 768,
    "GPQA-400": 512,
    "MMLU-Pro-1000": 512,
    "BFCL-200": 512,
    "ZebraLogic-200": 512,
    "HLE-100": 512,
    "DeepSWE-50": 2048,
    "TensorGraphDSL-300": 512,
    "AutonomousEvolution-200": 1024,
    "DialogueRecall-150": 384,
}


def checkpoint_history(cache: Dict[str, Any], item_id: str) -> Tuple[str, Any]:
    key = f"Phase 1: Baseline_{item_id}"
    if key in cache:
        value = cache[key]
        if value is True:
            return "PASS", True
        if value is False:
            return "FAIL", False
        return "DIRECT_NONBOOLEAN", None
    for prefix in LEGACY_MARKERS:
        if cache.get(prefix + key) is True:
            return "DONE_NO_DIRECT_RESULT", None
    return "NO_HISTORY", None


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        cache = data.get("completed_items", {})
        return cache if isinstance(cache, dict) else {}
    except Exception:
        return {}


def thought_stats(tokenizer, raw: str):
    raw = raw or ""
    closed = "</think>" in raw
    thinking = raw.split("</think>", 1)[0] if closed else raw
    try:
        count = len(tokenizer.encode(thinking, add_special_tokens=False))
    except TypeError:
        count = len(tokenizer.encode(thinking))
    except Exception:
        count = -1
    return closed, count


def changed_split(name: str) -> bool:
    return any(part in name for part in CHANGED_SPLITS)


def current_strict_or_existing(engine, split_name, item, existing_value):
    strict = scoring.strict_score(engine, split_name, item, str(getattr(engine, "last_raw_out", "") or ""))
    return bool(strict) if strict is not None else bool(existing_value)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default="eval_results/eval_checkpoint_4000.json",
        help="Checkpoint to READ for history; never modified.",
    )
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    assert rt.SYSTEM_PROMPT == VERIFIED_GEMINI_PROMPT
    assert p4.SYSTEM_PROMPT == VERIFIED_GEMINI_PROMPT
    assert getattr(suite.Master4000EvaluationEngine, "_code_prompt_hardening_installed", False)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    cache = load_checkpoint(checkpoint_path)

    live_stream.LIVE_GENERATION_LOG = "/tmp/random_all_split_smoke_live.log"
    rt.RAW_OUTPUT_LOG = "/tmp/random_all_split_smoke_raw.log"
    p4.RAW_OUTPUT_LOG = "/tmp/random_all_split_smoke_raw.log"

    rng = random.Random(args.seed) if args.seed is not None else random.SystemRandom()

    print("[✓] Exact verified Gemini system prompt is unchanged.")
    print("[✓] No global anti-unknown system suffix is installed.")
    print(f"[*] Historical checkpoint (read only): {checkpoint_path}")
    print("[*] Loading ONE real benchmark model...")

    engine = suite.Master4000EvaluationEngine(max_duration_hours=72.0)
    splits = rt._repair_suite(engine.provider.load_all_4000_items())
    tokenizer = engine.engine.tokenizer

    print(f"[*] Found {len(splits)} benchmark split types.")
    print("[*] One random item will be evaluated from EVERY split.\n")

    rows = []

    for split_name, items in splits.items():
        if not items:
            continue
        item = rng.choice(items)
        item_id = str(item.get("id", "missing-id"))
        historical_label, historical_value = checkpoint_history(cache, item_id)

        print("=" * 118)
        print(f"{split_name} | RANDOM ITEM {item_id}")
        print(f"Historical status: {historical_label}")
        print("=" * 118)

        reference_label = historical_label
        reference_value = historical_value

        # If this branch changed the task prompt and no direct old result survives,
        # run the exact pre-hardening evaluator on the same item. Score its output
        # with the current strict reader where one owns the split.
        if changed_split(split_name) and reference_value is None:
            engine._current_phase = "All-Split Smoke OLD Reference"
            engine._current_split = split_name
            engine._current_item_id = item_id
            engine.benchmark_max_tokens = SMOKE_CAPS.get(split_name, 1024)

            t0 = time.perf_counter()
            try:
                old_existing = bool(rt.evaluate_one(engine, split_name, item))
                reference_value = current_strict_or_existing(
                    engine, split_name, item, old_existing
                )
                reference_label = "RETEST_OLD_PASS" if reference_value else "RETEST_OLD_FAIL"
            except Exception as exc:
                reference_value = None
                reference_label = f"OLD_RETEST_ERROR:{type(exc).__name__}"
            old_wall = time.perf_counter() - t0
            print(f"Old-prompt reference: {reference_label} ({old_wall:.2f}s)")

        engine._current_phase = "All-Split Smoke NEW"
        engine._current_split = split_name
        engine._current_item_id = item_id
        engine.benchmark_max_tokens = SMOKE_CAPS.get(split_name, 1024)

        started = time.perf_counter()
        error = ""
        try:
            new_pass = bool(engine._evaluate_single_item(split_name, item))
        except Exception as exc:
            new_pass = False
            error = f"{type(exc).__name__}: {exc}"
        wall = time.perf_counter() - started

        raw = str(getattr(engine, "last_raw_out", "") or "")
        closed, think_tokens = thought_stats(tokenizer, raw)
        output_tokens = int(getattr(engine, "last_output_tokens", 0) or 0)

        print(f"NEW result:           {'PASS' if new_pass else 'FAIL'}")
        print(f"closed </think>:      {closed}")
        print(f"thinking tokens:      {think_tokens}")
        print(f"output tokens:        {output_tokens}")
        print(f"wall time:            {wall:.2f}s")
        if error:
            print(f"error:                {error}")
        if raw:
            print("RAW OUTPUT (tail):")
            print("-" * 118)
            print(raw[-2200:])
            print("-" * 118)

        comparable = reference_value is not None and changed_split(split_name)
        regression = bool(comparable and reference_value is True and new_pass is False)

        rows.append({
            "split": split_name,
            "id": item_id,
            "history": reference_label,
            "old": reference_value,
            "new": new_pass,
            "closed": closed,
            "think": think_tokens,
            "out": output_tokens,
            "wall": wall,
            "regression": regression,
        })

    print("\n" + "=" * 118)
    print("FINAL — ONE RANDOM ITEM FROM EVERY BENCHMARK TYPE")
    print("=" * 118)
    for row in rows:
        if row["old"] is True:
            old_text = "PASS"
        elif row["old"] is False:
            old_text = "FAIL"
        else:
            old_text = row["history"]
        print(
            f"{row['split']:<26} | {row['id']:<24} | "
            f"old={old_text:<20} | new={'PASS' if row['new'] else 'FAIL':<4} | "
            f"closed={str(row['closed']):5} | think={row['think']:4} | "
            f"out={row['out']:4} | wall={row['wall']:7.2f}s"
        )

    regressions = [row for row in rows if row["regression"]]
    print()
    if regressions:
        print("[X] REGRESSION on a changed-family sample that previously/retested PASS:")
        for row in regressions:
            print(f"    {row['split']} {row['id']}")
    else:
        print("[✓] No changed-family sample regressed from a known/retested old PASS.")

    print("[*] DialogueRecall may intentionally fail before Learn/RSI/Phase-3, then be retested in Phase 4.")
    print("[✓] Evaluation checkpoint was READ ONLY and was never modified.")
    print("[*] Live stream: /tmp/random_all_split_smoke_live.log")
    print("[*] Raw smoke log: /tmp/random_all_split_smoke_raw.log")
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
