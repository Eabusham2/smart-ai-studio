#!/usr/bin/env python3
"""Final fast live-model smoke: exactly one current-policy run per merged family.

One model load, five benchmark families, five generations total. No historical
old-prompt reruns. DialogueRecall is intentionally excluded pre-Learn; it belongs
in the real pipeline's Learn/RSI/Phase-3 -> Phase-4 retest path.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import master_4000_eval_suite as suite
import eval.code_prompt_hardening as hardening
import eval.live_generation_stream as live_stream
import eval.master_4000_runtime as rt
import eval.phase4_pro_rsi as p4

VERIFIED_GEMINI_PROMPT = (
    "You are a fast symbolic computing engine. "
    "Keep your internal scratchpad (<think>) strictly minimal: write only concise intermediate formulas or numbers. "
    "No conversational monologue, no self-reflection, and no verification loops. "
    "Close </think> immediately once calculated and output the answer."
)
EXPECTED_SYSTEM_PROMPT = VERIFIED_GEMINI_PROMPT + hardening.GLOBAL_SYSTEM_SUFFIX

FOCUS = (
    "LiveCodeBench-Hard",
    "AIME-150",
    "GPQA-400",
    "DeepSWE-50",
    "AutonomousEvolution-200",
)

CAPS = {
    "LiveCodeBench-Hard": 1024,
    "AIME-150": 768,
    "GPQA-400": 512,
    "DeepSWE-50": 2048,
    "AutonomousEvolution-200": 1024,
}


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    assert rt.SYSTEM_PROMPT == EXPECTED_SYSTEM_PROMPT
    assert p4.SYSTEM_PROMPT == EXPECTED_SYSTEM_PROMPT
    assert getattr(suite.Master4000EvaluationEngine, "_code_prompt_hardening_installed", False)

    live_stream.LIVE_GENERATION_LOG = "/tmp/random_changed_split_smoke_live.log"
    rt.RAW_OUTPUT_LOG = "/tmp/random_changed_split_smoke_raw.log"
    p4.RAW_OUTPUT_LOG = "/tmp/random_changed_split_smoke_raw.log"

    rng = random.Random(args.seed) if args.seed is not None else random.SystemRandom()

    print("[✓] Verified Gemini base retained; universal anti-loop rule active.")
    print("[*] Final merged-family smoke: EXACTLY 5 families, one generation each.")
    print("[*] LCB | AIME | GPQA | DeepSWE | AutonomousEvolution")
    print("[*] DialogueRecall intentionally deferred until after Learn/RSI/Phase-3.\n")
    print("[*] Loading ONE real benchmark model...")

    engine = suite.Master4000EvaluationEngine(max_duration_hours=72.0)
    splits = rt._repair_suite(engine.provider.load_all_4000_items())
    tokenizer = engine.engine.tokenizer

    rows = []

    for split_name in FOCUS:
        items = splits.get(split_name, [])
        if not items:
            raise RuntimeError(f"Missing required split: {split_name}")

        item = rng.choice(items)
        item_id = str(item.get("id", "missing-id"))

        print("=" * 116)
        print(f"{split_name} | ONE RANDOM ITEM | {item_id}")
        print("=" * 116)

        engine._current_phase = "Final Merged-Family Smoke"
        engine._current_split = split_name
        engine._current_item_id = item_id
        engine.benchmark_max_tokens = CAPS[split_name]

        started = time.perf_counter()
        error = ""
        try:
            passed = bool(engine._evaluate_single_item(split_name, item))
        except Exception as exc:
            passed = False
            error = f"{type(exc).__name__}: {exc}"
        wall = time.perf_counter() - started

        raw = str(getattr(engine, "last_raw_out", "") or "")
        closed, think_tokens = thought_stats(tokenizer, raw)
        out_tokens = int(getattr(engine, "last_output_tokens", 0) or 0)

        print(f"result:          {'PASS' if passed else 'FAIL'}")
        print(f"closed </think>: {closed}")
        print(f"thinking tokens: {think_tokens}")
        print(f"output tokens:   {out_tokens}")
        print(f"wall time:       {wall:.2f}s")
        if error:
            print(f"error:           {error}")
        if raw:
            print("RAW OUTPUT:")
            print("-" * 116)
            print(raw)
            print("-" * 116)

        rows.append((split_name, item_id, passed, closed, think_tokens, out_tokens, wall))

    print("\n" + "=" * 116)
    print("FINAL — EXACTLY FIVE MERGED FAMILIES, ONE GO EACH")
    print("=" * 116)
    for split_name, item_id, passed, closed, think_tokens, out_tokens, wall in rows:
        print(
            f"{split_name:<26} | {item_id:<22} | "
            f"{'PASS' if passed else 'FAIL':4} | closed={str(closed):5} | "
            f"think={think_tokens:4} | out={out_tokens:4} | wall={wall:7.2f}s"
        )

    bad = [row for row in rows if not row[2]]
    unclosed = [row for row in rows if not row[3]]
    print()
    if bad:
        print("[X] Failed family/families:", ", ".join(row[0] for row in bad))
    if unclosed:
        print("[X] Unclosed thinking:", ", ".join(row[0] for row in unclosed))
    if not bad and not unclosed:
        print("[✓] All five merged-family samples passed and closed </think>.")

    print("[✓] No checkpoint was read or modified.")
    print("[*] Live: /tmp/random_changed_split_smoke_live.log")
    print("[*] Raw : /tmp/random_changed_split_smoke_raw.log")
    return 1 if bad or unclosed else 0


if __name__ == "__main__":
    raise SystemExit(main())
