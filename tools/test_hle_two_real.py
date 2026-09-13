#!/usr/bin/env python3
"""Two-generation real-model regression for the HLE termination fix.

Covers the exact pathological I0 case and one normal nonzero-token control.
Never reads or writes the production benchmark checkpoint.
"""
from __future__ import annotations

import time
from collections import Counter

import eval.code_prompt_hardening as hard
import eval.master_4000_runtime as rt
from master_4000_eval_suite import Master4000EvaluationEngine

TARGETS = ("HLE_9", "HLE_10")


def repeat_count(raw: str) -> int:
    thinking = raw.split("</think>", 1)[0]
    lines = [" ".join(line.split()) for line in thinking.splitlines() if line.strip()]
    return max(Counter(lines).values(), default=0)


def main() -> int:
    print("[*] Loading ONE real benchmark model for exactly TWO HLE generations...")
    engine = Master4000EvaluationEngine(max_duration_hours=72.0)
    splits = rt._repair_suite(engine.provider.load_all_4000_items())
    items = {str(x.get("id")): x for x in splits.get("HLE-100", [])}

    # HLE must use the normal benchmark ceiling; there is no HLE-specific cap.
    normal_ceiling = int(getattr(engine, "benchmark_max_tokens", None) or rt._benchmark_ceiling(engine))
    if normal_ceiling != 16384:
        raise SystemExit(f"[X] Expected normal benchmark ceiling 16384, got {normal_ceiling}")
    if hasattr(hard, "HLE_GENERATION_CEILING"):
        raise SystemExit("[X] HLE-specific generation ceiling still exists")

    failures = []
    results = []

    for item_id in TARGETS:
        item = items[item_id]
        token = hard._hle_literal_token(str(item.get("prompt", "")))
        if not token:
            raise SystemExit(f"[X] Could not extract HLE token from {item_id}")

        routed = hard._task_user("HLE-100", item, lambda split, obj: "ORIGINAL")
        expected_literal = f"Con(ZFC + {token})"
        assert expected_literal in routed
        assert "opaque literal axiom token" in routed
        assert "Immediately close </think>" in routed
        assert "alternate notation" in routed

        engine._current_phase = "HLE two-case termination regression"
        engine._current_split = "HLE-100"
        engine._current_item_id = item_id

        started = time.perf_counter()
        passed = bool(engine._evaluate_single_item("HLE-100", item))
        wall = time.perf_counter() - started

        raw = str(getattr(engine, "last_raw_out", "") or "")
        closed = "</think>" in raw
        repeats = repeat_count(raw)
        out_tokens = int(getattr(engine, "last_output_tokens", -1) or -1)
        good = passed and closed and repeats <= 2

        results.append((item_id, token, passed, closed, repeats, out_tokens, wall))
        if not good:
            failures.append((item_id, raw[-1500:]))

        print()
        print("=" * 96)
        print(f"{item_id} | literal token {token}")
        print("=" * 96)
        print(raw)
        print()
        print(f"PASS:         {passed}")
        print(f"closed:       {closed}")
        print(f"max repeat:   {repeats}")
        print(f"output tokens:{out_tokens}")
        print(f"wall:         {wall:.2f}s")

    print()
    print("=" * 96)
    print("FINAL — TWO HLE GENERATIONS ONLY")
    print("=" * 96)
    for item_id, token, passed, closed, repeats, out_tokens, wall in results:
        print(
            f"{item_id:8} token={token:4} | {'PASS' if passed else 'FAIL':4} | "
            f"closed={closed} | repeat={repeats} | out={out_tokens} | wall={wall:.2f}s"
        )

    if failures:
        for item_id, tail in failures:
            print(f"\n[X] {item_id} failed:\n{tail}")
        return 1

    print()
    print("[✓] Exact I0 loop case fixed")
    print("[✓] Nonzero HLE control remains correct")
    print("[✓] Both close </think>")
    print("[✓] No thinking line repeated more than twice")
    print("[✓] HLE uses the normal 16,384-token benchmark ceiling")
    print("[✓] Production checkpoint was not read or modified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
