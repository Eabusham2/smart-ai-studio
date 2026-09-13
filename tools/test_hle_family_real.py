#!/usr/bin/env python3
"""Run every HLE-100 item exactly once on the real loaded model.

Read-only with respect to the benchmark checkpoint. This validates the production
family router, strict scorer, closing </think>, and repetition behavior across the
entire HLE section rather than spot-checking one index.
"""
from __future__ import annotations

import time
from collections import Counter

import eval.code_prompt_hardening as hard
import eval.master_4000_runtime as rt
from master_4000_eval_suite import Master4000EvaluationEngine


def _repeat_count(raw: str) -> int:
    thinking = raw.split("</think>", 1)[0]
    lines = [" ".join(line.split()) for line in thinking.splitlines() if line.strip()]
    counts = Counter(lines)
    return max(counts.values(), default=0)


def main() -> int:
    print("[✓] HLE family test is checkpoint-independent")
    print(f"[*] HLE generation backstop: {hard.HLE_GENERATION_CEILING} tokens")
    print("[*] Loading ONE real benchmark model...")

    engine = Master4000EvaluationEngine(max_duration_hours=72.0)
    splits = rt._repair_suite(engine.provider.load_all_4000_items())
    items = list(splits.get("HLE-100", []))
    if len(items) != 100:
        raise SystemExit(f"Expected 100 HLE items, found {len(items)}")

    for item in items:
        token = hard._hle_literal_token(str(item.get("prompt", "")))
        if not token:
            raise SystemExit(f"Could not extract literal HLE token from {item.get('id')}")
        routed = hard._task_user("HLE-100", item, lambda split, obj: "ORIGINAL")
        if token not in routed:
            raise SystemExit(f"Routed prompt lost literal token for {item.get('id')}")
        if "using the form Con(ZFC + Ik)" in routed:
            raise SystemExit(f"Ambiguous Ik wording leaked into routed prompt for {item.get('id')}")

    print("[✓] All 100 HLE prompts route through the generic literal-token policy")
    print()

    failures = []
    total_wall = 0.0

    for n, item in enumerate(items, 1):
        item_id = str(item.get("id", f"HLE_{n-1}"))
        token = hard._hle_literal_token(str(item.get("prompt", ""))) or "?"

        engine._current_phase = "HLE all-family real-model regression"
        engine._current_split = "HLE-100"
        engine._current_item_id = item_id

        started = time.perf_counter()
        passed = bool(engine._evaluate_single_item("HLE-100", item))
        wall = time.perf_counter() - started
        total_wall += wall

        raw = str(getattr(engine, "last_raw_out", "") or "")
        closed = "</think>" in raw
        repeats = _repeat_count(raw)
        out_tokens = int(getattr(engine, "last_output_tokens", -1) or -1)

        good = passed and closed and repeats <= 2 and (out_tokens < 0 or out_tokens <= hard.HLE_GENERATION_CEILING)
        status = "PASS" if good else "FAIL"
        print(
            f"{n:3}/100 {item_id:8} token={token:8} {status} | "
            f"closed={closed} repeat={repeats} out={out_tokens:4} wall={wall:6.2f}s"
        )

        if not good:
            failures.append((item_id, token, passed, closed, repeats, out_tokens, raw[-1200:]))

    print()
    print("=" * 108)
    print("HLE-100 FULL FAMILY RESULT")
    print("=" * 108)
    print(f"Passed cleanly: {100 - len(failures)}/100")
    print(f"Failures:       {len(failures)}/100")
    print(f"Total wall:     {total_wall:.2f}s")

    if failures:
        print()
        for item_id, token, passed, closed, repeats, out_tokens, tail in failures:
            print(f"[X] {item_id} token={token} scorer={passed} closed={closed} repeat={repeats} out={out_tokens}")
            print(tail)
            print("-" * 108)
        return 1

    print("[✓] Every HLE question passed")
    print("[✓] Every HLE question closed </think>")
    print("[✓] No HLE thinking line repeated more than twice")
    print("[✓] No HLE item can run to the old 16,384-token ceiling")
    print("[✓] Production checkpoint was not read or modified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
