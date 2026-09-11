#!/usr/bin/env python3
"""Live terminal dashboard for the last 20 completed benchmark questions."""
from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path

TELEMETRY_PATH = Path("eval_results/recent20_questions.jsonl")
REFRESH_SECONDS = 2.0


def _last_records(limit=20):
    rows = deque(maxlen=limit)
    if not TELEMETRY_PATH.exists():
        return list(rows)
    try:
        with TELEMETRY_PATH.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict):
                    rows.append(rec)
    except Exception:
        pass
    return list(rows)


def _phase_short(value):
    text = str(value or "")
    if "Phase 1" in text:
        return "P1"
    if "Phase 4" in text:
        return "P4"
    return text[:4] or "--"


def _render(rows):
    print("\033[2J\033[H", end="")
    print("=" * 116)
    print(" SMART AI STUDIO — LAST 20 COMPLETED BENCHMARK QUESTIONS")
    print("=" * 116)

    if not rows:
        print("\nWaiting for newly completed questions...\n")
        print(f"Telemetry source: {TELEMETRY_PATH}")
        return

    correct = sum(1 for r in rows if r.get("correct") is True)
    total = len(rows)
    prompt_tokens = sum(int(r.get("prompt_tokens", 0) or 0) for r in rows)
    output_tokens = sum(int(r.get("output_tokens", 0) or 0) for r in rows)
    all_tokens = prompt_tokens + output_tokens
    accuracy = 100.0 * correct / max(1, total)
    avg_tokens = all_tokens / max(1, total)

    print(
        f" Last-{total} accuracy: {correct}/{total} ({accuracy:.1f}%)"
        f"   |   Tokens: {all_tokens:,} total ({prompt_tokens:,} prompt + {output_tokens:,} output)"
        f"   |   Avg: {avg_tokens:,.0f}/question"
    )
    print("-" * 116)
    print(f" {'#':>2} {'Ph':<3} {'Split':<22} {'Question':<22} {'Prompt':>7} {'Output':>7} {'Total':>7} {'Result':>7} {'t/s':>6} {'sec':>7}")
    print("-" * 116)

    start_index = max(1, 21 - len(rows))
    for n, r in enumerate(rows, start=start_index):
        phase = _phase_short(r.get("phase"))
        split_name = str(r.get("split", ""))[:22]
        item_id = str(r.get("item_id", ""))[:22]
        p = int(r.get("prompt_tokens", 0) or 0)
        o = int(r.get("output_tokens", 0) or 0)
        t = int(r.get("total_tokens", p + o) or 0)
        result = "PASS" if r.get("correct") is True else "FAIL"
        tps = float(r.get("tok_per_sec", 0.0) or 0.0)
        sec = float(r.get("duration_seconds", 0.0) or 0.0)
        print(f" {n:>2} {phase:<3} {split_name:<22} {item_id:<22} {p:>7,} {o:>7,} {t:>7,} {result:>7} {tps:>6.1f} {sec:>7.1f}")

    print("-" * 116)
    print(" Refreshes every 2s. Ctrl+C closes only this monitor; it does NOT stop the benchmark.")


def main():
    try:
        while True:
            _render(_last_records(20))
            time.sleep(REFRESH_SECONDS)
    except KeyboardInterrupt:
        print("\nMonitor closed.")


if __name__ == "__main__":
    main()
