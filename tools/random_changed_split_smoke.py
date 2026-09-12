#!/usr/bin/env python3
"""Run the read-only real-model smoke only on families still changing now.

Already-verified HumanEval/GPQA/DSL behavior and intentionally pre-Learn
DialogueRecall are skipped. The wrapper reuses random_all_split_smoke so old/new
comparison, strict scoring, checkpoint read-only behavior, live logging, and one
model load remain identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import random_all_split_smoke as smoke


_FOCUS_EXACT = {
    "LiveCodeBench-Hard",
    "AIME-150",
    "MMLU-Pro-1000",
    "BFCL-200",
    "HLE-100",
    "DeepSWE-50",
    "AutonomousEvolution-200",
}


def main() -> int:
    provider_cls = smoke.suite.BenchmarkDatasetProvider
    original = provider_cls.load_all_4000_items

    def focus_only(self):
        splits = original(self)
        return {name: items for name, items in splits.items() if name in _FOCUS_EXACT}

    provider_cls.load_all_4000_items = focus_only

    print("[*] Focused changed-family mode: 7 benchmark families.")
    print("[*] Skipped: already-verified HumanEval/GPQA/DSL and pre-Learn DialogueRecall.")
    print("[*] Also skipped unchanged GSM8K/MATH/Zebra.\n")
    return smoke.main()


if __name__ == "__main__":
    raise SystemExit(main())
