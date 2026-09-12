#!/usr/bin/env python3
"""Run the existing read-only smoke harness on changed benchmark families only.

This wrapper deliberately reuses tools/random_all_split_smoke.py so old/new
comparison, strict scoring, checkpoint read-only behavior, live logging, and
model loading remain identical. It only filters the provider to families whose
prompt/reader/verifier behavior changed on code-low-think-hardening.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import random_all_split_smoke as smoke


_CHANGED_EXACT = {
    "HumanEval-164",
    "LiveCodeBench-Hard",
    "AIME-150",
    "GPQA-400",
    "MMLU-Pro-1000",
    "BFCL-200",
    "HLE-100",
    "DeepSWE-50",
    "TensorGraphDSL-300",
    "AutonomousEvolution-200",
    "DialogueRecall-150",
}


def main() -> int:
    provider_cls = smoke.suite.BenchmarkDatasetProvider
    original = provider_cls.load_all_4000_items

    def changed_only(self):
        splits = original(self)
        return {name: items for name, items in splits.items() if name in _CHANGED_EXACT}

    provider_cls.load_all_4000_items = changed_only

    print("[*] Changed-family-only mode: 11 benchmark families.")
    print("[*] Unchanged good families (GSM8K/MATH/Zebra) are skipped.\n")
    return smoke.main()


if __name__ == "__main__":
    raise SystemExit(main())
