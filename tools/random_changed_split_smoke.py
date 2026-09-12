#!/usr/bin/env python3
"""Run the read-only real-model smoke only on representative changed behaviors.

The focused gate deliberately skips already-proven families to save 27B inference
time. It reuses random_all_split_smoke so old/new comparison, strict scoring,
checkpoint read-only behavior, live logging, and one model load remain identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import random_all_split_smoke as smoke


_FOCUS_EXACT = {
    "LiveCodeBench-Hard",       # code-generation overthinking
    "AIME-150",                # long arithmetic/recheck loop
    "GPQA-400",                # direct choice + strict reader
    "DeepSWE-50",              # repo-repair reasoning/verifier
    "AutonomousEvolution-200", # symbolic derivation overthinking
    "DialogueRecall-150",      # fast pre-Learn miss / post-Learn recall policy
}


def main() -> int:
    provider_cls = smoke.suite.BenchmarkDatasetProvider
    original = provider_cls.load_all_4000_items

    def focus_only(self):
        splits = original(self)
        return {name: items for name, items in splits.items() if name in _FOCUS_EXACT}

    provider_cls.load_all_4000_items = focus_only

    print("[*] Focused representative mode: 6 benchmark families.")
    print("[*] Testing: LCB, AIME, GPQA, DeepSWE, AutonomousEvolution, DialogueRecall.")
    print("[*] Skipped: proven-good HumanEval/GSM8K/MATH/Zebra/BFCL/MMLU/HLE/DSL.\n")
    return smoke.main()


if __name__ == "__main__":
    raise SystemExit(main())
