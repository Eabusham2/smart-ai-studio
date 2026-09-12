#!/usr/bin/env python3
"""Run the read-only real-model smoke only on the one behavior still failing.

Current evidence already cleared LCB, AIME, DeepSWE and AutonomousEvolution.
DialogueRecall is intentionally not meaningful before Learn/RSI/Phase-3 and is
therefore excluded from this pre-Learn smoke. Reuse the common smoke harness so
strict scoring, checkpoint read-only behavior, live logging and one model load
remain identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import random_all_split_smoke as smoke


_FOCUS_EXACT = {"GPQA-400"}


def main() -> int:
    provider_cls = smoke.suite.BenchmarkDatasetProvider
    original = provider_cls.load_all_4000_items

    def focus_only(self):
        splits = original(self)
        return {name: items for name, items in splits.items() if name in _FOCUS_EXACT}

    provider_cls.load_all_4000_items = focus_only

    print("[*] Final bad-family mode: GPQA only.")
    print("[*] LCB/AIME/DeepSWE/AutonomousEvolution already cleared by real-model smoke.")
    print("[*] DialogueRecall intentionally deferred until after Learn/RSI/Phase-3.\n")
    return smoke.main()


if __name__ == "__main__":
    raise SystemExit(main())
