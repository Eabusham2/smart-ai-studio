"""Keep public replacement benchmark names honest in logs/checkpoints.

Only renames split keys after the real public datasets are loaded. It does not
change evaluation order, counts, stage transitions, prompts, scoring, or training.
"""
from __future__ import annotations

_LABELS = {
    "AIME-150": "OlympiadBench-150",
    "GPQA-400": "SuperGPQA-400",
    "ZebraLogic-200": "LogiQA-200",
    "HLE-100": "LiveBench-Reasoning-100",
    # SWE-bench Verified is a strong 32K-compatible replacement. It is NOT DeepSWE.
    "DeepSWE-50": "SWE-bench-Verified-50",
}


def install(real_module) -> None:
    original = real_module.load_real_4000_suite

    def renamed(cache_dir="eval_datasets"):
        suite = original(cache_dir)
        for old, new in _LABELS.items():
            if old in suite:
                suite[new] = suite.pop(old)
        return suite

    real_module.load_real_4000_suite = renamed
    counts = {
        _LABELS.get(name, name): count
        for name, count in real_module.REAL_SPLIT_COUNTS.items()
    }
    real_module.REAL_SPLIT_COUNTS = counts
    real_module._PUBLIC_KEYS = tuple(counts)
