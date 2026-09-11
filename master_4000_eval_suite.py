"""Canonical entry point for the merged 4,014-item evaluation suite."""
from eval._master_4000_base import *
from eval.master_4000_runtime import install


# Repair the recovered MATH-500 prompt text at suite-load time without changing
# item IDs, expected answers, scoring, checkpoints, or any other split.
# The prior source emitted a tab for ``\times`` and the literal text
# ``(i % 5) + 1`` instead of the evaluated exponent.
_original_load_all_4000_items = BenchmarkDatasetProvider.load_all_4000_items


def _load_all_4000_items_with_repaired_math(self):
    suite = _original_load_all_4000_items(self)
    for i, item in enumerate(suite.get("MATH-500", [])):
        item["prompt"] = rf"Compute the exact value of the modular residue: ({i * 7} \times 13^{{{(i % 5) + 1}}} + 29) \pmod{{{17 + (i % 13)}}}. Output \boxed{{answer}}."
    return suite


BenchmarkDatasetProvider.load_all_4000_items = _load_all_4000_items_with_repaired_math

install(Master4000EvaluationEngine)

if __name__ == "__main__":
    Master4000EvaluationEngine(max_duration_hours=72.0).run_full_suite()
