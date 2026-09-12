"""Dataset-level repairs applied in memory before every 4K evaluation.

The checked-in/cached LiveCodeBench-Hard fallback is synthetic. Earlier recovery
stopped it from redefining the model function, but left an assertion of only
`>= 0`, which was too weak to validate the requested algorithm. This layer keeps
the same 100 IDs while making the fallback task deterministic and genuinely
verifiable: minimum adjacent swaps to sort == inversion count.
"""
from __future__ import annotations


def _harden_lcb_fallback(splits):
    for item in splits.get("LiveCodeBench-Hard", []):
        item_id = str(item.get("id", ""))
        entry = str(item.get("entry_point", "")).strip()
        prompt = str(item.get("prompt", ""))
        if not (item_id.startswith("LCB_Hard_") and entry):
            continue

        # Only rewrite the known synthetic fallback family. Never rewrite a real
        # upstream LiveCodeBench item that happens to use this split name.
        if "minimum operations to sort array with shift step" not in prompt and "minimum adjacent swaps" not in prompt:
            continue

        item["prompt"] = (
            f"Write a Python function `{entry}(arr)` that returns the exact minimum number of "
            "adjacent swaps required to sort the integer array in nondecreasing order. "
            "This is exactly the array inversion count. Do not mutate the caller's input."
        )
        item["test"] = (
            f"assert {entry}([]) == 0\n"
            f"assert {entry}([1]) == 0\n"
            f"assert {entry}([1, 2, 3, 4]) == 0\n"
            f"assert {entry}([3, 1, 2]) == 2\n"
            f"assert {entry}([4, 3, 2, 1]) == 6\n"
            f"assert {entry}([2, 1, 1]) == 2\n"
        )
    return splits


def install(runtime_module, phase4_module) -> None:
    original = runtime_module._repair_suite
    if getattr(original, "_dataset_hardening_installed", False):
        phase4_module._repair_suite = runtime_module._repair_suite
        return

    def hardened_repair(splits):
        return _harden_lcb_fallback(original(splits))

    hardened_repair._dataset_hardening_installed = True
    runtime_module._repair_suite = hardened_repair
    # phase4_pro_rsi imported _repair_suite by value, so update its module-global
    # reference too; run_full_rsi resolves this global at execution time.
    phase4_module._repair_suite = hardened_repair
