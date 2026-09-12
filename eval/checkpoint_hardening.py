"""Checkpoint integrity hardening for recovered benchmark state.

Legacy completion markers can say an item once finished, but if the actual boolean
PASS/FAIL value is gone there is nothing trustworthy to score. Such marker-only
entries must be rerun instead of silently skipped.
"""
from __future__ import annotations


def install(runtime_module) -> None:
    def completed(cache, key):
        return isinstance(cache.get(key), bool)

    runtime_module._completed = completed
