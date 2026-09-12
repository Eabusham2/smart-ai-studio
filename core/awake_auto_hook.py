"""Activate awake/background learning for normal ProReasoningEngine chat paths.

The rich GUI streams through ProReasoningEngine.stream_solve() and only falls back
to solve(); historically the rolling AwakeOnlineConsolidator was invoked only by
chat(), so ordinary GUI conversations could bypass automatic learning. This hook
routes any history-bearing solve/stream call through the existing consolidator
without changing UI behavior or model generation semantics.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def install_awake_auto_learning(cls) -> None:
    if getattr(cls, "_awake_auto_learning_installed", False):
        return

    original_stream_solve = cls.stream_solve
    original_solve = cls.solve

    def _apply_awake_learning(self, history):
        if not history:
            return history

        consolidator = getattr(self, "awake_consolidator", None)
        if consolidator is None:
            return history

        try:
            retained, triggered = consolidator.check_and_prune(history)
        except Exception:
            return history

        if triggered and isinstance(history, list):
            # Mutate the caller-owned list in place so the GUI's active history no
            # longer re-feeds the same evicted turns on the next message.
            history[:] = retained
            return history

        return retained

    def stream_solve_with_awake_learning(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        *args,
        **kwargs,
    ):
        history = _apply_awake_learning(self, history)
        yield from original_stream_solve(
            self,
            prompt,
            history=history,
            *args,
            **kwargs,
        )

    def solve_with_awake_learning(
        self,
        prompt: str,
        *args,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ):
        history = _apply_awake_learning(self, history)
        return original_solve(
            self,
            prompt,
            *args,
            history=history,
            **kwargs,
        )

    cls.stream_solve = stream_solve_with_awake_learning
    cls.solve = solve_with_awake_learning
    cls._awake_auto_learning_installed = True
