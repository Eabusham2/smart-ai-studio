"""Activate awake/background learning for normal ProReasoningEngine chat paths.

The rich GUI streams through ProReasoningEngine.stream_solve(). Preserve the original
Pro contract by consulting the existing entropy router first: easy N=1 prompts keep
the newer real-time stream, while N>1 prompts delegate back to the engine's existing
solve() path (the historical Pro-routed chat path). Awake learning still wraps both
paths without changing the Pro router, temperature ladder, branch generator, or model.
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
        temperature: float = 0.75,
        top_p: float = 0.92,
        cancel_event: Optional[Any] = None,
    ):
        history = _apply_awake_learning(self, history)

        # Restore the historical chat decision point: the existing entropy router
        # decides whether this is Instant N=1 or Pro N=8/N=16 before generation.
        # Temperature is still the branch-diversity ladder inside Pro; it does not
        # replace the entropy thresholds that decide the compute budget.
        entropy = float(self.calculate_token_entropy(prompt))
        mode, branch_count = self.router.route(entropy, has_test_cases=False)

        if int(branch_count) <= 1:
            # Keep the later real-time streaming UX but preserve the historical
            # Instant-path anchor temperature from get_ladder_temperatures(1).
            yield from original_stream_solve(
                self,
                prompt,
                history=history,
                temperature=0.20,
                top_p=top_p,
                cancel_event=cancel_event,
            )
            return

        # Hard prompts use the same historical solve() Pro path. self.solve is looked
        # up at call time so later runtime-hardening wrappers still apply around the
        # original engine rather than creating a second Pro engine.
        self._awake_stream_history_prepared = True
        try:
            response, metadata = self.solve(
                prompt,
                history=history,
                cancel_event=cancel_event,
                force_branch_count=int(branch_count),
                temperature=None,
            )
        finally:
            self._awake_stream_history_prepared = False

        metadata = dict(metadata or {})
        metadata["entropy"] = entropy
        metadata["mode"] = mode
        self._last_stream_pro_meta = metadata

        # Pro must finish its parallel reasoning/selection before one final answer exists.
        # Feed that completed answer through the GUI's existing stream renderer in chunks.
        text = str(response or "")
        step = 64
        for idx in range(0, len(text), step):
            if cancel_event and cancel_event.is_set():
                break
            yield text[idx : idx + step]

    def solve_with_awake_learning(
        self,
        prompt: str,
        test_cases: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        cancel_event: Optional[Any] = None,
        force_branch_count: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        if not getattr(self, "_awake_stream_history_prepared", False):
            history = _apply_awake_learning(self, history)
        return original_solve(
            self,
            prompt,
            test_cases=test_cases,
            history=history,
            cancel_event=cancel_event,
            force_branch_count=force_branch_count,
            temperature=temperature,
        )

    cls.stream_solve = stream_solve_with_awake_learning
    cls.solve = solve_with_awake_learning
    cls._awake_auto_learning_installed = True
