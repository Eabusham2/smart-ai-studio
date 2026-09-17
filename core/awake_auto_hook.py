"""Activate awake learning while preserving the historical Pro chat path.

The rich GUI streams through ProReasoningEngine.stream_solve(). The existing entropy
router remains authoritative: easy N=1 prompts keep real-time streaming at the old
T=0.20 anchor, while N>1 prompts delegate to the historical solve() Pro path.

Chat now has one total Context budget (prompt/history + generated tokens), not a
separate output-token allowance. At the Gemini-designed 80% context watermark, old
completed dialogue pairs are synchronously consolidated into the real trainable
adapter and are removed only after that real parameter update succeeds. Generation
then owns every token remaining in Context until natural EOS.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.platform import get_auto_context_window_size


def _model_context_limit(engine) -> Optional[int]:
    backend = getattr(engine, "mlx_backend", None)
    model = getattr(backend, "model", None)
    tokenizer = getattr(backend, "tokenizer", None)
    values = []
    for obj in (getattr(model, "args", None), getattr(model, "config", None), tokenizer):
        if obj is None:
            continue
        for name in (
            "max_position_embeddings",
            "max_seq_len",
            "max_sequence_length",
            "context_length",
            "model_max_length",
        ):
            try:
                value = int(getattr(obj, name))
            except Exception:
                continue
            if 1024 <= value <= 10_000_000:
                values.append(value)
    return min(values) if values else None


def _selected_context_budget(engine) -> int:
    """Resolve the session's one prompt+output Context budget."""
    requested = int(getattr(engine, "_context_budget_tokens", 0) or 0)
    if requested <= 0:
        requested = int(get_auto_context_window_size())
        engine._context_budget_tokens = requested

    physical = _model_context_limit(engine)
    effective = min(requested, int(physical)) if physical else requested
    effective = max(1024, int(effective))
    engine._effective_context_budget_tokens = effective

    backend = getattr(engine, "mlx_backend", None)
    if backend is not None:
        # Preserve the user's requested value; MLX's context-budget layer clamps it
        # against the physical model window when calculating generation room.
        backend.context_budget_tokens = max(1024, int(requested))
    return effective


def _oldest_completed_chunk(history: List[Dict[str, str]], ratio: float) -> tuple[list, list]:
    """Take an oldest prefix ending on assistant, while always retaining recent turns."""
    if len(history) <= 2:
        return [], history
    target = max(2, int(len(history) * max(0.05, min(float(ratio), 0.80))))
    target = min(target, len(history) - 1)

    last_assistant = -1
    saw_user = False
    for idx, message in enumerate(history[:target]):
        role = str(message.get("role", "")).strip().lower()
        if role == "user":
            saw_user = True
        elif role == "assistant" and saw_user:
            last_assistant = idx
    if last_assistant < 0:
        for idx, message in enumerate(history[target:-1], start=target):
            role = str(message.get("role", "")).strip().lower()
            if role == "user":
                saw_user = True
            elif role == "assistant" and saw_user:
                last_assistant = idx
                break
    if last_assistant < 0:
        return [], history
    return history[: last_assistant + 1], history[last_assistant + 1 :]


def install_awake_auto_learning(cls) -> None:
    if getattr(cls, "_awake_auto_learning_installed", False):
        return

    original_stream_solve = cls.stream_solve
    original_solve = cls.solve

    def _apply_awake_learning(self, history, prompt: Optional[str] = None):
        if not prompt:
            return history

        context_budget = _selected_context_budget(self)
        consolidator = getattr(self, "awake_consolidator", None)
        backend = getattr(self, "mlx_backend", None)
        tokenizer = getattr(backend, "tokenizer", None)
        if consolidator is None or tokenizer is None:
            return history

        # Gemini's rolling-memory design: start learning old turns before the hard
        # boundary, at 80% of total Context. After triggering, reduce the active
        # textual working set toward 60% so the answer has substantial free room.
        trigger_tokens = max(1, int(context_budget * 0.80))
        target_tokens = max(1, int(context_budget * 0.60))
        consolidator.max_context = int(context_budget)
        consolidator.watermark_tokens = int(trigger_tokens)

        def packed_token_count(active_history) -> int:
            formatted = self._format_prompt_with_history(prompt, active_history)
            return len(tokenizer.encode(formatted))

        try:
            current_tokens = packed_token_count(history or [])
        except Exception:
            return history

        if current_tokens < trigger_tokens:
            self._last_context_consolidation_note = (
                f"Context {context_budget:,}: prompt/history {current_tokens:,}; "
                f"generation may use the remaining {max(0, context_budget-current_tokens):,} tokens until EOS."
            )
            return history

        if not history:
            self._last_context_consolidation_note = (
                f"Context {context_budget:,}: current prompt alone uses {current_tokens:,}; "
                "there is no older completed dialogue to consolidate."
            )
            return history

        retained = list(history)
        consolidated_turns = 0
        original_tokens = current_tokens
        while current_tokens > target_tokens:
            chunk, candidate_retained = _oldest_completed_chunk(
                retained,
                getattr(consolidator, "evict_ratio", 0.40),
            )
            if not chunk:
                break
            try:
                learned = bool(consolidator.consolidate_chunk_sync(chunk))
            except Exception:
                learned = False
            if not learned:
                # Never remove dialogue that was not successfully consolidated.
                break
            consolidated_turns += len(chunk)
            retained = candidate_retained
            try:
                current_tokens = packed_token_count(retained)
            except Exception:
                break

        if consolidated_turns and isinstance(history, list):
            history[:] = retained
            self._last_context_consolidation_note = (
                f"Context {context_budget:,}: consolidated {consolidated_turns} old dialogue turns into "
                f"trainable weights ({original_tokens:,} → {current_tokens:,} active prompt tokens); "
                f"generation may use all {max(0, context_budget-current_tokens):,} remaining tokens until EOS."
            )
            return history

        self._last_context_consolidation_note = (
            f"Context {context_budget:,}: active prompt uses {original_tokens:,} tokens; no completed old dialogue "
            "pair could be safely consolidated, so the physical Context boundary remains authoritative."
        )
        return history

    def stream_solve_with_awake_learning(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.75,
        top_p: float = 0.92,
        cancel_event: Optional[Any] = None,
    ):
        history = _apply_awake_learning(self, history, prompt)
        self._last_stream_pro_meta = None

        # Restore the historical chat decision point: entropy chooses N=1/8/16.
        entropy = float(self.calculate_token_entropy(prompt))
        mode, branch_count = self.router.route(entropy, has_test_cases=False)

        if int(branch_count) <= 1:
            # Historical good solve()-routed N=1 uses get_ladder_temperatures(1)=[0.20].
            yield from original_stream_solve(
                self,
                prompt,
                history=history,
                temperature=0.20,
                top_p=top_p,
                cancel_event=cancel_event,
            )
            return

        # N>1 delegates to the original Pro engine: same entropy router, branch
        # generator, convex temperature ladder, verifier/consensus and metadata.
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
        note = str(getattr(self, "_last_context_consolidation_note", "") or "")
        if note:
            metadata["context_consolidation"] = note
        self._last_stream_pro_meta = metadata

        # Pro must finish its search before one final answer exists. Preserve the
        # newer GUI renderer by yielding that completed answer in small chunks.
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
            history = _apply_awake_learning(self, history, prompt)
        else:
            _selected_context_budget(self)
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
