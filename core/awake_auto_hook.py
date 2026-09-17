"""Activate awake/background learning for normal ProReasoningEngine chat paths.

The rich GUI streams through ProReasoningEngine.stream_solve(). Preserve the original
Pro contract by consulting the existing entropy router first: easy N=1 prompts keep
the newer real-time stream, while N>1 prompts delegate back to the engine's existing
solve() path (the historical Pro-routed chat path).

Context pressure is handled without silent history loss: if the complete packed prompt
would leave less room than the user-requested output cap, oldest completed dialogue
pairs are synchronously consolidated into the real trainable adapter before they are
removed from active history. The old fixed 8K asynchronous prune threshold is not used
for this chat path.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


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
        # Search a little farther only if needed to finish the oldest real pair.
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
        if not history or not prompt:
            return history

        consolidator = getattr(self, "awake_consolidator", None)
        backend = getattr(self, "mlx_backend", None)
        tokenizer = getattr(backend, "tokenizer", None)
        context_limit = _model_context_limit(self)
        if consolidator is None or tokenizer is None or context_limit is None:
            return history

        requested = max(1, int(getattr(self.settings, "max_new_tokens", 1536) or 1536))
        # A user cap larger than the model window is itself clamped by the backend.
        # Reserve as much of it as the model can possibly provide after at least one
        # prompt token; consolidation is used only to recover room occupied by old chat.
        reserve = min(requested, max(1, context_limit - 1))
        target_prompt_tokens = max(1, context_limit - reserve)
        consolidator.max_context = int(context_limit)
        consolidator.watermark_tokens = int(target_prompt_tokens)

        def packed_token_count(active_history) -> int:
            formatted = self._format_prompt_with_history(prompt, active_history)
            return len(tokenizer.encode(formatted))

        try:
            current_tokens = packed_token_count(history)
        except Exception:
            return history

        if current_tokens <= target_prompt_tokens:
            self._last_context_consolidation_note = ""
            return history

        retained = list(history)
        consolidated_turns = 0
        original_tokens = current_tokens
        while current_tokens > target_prompt_tokens:
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
                f"Context capacity: consolidated {consolidated_turns} old dialogue turns into "
                f"trainable weights before generation ({original_tokens:,} → {current_tokens:,} prompt tokens)."
            )
            return history

        self._last_context_consolidation_note = (
            f"Context capacity: packed prompt uses {original_tokens:,} tokens; no completed old dialogue "
            "pair could be safely consolidated, so the model/user cap remains authoritative."
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
        note = str(getattr(self, "_last_context_consolidation_note", "") or "")
        if note:
            metadata["context_consolidation"] = note
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
            history = _apply_awake_learning(self, history, prompt)
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
