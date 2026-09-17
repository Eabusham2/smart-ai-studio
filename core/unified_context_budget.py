"""One total Context budget for native MLX chat/Pro generation.

The legacy Pro engine passes ``max_new_tokens`` to backends. For the native MLX path
we intentionally reinterpret that value only as a compatibility fallback: the active
chat session owns one total context budget (prompt/history + generated tokens).
Generation may consume every token remaining in that budget until natural EOS.
There is no second artificial output-token cap.
"""
from __future__ import annotations

from typing import Optional


def _model_context_limit(model, tokenizer) -> Optional[int]:
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


def install(cls) -> None:
    if getattr(cls, "_unified_context_budget_installed", False):
        return

    def remaining_context_for_generation(self, prompt: str, requested: int) -> int:
        try:
            prompt_tokens = len(self.tokenizer.encode(prompt))
        except Exception:
            prompt_tokens = 0

        selected = int(getattr(self, "context_budget_tokens", 0) or 0)
        if selected <= 0:
            # Compatibility fallback for callers that have not yet installed a
            # session Context value. The old argument is treated as total context,
            # never as an additional output-only allowance.
            selected = max(1, int(requested))

        physical = _model_context_limit(self.model, self.tokenizer)
        context_budget = min(selected, int(physical)) if physical else selected
        if context_budget <= 0:
            raise RuntimeError("Context budget must be positive")

        remaining = int(context_budget) - int(prompt_tokens)
        self.last_model_context_limit = physical
        self.last_context_budget_tokens = int(context_budget)
        self.last_effective_max_tokens = max(0, int(remaining))

        if remaining <= 0:
            self.last_generation_cap_reason = (
                f"Context full: packed prompt uses {prompt_tokens:,} tokens but active Context is "
                f"{context_budget:,}. Old completed dialogue must consolidate before generation."
            )
            raise RuntimeError(self.last_generation_cap_reason)

        if physical and selected > physical:
            self.last_generation_cap_reason = (
                f"Context {selected:,} requested; model physically exposes {physical:,}. "
                f"Prompt uses {prompt_tokens:,}; generation may use all {remaining:,} remaining tokens until EOS."
            )
        else:
            self.last_generation_cap_reason = (
                f"Context {context_budget:,} total = prompt/history + output. Prompt uses {prompt_tokens:,}; "
                f"generation may use all {remaining:,} remaining tokens until EOS."
            )
        return max(1, int(remaining))

    cls._effective_generation_cap = remaining_context_for_generation
    cls._unified_context_budget_installed = True
