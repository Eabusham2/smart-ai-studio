"""Enforce one total prompt+output Context budget across the evaluation lifecycle.

Normal public benchmark stages use the already-configured benchmark context ceiling
(32K after the real-data adapter). Flagship DeepSWE uses its configured 226K context.
The historical decoder temperatures, prompts, scoring, RSI rounds, Pro routing,
training and verification remain untouched. This layer removes only artificial
output-only caps: generation may use every token left after the formatted prompt
until natural EOS.
"""
from __future__ import annotations

from typing import Optional


def _physical_context(engine) -> Optional[int]:
    model = getattr(engine, "model", None)
    tokenizer = getattr(engine, "tokenizer", None)
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


def _remaining(self, tokenizer, context_budget: int, prompt: str) -> int:
    prompt_tokens = len(tokenizer.encode(prompt))
    physical = _physical_context(self.engine)
    effective = min(int(context_budget), int(physical)) if physical else int(context_budget)
    remaining = effective - prompt_tokens
    if remaining <= 0:
        raise RuntimeError(
            f"Formatted prompt requires {prompt_tokens:,} tokens but Context is {effective:,}; "
            "refusing truncation/compaction."
        )
    self.last_context_budget_tokens = effective
    self.last_prompt_tokens = prompt_tokens
    return int(remaining)


def install(runtime_module, phase4_module, deepswe_module, cls) -> None:
    if getattr(cls, "_unified_eval_context_budget_installed", False):
        return

    original_fast = cls._fast_generate
    original_branch_generate = phase4_module._generate_branches_same_model

    def context_fast_generate(self, prompt, max_tokens=16384, stream=False):
        # ``max_tokens`` remains accepted for compatibility, but the benchmark's one
        # total Context budget is authoritative. Natural EOS normally ends far sooner.
        ceiling = int(runtime_module._benchmark_ceiling(self))
        remaining = _remaining(self, self.engine.tokenizer, ceiling, prompt)
        return original_fast(self, prompt, max_tokens=remaining, stream=stream)

    def context_branch_generate(
        self,
        formatted_prompt: str,
        temperatures,
        max_tokens: int,
        top_p: float = 0.92,
    ):
        flagship = int(getattr(self, "_deepswe_flagship_context_tokens", 0) or 0)
        ceiling = flagship if flagship > 0 else int(phase4_module._benchmark_ceiling(self))
        remaining = _remaining(self, self.engine.tokenizer, ceiling, formatted_prompt)
        return original_branch_generate(
            self,
            formatted_prompt,
            temperatures,
            max_tokens=remaining,
            top_p=top_p,
        )

    cls._fast_generate = context_fast_generate
    runtime_module.fast_generate = context_fast_generate
    phase4_module._generate_branches_same_model = context_branch_generate

    # DeepSWE previously had 226K total context plus a separate 8K output ceiling.
    # Make its legacy output variable equal the total context so the existing bridge
    # naturally resolves max generation to (Context - prompt tokens).
    deepswe_module.DEEPSWE_MAX_OUTPUT_TOKENS = int(deepswe_module.DEEPSWE_CONTEXT_TOKENS)

    bridge_cls = getattr(deepswe_module, "_LocalModelBridge", None)
    if bridge_cls is not None and not getattr(bridge_cls, "_unified_context_installed", False):
        original_start = bridge_cls.start
        original_stop = bridge_cls.stop

        def start_with_context(self):
            self.owner._deepswe_flagship_context_tokens = int(deepswe_module.DEEPSWE_CONTEXT_TOKENS)
            return original_start(self)

        def stop_with_context(self):
            try:
                return original_stop(self)
            finally:
                try:
                    delattr(self.owner, "_deepswe_flagship_context_tokens")
                except Exception:
                    pass

        bridge_cls.start = start_with_context
        bridge_cls.stop = stop_with_context
        bridge_cls._unified_context_installed = True

    cls._unified_eval_context_budget_installed = True
