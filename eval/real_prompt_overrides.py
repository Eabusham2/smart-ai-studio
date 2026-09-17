"""Tiny real-benchmark prompt specializations.

Preserves the existing universal low-thinking/anti-loop system prompt. This only
adds family-specific user instructions where the recovered hardening remains
appropriate for the real replacement benchmark.
"""
from __future__ import annotations


def install(real_module) -> None:
    original = real_module._real_user_prompt

    def real_user_prompt(split, item):
        kind = item.get("_real_kind")
        if kind == "choice" and "SuperGPQA" in split:
            return (
                f"{item['prompt']}\n"
                "Map the evidence directly to the best option. Use one terse reasoning line, "
                "do not restate the question or options, then output ONLY the final option letter."
            )
        if kind == "choice" and "MMLU-Pro" in split:
            return (
                f"{item['prompt']}\n"
                "Use the stated premises and relevant domain knowledge directly. Use one concise deduction, "
                "do not repeat or re-check a settled answer, then output ONLY the final option letter."
            )
        return original(split, item)

    real_module._real_user_prompt = real_user_prompt
