"""Tiny real-benchmark prompt specializations.

Preserves the existing universal low-thinking/anti-loop system prompt. This only
adds family-specific user instructions where the recovered hardening remains
appropriate for the real replacement benchmark. Existing prompt text is reused
rather than copied or rewritten.
"""
from __future__ import annotations


def install(real_module) -> None:
    from eval import code_prompt_hardening as prompt_module
    from eval import phase4_pro_rsi as phase4_module
    from eval._master_4000_base import Master4000EvaluationEngine
    from eval.real_system_prompt_routing import install as install_system_routing

    original = real_module._real_user_prompt

    def real_user_prompt(split, item):
        kind = item.get("_real_kind")
        if kind == "livecodebench":
            return prompt_module._lcb_user(item)
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

    # real_benchmark_runtime.install() is called after this function by the canonical
    # entry point. Wrap that installer so the prompt-system router is attached after
    # the real evaluator wrappers exist. Stage order/RSI logic remain untouched.
    original_install = real_module.install

    def install_with_prompt_routing(provider_cls, runtime_module, phase4, cls):
        original_install(provider_cls, runtime_module, phase4, cls)
        install_system_routing(prompt_module, phase4_module, Master4000EvaluationEngine)

    real_module.install = install_with_prompt_routing
