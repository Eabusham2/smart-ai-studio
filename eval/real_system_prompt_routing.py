"""Route existing benchmark system hardening to real benchmark adapters.

This does not define new prompt text. It reuses code_prompt_hardening's existing
system routing so real benchmark wrappers cannot bypass the universal anti-loop
rule or the still-applicable family-specific suffixes.
"""
from __future__ import annotations


def install(prompt_module, phase4_module, cls) -> None:
    original_eval = cls._evaluate_single_item
    original_task_prompt = phase4_module._task_user_prompt
    base_system = phase4_module.SYSTEM_PROMPT

    def evaluate(self, split, item):
        if not item.get("_real_kind"):
            return original_eval(self, split, item)
        previous = phase4_module.SYSTEM_PROMPT
        phase4_module.SYSTEM_PROMPT = prompt_module._system_for_split(split, base_system)
        try:
            return original_eval(self, split, item)
        finally:
            phase4_module.SYSTEM_PROMPT = previous

    def task_prompt(split, item):
        if item.get("_real_kind") and getattr(
            phase4_module, "_task_system_routing_active", False
        ):
            phase4_module.SYSTEM_PROMPT = prompt_module._system_for_split(
                split, base_system
            )
        return original_task_prompt(split, item)

    cls._evaluate_single_item = evaluate
    phase4_module._task_user_prompt = task_prompt
