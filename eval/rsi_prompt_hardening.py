"""Keep RSI on the verified Gemini base without the global anti-loop suffix."""
from __future__ import annotations

from eval.code_prompt_hardening import GLOBAL_SYSTEM_SUFFIX


def _without_global_rule(system: str) -> str:
    """Remove only the additive global anti-loop block; preserve all other text."""
    return str(system).replace(GLOBAL_SYSTEM_SUFFIX, "", 1)


def install(phase4_module) -> None:
    """Strip the global anti-loop block only for RSI chat formatting.

    Task-specific RSI instructions remain in the user prompt/system routing. All
    non-RSI generations keep the normal global anti-loop system protection.
    """
    if getattr(phase4_module, "_rsi_clean_gemini_system_installed", False):
        return

    original_chat = phase4_module._chat

    def rsi_clean_chat(tokenizer, user, system=None):
        if "Recursive Self-Improvement" in str(user):
            chosen = phase4_module.SYSTEM_PROMPT if system is None else system
            chosen = _without_global_rule(chosen)
            return original_chat(tokenizer, user, system=chosen)
        if system is None:
            return original_chat(tokenizer, user)
        return original_chat(tokenizer, user, system=system)

    phase4_module._chat = rsi_clean_chat
    phase4_module._rsi_clean_gemini_system_installed = True
