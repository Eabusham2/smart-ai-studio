"""Run RSI without a system message while preserving family-specific guidance."""
from __future__ import annotations

from eval.code_prompt_hardening import SYSTEM_SUFFIXES


RSI_REPEAT_GUARD = (
    "Continue while your reasoning is making new progress. Once the solution is determined, "
    "if the same point or check repeats without new information, stop the repetition and output the answer."
)


def _family_guidance_from_system(system: str | None) -> str:
    text = str(system or "")
    for suffix in SYSTEM_SUFFIXES.values():
        if suffix and suffix in text:
            return suffix.strip()
    return ""


def _chat_without_system(tokenizer, user: str) -> str:
    messages = [{"role": "user", "content": str(user)}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def install(phase4_module) -> None:
    """Remove RSI's system role, retain family guidance, and stop settled repeat loops.

    The verified Gemini base and global anti-loop suffix are omitted only for
    Recursive Self-Improvement generations. Any family-specific system guidance
    already selected for the split is copied into the RSI user message instead.
    RSI keeps its normal reasoning budget; the repeat guard explicitly allows
    continued reasoning while it is making new progress and only stops cycles
    after the solution is determined. Non-RSI generation is unchanged.
    """
    if getattr(phase4_module, "_rsi_no_system_installed", False):
        return

    original_chat = phase4_module._chat

    def rsi_no_system_chat(tokenizer, user, system=None):
        if "Recursive Self-Improvement" in str(user):
            family_guidance = _family_guidance_from_system(system)
            rsi_user = str(user)
            if family_guidance and family_guidance not in rsi_user:
                rsi_user += "\n\n" + family_guidance
            if RSI_REPEAT_GUARD not in rsi_user:
                rsi_user += "\n\n" + RSI_REPEAT_GUARD
            return _chat_without_system(tokenizer, rsi_user)
        if system is None:
            return original_chat(tokenizer, user)
        return original_chat(tokenizer, user, system=system)

    phase4_module._chat = rsi_no_system_chat
    phase4_module._rsi_no_system_installed = True
