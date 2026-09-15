"""Run RSI with no system message; keep task guidance in the user prompt only."""
from __future__ import annotations


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
    """Remove the system role entirely for Recursive Self-Improvement generations.

    RSI still receives its benchmark/task-specific instructions through the user
    prompt. Non-RSI generation keeps the normal Gemini/global/task system routing.
    """
    if getattr(phase4_module, "_rsi_no_system_installed", False):
        return

    original_chat = phase4_module._chat

    def rsi_no_system_chat(tokenizer, user, system=None):
        if "Recursive Self-Improvement" in str(user):
            return _chat_without_system(tokenizer, user)
        if system is None:
            return original_chat(tokenizer, user)
        return original_chat(tokenizer, user, system=system)

    phase4_module._chat = rsi_no_system_chat
    phase4_module._rsi_no_system_installed = True
