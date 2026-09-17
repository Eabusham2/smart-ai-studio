"""Preserve complete requested chat history until the real model context boundary.

The historical Pro prompt packer reduced the usable history window based on transient
free-RAM thresholds and then silently kept only the newest turns. That changes the
information available to the model. This hardening keeps the same chat template and
filters only the existing synthetic warning messages, but no longer drops legitimate
history for memory pressure. The MLX backend enforces the actual model/user output
cap and fails visibly if the packed prompt itself cannot fit.
"""
from __future__ import annotations


def install(cls) -> None:
    if getattr(cls, "_full_context_hardening_installed", False):
        return

    def full_history_prompt(self, prompt, history=None):
        tok = getattr(self.mlx_backend, "tokenizer", None) or getattr(self, "tokenizer", None)
        messages = []
        for turn in (history or []):
            role = str(turn.get("role", "user"))
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            # Preserve the old protection against feeding product warning UI text back
            # into the model. This is not user conversation content.
            if "weights are not currently loaded" in content or "Click '⬇️ Download" in content:
                continue
            messages.append({"role": role, "content": content})
        if not messages or messages[-1].get("content") != prompt:
            messages.append({"role": "user", "content": prompt})

        if tok is not None and hasattr(tok, "apply_chat_template"):
            try:
                return tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        packed = []
        for item in messages:
            packed.append(
                f"<|im_start|>{item.get('role', 'user')}\n"
                f"{item.get('content', '')}<|im_end|>\n"
            )
        packed.append("<|im_start|>assistant\n")
        return "".join(packed)

    cls._format_prompt_with_history = full_history_prompt
    cls._full_context_hardening_installed = True
