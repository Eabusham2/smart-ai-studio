"""Narrow reader fixes for outputs that are clearly formatted but were rejected.

This does not relax grading to substring matching. It only accepts a final explicit
multiple-choice label such as `(D) Asymptotic` or `D. Asymptotic`, which is an
unambiguous final choice.
"""
from __future__ import annotations

import re
from typing import Optional


def install(scoring_module) -> None:
    original = scoring_module._final_choice
    if getattr(original, "_explicit_label_hardened", False):
        return

    def final_choice(text: str) -> Optional[str]:
        value = original(text)
        if value is not None:
            return value

        line = scoring_module._last_nonempty_line(text)

        # Explicit parenthesized label followed by descriptive option text.
        m = re.fullmatch(r"\s*\(([A-Da-d])\)\s+\S.*", line)
        if m:
            return m.group(1).upper()

        # Explicit `D. text`, `D) text`, or `D: text` final-choice forms.
        m = re.fullmatch(r"\s*([A-Da-d])[\).:]\s+\S.*", line)
        if m:
            return m.group(1).upper()

        # Explicit `Option D ...` form.
        m = re.fullmatch(r"(?i)\s*option\s+([A-D])(?:\s*[:\-])?\s+\S.*", line)
        return m.group(1).upper() if m else None

    final_choice._explicit_label_hardened = True
    scoring_module._final_choice = final_choice
