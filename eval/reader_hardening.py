"""Narrow reader fixes for outputs that are clearly formatted but were rejected.

This does not relax grading to substring matching. It accepts only an explicit
final multiple-choice label on the final non-empty line, such as `D`,
`(D) Asymptotic`, `D. Asymptotic`, or `Answer: D`.
"""
from __future__ import annotations

import re
from typing import Optional


def install(scoring_module) -> None:
    original = scoring_module._final_choice
    if getattr(original, "_explicit_label_hardened", False):
        return

    def final_choice(text: str) -> Optional[str]:
        # Boxed answers remain explicit final answers.
        boxed_values = getattr(scoring_module, "_boxed_values", lambda _text: [])(text or "")
        for value in reversed(boxed_values):
            m = re.fullmatch(r"\s*\(?([A-Da-d])\)?\s*", value)
            if m:
                return m.group(1).upper()

        line = scoring_module._last_nonempty_line(text)

        # Bare final label: D, (D), D., D), D:, optionally markdown-bolded.
        m = re.fullmatch(
            r"\s*(?:\*\*)?\(?([A-Da-d])\)?[\).:]?(?:\*\*)?\s*",
            line,
        )
        if m:
            return m.group(1).upper()

        # Parenthesized label followed by descriptive option text.
        m = re.fullmatch(r"\s*\(([A-Da-d])\)\s+\S.*", line)
        if m:
            return m.group(1).upper()

        # `D. text`, `D) text`, or `D: text` final-choice forms.
        m = re.fullmatch(r"\s*([A-Da-d])[\).:]\s+\S.*", line)
        if m:
            return m.group(1).upper()

        # `Option D`, `Option D: text`, or `Option D - text` only when the
        # entire final line is itself the answer. Never search arbitrary prose.
        m = re.fullmatch(
            r"(?i)\s*option\s+([A-D])(?:\s*(?:[:\-]\s*\S.*))?\s*",
            line,
        )
        if m:
            return m.group(1).upper()

        # Explicit answer/choice/result labels anchored to the whole final line.
        m = re.fullmatch(
            r"(?i)\s*(?:final\s+answer|answer|choice|result)\s*(?:is\s*)?"
            r"[:=\-]?\s*\(?([A-D])\)?(?:[\).:]?)\s*",
            line,
        )
        return m.group(1).upper() if m else None

    final_choice._explicit_label_hardened = True
    scoring_module._final_choice = final_choice
