"""Allow real multiple-choice benchmarks with A-J options.

MMLU-Pro and SuperGPQA can expose more than four choices. This only broadens the
existing final-choice parser; it does not change any stage or training behavior.
"""
from __future__ import annotations

import re


def install(scoring_module) -> None:
    def final_choice(text: str):
        boxed = scoring_module._boxed_values(text)
        for value in reversed(boxed):
            match = re.fullmatch(r"\s*\(?([A-Ja-j])\)?\s*", value)
            if match:
                return match.group(1).upper()

        line = scoring_module._last_nonempty_line(text)
        match = re.fullmatch(
            r"\s*(?:\*\*)?\(?([A-Ja-j])\)?[\).:]?(?:\*\*)?\s*",
            line,
        )
        if match:
            return match.group(1).upper()

        matches = re.findall(
            r"(?i)(?:final\s+answer|answer|choice|option)\s*(?:is\s*)?[:=\-]?\s*\(?([A-J])\)?\b",
            line,
        )
        return matches[-1].upper() if matches else None

    scoring_module._final_choice = final_choice
