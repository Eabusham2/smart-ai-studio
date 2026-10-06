"""Allow real multiple-choice benchmarks with A-J options.

MMLU-Pro and SuperGPQA can expose more than four choices. This broadens the
strict final-choice parser while preserving the reader-hardening invariant:
only an explicit final-line choice is accepted; arbitrary prose is never scanned
for a stray option letter.
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

        # Bare final label: D, (D), D., D), D:
        match = re.fullmatch(
            r"\s*(?:\*\*)?\(?([A-Ja-j])\)?[\).:]?(?:\*\*)?\s*",
            line,
        )
        if match:
            return match.group(1).upper()

        # Explicit labeled option with descriptive text: (D) text / D. text / D) text.
        match = re.fullmatch(r"\s*\(([A-Ja-j])\)\s+\S.*", line)
        if match:
            return match.group(1).upper()
        match = re.fullmatch(r"\s*([A-Ja-j])[\).:]\s+\S.*", line)
        if match:
            return match.group(1).upper()

        # Option D / Option D: text / Option D - text, anchored to the whole final line.
        match = re.fullmatch(
            r"(?i)\s*option\s+([A-J])(?:\s*(?:[:\-]\s*\S.*))?\s*",
            line,
        )
        if match:
            return match.group(1).upper()

        # Explicit answer/choice/result labels only when the whole final line is the answer.
        match = re.fullmatch(
            r"(?i)\s*(?:final\s+answer|answer|choice|result)\s*(?:is\s*)?"
            r"[:=\-]?\s*\(?([A-J])\)?(?:[\).:]?)\s*",
            line,
        )
        return match.group(1).upper() if match else None

    scoring_module._final_choice = final_choice
