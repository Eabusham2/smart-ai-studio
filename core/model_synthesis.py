"""Shared prompt and pairwise reduction for model-written candidate synthesis."""
from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple


def synthesis_prompt(request: str, left: str, right: str) -> str:
    """Keep the established benchmark synthesis prompt unchanged."""
    return (
        request
        + "\n\nPro synthesis step: below are two independent candidate responses generated for "
          "this exact task. Deliberate over both, reconcile disagreements, preserve the strongest "
          "reasoning from either one, and correct weaknesses you can identify from the task itself. "
          "You have NO hidden answer, reward, verifier result, pass/fail signal, or test outcome. "
          "Synthesize one best final response in the task's originally requested output format. "
          "Do not mention the candidates or the merge process.\n\n"
        + "CANDIDATE A:\n"
        + str(left)
        + "\n\nCANDIDATE B:\n"
        + str(right)
    )


def synthesize_candidates(
    branches: Sequence[Any], merge_pair: Callable[[str, str], str]
) -> Tuple[str, int]:
    """Merge adjacent pairs in rounds; carry an unpaired candidate forward."""
    contenders = [str(branch) for branch in branches]
    if not contenders:
        return "", 0
    merge_calls = 0
    while len(contenders) > 1:
        next_round = []
        for index in range(0, len(contenders), 2):
            if index + 1 >= len(contenders):
                next_round.append(contenders[index])
                continue
            next_round.append(merge_pair(contenders[index], contenders[index + 1]))
            merge_calls += 1
        contenders = next_round
    return contenders[0], merge_calls
