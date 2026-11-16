"""Deterministic, model-free budget choice for the desktop Best-of-N mode.

A simple chat must never pay for a hidden entropy forward pass. Complex user
tasks may opt into the existing verified N+1 Pro search. No prompt is sent
off-device, and selecting the budget does not invoke an inference backend.
"""
import re

_COMPLEX = re.compile(
    r"\b(?:implement|refactor|debug|investigate|benchmark|optimi[sz]e|"
    r"design|architect|prove|verify|research|compare|analy[sz]e|"
    r"build|develop|integrat(?:e|ion)|test(?:ing)?|"
    r"multiple approaches|step[- ]by[- ]step|edge cases|"
    r"root cause|trade[- ]offs|codebase|repository|ci|deploy)\b",
    re.I,
)
_CONJUNCTION = re.compile(r"\b(?:and|then|also|after|before|while)\b", re.I)
_FAST = re.compile(r"\b(?:single pass|no branches|without branches|instant only|just answer|quick answer)\b", re.I)


def select_branch_budget(prompt: str, *, manual: bool = False, auto: bool = True,
                         requested: int = 4) -> int:
    """Return 1, 4, 8 or 16, using only supplied prompt and explicit UI flags."""
    chosen = requested if type(requested) is int and requested in (4, 8, 16) else 4
    if manual:
        return chosen
    if not auto or _FAST.search(prompt or ""):
        return 1
    words = re.findall(r"\b[\w'-]+\b", prompt or "")
    n = len(words)
    flags = len(_COMPLEX.findall(prompt or ""))
    clauses = len(_CONJUNCTION.findall(prompt or ""))
    # A single technical word in a normal question is not enough to schedule
    # expensive search. Complexity needs either an extended prompt or multiple
    # explicit work/verification requirements.
    complex_task = (n >= 105 and flags >= 1) or (n >= 45 and flags >= 2 and clauses >= 1) or flags >= 4
    if not complex_task:
        return 1
    return 8 if n >= 180 and flags >= 3 else chosen
