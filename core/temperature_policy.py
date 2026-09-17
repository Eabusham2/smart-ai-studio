"""Temperature policy tuned for the native low-bit/ternary reasoning model.

Keep Gemini's convex Pro-search shape while separating single-pass behavior:
- chat N=1: 0.65 for natural/human-like but still controlled sampling;
- eval N=1: 0.55 for a more conservative single-pass benchmark sample;
- Pro N>1: T(i) = 0.20 + (0.95 - 0.20) * (i/(N-1))**1.35.

This module changes sampling policy only. It does not alter routing, prompts, context,
weights, KV precision, branch count, verification, scoring, or training.
"""
from __future__ import annotations

from typing import List

import numpy as np

CHAT_N1_TEMPERATURE = 0.65
EVAL_N1_TEMPERATURE = 0.55
PRO_T_MIN = 0.20
PRO_T_MAX = 0.95
PRO_GAMMA = 1.35


def pro_ladder(num_branches: int) -> List[float]:
    n = max(1, int(num_branches))
    if n == 1:
        return [CHAT_N1_TEMPERATURE]
    indices = np.arange(n)
    normalized = indices / (n - 1)
    temperatures = PRO_T_MIN + (PRO_T_MAX - PRO_T_MIN) * (normalized ** PRO_GAMMA)
    return [float(round(t, 2)) for t in temperatures]


def eval_ladder(num_branches: int) -> List[float]:
    n = max(1, int(num_branches))
    if n == 1:
        return [EVAL_N1_TEMPERATURE]
    return pro_ladder(n)


def install_chat(pro_module) -> None:
    """Replace only the temperature-ladder function used by the existing Pro engine."""
    if getattr(pro_module, "_ternary_temperature_policy_installed", False):
        return
    pro_module.get_ladder_temperatures = pro_ladder
    pro_module._ternary_temperature_policy_installed = True


def install_eval(phase4_module) -> None:
    """Give eval N=1 its conservative temperature while sharing the same Pro ladder."""
    phase4_module.get_ladder_temperatures = eval_ladder
