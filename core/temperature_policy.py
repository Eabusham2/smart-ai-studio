"""Temperature policy tuned for the native low-bit/ternary reasoning model.

Single-pass policy:
- chat N=1: 0.65 for natural/human-like but still controlled sampling;
- eval N=1: 0.55 for a more conservative benchmark sample.

Pro N>1 keeps the Gemini-style convex search shape but improves it dynamically:
- Tmin = 0.20, Tmax = 0.95;
- gamma rises smoothly with routed branch count (~1.26 at N=8, 1.40 at N=16),
  keeping larger searches conservative through more of the pool while preserving a
  high-diversity tail;
- one additional fixed T=0.65 candidate is always generated on top of the routed N.

This module changes sampling/search policy only. It does not alter entropy routing,
prompts, Context, weights, KV precision, verification, scoring, or training.
"""
from __future__ import annotations

from typing import List

import numpy as np

CHAT_N1_TEMPERATURE = 0.65
EVAL_N1_TEMPERATURE = 0.55
FIXED_PRO_EXTRA_TEMPERATURE = 0.65
PRO_T_MIN = 0.20
PRO_T_MAX = 0.95
PRO_GAMMA_MIN = 1.15
PRO_GAMMA_MAX = 1.40


def pro_gamma(num_branches: int) -> float:
    """More routed branches => slightly more convex/cautious temperature spacing."""
    n = max(2, int(num_branches))
    progress = min(1.0, max(0.0, (n - 2) / 14.0))
    return PRO_GAMMA_MIN + (PRO_GAMMA_MAX - PRO_GAMMA_MIN) * progress


def _routed_pro_ladder(num_branches: int) -> List[float]:
    n = max(2, int(num_branches))
    gamma = pro_gamma(n)
    indices = np.arange(n)
    normalized = indices / (n - 1)
    temperatures = PRO_T_MIN + (PRO_T_MAX - PRO_T_MIN) * (normalized ** gamma)
    return [float(round(t, 2)) for t in temperatures]


def pro_ladder(num_branches: int) -> List[float]:
    """Return routed temperatures plus one always-extra T=0.65 Pro candidate."""
    n = max(1, int(num_branches))
    if n == 1:
        return [CHAT_N1_TEMPERATURE]
    return _routed_pro_ladder(n) + [FIXED_PRO_EXTRA_TEMPERATURE]


def eval_ladder(num_branches: int) -> List[float]:
    n = max(1, int(num_branches))
    if n == 1:
        return [EVAL_N1_TEMPERATURE]
    return pro_ladder(n)


def _ensure_extra_anchor(temperatures) -> List[float]:
    temps = [float(t) for t in temperatures]
    if len(temps) <= 1:
        return temps
    if not any(abs(t - FIXED_PRO_EXTRA_TEMPERATURE) < 1e-9 for t in temps):
        temps.append(FIXED_PRO_EXTRA_TEMPERATURE)
    return temps


def install_chat(pro_module) -> None:
    """Patch only temperature scheduling around the existing historical Pro engine."""
    if getattr(pro_module, "_ternary_temperature_policy_installed", False):
        return

    cls = pro_module.ProReasoningEngine
    original_generate = cls.generate_parallel_branches

    def generate_with_fixed_anchor(
        self,
        prompt: str,
        branch_count: int = 16,
        history=None,
        temperatures=None,
    ):
        n = max(1, int(branch_count))
        if n <= 1:
            temps = list(temperatures) if temperatures is not None else [CHAT_N1_TEMPERATURE]
            return original_generate(
                self,
                prompt,
                branch_count=1,
                history=history,
                temperatures=temps[:1],
            )

        # solve() receives N+1 temperatures from pro_ladder(): N routed curve values
        # followed by the fixed 0.65 extra candidate. Generate the historical routed N
        # exactly as before, then add one independent extra branch so N=16 really yields 17.
        scheduled = list(temperatures) if temperatures is not None else pro_ladder(n)
        if len(scheduled) >= n + 1:
            routed = scheduled[:n]
            extra_temp = float(scheduled[n])
        else:
            routed = _routed_pro_ladder(n)
            extra_temp = FIXED_PRO_EXTRA_TEMPERATURE

        branches = original_generate(
            self,
            prompt,
            branch_count=n,
            history=history,
            temperatures=routed,
        )
        extra = original_generate(
            self,
            prompt,
            branch_count=1,
            history=history,
            temperatures=[extra_temp],
        )
        return list(branches or []) + list(extra or [])

    pro_module.get_ladder_temperatures = pro_ladder
    cls.generate_parallel_branches = generate_with_fixed_anchor
    pro_module._ternary_temperature_policy_installed = True


def install_eval(phase4_module) -> None:
    """Use eval N=1=0.55 and add the same fixed 0.65 branch to any multi-branch eval search."""
    if getattr(phase4_module, "_ternary_eval_temperature_policy_installed", False):
        return

    original_generate = phase4_module._generate_branches_same_model

    def generate_with_fixed_anchor(
        self,
        formatted_prompt: str,
        temperatures,
        max_tokens: int,
        top_p: float = 0.92,
    ):
        temps = _ensure_extra_anchor(temperatures)
        return original_generate(
            self,
            formatted_prompt,
            temps,
            max_tokens=max_tokens,
            top_p=top_p,
        )

    phase4_module.get_ladder_temperatures = eval_ladder
    phase4_module._generate_branches_same_model = generate_with_fixed_anchor
    phase4_module._ternary_eval_temperature_policy_installed = True
