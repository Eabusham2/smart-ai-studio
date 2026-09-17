"""Temperature policy tuned for the native low-bit/ternary reasoning model.

Single-pass policy:
- chat N=1: 0.65 for natural/human-like but still controlled sampling;
- eval N=1: 0.55 for a more conservative benchmark sample.

Pro N>1 keeps the established Gemini convex search shape:
- Tmin = 0.20, Tmax = 0.95, gamma = 1.35;
- one additional fixed T=0.65 candidate is always generated on top of the routed N.

Gamma stays fixed because there is no benchmark evidence that dynamically changing the
curve improves this model. The fixed 0.65 branch has a separate purpose: it guarantees
one balanced mid-temperature candidate regardless of routed branch count or rounding.

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
PRO_GAMMA = 1.35


def pro_gamma(num_branches: int) -> float:
    """Compatibility/telemetry helper: Pro uses one fixed calibrated gamma."""
    return PRO_GAMMA


def _routed_pro_ladder(num_branches: int) -> List[float]:
    n = max(2, int(num_branches))
    indices = np.arange(n)
    normalized = indices / (n - 1)
    temperatures = PRO_T_MIN + (PRO_T_MAX - PRO_T_MIN) * (normalized ** PRO_GAMMA)
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
    # A naturally occurring 0.65 inside the routed curve does not replace the
    # dedicated extra branch. Only an already-appended final 0.65 suppresses another.
    if abs(temps[-1] - FIXED_PRO_EXTRA_TEMPERATURE) >= 1e-9:
        temps.append(FIXED_PRO_EXTRA_TEMPERATURE)
    return temps


def install_chat(pro_module) -> None:
    """Patch only temperature scheduling around the existing historical Pro engine."""
    if getattr(pro_module, "_ternary_temperature_policy_installed", False):
        return

    cls = pro_module.ProReasoningEngine
    original_generate = cls.generate_parallel_branches
    original_solve = cls.solve

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

    def solve_with_actual_candidate_metadata(self, *args, **kwargs):
        response, metadata = original_solve(self, *args, **kwargs)
        metadata = dict(metadata or {})
        candidates = list(metadata.get("raw_branches") or [])
        temps = list(metadata.get("temp_ladder") or [])
        actual_n = len(candidates)
        if actual_n > 1 and len(temps) == actual_n:
            routed_n = max(1, actual_n - 1)
            metadata["routed_branch_count"] = routed_n
            metadata["branch_count"] = actual_n
            metadata["fixed_extra_temperature"] = FIXED_PRO_EXTRA_TEMPERATURE
            metadata["pro_gamma"] = PRO_GAMMA
            metadata["mode"] = f"Pro Search (N={routed_n}+1 fixed)"

            # The historical verified-branch surprise formula normalized by routed N.
            # Re-normalize only that metadata score to the actual candidate pool so an
            # extra-branch win can never exceed the intended 0..1-ish range.
            if bool(metadata.get("verified")):
                try:
                    idx = int(metadata.get("winning_branch", 0) or 0)
                    entropy = float(metadata.get("entropy", 0.0) or 0.0)
                    if idx == 0:
                        surprise = 0.10 + 0.10 * entropy
                    else:
                        surprise = 0.50 + 0.40 * (idx / max(1, actual_n - 1)) + 0.10 * entropy
                    metadata["surprise_score"] = round(min(1.0, max(0.0, surprise)), 4)
                except Exception:
                    pass
        return response, metadata

    pro_module.get_ladder_temperatures = pro_ladder
    cls.generate_parallel_branches = generate_with_fixed_anchor
    cls.solve = solve_with_actual_candidate_metadata
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
