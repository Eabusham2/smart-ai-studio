"""Group-relative policy updates on complete, bounded sampled continuations."""
from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from numbers import Real

from core.learning_data import IGNORE_INDEX, TrainingWindow, encode_text, format_training_row


@dataclass(frozen=True)
class RewardGroup:
    windows: tuple[TrainingWindow, ...]
    advantages: tuple[float, ...]
    rewards: tuple[float, ...]
    diagnostics: dict

    @property
    def target_count(self):
        return sum(window.supervised_tokens for window in self.windows)


def _token_ids(values):
    if not isinstance(values, (list, tuple)) or not values or any(type(i) is not int or i < 0 for i in values):
        raise ValueError("Rollout IDs must be a nonempty sequence of token IDs")
    return tuple(values)


def reward_group(tokenizer, rows, max_tokens, *, reward_weighting="group_relative", entropic_kl_target=None):
    """Center rewards within one prompt and mask every conditioning token.

    Prefer the actual sampled IDs when supplied. Text-only providers must
    provide a separable boundary: training an invented prompt-ending token
    as part of the sampled response would bias the policy gradient.
    """
    diagnostics = {}
    rewards, advantages = reward_advantages(rows, reward_weighting=reward_weighting,
                                            entropic_kl_target=entropic_kl_target, diagnostics=diagnostics)
    windows = []
    conditioning = None
    for index, row in enumerate(rows):
        if "rollout_prompt_ids" in row or "rollout_completion_ids" in row:
            prefix = _token_ids(row.get("rollout_prompt_ids"))
            actions = _token_ids(row.get("rollout_completion_ids"))
        else:
            text = row.get("rollout_prefix")
            if text is None:
                _, text = format_training_row(tokenizer, row)
            answer = row.get("completion")
            if not isinstance(text, str) or not text or not isinstance(answer, str) or not answer.strip():
                raise ValueError("Reward updates require a nonempty generation prefix and sampled completion")
            prefix = tuple(encode_text(tokenizer, text))
            full = tuple(encode_text(tokenizer, text + answer))
            if not prefix or full[:len(prefix)] != prefix or len(full) <= len(prefix):
                raise ValueError("Cannot verify sampled completion boundary; provide original rollout token IDs")
            actions = full[len(prefix):]
        if conditioning is not None and prefix != conditioning:
            raise ValueError("Reward group contains different conditioning prompts")
        conditioning = prefix
        ids = prefix + actions
        if len(ids) > max_tokens:
            raise ValueError("The complete sampled rollout exceeds the policy update token budget")
        labels = (IGNORE_INDEX,) * len(prefix) + actions
        windows.append(TrainingWindow(ids, labels, index, 0, len(ids)))
    return RewardGroup(tuple(windows), advantages, rewards, diagnostics)


def _entropic_advantages(rewards, gamma):
    """Solve the empirical exponential tilt in Appendix A.1, arXiv:2601.16175."""
    if gamma is None:
        gamma = math.log(2)
    if isinstance(gamma, bool) or not isinstance(gamma, Real) or not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("Entropic KL target must be finite and positive")
    maximum = max(rewards)
    spread = maximum - min(rewards)
    if spread == 0:
        raise ValueError("Uniform rollout rewards supply no policy gradient")
    # Solve in reward-range units. Adding a constant or scaling rewards leaves
    # this problem unchanged, even for differences too small to square safely.
    gaps = [(maximum - r) / spread for r in rewards]
    count = len(gaps)
    limit_kl = math.log(count / gaps.count(0.0))

    def tilt(scale):
        weights = [math.exp(-scale * gap) for gap in gaps]
        total = math.fsum(weights)
        probabilities = [value / total for value in weights]
        kl = math.fsum(p * math.log(count * p) for p in probabilities if p > 0)
        return max(0.0, kl), weights

    nearest = min(gap for gap in gaps if gap > 0)
    # At this finite scaled bound nonmaximal mass is below machine precision.
    cap = 64.0 / nearest
    target = min(gamma, limit_kl)
    low, high = 0.0, min(1.0, cap)
    while high < cap and tilt(high)[0] < target:
        low, high = high, min(high * 2, cap)
    if tilt(high)[0] >= target:
        for _ in range(64):
            middle = low + (high - low) * .5
            if tilt(middle)[0] < target:
                low = middle
            else:
                high = middle
    achieved, weights = tilt(high)
    epsilon = 1e-8
    # Summing peers directly avoids cancellation for a single dominant winner.
    advantages = []
    for i, weight in enumerate(weights):
        peers = math.fsum(weights[:i] + weights[i + 1:]) / (count - 1)
        advantages.append(weight / (peers + epsilon) - 1.0)
    if max(advantages) <= 0:
        raise ValueError("Entropic KL target yields no resolvable positive advantage")
    beta = high / spread
    metrics = {"entropic_beta": min(beta, sys.float_info.max),
               "entropic_beta_clamped": not math.isfinite(beta), "entropic_beta_times_range": high,
               "entropic_kl_target": float(gamma), "entropic_kl_achieved": achieved,
               "entropic_kl_saturated": gamma >= limit_kl or achieved + 1e-12 < target,
               "entropic_loo_epsilon": epsilon}
    return tuple(advantages), metrics


def reward_advantages(rows, *, reward_weighting="group_relative", entropic_kl_target=None, diagnostics=None):
    if not 2 <= len(rows) <= 16:
        raise ValueError("A reward group requires between two and sixteen rollouts")
    rewards = tuple(row.get("reward") for row in rows)
    if any(isinstance(score, bool) or not isinstance(score, Real) or
           not math.isfinite(score) or not 0 <= score <= 1 for score in rewards):
        raise ValueError("Each rollout requires a finite numeric reward in [0, 1]")
    rewards = tuple(float(value) for value in rewards)
    if reward_weighting == "entropic":
        advantages, metrics = _entropic_advantages(rewards, entropic_kl_target)
    elif reward_weighting == "group_relative":
        mean = math.fsum(rewards) / len(rewards)
        centered = [value - mean for value in rewards]
        std = math.sqrt(math.fsum(value * value for value in centered) / len(centered))
        if std <= 1e-8:
            raise ValueError("Uniform rollout rewards supply no policy gradient")
        advantages, metrics = tuple(value / std for value in centered), {}
    else:
        raise ValueError("Unknown reward weighting; use group_relative or entropic")
    if diagnostics is not None:
        diagnostics.update(metrics, reward_weighting=reward_weighting, advantages=list(advantages))
    return rewards, advantages


def validation_contrast_weights(advantages):
    """Zero-sum likelihood contrast, independent of a common change in all losses."""
    mean = math.fsum(advantages) / len(advantages)
    centered = tuple(value - mean for value in advantages)
    largest = max(map(abs, centered))
    if not math.isfinite(largest) or not largest:
        raise ValueError("Reward validation requires a finite nonzero likelihood contrast")
    return tuple(value / largest for value in centered)


def validation_gain_tolerance(before, after):
    """Allow one float32 scoring unit at the scale of the completion NLLs."""
    return 2.0 ** -23 * max(1.0, *(abs(value) for value in (*before, *after)))


def policy_advantages(group, rows, current_nlls, coefficient):
    """Apply the optional, detached initial-policy penalty on full trajectories."""
    if (isinstance(coefficient, bool) or not isinstance(coefficient, Real) or
            not math.isfinite(coefficient) or coefficient < 0):
        raise ValueError("Initial-reference KL coefficient must be finite and nonnegative")
    diagnostics = {**group.diagnostics, "reference_kl_coef": float(coefficient),
                   "reference_logprob_source": "disabled"}
    advantages = group.advantages
    if coefficient:
        initial = [row.get("reference_logprob") for row in rows]
        if any("rollout_prompt_ids" not in row or "rollout_completion_ids" not in row for row in rows):
            raise ValueError("Initial-reference KL requires original prompt and sampled completion IDs")
        if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) or x > 0 for x in initial):
            raise ValueError("Each rollout requires its frozen initial-policy trajectory reference_logprob")
        # log(pi_current / pi_initial) is -NLL_current - log(pi_initial).
        log_ratios = [-nll - logp for nll, logp in zip(current_nlls, initial)]
        advantages = tuple(a - coefficient * ratio for a, ratio in zip(advantages, log_ratios))
        if not all(math.isfinite(a) for a in advantages):
            raise ValueError("Non-finite initial-reference shaped advantage")
        diagnostics.update(reference_logprob_source="supplied_initial_trajectory_logprobs",
                           sampled_initial_log_ratios=log_ratios, shaped_advantages=list(advantages))
    # Extreme LOO weights (e.g., one winner in a two-sample group) can reach
    # 1/epsilon. Scale the whole group uniformly before mixed-precision backward.
    # Preserve the exact LOO estimates above and the gradient direction.
    divisor = max(1.0, max(map(abs, advantages)) / 1024.0)
    diagnostics["policy_loss_divisor"] = divisor
    return tuple(a / divisor for a in advantages), diagnostics


def causal_row_weights(rows, weights):
    values = [row.get("weight", 1.0) for row in rows] if weights is None else list(weights)
    if (len(values) != len(rows) or not values or
            any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v < 0 for v in values) or
            not any(values)):
        raise ValueError("Causal learning needs one finite nonnegative weight per row and at least one positive weight")
    return values
