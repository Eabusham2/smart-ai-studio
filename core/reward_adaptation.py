"""Group-relative policy updates on complete, bounded sampled continuations."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from core.learning_data import IGNORE_INDEX, TrainingWindow, encode_text, format_training_row


@dataclass(frozen=True)
class RewardGroup:
    windows: tuple[TrainingWindow, ...]
    advantages: tuple[float, ...]
    rewards: tuple[float, ...]

    @property
    def target_count(self):
        return sum(window.supervised_tokens for window in self.windows)


def _token_ids(values):
    if not isinstance(values, (list, tuple)) or not values or any(type(i) is not int or i < 0 for i in values):
        raise ValueError("Rollout IDs must be a nonempty sequence of token IDs")
    return tuple(values)


def reward_group(tokenizer, rows, max_tokens):
    """Center rewards within one prompt and mask every conditioning token.

    Prefer the actual sampled IDs when supplied. Text-only providers must
    provide a separable boundary: training an invented prompt-ending token
    as part of the sampled response would bias the policy gradient.
    """
    rewards, advantages = reward_advantages(rows)
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
    return RewardGroup(tuple(windows), advantages, rewards)


def reward_advantages(rows):
    if not 2 <= len(rows) <= 16:
        raise ValueError("A reward group requires between two and sixteen rollouts")
    rewards = tuple(row.get("reward") for row in rows)
    if any(isinstance(score, bool) or not isinstance(score, Real) or
           not math.isfinite(score) or not 0 <= score <= 1 for score in rewards):
        raise ValueError("Each rollout requires a finite numeric reward in [0, 1]")
    mean = math.fsum(rewards) / len(rewards)
    centered = [value - mean for value in rewards]
    std = math.sqrt(math.fsum(value * value for value in centered) / len(centered))
    if std <= 1e-8:
        raise ValueError("Uniform rollout rewards supply no policy gradient")
    return rewards, tuple(value / std for value in centered)


def causal_row_weights(rows, weights):
    values = [row.get("weight", 1.0) for row in rows] if weights is None else list(weights)
    if (len(values) != len(rows) or not values or
            any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v < 0 for v in values) or
            not any(values)):
        raise ValueError("Causal learning needs one finite nonnegative weight per row and at least one positive weight")
    return values
