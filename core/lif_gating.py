import math
from dataclasses import dataclass, field
from typing import List, Tuple


def convex_temperature_ladder(
    num_branches: int,
    t_min: float = 0.20,
    t_max: float = 0.88,
) -> List[float]:
    """Pure compatibility helper; active chat/eval temperature policy stays authoritative."""
    n = max(1, int(num_branches))
    if n == 1:
        return [0.0]
    if t_min <= 0 or t_max <= 0:
        raise ValueError("temperature bounds must be positive")
    ratio = float(t_max) / float(t_min)
    return [
        float(round(float(t_min) * (ratio ** (i / (n - 1))), 4))
        for i in range(n)
    ]


def normalized_shannon_from_probabilities(probabilities) -> Tuple[float, float]:
    values = [max(0.0, float(x)) for x in probabilities]
    total = sum(values)
    if total <= 0:
        return 0.0, 0.0
    probs = [x / total for x in values if x > 0]
    raw = -sum(x * math.log2(max(x, 1e-12)) for x in probs)
    denom = math.log2(max(2, len(probs)))
    return raw, max(0.0, min(1.0, raw / denom))


def topk_shannon_from_logits(logits, top_k: int = 40) -> Tuple[float, float]:
    """Diagnostic entropy helper; it does not change the active router."""
    try:
        import mlx.core as mx
        flat = logits.reshape(-1).astype(mx.float32)
        k = max(2, min(int(top_k), int(flat.shape[0])))
        idx = mx.argpartition(-flat, kth=k - 1)[:k]
        probs = mx.softmax(flat[idx], axis=-1)
        raw_arr = -mx.sum(probs * mx.log2(mx.clip(probs, 1e-12, 1.0)))
        mx.eval(raw_arr)
        raw = float(raw_arr.item())
        return raw, max(0.0, min(1.0, raw / math.log2(k)))
    except Exception:
        pass

    try:
        values = sorted((float(x) for x in logits), reverse=True)
    except Exception:
        return 0.0, 0.0
    values = values[: max(2, min(int(top_k), len(values)))]
    if not values:
        return 0.0, 0.0
    vmax = max(values)
    exps = [math.exp(v - vmax) for v in values]
    total = sum(exps)
    return normalized_shannon_from_probabilities([x / total for x in exps])

@dataclass
class LIFNeuronState:
    v_mem: float = 0.0
    v_thresh: float = 0.55
    v_rest: float = 0.0
    beta: float = 0.85
    spike_history: List[int] = field(default_factory=list)

    def step(self, entropy_current: float) -> Tuple[int, float]:
        self.v_mem = (self.beta * self.v_mem) + ((1.0 - self.beta) * entropy_current)
        if self.v_mem >= self.v_thresh:
            spike = 1
            self.v_mem = self.v_rest
        else:
            spike = 0
        self.spike_history.append(spike)
        return spike, self.v_mem

    def determine_branch_budget(self, entropy: float) -> Tuple[int, List[float], int]:
        spike, _ = self.step(entropy)
        if spike == 1 or entropy >= 0.65:
            branch_count = 4
            ladder = [0.20, 0.40, 0.65, 0.88]
        elif entropy >= 0.30:
            branch_count = 2
            ladder = [0.20, 0.45]
        else:
            branch_count = 1
            ladder = [0.0]
        return branch_count, ladder, spike

    @property
    def spike_count(self) -> int:
        return int(sum(self.spike_history))
