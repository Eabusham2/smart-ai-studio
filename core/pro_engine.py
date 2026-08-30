import math
from typing import List
from config.settings import get_settings

def get_ladder_temperatures(n_branches: int) -> List[float]:
    if n_branches <= 1:
        return [0.0]
    t_min = 0.20
    t_max = 0.88
    return [round(t_min * math.pow(t_max / t_min, i / (n_branches - 1)), 2) for i in range(n_branches)]

class EntropyRouter:
    def __init__(self):
        settings = get_settings()
        self.low_threshold = settings.entropy_low_threshold
        self.high_threshold = settings.entropy_high_threshold

    def compute_entropy(self, logprobs: List[float]) -> float:
        if not logprobs:
            return 0.0
        probs = [math.exp(lp) for lp in logprobs]
        s = sum(probs)
        if s <= 0:
            return 0.0
        norm_probs = [p / s for p in probs]
        return -sum(p * math.log2(p + 1e-12) for p in norm_probs if p > 0)

class ProReasoningEngine:
    def __init__(self):
        self.router = EntropyRouter()
