import math
from dataclasses import dataclass, field
from typing import List, Tuple

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
