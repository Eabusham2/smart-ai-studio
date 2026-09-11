"""Gradient-variance adaptive hyperparameter scheduling."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence


def _flatten_numeric(values: Any) -> Iterable[float]:
    if values is None:
        return ()
    if isinstance(values, (int, float)):
        return (float(values),)
    if isinstance(values, dict):
        out = []
        for value in values.values():
            out.extend(_flatten_numeric(value))
        return out
    if isinstance(values, (list, tuple)):
        out = []
        for value in values:
            out.extend(_flatten_numeric(value))
        return out
    try:
        return [float(x) for x in values.reshape(-1).tolist()]
    except Exception:
        return ()


def gradient_variance(grads: Any) -> float:
    vals = list(_flatten_numeric(grads))
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)


@dataclass
class AdaptiveHyperparameterScheduler:
    base_learning_rate: float = 1e-4
    base_ewc_lambda: float = 400.0
    min_learning_rate: float = 1e-6
    max_learning_rate: float = 3e-4
    min_ewc_lambda: float = 50.0
    max_ewc_lambda: float = 2000.0
    base_rank: int = 4
    max_rank: int = 8

    def schedule(self, gradient_var: float, validation_curvature: float = 1.0, free_ram_gb: Optional[float] = None) -> Dict[str, float]:
        var = max(0.0, float(gradient_var))
        curvature = max(1e-6, float(validation_curvature))
        lr = self.base_learning_rate / math.sqrt(1.0 + var)
        lr = min(self.max_learning_rate, max(self.min_learning_rate, lr))
        lam = self.base_ewc_lambda * math.sqrt(curvature)
        lam = min(self.max_ewc_lambda, max(self.min_ewc_lambda, lam))
        rank = self.base_rank
        if free_ram_gb is not None and float(free_ram_gb) >= 6.0 and var < 0.25:
            rank = min(self.max_rank, max(self.base_rank, 8))
        alpha = float(rank * 2)
        return {"learning_rate": float(lr), "ewc_lambda": float(lam), "lora_rank": float(rank), "lora_alpha": alpha, "gradient_variance": var, "validation_curvature": curvature}
