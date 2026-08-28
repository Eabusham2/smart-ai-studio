"""
Elastic Weight Consolidation (EWC) Loss Penalty.
Computes the quadratic penalty constraining weight updates against critical anchor parameters:
L_EWC = (λ / 2) * Σ F_i * (θ_i - θ_anchor)^2
"""

from typing import Any, Dict, Optional


class EWCLossCalculator:
    def __init__(self, lambda_ewc: float = 400.0, device: str = "cpu"):
        self.lambda_ewc = lambda_ewc
        self.device = device

    def calculate_penalty(
        self,
        named_parameters: Any,
        fisher_matrix: Dict[str, Any],
        anchor_weights: Dict[str, Any]
    ) -> Any:
        """
        Computes EWC quadratic loss penalty across trainable parameters.
        Returns a scalar loss tensor.
        """
        try:
            import torch

            penalty = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            if not fisher_matrix or not anchor_weights:
                return penalty

            for name, param in named_parameters:
                if name in fisher_matrix and name in anchor_weights:
                    f_diag = fisher_matrix[name]
                    theta_anchor = anchor_weights[name]

                    # Convert to tensor if passed as list
                    p_tensor = torch.as_tensor(param, device=self.device, dtype=torch.float32) if not isinstance(param, torch.Tensor) else param
                    f_tensor = torch.as_tensor(f_diag, device=self.device, dtype=torch.float32) if not isinstance(f_diag, torch.Tensor) else f_diag
                    a_tensor = torch.as_tensor(theta_anchor, device=self.device, dtype=torch.float32) if not isinstance(theta_anchor, torch.Tensor) else theta_anchor

                    if f_tensor.shape == p_tensor.shape and a_tensor.shape == p_tensor.shape:
                        diff_sq = (p_tensor - a_tensor) ** 2
                        penalty = penalty + (f_tensor * diff_sq).sum()

            return (self.lambda_ewc / 2.0) * penalty

        except ImportError:
            # Fallback pure-Python scalar computation for mock testing
            penalty = 0.0
            for name, param in named_parameters:
                if name in fisher_matrix and name in anchor_weights:
                    f_vals = fisher_matrix[name]
                    anchor_vals = anchor_weights[name]
                    if isinstance(f_vals, list) and isinstance(anchor_vals, list) and isinstance(param, list):
                        for f, p, a in zip(f_vals, param, anchor_vals):
                            penalty += f * ((p - a) ** 2)
            return (self.lambda_ewc / 2.0) * penalty

    def combine_losses(
        self,
        task_loss: Any,
        ewc_loss: Any
    ) -> Any:
        """Combines task next-token prediction loss with EWC regularization penalty."""
        return task_loss + ewc_loss


# Alias for backward compatibility
EWCLoss = EWCLossCalculator
