"""
Token Entropy Router.
Evaluates model next-token uncertainty H(Y) to dynamically allocate compute:
- H(Y) < 0.25 (and no assertions): Instant Single Pass (N=1)
- 0.25 <= H(Y) < 0.70: Pro Search (N=8)
- H(Y) >= 0.70 (or test assertions present): Pro Parallel Search (N=16)
"""

import math
from typing import Any, List, Optional, Tuple, Union


class EntropyRouter:
    def __init__(
        self,
        low_threshold: float = 0.25,
        high_threshold: float = 0.70,
        instant_branches: int = 1,
        pro_branches_mid: int = 8,
        pro_branches_high: int = 16
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.instant_branches = instant_branches
        self.pro_branches_mid = pro_branches_mid
        self.pro_branches_high = pro_branches_high

    @staticmethod
    def compute_entropy_from_logits(logits: Any, normalize: bool = False) -> float:
        """
        Computes Shannon entropy H(P) = -sum(p * log(p)) over next-token logits.
        Supports PyTorch tensors or Python lists/arrays.
        If normalize=True, scales entropy by ln(vocab_size) to [0, 1].
        """
        try:
            import torch
            import torch.nn.functional as F

            if isinstance(logits, torch.Tensor):
                # Ensure 2D (batch_size, vocab_size) or 1D (vocab_size,)
                if logits.dim() == 3:
                    last_logits = logits[:, -1, :]
                elif logits.dim() == 2:
                    last_logits = logits[-1:, :] if logits.shape[0] > 1 else logits
                else:
                    last_logits = logits.unsqueeze(0)

                probs = F.softmax(last_logits, dim=-1)
                log_probs = F.log_softmax(last_logits, dim=-1)
                entropy = -torch.sum(probs * log_probs, dim=-1).mean().item()
                if normalize:
                    vocab_size = last_logits.shape[-1]
                    max_entropy = math.log(max(vocab_size, 2))
                    entropy = entropy / max_entropy if max_entropy > 0 else entropy
                return float(entropy)
        except ImportError:
            pass

        # Fallback pure-Python computation
        if hasattr(logits, "tolist"):
            logits = logits.tolist()

        if isinstance(logits, list) and logits and isinstance(logits[0], list):
            logits = logits[-1]

        # Numerically stable softmax
        max_val = max(logits)
        exp_vals = [math.exp(v - max_val) for v in logits]
        sum_exp = sum(exp_vals)
        if sum_exp <= 0:
            return 0.0

        probs = [e / sum_exp for e in exp_vals]
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs if p > 0)
        if normalize:
            max_entropy = math.log(max(len(logits), 2))
            entropy = entropy / max_entropy if max_entropy > 0 else entropy
        return float(entropy)

    @staticmethod
    def estimate_prompt_entropy_heuristic(prompt: str, test_cases: Optional[str] = None) -> float:
        """
        Heuristic entropy estimator for mock and CPU environments without full model passes.
        Factors in ambiguity markers, code requests, and math constraints.
        """
        if test_cases:
            return 0.85

        prompt_lower = prompt.lower()
        complexity_score = 0.15

        # Conversational / greeting triggers (ultra-low entropy)
        greetings = ["hi", "hello", "hey", "good morning", "good evening", "howdy", "sup", "what's up", "yo"]
        if prompt_lower.strip() in greetings or any(prompt_lower.startswith(g + " ") for g in greetings) or prompt_lower.startswith(("hi,", "hello,")):
            return 0.10

        # Ambiguity / high-entropy triggers
        high_entropy_keywords = [
            "regex", "regular expression", "dynamic programming", "optimize",
            "proof", "theorem", "np-hard", "combinatorics", "recursion",
            "concurrency", "async", "backtracking", "graph algorithm"
        ]
        for kw in high_entropy_keywords:
            if kw in prompt_lower:
                complexity_score += 0.35

        # Code / logic markers
        code_markers = ["def ", "class ", "function", "write a program", "implement", "script", "algorithm"]
        for marker in code_markers:
            if marker in prompt_lower:
                complexity_score += 0.20

        # Simple / factual markers (low entropy)
        simple_markers = ["what is", "capital of", "define", "who is", "translate", "summarize"]
        for marker in simple_markers:
            if prompt_lower.startswith(marker):
                complexity_score -= 0.15

        return max(0.05, min(0.95, round(complexity_score, 3)))

    def route(
        self,
        entropy: float,
        has_test_cases: bool = False
    ) -> Tuple[str, int]:
        """
        Determines execution mode and branch allocation based on entropy and constraints.
        Returns: (mode_name, branch_count)
        """
        if has_test_cases:
            return "Pro-RLVR (N=16)", self.pro_branches_high

        if entropy < self.low_threshold:
            return "Instant (N=1)", self.instant_branches
        elif entropy >= self.high_threshold:
            return "Pro-Search (N=16)", self.pro_branches_high
        else:
            return "Pro-Search (N=8)", self.pro_branches_mid
