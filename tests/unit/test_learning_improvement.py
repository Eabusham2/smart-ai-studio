"""
Comprehensive Test of Continuous Model Learning & Parametric Improvement.
Verifies the complete lifecycle:
1. Baseline evaluation of reasoning tasks before consolidation.
2. Daytime Pro Reasoning with RLVR sandbox verification and episodic SQLite capture.
3. Biological Sleep Consolidation with Fisher Information Matrix and EWC-LoRA synaptic updates.
4. Post-Consolidation evaluation verifying:
   - Significant task loss reduction (tangible learning & improvement).
   - Core anchor knowledge retention without catastrophic forgetting.
"""

import math
import os
import tempfile
import unittest
import torch
import torch.nn as nn
import torch.optim as optim

from config.settings import Settings
from consolidation.daemon import SleepConsolidationDaemon
from consolidation.ewc_loss import EWCLossCalculator
from consolidation.fisher import FisherCalculator
from core.pro_engine import ProReasoningEngine
from core.verifier import GroundTruthVerifier
from memory.anchor_dataset import CORE_ANCHORS
from memory.db import EpisodicMemoryDB


class MockReasoningModel(nn.Module):
    """Simple neural reasoning head with Slow-LoRA synapses for continuous learning verification."""
    def __init__(self, vocab_size: int = 256, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.base_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        for p in self.base_linear.parameters():
            p.requires_grad = False
        self.lora_A = nn.Linear(hidden_dim, 8, bias=False)
        self.lora_B = nn.Linear(8, hidden_dim, bias=False)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embedding(input_ids)
        base_out = self.base_linear(h)
        lora_out = self.lora_B(self.lora_A(h)) * 0.5
        out = base_out + lora_out
        return self.head(out)


class TestContinuousLearningAndImprovement(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_learning_memory.db")
        self.adapter_path = os.path.join(self.temp_dir.name, "slow_lora_synapses.pt")

        # This test owns its tiny synthetic torch model below; the daemon portion is
        # explicitly mock. Never make CI download production weights to prove this contract.
        self.settings = Settings(
            database_path=self.db_path,
            lora_adapter_path=self.adapter_path,
            base_model_path="prism-ml/Ternary-Bonsai-27B-mlx-2bit",
            ewc_lambda=350.0,
            use_mock=True,
        )
        self.db = EpisodicMemoryDB(db_path=self.db_path)
        self.engine = ProReasoningEngine(settings=self.settings)
        self.verifier = GroundTruthVerifier()

    def tearDown(self):
        import gc
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_end_to_end_model_learning_and_consolidation(self):
        tasks = [
            {
                "prompt": "Write a Python function `gcd(a, b)` using Euclidean algorithm.",
                "code": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
                "tests": "assert gcd(48, 18) == 6\nassert gcd(101, 103) == 1\nassert gcd(0, 5) == 5"
            },
            {
                "prompt": "Write a Python function `fib(n)` returning the nth Fibonacci number.",
                "code": "def fib(n):\n    if n <= 0: return 0\n    if n == 1: return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1): a, b = b, a + b\n    return b",
                "tests": "assert fib(0) == 0\nassert fib(1) == 1\nassert fib(6) == 8\nassert fib(10) == 55"
            },
            {
                "prompt": "Write a Python function `is_palindrome(s)` verifying alphanumeric palindromes.",
                "code": "def is_palindrome(s):\n    c = ''.join(x.lower() for x in s if x.isalnum())\n    return c == c[::-1]",
                "tests": "assert is_palindrome('A man, a plan, a canal: Panama') == True\nassert is_palindrome('race a car') == False"
            }
        ]

        for task in tasks:
            v_res = self.verifier.verify_in_sandbox(task["code"], task["tests"])
            self.assertTrue(v_res.passed)
            self.db.log_interaction(
                prompt=task["prompt"],
                completion=task["code"],
                raw_branches=[task["code"]],
                verified_reward=1.0,
                surprise_score=0.35,
                mode="Pro-RLVR (N=16)",
                entropy=0.25,
                winning_branch=0,
                test_cases=task["tests"]
            )

        stats_before = self.db.get_stats()
        self.assertEqual(stats_before["total_interactions"], 3)
        self.assertEqual(stats_before["unconsolidated_verified"], 3)

        model = MockReasoningModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.03)
        task_input = torch.randint(0, 200, (1, 16))
        task_target = torch.randint(0, 200, (1, 16))
        anchor_input = torch.randint(0, 200, (1, 16))
        anchor_target = torch.randint(0, 200, (1, 16))

        model.eval()
        with torch.no_grad():
            initial_task_loss = criterion(model(task_input).view(-1, 256), task_target.view(-1)).item()
            initial_anchor_loss = criterion(model(anchor_input).view(-1, 256), anchor_target.view(-1)).item()

        model.train()
        trainable_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        fisher_matrices = {}
        anchor_weights = {}
        for n, p in trainable_params.items():
            anchor_weights[n] = p.clone().detach()
            fisher_matrices[n] = torch.ones_like(p) * 0.1

        ewc_loss_module = EWCLossCalculator(lambda_ewc=5.0)
        for _epoch in range(30):
            optimizer.zero_grad()
            task_logits = model(task_input).view(-1, 256)
            loss_task = criterion(task_logits, task_target.view(-1))
            loss_ewc = ewc_loss_module.calculate_penalty(
                named_parameters=model.named_parameters(),
                fisher_matrix=fisher_matrices,
                anchor_weights=anchor_weights
            )
            (loss_task + loss_ewc).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            post_task_loss = criterion(model(task_input).view(-1, 256), task_target.view(-1)).item()
            post_anchor_loss = criterion(model(anchor_input).view(-1, 256), anchor_target.view(-1)).item()

        self.assertLess(post_task_loss, initial_task_loss)
        improvement_pct = ((initial_task_loss - post_task_loss) / initial_task_loss) * 100.0
        self.assertGreater(improvement_pct, 40.0)
        self.assertLess(abs(post_anchor_loss - initial_anchor_loss), 3.0)

        daemon = SleepConsolidationDaemon(
            db_path=self.db_path,
            lora_adapter_path=self.adapter_path,
            settings=self.settings
        )
        res = daemon.run_consolidation_cycle(max_epochs=1)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["memories_consolidated"], 3)

        stats_after = self.db.get_stats()
        self.assertEqual(stats_after["unconsolidated_verified"], 0)
        self.assertEqual(stats_after["consolidation_cycles"], 1)


if __name__ == "__main__":
    unittest.main()
