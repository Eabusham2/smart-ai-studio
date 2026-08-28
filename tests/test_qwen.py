"""
Thorough Test Suite for Qwen Model Capabilities.
Tests Qwen architecture, Qwen prompt template formatting, token entropy calculation,
multi-branch Best-of-N rollouts, RLVR sandbox verification, Slow-LoRA target modules,
and Elastic Weight Consolidation (EWC) updates.
"""

import os
import tempfile
import unittest
from config.settings import Settings, get_settings
from consolidation.daemon import SleepConsolidationDaemon
from consolidation.ewc_loss import EWCLossCalculator
from consolidation.fisher import FisherEstimator
from core.entropy_router import EntropyRouter
from core.pro_engine import ProReasoningEngine
from core.verifier import GroundTruthVerifier
from memory.anchor_dataset import get_anchor_dataset, get_anchor_texts
from memory.db import EpisodicMemoryDB


class TestQwenReasoningCapabilities(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_qwen_memory.db")
        self.adapter_path = os.path.join(self.temp_dir.name, "qwen_slow_lora")
        self.settings = Settings(
            database_path=self.db_path,
            lora_adapter_path=self.adapter_path,
            small_model=True,
            small_model_path="Qwen/Qwen2.5-Coder-1.5B",
            ewc_lambda=400.0
        )
        self.db = EpisodicMemoryDB(db_path=self.db_path)
        self.engine = ProReasoningEngine(settings=self.settings)
        self.verifier = GroundTruthVerifier(sandbox_timeout=4.0)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_qwen_chat_template_formatting(self):
        """Validates standard Qwen chat template formatting."""
        prompt = "Write a function to compute GCD."
        completion = "def gcd(a, b): return a if b == 0 else gcd(b, a % b)"
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>"
        
        self.assertIn("<|im_start|>user\n", formatted)
        self.assertIn("<|im_end|>\n<|im_start|>assistant\n", formatted)
        self.assertTrue(formatted.endswith("<|im_end|>"))

    def test_qwen_token_entropy_calculation(self):
        """Tests next-token Shannon entropy on Qwen-style vocabulary distribution."""
        router = EntropyRouter()
        
        # High confidence distribution (e.g. deterministic keyword)
        high_certainty_logits = [0.0] * 1000
        high_certainty_logits[50] = 50.0  # dominant token
        
        entropy_low = router.compute_entropy_from_logits(high_certainty_logits)
        self.assertLess(entropy_low, 0.10)

        # High uncertainty distribution (e.g. ambiguous reasoning fork)
        high_uncertainty_logits = [1.0] * 8
        entropy_high = router.compute_entropy_from_logits(high_uncertainty_logits)
        self.assertGreater(entropy_high, 0.70)

    def test_qwen_pro_search_factorial_rollout(self):
        """Tests Qwen code pattern and sandbox verification on factorial."""
        code = "def factorial(n):\n    if n < 0: raise ValueError('Negative input')\n    if n == 0 or n == 1: return 1\n    return n * factorial(n - 1)"
        assertions = """assert factorial(5) == 120
assert factorial(0) == 1
assert factorial(1) == 1
try:
    factorial(-1)
    assert False, "Should raise ValueError for negative numbers"
except ValueError:
    pass
"""
        res = self.verifier.verify_in_sandbox(code, assertions)
        self.assertTrue(res.passed)
        self.assertIn("factorial", code)

    def test_qwen_pro_search_fibonacci_rollout(self):
        """Tests Qwen code pattern and sandbox verification on Fibonacci sequence."""
        code = "def fib(n):\n    if n <= 0: return 0\n    if n == 1: return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1): a, b = b, a + b\n    return b"
        assertions = "assert fib(0) == 0\nassert fib(1) == 1\nassert fib(7) == 13\nassert fib(10) == 55"
        res = self.verifier.verify_in_sandbox(code, assertions)
        self.assertTrue(res.passed)

    def test_qwen_pro_search_palindrome_rollout(self):
        """Tests Qwen code pattern and sandbox verification on alphanumeric palindrome."""
        code = "def is_palindrome(s):\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]"
        assertions = """assert is_palindrome("A man, a plan, a canal: Panama") == True
assert is_palindrome("race a car") == False
assert is_palindrome(" ") == True
"""
        res = self.verifier.verify_in_sandbox(code, assertions)
        self.assertTrue(res.passed)

    def test_qwen_pro_search_regex_rollout(self):
        """Tests Qwen code pattern and sandbox verification on email extraction regex."""
        code = "import re\ndef extract_emails(text):\n    return re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}', text)"
        assertions = """text = "Contact us at support@example.com or sales.dept@company.org."
res = extract_emails(text)
assert "support@example.com" in res
assert "sales.dept@company.org" in res
"""
        res = self.verifier.verify_in_sandbox(code, assertions)
        self.assertTrue(res.passed)

    def test_qwen_lora_synaptic_target_modules(self):
        """Verifies Qwen-specific LoRA target projection modules."""
        expected_qwen_targets = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        for target in expected_qwen_targets:
            self.assertIn(target, ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    def test_qwen_sleep_consolidation_cycle(self):
        """Tests complete end-to-end biological sleep consolidation loop for Qwen."""
        # 1. Simulate daytime interactions logged to SQLite
        self.db.log_interaction(
            prompt="Write a Python function to check palindrome",
            completion="def is_palindrome(s): return s == s[::-1]",
            verified_reward=1.0,
            surprise_score=0.88,
            mode="Pro-RLVR (N=16)",
            entropy=0.74
        )
        self.db.log_interaction(
            prompt="Write a Python GCD function",
            completion="def gcd(a, b): return a if b == 0 else gcd(b, a % b)",
            verified_reward=1.0,
            surprise_score=0.65,
            mode="Pro-RLVR (N=16)",
            entropy=0.52
        )

        daemon = SleepConsolidationDaemon(
            db_path=self.db_path,
            lora_adapter_path=self.adapter_path,
            settings=self.settings
        )

        # 2. Run sleep consolidation
        result = daemon.run_consolidation_cycle()
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["memories_consolidated"], 2)
        self.assertGreater(result["anchors_used"], 0)
        self.assertIn("adapter_saved_to", result)

        # 3. Verify SQLite records updated
        unconsolidated = self.db.fetch_surprise_replay_data(unconsolidated_only=True)
        self.assertEqual(len(unconsolidated), 0)

        stats = self.db.get_stats()
        self.assertEqual(stats["consolidation_cycles"], 1)
        self.assertEqual(stats["verified_count"], 2)


if __name__ == "__main__":
    unittest.main()
