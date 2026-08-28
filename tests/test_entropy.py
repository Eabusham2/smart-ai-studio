"""
Unit tests for Entropy Router & Branch Allocation.
"""

import unittest
from core.entropy_router import EntropyRouter


class TestEntropyRouter(unittest.TestCase):
    def setUp(self):
        self.router = EntropyRouter(
            low_threshold=0.25,
            high_threshold=0.70,
            instant_branches=1,
            pro_branches_mid=8,
            pro_branches_high=16
        )

    def test_pure_python_entropy_calculation(self):
        # Uniform distribution across 4 tokens: H = ln(4) ~= 1.386
        uniform_logits = [1.0, 1.0, 1.0, 1.0]
        entropy = self.router.compute_entropy_from_logits(uniform_logits)
        self.assertAlmostEqual(entropy, 1.386, places=2)

        # High certainty distribution: one dominant token
        certain_logits = [10.0, -10.0, -10.0, -10.0]
        low_entropy = self.router.compute_entropy_from_logits(certain_logits)
        self.assertLess(low_entropy, 0.01)

    def test_routing_decisions(self):
        # Low entropy -> Instant N=1
        mode, branches = self.router.route(0.10, has_test_cases=False)
        self.assertEqual(mode, "Instant (N=1)")
        self.assertEqual(branches, 1)

        # Medium entropy -> Pro Search N=8
        mode, branches = self.router.route(0.45, has_test_cases=False)
        self.assertEqual(mode, "Pro-Search (N=8)")
        self.assertEqual(branches, 8)

        # High entropy -> Pro Search N=16
        mode, branches = self.router.route(0.85, has_test_cases=False)
        self.assertEqual(mode, "Pro-Search (N=16)")
        self.assertEqual(branches, 16)

        # Ground-truth test cases override entropy -> Pro-RLVR N=16
        mode, branches = self.router.route(0.05, has_test_cases=True)
        self.assertEqual(mode, "Pro-RLVR (N=16)")
        self.assertEqual(branches, 16)

    def test_prompt_heuristic_entropy(self):
        # High complexity code request
        high_e = self.router.estimate_prompt_entropy_heuristic(
            "Write a dynamic programming algorithm to optimize regex parsing"
        )
        self.assertGreaterEqual(high_e, 0.50)

        # Simple factual query
        low_e = self.router.estimate_prompt_entropy_heuristic("What is the capital of France?")
        self.assertLess(low_e, 0.40)


if __name__ == "__main__":
    unittest.main()
