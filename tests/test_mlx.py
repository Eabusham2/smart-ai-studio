"""
Unit tests for Apple Silicon native MLX reasoning engine.
"""

import unittest
from core.mlx_engine import MLXReasoningBackend


class TestMLXBackend(unittest.TestCase):
    def setUp(self):
        self.backend = MLXReasoningBackend()

    def test_mock_entropy_calculation(self):
        entropy = self.backend.calculate_token_entropy("Write an optimal sorting function")
        self.assertIsInstance(entropy, float)
        self.assertGreater(entropy, 0.0)

    def test_mock_fisher_matrix(self):
        anchors = ["def test(): pass", "def add(a, b): return a + b"]
        fisher = self.backend.compute_mlx_fisher(anchors)
        self.assertIsInstance(fisher, dict)
        self.assertTrue(len(fisher) > 0)


if __name__ == "__main__":
    unittest.main()
