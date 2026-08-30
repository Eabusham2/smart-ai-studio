"""
Unit tests for Speculative Acceleration & Zero-VRAM Drafting Engine.
Tests Prompt Lookup Decoding (PLD), Lookahead/Jacobi drafting, rejection sampling,
and speedup telemetry.
"""

import unittest
from config.settings import Settings
from core.pro_engine import ProReasoningEngine
from core.speculative_engine import (
    LookaheadJacobiDrafter,
    PromptLookupDrafter,
    SpeculativeEngine,
    SpeculativeStats,
)


class TestSpeculativeAcceleration(unittest.TestCase):
    def setUp(self):
        self.pld = PromptLookupDrafter(min_ngram=3, max_ngram=5, max_draft_tokens=4)
        self.jacobi = LookaheadJacobiDrafter(max_draft_tokens=4)
        self.spec_engine = SpeculativeEngine(mode="pld", max_draft_tokens=4)

    def test_pld_ngram_pattern_detection(self):
        """Tests that PLD successfully detects repeated n-grams from historical context."""
        # Simulated sequence with repeating pattern: [10, 20, 30, 40, 50, 99, 10, 20, 30]
        # Target matching suffix is [10, 20, 30]. Historical continuation is [40, 50, 99].
        tokens = [10, 20, 30, 40, 50, 99, 100, 200, 10, 20, 30]
        draft = self.pld.find_draft_tokens(tokens)
        
        self.assertEqual(draft, [40, 50, 99, 100])

    def test_pld_no_match_returns_empty(self):
        """Tests that unmatched prefixes return empty draft list gracefully."""
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        draft = self.pld.find_draft_tokens(tokens)
        self.assertEqual(draft, [])

    def test_jacobi_lookahead_caching(self):
        """Tests Lookahead/Jacobi association cache and draft retrieval."""
        tokens = [101, 102, 103, 104, 105, 106]
        self.jacobi.update_cache(tokens)
        
        # Suffix [101, 102] should draft [103, 104, 105, 106]
        draft = self.jacobi.find_draft_tokens([101, 102])
        self.assertIn(103, draft)

    def test_speculative_rejection_sampling(self):
        """Tests lossless rejection sampling verification and telemetry calculation."""
        context = [10, 20, 30, 40, 50, 99, 100, 10, 20, 30]
        draft = self.spec_engine.propose_draft_tokens(context)
        self.assertEqual(len(draft), 4)

        accepted, bonus = self.spec_engine.verify_draft_tokens_rejection_sampling(context, draft)
        self.assertIsInstance(accepted, list)
        
        telemetry = self.spec_engine.get_telemetry()
        self.assertEqual(telemetry["mode"], "PLD")
        self.assertGreater(telemetry["draft_tokens_proposed"], 0)
        self.assertEqual(telemetry["vram_overhead_mb"], 0.0)  # Zero VRAM

    def test_pro_engine_speculative_telemetry_integration(self):
        """Tests ProReasoningEngine returns speculative metrics in metadata."""
        settings = Settings(
            speculative_mode="pld",
            speculative_tokens=4,
            mlx_model_path="prism-ml/Ternary-Bonsai-27B-mlx-2bit"
        )
        engine = ProReasoningEngine(settings=settings)
        response, meta = engine.solve(
            prompt="Write a Python function to check palindrome",
            test_cases="assert is_palindrome('aba') == True"
        )
        self.assertIn("speculative", meta)
        self.assertEqual(meta["speculative"]["mode"], "PLD")
        self.assertEqual(meta["speculative"]["vram_overhead_mb"], 0.0)


if __name__ == "__main__":
    unittest.main()
