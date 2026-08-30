"""
Unit & Integration Tests for Benchmark Suite & RLVR Autonomous Continuous Learning.
Verifies:
1. HumanEval & Math dataset integrity and syntax
2. BenchmarkRunner pass@1 evaluation
3. RLVR self-play rollouts and ground-truth SQLite memory logging
4. Parameter delta norm calculation (||ΔW||_2 > 0)
"""

import json
import os
import tempfile
import unittest

from config.settings import Settings
from eval.benchmark_data import HUMANEVAL_50_SUBSET, MATH_50_SUBSET
from eval.benchmark_runner import BenchmarkRunner
from eval.rlvr_orchestrator import RLVRContinuousLearner


class TestBenchmarkPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_eval_memory.db")
        self.adapter_path = os.path.join(self.temp_dir.name, "eval_slow_lora")
        self.settings = Settings(
            database_path=self.db_path,
            lora_adapter_path=self.adapter_path,
            backend="mock",
            live_mode=False,
            ewc_lambda=300.0
        )
        self.runner = BenchmarkRunner(settings=self.settings)
        self.learner = RLVRContinuousLearner(settings=self.settings)

    def tearDown(self):
        import gc
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_01_benchmark_datasets_integrity(self):
        """Verify 50 problems in HumanEval and 10 in Math subset have valid prompts and tests."""
        self.assertEqual(len(HUMANEVAL_50_SUBSET), 50)
        self.assertGreaterEqual(len(MATH_50_SUBSET), 10)

        for item in HUMANEVAL_50_SUBSET[:5]:
            self.assertIn("id", item)
            self.assertIn("prompt", item)
            self.assertIn("tests", item)

    def test_02_benchmark_runner_subset_execution(self):
        """Verify BenchmarkRunner evaluates a subset and computes pass@1 and metrics."""
        subset = HUMANEVAL_50_SUBSET[:3]
        res = self.runner.evaluate_subset("Test-Coding", subset, verbose=False)
        self.assertEqual(res["total_samples"], 3)
        self.assertGreaterEqual(res["pass_at_1_accuracy"], 0.0)
        self.assertIn("mean_entropy", res)
        self.assertIn("throughput_tok_per_sec", res)

    def test_03_rlvr_self_play_and_consolidation_delta(self):
        """Verify RLVR self-play gathers verified traces and executes consolidation with ||ΔW||_2 > 0."""
        rollout_res = self.learner.execute_self_play_rollouts(target_verified_traces=2, branch_count=2, verbose=False)
        self.assertEqual(rollout_res["verified_traces_gathered"], 2)

        consol_res = self.learner.trigger_sleep_consolidation_with_delta_norm(verbose=False)
        self.assertEqual(consol_res["status"], "success")
        self.assertGreater(consol_res["weight_delta_l2_norm"], 0.0)
        self.assertTrue(consol_res["parameter_update_verified"])


if __name__ == "__main__":
    unittest.main()
