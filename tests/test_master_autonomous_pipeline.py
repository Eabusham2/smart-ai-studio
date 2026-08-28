"""
Unit & Integration Tests for Master Autonomous Continuous-Learning Pipeline.
Tests:
1. All benchmark splits including HLE, DeepSWE, and Autonomous Evolution (Unlabeled Discovery)
2. Unsupervised evolution test synthesis & Environmental RLVR error recovery
3. 3-pass multi-temperature baseline and post-consolidation evaluation
4. Checkpoint saving & layer-by-layer Frobenius parameter delta telemetry (||ΔW||_2 >= 0.035)
"""

import os
import unittest

from config.settings import get_settings
from eval.master_benchmarks import (
    MASTER_AUTONOMOUS_EVOLUTION_SPLIT,
    MASTER_DEEPSWE_SPLIT,
    MASTER_HLE_SPLIT,
    MasterBenchmarkRunner
)
from rlvr.master_curriculum import MasterCurriculumOrchestrator


class TestMasterAutonomousPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings()

    def test_01_new_flagship_splits_counts(self):
        """Verify HLE, DeepSWE, and Autonomous Evolution splits have exact required counts."""
        self.assertEqual(len(MASTER_HLE_SPLIT), 15)
        self.assertEqual(len(MASTER_DEEPSWE_SPLIT), 10)
        self.assertEqual(len(MASTER_AUTONOMOUS_EVOLUTION_SPLIT), 12)

    def test_02_unsupervised_evolution_and_environmental_rlvr(self):
        """Verify unsupervised evolution self-synthesizes tests and environmental RLVR recovers from errors."""
        orchestrator = MasterCurriculumOrchestrator(settings=self.settings)
        evol_res = orchestrator.execute_autonomous_unsupervised_evolution(target_traces=20, verbose=False)
        self.assertEqual(evol_res["status"], "success")
        self.assertGreaterEqual(evol_res["discovery_traces_logged"], 20)

        rlvr_res = orchestrator.execute_environmental_rlvr_recovery(target_traces=20, max_attempts=4, verbose=False)
        self.assertEqual(rlvr_res["status"], "success")
        self.assertGreaterEqual(rlvr_res["recovery_traces_logged"], 20)

    def test_03_master_multipass_suite_14_splits(self):
        """Verify 3-pass multi-temperature evaluation runner calculates mean and variance across 14 splits."""
        runner = MasterBenchmarkRunner(settings=self.settings)
        b_res = runner.run_multi_pass_suite(temperatures=[0.2, 0.6, 0.8], is_post_training=False, verbose=False)
        self.assertGreater(b_res["overall_master_mean"], 50.0)
        self.assertEqual(b_res["splits"]["Autonomous Evolution"]["mean_accuracy"], 0.0)

        p_res = runner.run_multi_pass_suite(temperatures=[0.2, 0.6, 0.8], is_post_training=True, verbose=False)
        self.assertGreater(p_res["overall_master_mean"], b_res["overall_master_mean"])
        self.assertGreaterEqual(p_res["splits"]["Autonomous Evolution"]["mean_accuracy"], 80.0)

    def test_04_lora_backprop_and_layer_deltas(self):
        """Verify live AdamW backprop updates parameters, exceeds ||ΔW||_2 >= 0.035, and saves checkpoint."""
        orchestrator = MasterCurriculumOrchestrator(settings=self.settings)
        res = orchestrator.execute_live_lora_backpropagation(verbose=False)
        self.assertEqual(res["status"], "success")
        self.assertGreaterEqual(res["total_weight_delta_frobenius"], 0.035)
        self.assertTrue(res["target_delta_met"])
        self.assertTrue(os.path.exists(res["checkpoint_file"]))


if __name__ == "__main__":
    unittest.main()
