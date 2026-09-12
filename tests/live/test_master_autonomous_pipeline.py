"""
Integration Tests for Master Autonomous Continuous-Learning Pipeline.
Tests orchestration contracts without downloading production neural weights in CI.
"""

import os
import tempfile
import unittest

from config.settings import Settings
from eval.master_benchmarks import (
    MASTER_AUTONOMOUS_EVOLUTION_SPLIT,
    MASTER_DEEPSWE_SPLIT,
    MASTER_HLE_SPLIT,
    MasterBenchmarkRunner,
)
from rlvr.master_curriculum import MasterCurriculumOrchestrator


class TestMasterAutonomousPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.settings = Settings(
            database_path=os.path.join(cls.temp_dir.name, "master_autonomous_ci.db"),
            lora_adapter_path=os.path.join(cls.temp_dir.name, "adapters.pt"),
            backend="mock",
            live_mode=False,
            use_mock=True,
            auto_download=False,
        )

    @classmethod
    def tearDownClass(cls):
        try:
            cls.temp_dir.cleanup()
        except Exception:
            pass

    def test_01_new_flagship_splits_counts(self):
        self.assertEqual(len(MASTER_HLE_SPLIT), 15)
        self.assertEqual(len(MASTER_DEEPSWE_SPLIT), 10)
        self.assertEqual(len(MASTER_AUTONOMOUS_EVOLUTION_SPLIT), 12)

    def test_02_unsupervised_evolution_and_environmental_rlvr(self):
        orchestrator = MasterCurriculumOrchestrator(settings=self.settings)
        evol_res = orchestrator.execute_autonomous_unsupervised_evolution(target_traces=4, verbose=False)
        self.assertEqual(evol_res["status"], "success")
        self.assertGreaterEqual(evol_res["discovery_traces_logged"], 4)

        rlvr_res = orchestrator.execute_environmental_rlvr_recovery(
            target_traces=4, max_attempts=2, verbose=False
        )
        self.assertEqual(rlvr_res["status"], "success")
        self.assertGreaterEqual(rlvr_res["recovery_traces_logged"], 4)

    def test_03_master_multipass_suite_14_splits(self):
        runner = MasterBenchmarkRunner(settings=self.settings)
        b_res = runner.run_multi_pass_suite(
            temperatures=[0.2, 0.6, 0.8], is_post_training=False, verbose=False
        )
        self.assertGreater(b_res["overall_master_mean"], 10.0)
        self.assertEqual(b_res["splits"]["Autonomous Evolution"]["mean_accuracy"], 0.0)

        p_res = runner.run_multi_pass_suite(
            temperatures=[0.2, 0.6, 0.8], is_post_training=True, verbose=False
        )
        self.assertGreater(p_res["overall_master_mean"], b_res["overall_master_mean"])
        self.assertGreaterEqual(
            p_res["splits"]["Autonomous Evolution"]["mean_accuracy"], 80.0
        )

    def test_04_lora_backprop_and_layer_deltas(self):
        orchestrator = MasterCurriculumOrchestrator(settings=self.settings)
        res = orchestrator.execute_live_lora_backpropagation(verbose=False)
        self.assertEqual(res["status"], "success")
        self.assertGreaterEqual(res["total_weight_delta_frobenius"], 0.035)
        self.assertTrue(res["target_delta_met"])
        self.assertTrue(os.path.exists(res["checkpoint_file"]))


if __name__ == "__main__":
    unittest.main()
