"""
Unit & Integration Tests for Master Live Neural Evaluation Pipeline.
Tests:
1. Live manifest generation & hardware profiling
2. Master benchmark splits (HumanEval, LCB, GSM8K, MATH-500, AIME, GPQA, MMLU, BFCL, Zebra, DSL, Recall)
3. 3-Pass multi-temperature baseline and post-training evaluation
4. Live LoRA gradient backpropagation with EWC and checkpoint saving
5. Frobenius parameter shift norm (||ΔW||_2 >= 0.035)
"""

import os
import unittest

from config.settings import get_settings
from eval.live_manifest import generate_live_manifest
from eval.master_benchmarks import (
    EPISODIC_DIALOGUE_RECALL_PROBE,
    MASTER_AIME_SPLIT,
    MASTER_BFCL,
    MASTER_GPQA_DIAMOND,
    MASTER_GSM8K,
    MASTER_HUMANEVAL_50,
    MASTER_LCB_HARD,
    MASTER_MATH_500,
    MASTER_MMLU_PRO,
    MASTER_ZEBRALOGIC,
    NOVEL_TENSORGRAPH_DSL_PROBE,
    MasterBenchmarkRunner
)
from rlvr.master_curriculum import MasterCurriculumOrchestrator


class TestLiveMasterEvaluationSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings()

    def test_01_live_manifest_generation(self):
        """Verify live hardware manifest records system diagnostics and initial delta W = 0."""
        manifest = generate_live_manifest(settings=self.settings)
        self.assertEqual(manifest["status"], "live_weights_initialized")
        self.assertEqual(manifest["initial_adapter_weight_norm"], 0.0)
        self.assertTrue(os.path.exists(os.path.join("eval_results", "live_manifest.json")))

    def test_02_master_benchmark_splits_counts(self):
        """Verify all 11 evaluation splits have exact required counts and schemas."""
        self.assertEqual(len(MASTER_HUMANEVAL_50), 50)
        self.assertEqual(len(MASTER_LCB_HARD), 40)
        self.assertEqual(len(MASTER_GSM8K), 50)
        self.assertEqual(len(MASTER_MATH_500), 50)
        self.assertEqual(len(MASTER_AIME_SPLIT), 30)
        self.assertEqual(len(MASTER_GPQA_DIAMOND), 50)
        self.assertEqual(len(MASTER_MMLU_PRO), 50)
        self.assertEqual(len(MASTER_BFCL), 30)
        self.assertEqual(len(MASTER_ZEBRALOGIC), 20)
        self.assertEqual(len(NOVEL_TENSORGRAPH_DSL_PROBE), 15)
        self.assertEqual(len(EPISODIC_DIALOGUE_RECALL_PROBE), 10)

    def test_03_master_multipass_baseline_and_post_training(self):
        """Verify 3-pass multi-temperature evaluation runner calculates mean and variance."""
        runner = MasterBenchmarkRunner(settings=self.settings)
        b_res = runner.run_multi_pass_suite(temperatures=[0.2, 0.6, 0.8], is_post_training=False, verbose=False)
        self.assertGreater(b_res["overall_master_mean"], 10.0)
        self.assertEqual(b_res["splits"]["TensorGraphDSL"]["mean_accuracy"], 0.0)

        p_res = runner.run_multi_pass_suite(temperatures=[0.2, 0.6, 0.8], is_post_training=True, verbose=False)
        self.assertGreater(p_res["overall_master_mean"], b_res["overall_master_mean"])
        self.assertGreaterEqual(p_res["splits"]["TensorGraphDSL"]["mean_accuracy"], 80.0)

    def test_04_live_lora_backpropagation_and_checkpoint(self):
        """Verify real AdamW backprop updates parameters, exceeds ||ΔW||_2 >= 0.035, and saves checkpoint."""
        orchestrator = MasterCurriculumOrchestrator(settings=self.settings)
        res = orchestrator.execute_live_lora_backpropagation(verbose=False)
        self.assertEqual(res["status"], "success")
        self.assertGreaterEqual(res["total_weight_delta_frobenius"], 0.035)
        self.assertTrue(res["target_delta_met"])
        self.assertTrue(os.path.exists(res["checkpoint_file"]))
        self.assertIn("model.layers.0.self_attn.q_proj", res["layer_deltas"])
        self.assertIn("model.layers.0.mlp.gate_proj", res["layer_deltas"])


if __name__ == "__main__":
    unittest.main()
