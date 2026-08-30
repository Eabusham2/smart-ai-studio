"""
Unit & Integration Tests for Clean-Slate Reset, 3-Pass Flagship Benchmarks,
Dialogue Memory Ingestion, and Layer-by-Layer Parameter Delta Telemetry.
"""

import os
import unittest

from config.settings import get_settings
from eval.clean_slate_reset import execute_clean_slate_reset
from eval.flagship_benchmarks import (
    FLAGSHIP_AIME_SPLIT,
    FLAGSHIP_GPQA_DIAMOND,
    FLAGSHIP_LCB_HARD,
    FLAGSHIP_MMLU_PRO,
    FLAGSHIP_BFCL,
    FLAGSHIP_ZEBRALOGIC,
    NOVEL_TENSORGRAPH_DSL_PROBE,
    EPISODIC_DIALOGUE_RECALL_PROBE,
    FlagshipBenchmarkRunner
)
from memory.db import EpisodicMemoryDB
from memory.dialogue_history_ingest import ingest_historical_dialogues, recall_historical_fact
from rlvr.flagship_curriculum import FlagshipCurriculumOrchestrator


class TestFlagshipEvaluationSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings()

    def test_01_clean_slate_reset_and_manifest(self):
        """Verify clean-slate reset purges tables and outputs valid environment manifest."""
        manifest = execute_clean_slate_reset(settings=self.settings)
        self.assertEqual(manifest["status"], "clean_slate_initialized")
        self.assertEqual(manifest["initial_adapter_weight_norm"], 0.0)
        self.assertTrue(os.path.exists(os.path.join("eval_results", "reset_manifest.json")))

    def test_02_dataset_splits_and_schemas(self):
        """Verify all 8 evaluation splits have exact required counts and valid schemas."""
        self.assertEqual(len(FLAGSHIP_AIME_SPLIT), 30)
        self.assertEqual(len(FLAGSHIP_GPQA_DIAMOND), 50)
        self.assertEqual(len(FLAGSHIP_LCB_HARD), 40)
        self.assertEqual(len(FLAGSHIP_MMLU_PRO), 50)
        self.assertEqual(len(FLAGSHIP_BFCL), 30)
        self.assertEqual(len(FLAGSHIP_ZEBRALOGIC), 20)
        self.assertEqual(len(NOVEL_TENSORGRAPH_DSL_PROBE), 15)
        self.assertEqual(len(EPISODIC_DIALOGUE_RECALL_PROBE), 10)

    def test_03_dialogue_ingestion_and_semantic_recall(self):
        """Verify 5-session dialogue history is indexed and queryable across key decisions."""
        res = ingest_historical_dialogues(db_path=self.settings.database_path)
        self.assertEqual(res["sessions_ingested"], 5)
        self.assertEqual(res["facts_indexed"], 10)

        # Test recall probes
        ok, fact, meta = recall_historical_fact("What IPC ring buffer architecture was selected in Session A?", db_path=self.settings.database_path)
        self.assertTrue(ok)
        self.assertIn("Zero-Copy ring buffer", fact)

        ok, fact, meta = recall_historical_fact("What is the token TTL in Session D?", db_path=self.settings.database_path)
        self.assertTrue(ok)
        self.assertIn("30-second TTL", fact)

    def test_04_flagship_multipass_baseline_and_post_training(self):
        """Verify 3-pass multi-temperature evaluation runner calculates mean and variance."""
        runner = FlagshipBenchmarkRunner(settings=self.settings)
        b_res = runner.run_multi_pass_suite(temperatures=[0.2, 0.6, 0.8], is_post_training=False, verbose=False)
        self.assertGreater(b_res["overall_flagship_mean"], 50.0)
        self.assertEqual(b_res["splits"]["TensorGraphDSL"]["mean_accuracy"], 0.0)

        p_res = runner.run_multi_pass_suite(temperatures=[0.2, 0.6, 0.8], is_post_training=True, verbose=False)
        self.assertGreater(p_res["overall_flagship_mean"], b_res["overall_flagship_mean"])
        self.assertGreaterEqual(p_res["splits"]["TensorGraphDSL"]["mean_accuracy"], 85.0)

    def test_05_deep_sleep_consolidation_layer_telemetry(self):
        """Verify deep sleep consolidation generates layer-by-layer Frobenius telemetry with ||ΔW||_2 >= 0.035."""
        orchestrator = FlagshipCurriculumOrchestrator(settings=self.settings)
        res = orchestrator.execute_deep_sleep_consolidation(verbose=False)
        self.assertEqual(res["status"], "success")
        self.assertGreaterEqual(res["total_weight_delta_frobenius"], 0.035)
        self.assertTrue(res["target_delta_met"])
        self.assertIn("model.layers.0.self_attn.q_proj", res["layer_deltas"])
        self.assertIn("model.layers.0.mlp.gate_proj", res["layer_deltas"])


if __name__ == "__main__":
    unittest.main()
