"""
Unit & Integration Tests for Frontier Industry Benchmarks & Dual-Memory Probing Suite.
Verifies:
1. Dataset split integrity (GPQA 50, AIME 30, LCB 40, MMLU-Pro 50, BFCL 30, DSL 10, Recall 10)
2. Frontier benchmark runner baseline and post-training delta validation
3. Targeted RLVR self-play curriculum execution
4. EWC consolidation parameter update delta ||ΔW||_2 >= 0.020
5. Zero-context novel DSL probe retention and episodic memory recall
"""

import os
import unittest

from config.settings import get_settings
from core.pro_engine import ProReasoningEngine
from eval.frontier_benchmarks import (
    GPQA_DIAMOND_SUBSET,
    AIME_SUBSET,
    LIVECODEBENCH_SUBSET,
    MMLU_PRO_SUBSET,
    BFCL_SUBSET,
    NOVEL_SKILL_DSL_PROBE,
    EPISODIC_RECALL_PROBE,
    FrontierBenchmarkRunner
)
from memory.db import EpisodicMemoryDB
from rlvr.frontier_curriculum import FrontierCurriculumOrchestrator


class TestFrontierBenchmarkPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings()
        cls.db = EpisodicMemoryDB(db_path=cls.settings.database_path)
        cls.engine = ProReasoningEngine(settings=cls.settings)
        cls.runner = FrontierBenchmarkRunner(engine=cls.engine, db=cls.db, settings=cls.settings)
        cls.orchestrator = FrontierCurriculumOrchestrator(engine=cls.engine, db=cls.db, settings=cls.settings)

    def test_01_dataset_split_counts_and_schemas(self):
        """Verify all standardized frontier splits meet required size and schema constraints."""
        self.assertEqual(len(GPQA_DIAMOND_SUBSET), 50)
        self.assertEqual(len(AIME_SUBSET), 30)
        self.assertEqual(len(LIVECODEBENCH_SUBSET), 40)
        self.assertEqual(len(MMLU_PRO_SUBSET), 50)
        self.assertEqual(len(BFCL_SUBSET), 30)
        self.assertEqual(len(NOVEL_SKILL_DSL_PROBE), 10)
        self.assertEqual(len(EPISODIC_RECALL_PROBE), 10)

        # Check schema keys
        self.assertIn("answer", GPQA_DIAMOND_SUBSET[0])
        self.assertIn("answer", AIME_SUBSET[0])
        self.assertIn("tests", LIVECODEBENCH_SUBSET[0])
        self.assertIn("answer", MMLU_PRO_SUBSET[0])
        self.assertIn("expected_tool", BFCL_SUBSET[0])
        self.assertIn("expected", NOVEL_SKILL_DSL_PROBE[0])
        self.assertIn("expected_fact", EPISODIC_RECALL_PROBE[0])

    def test_02_baseline_evaluation_accuracy(self):
        """Verify zero-shot baseline execution produces expected baseline metrics."""
        res = self.runner.run_full_frontier_suite(is_post_training=False, verbose=False)
        self.assertGreater(res["frontier_combined_accuracy"], 70.0)
        self.assertEqual(res["novel_skill_dsl"]["accuracy_percent"], 0.0)  # Unseen DSL baseline
        self.assertEqual(res["episodic_recall"]["accuracy_percent"], 100.0)

    def test_03_curriculum_mining_and_self_play(self):
        """Verify targeted RLVR curriculum mines tasks and accumulates verified traces."""
        tasks = self.orchestrator.mine_training_curriculum()
        self.assertGreater(len(tasks), 100)

        res = self.orchestrator.execute_self_play_curriculum(target_verified_traces=5, branch_count=2, verbose=False)
        self.assertGreaterEqual(res["verified_traces_gathered"], 5)

    def test_04_neuromorphic_consolidation_parameter_shift(self):
        """Verify EWC consolidation executes and confirms parameter shift ||ΔW||_2 >= 0.020."""
        res = self.orchestrator.trigger_neuromorphic_consolidation(verbose=False)
        self.assertEqual(res["status"], "success")
        self.assertGreaterEqual(res["weight_delta_l2_norm"], 0.020)
        self.assertTrue(res["target_delta_met"])

    def test_05_post_training_multi_tier_validation(self):
        """Verify post-training evaluation achieves positive delta and DSL retention >= 80%."""
        res = self.runner.run_full_frontier_suite(is_post_training=True, verbose=False)
        self.assertGreater(res["frontier_combined_accuracy"], 85.0)
        self.assertGreaterEqual(res["novel_skill_dsl"]["accuracy_percent"], 80.0)
        self.assertEqual(res["episodic_recall"]["accuracy_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
