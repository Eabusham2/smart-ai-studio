"""
Unit tests for Fisher Matrix, EWC Loss, and Sleep Consolidation Daemon.
"""

import os
import tempfile
import unittest
from consolidation.daemon import SleepConsolidationDaemon
from consolidation.ewc_loss import EWCLossCalculator
from consolidation.fisher import FisherEstimator
from memory.anchor_dataset import get_anchor_dataset
from memory.db import EpisodicMemoryDB


class TestConsolidation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_memory.db")
        self.adapter_path = os.path.join(self.temp_dir.name, "test_slow_lora")
        self.db = EpisodicMemoryDB(db_path=self.db_path)
        self.daemon = SleepConsolidationDaemon(
            db_path=self.db_path,
            lora_adapter_path=self.adapter_path,
            ewc_lambda=400.0
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_ewc_loss_calculation(self):
        calculator = EWCLossCalculator(lambda_ewc=400.0)

        # Mock parameters: param deviates from anchor
        params = [("layer_1", [1.1, 2.2, 3.3])]
        fisher = {"layer_1": [0.5, 0.5, 0.5]}
        anchors = {"layer_1": [1.0, 2.0, 3.0]}

        penalty = calculator.calculate_penalty(params, fisher, anchors)
        # diffs: (0.1)^2 * 0.5 + (0.2)^2 * 0.5 + (0.3)^2 * 0.5 = 0.005 + 0.02 + 0.045 = 0.07
        # (400 / 2) * 0.07 = 200 * 0.07 = 14.0
        self.assertAlmostEqual(penalty, 14.0, places=2)

    def test_interleave_replay_batches(self):
        memories = [
            {"prompt": "User Prompt 1", "completion": "User Comp 1"},
            {"prompt": "User Prompt 2", "completion": "User Comp 2"}
        ]
        anchors = get_anchor_dataset()

        interleaved = SleepConsolidationDaemon.interleave_replay_batches(
            user_memories=memories,
            anchor_dataset=anchors,
            episodic_ratio=0.25
        )
        self.assertGreater(len(interleaved), len(memories))
        # Ensure user prompts are present
        prompts = [item["prompt"] for item in interleaved]
        self.assertIn("User Prompt 1", prompts)
        self.assertIn("User Prompt 2", prompts)

    def test_consolidation_cycle_execution(self):
        # Insert verified interaction with high surprise
        self.db.log_interaction(
            prompt="Optimize regex boundary parser",
            completion="def parse(): pass",
            verified_reward=1.0,
            surprise_score=0.92
        )

        result = self.daemon.run_consolidation_cycle()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["memories_consolidated"], 1)
        self.assertGreater(result["anchors_used"], 0)

        # Check DB was updated: no unconsolidated memories left
        unconsolidated = self.db.fetch_surprise_replay_data(unconsolidated_only=True)
        self.assertEqual(len(unconsolidated), 0)

        # Check consolidation logs in DB
        stats = self.db.get_stats()
        self.assertEqual(stats["consolidation_cycles"], 1)


if __name__ == "__main__":
    unittest.main()
