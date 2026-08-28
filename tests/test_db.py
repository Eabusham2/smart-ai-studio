"""
Unit tests for Episodic SQLite memory manager.
"""

import os
import tempfile
import unittest
from memory.db import EpisodicMemoryDB


class TestEpisodicMemoryDB(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_memory.db")
        self.db = EpisodicMemoryDB(db_path=self.db_path)

    def tearDown(self):
        import gc
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_database_initialization(self):
        stats = self.db.get_stats()
        self.assertEqual(stats["total_interactions"], 0)
        self.assertEqual(stats["verified_count"], 0)
        self.assertEqual(stats["consolidation_cycles"], 0)

    def test_log_interaction_and_retrieval(self):
        row_id = self.db.log_interaction(
            prompt="Write a fibonacci function",
            completion="def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
            raw_branches=["def fib(n): ...", "def fib(n): return n if n<=1 ..."],
            verified_reward=1.0,
            surprise_score=0.85,
            mode="Pro-RLVR (N=16)",
            entropy=0.78,
            winning_branch=1,
            test_cases="assert fib(5) == 5"
        )
        self.assertGreater(row_id, 0)

        # Retrieve surprise replay data
        memories = self.db.fetch_surprise_replay_data(limit=10, unconsolidated_only=True)
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["id"], row_id)
        self.assertEqual(memories[0]["surprise_score"], 0.85)

    def test_surprise_ordering(self):
        # Insert multiple items with different surprise scores
        self.db.log_interaction("Task A", "Code A", verified_reward=1.0, surprise_score=0.2)
        self.db.log_interaction("Task B", "Code B", verified_reward=1.0, surprise_score=0.95)
        self.db.log_interaction("Task C", "Code C", verified_reward=1.0, surprise_score=0.6)
        # Unverified should not be included
        self.db.log_interaction("Task D", "Code D", verified_reward=0.0, surprise_score=0.99)

        memories = self.db.fetch_surprise_replay_data(limit=10, unconsolidated_only=True)
        self.assertEqual(len(memories), 3)
        self.assertEqual(memories[0]["prompt"], "Task B")  # highest surprise 0.95
        self.assertEqual(memories[1]["prompt"], "Task C")  # 0.6
        self.assertEqual(memories[2]["prompt"], "Task A")  # 0.2

    def test_mark_consolidated(self):
        row_id = self.db.log_interaction("Task 1", "Code 1", verified_reward=1.0, surprise_score=0.5)
        self.db.mark_consolidated([row_id])

        # Should be empty when unconsolidated_only is True
        memories = self.db.fetch_surprise_replay_data(limit=10, unconsolidated_only=True)
        self.assertEqual(len(memories), 0)

        # Should be returned when unconsolidated_only is False
        all_memories = self.db.fetch_surprise_replay_data(limit=10, unconsolidated_only=False)
        self.assertEqual(len(all_memories), 1)

    def test_log_consolidation_run(self):
        log_id = self.db.log_consolidation(
            memories_count=10,
            anchors_count=30,
            ewc_lambda=400.0,
            avg_task_loss=0.12,
            avg_ewc_loss=0.04,
            adapter_path="./consolidated_slow_lora"
        )
        self.assertGreater(log_id, 0)
        stats = self.db.get_stats()
        self.assertEqual(stats["consolidation_cycles"], 1)


if __name__ == "__main__":
    unittest.main()
