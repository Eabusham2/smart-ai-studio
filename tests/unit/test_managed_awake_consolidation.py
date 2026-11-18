"""Tests for honest awake online consolidation and GUI thinking display."""

import os
import tempfile
import time
import unittest

from core.online_consolidator import ManagedAwakeOnlineConsolidator
from memory.db import EpisodicMemoryDB
from tests.unit.context_backend_fixture import TransactionalBackend



class TestManagedAwakeConsolidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state_root = tempfile.TemporaryDirectory()
        cls.db_fd, cls.db_path = tempfile.mkstemp(suffix=".db")
        cls.db = EpisodicMemoryDB(db_path=cls.db_path)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.state_root.cleanup()
            os.close(cls.db_fd)
            if os.path.exists(cls.db_path):
                os.remove(cls.db_path)
        except Exception:
            pass

    def test_01_watermark_detection_and_no_prune_below_threshold(self):
        consolidator = ManagedAwakeOnlineConsolidator(
            mlx_engine=TransactionalBackend(self.state_root.name),
            memory_db=self.db,
            max_context=1000,
            watermark=0.80,
            evict_ratio=0.40,
        )
        short_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help you today?"},
        ]
        retained, triggered = consolidator.check_and_prune(short_history)
        self.assertFalse(triggered)
        self.assertEqual(len(retained), 2)

    def test_02_watermark_prune_requires_real_training_updates_and_persists(self):
        trainable = TransactionalBackend(self.state_root.name)
        consolidator = ManagedAwakeOnlineConsolidator(
            mlx_engine=trainable,
            memory_db=self.db,
            max_context=100,
            watermark=0.80,
            evict_ratio=0.40,
        )
        long_history = []
        for i in range(8):
            long_history.append({"role": "user", "content": f"The archive label for sample {i} is cedar{i}."})
            long_history.append({"role": "assistant", "content": f"I recorded sample {i} as cedar{i}."})

        retained, triggered = consolidator.check_and_prune(long_history)
        self.assertTrue(triggered)
        self.assertLess(len(retained), len(long_history))
        self.assertEqual(retained[-1]["content"], long_history[-1]["content"])
        self.assertEqual(retained[-2]["content"], long_history[-2]["content"])

        deadline = time.time() + 2.0
        while consolidator.is_consolidating and time.time() < deadline:
            time.sleep(0.01)
        self.assertGreaterEqual(consolidator.consolidation_count, 1)
        self.assertGreater(consolidator.total_param_shift, 0.0)
        self.assertGreater(trainable.adapters["layer_0_lora"], 0.05)
        self.assertTrue(os.path.isfile(trainable.learning_state.current_checkpoint()))
        self.assertEqual(trainable.learning_state.current_manifest()["metadata"]["mode"], "awake")
