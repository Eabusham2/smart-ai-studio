"""Tests for honest awake online consolidation and GUI thinking display."""

import os
import tempfile
import time
import tkinter as tk
import unittest

from config.settings import get_settings
from core.mlx_engine import MLXReasoningBackend
from core.online_consolidator import AwakeOnlineConsolidator
from core.pro_engine import ProReasoningEngine
from memory.db import EpisodicMemoryDB
from tests.unit.context_backend_fixture import TransactionalBackend
from app_gui import SmartAIChatbotApp



class TestAwakeConsolidationSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state_root = tempfile.TemporaryDirectory()
        cls.settings = get_settings(use_mock=True)
        cls.db_fd, cls.db_path = tempfile.mkstemp(suffix=".db")
        cls.db = EpisodicMemoryDB(db_path=cls.db_path)
        cls.mlx_engine = MLXReasoningBackend(model_path="prism-ml/Ternary-Bonsai-27B-mlx-2bit")
        cls.engine = ProReasoningEngine(settings=cls.settings)

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
        consolidator = AwakeOnlineConsolidator(
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
        consolidator = AwakeOnlineConsolidator(
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

    def test_03_unloaded_engine_never_prunes_or_claims_learning(self):
        consolidator = AwakeOnlineConsolidator(
            mlx_engine=self.mlx_engine,
            memory_db=self.db,
            max_context=20,
            watermark=0.5,
            evict_ratio=0.40,
        )
        history = [
            {"role": "user", "content": "x" * 200},
            {"role": "assistant", "content": "y" * 200},
            {"role": "user", "content": "z" * 200},
            {"role": "assistant", "content": "q" * 200},
        ]
        retained, triggered = consolidator.check_and_prune(history)
        self.assertFalse(triggered)
        self.assertEqual(retained, history)
        self.assertEqual(consolidator.consolidation_count, 0)
        self.assertEqual(consolidator.total_param_shift, 0.0)

    def test_04_engine_chat_rolling_context_does_not_fake_learning_in_mock_mode(self):
        self.engine.awake_consolidator.max_context = 60
        self.engine.awake_consolidator.watermark_tokens = 40
        history = [
            {"role": "user", "content": "What is quantum teleportation?" * 4},
            {"role": "assistant", "content": "It is a protocol for quantum state transfer." * 4},
            {"role": "user", "content": "Tell me how Bell states are used." * 4},
            {"role": "assistant", "content": "Bell states provide maximally entangled qubit pairs." * 4},
            {"role": "user", "content": "Hello again!"},
        ]
        resp, pruned = self.engine.chat(history)
        self.assertTrue(len(resp) > 0)
        self.assertEqual(len(pruned), len(history))
        self.assertEqual(self.engine.awake_consolidator.consolidation_count, 0)

    def test_05_gui_thinking_dropdown_toggle(self):
        root = tk.Tk()
        root.withdraw()
        app = SmartAIChatbotApp(root, settings=self.settings)
        app._append_ai_message(
            "Here is the final verified answer.",
            thinking_text="Step 1: Analyzed input.\nStep 2: Applied formal logic rules.\nStep 3: Verification complete.",
            thinking_tokens=42,
            duration_s=0.35,
            tok_per_sec=120.0,
        )
        content = app.chat_stream.get("1.0", "end")
        self.assertIn("Thought for", content)
        self.assertIn("[Click to Expand]", content)
        self.assertNotIn("Step 1: Analyzed input.", content)

        think_id = "think_1"
        app._on_toggle_thinking_dropdown(think_id)
        content_expanded = app.chat_stream.get("1.0", "end")
        self.assertIn("[Click to Collapse]", content_expanded)
        self.assertIn("Step 1: Analyzed input.", content_expanded)

        app._on_toggle_thinking_dropdown(think_id)
        content_collapsed = app.chat_stream.get("1.0", "end")
        self.assertIn("[Click to Expand]", content_collapsed)
        self.assertNotIn("Step 1: Analyzed input.", content_collapsed)

        if hasattr(app, "watchdog") and app.watchdog:
            app.watchdog.stop_monitoring()
        root.destroy()


if __name__ == "__main__":
    unittest.main()
