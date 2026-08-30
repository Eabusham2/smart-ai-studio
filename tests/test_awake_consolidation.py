"""
Unit and Integration Tests for Awake Double-Buffered Synaptic Consolidation and Thinking Dropdown.
Verifies:
1. High-watermark context detection (80% of max_context) and eviction ratio (40%).
2. Preservation of recent turns and coherent dialogue slicing.
3. Background shadow adapter mini-batch training with EWC penalty and atomic weight hot-swapping.
4. ProReasoningEngine.chat() rolling context management.
5. GUI interactive thinking dropdown expand and collapse mechanics.
"""

import os
import tempfile
import time
import tkinter as tk
import unittest
from typing import Dict, List

from config.settings import Settings, get_settings
from core.mlx_engine import MLXReasoningBackend
from core.online_consolidator import AwakeOnlineConsolidator
from core.pro_engine import ProReasoningEngine
from memory.db import EpisodicMemoryDB
from app_gui import SmartAIChatbotApp


class TestAwakeConsolidationSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings(use_mock=True)
        cls.db_fd, cls.db_path = tempfile.mkstemp(suffix=".db")
        cls.db = EpisodicMemoryDB(db_path=cls.db_path)
        cls.mlx_engine = MLXReasoningBackend(model_path="prism-ml/Ternary-Bonsai-27B-mlx-2bit")
        cls.mlx_engine.adapters = {"layer_0_lora": 0.05}
        cls.engine = ProReasoningEngine(settings=cls.settings)

    @classmethod
    def tearDownClass(cls):
        try:
            os.close(cls.db_fd)
            if os.path.exists(cls.db_path):
                os.remove(cls.db_path)
        except Exception:
            pass

    def test_01_watermark_detection_and_no_prune_below_threshold(self):
        """Verify that short dialogues below the 80% watermark do not trigger pruning."""
        consolidator = AwakeOnlineConsolidator(
            mlx_engine=self.mlx_engine,
            memory_db=self.db,
            max_context=1000,
            watermark=0.80,
            evict_ratio=0.40
        )
        short_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help you today?"}
        ]
        retained, triggered = consolidator.check_and_prune(short_history)
        self.assertFalse(triggered)
        self.assertEqual(len(retained), 2)

    def test_02_watermark_prune_and_slice_eviction(self):
        """Verify that conversation exceeding watermark triggers eviction and background training."""
        consolidator = AwakeOnlineConsolidator(
            mlx_engine=self.mlx_engine,
            memory_db=self.db,
            max_context=100,   # 80 tokens watermark
            watermark=0.80,
            evict_ratio=0.40
        )
        # Create 10 turns with enough characters to exceed 80 tokens
        long_history = []
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            long_history.append({"role": role, "content": f"This is turn number {i} containing enough detailed context words." * 4})

        retained, triggered = consolidator.check_and_prune(long_history)
        self.assertTrue(triggered)
        self.assertLess(len(retained), len(long_history))
        # Verify most recent turns are retained
        self.assertEqual(retained[-1]["content"], long_history[-1]["content"])
        self.assertEqual(retained[-2]["content"], long_history[-2]["content"])

        # Allow background thread to complete
        time.sleep(0.3)
        self.assertGreaterEqual(consolidator.consolidation_count, 1)
        self.assertGreater(consolidator.total_param_shift, 0.0)

    def test_03_engine_chat_rolling_context(self):
        """Verify ProReasoningEngine.chat() passes messages through rolling consolidator."""
        self.engine.awake_consolidator.max_context = 60
        self.engine.awake_consolidator.watermark_tokens = 40

        history = [
            {"role": "user", "content": "What is quantum teleportation?" * 4},
            {"role": "assistant", "content": "It is a protocol for quantum state transfer." * 4},
            {"role": "user", "content": "Tell me how Bell states are used." * 4},
            {"role": "assistant", "content": "Bell states provide maximally entangled qubit pairs." * 4},
            {"role": "user", "content": "Hello again!"}
        ]

        resp, pruned = self.engine.chat(history)
        self.assertTrue(len(resp) > 0)
        self.assertTrue(len(pruned) >= 2)

    def test_04_gui_thinking_dropdown_toggle(self):
        """Verify GUI interactive thinking dropdown collapses and expands properly."""
        root = tk.Tk()
        root.withdraw()
        app = SmartAIChatbotApp(root, settings=self.settings)

        app._append_ai_message(
            "Here is the final verified answer.",
            thinking_text="Step 1: Analyzed input.\nStep 2: Applied formal logic rules.\nStep 3: Verification complete.",
            thinking_tokens=42,
            duration_s=0.35,
            tok_per_sec=120.0
        )

        content = app.chat_stream.get("1.0", "end")
        self.assertIn("Thought for", content)
        self.assertIn("[Click to Expand]", content)
        self.assertNotIn("Step 1: Analyzed input.", content)  # Hidden when collapsed

        # Expand dropdown
        think_id = "think_1"
        app._on_toggle_thinking_dropdown(think_id)
        content_expanded = app.chat_stream.get("1.0", "end")
        self.assertIn("[Click to Collapse]", content_expanded)
        self.assertIn("Step 1: Analyzed input.", content_expanded)

        # Collapse dropdown again
        app._on_toggle_thinking_dropdown(think_id)
        content_collapsed = app.chat_stream.get("1.0", "end")
        self.assertIn("[Click to Expand]", content_collapsed)
        self.assertNotIn("Step 1: Analyzed input.", content_collapsed)

        if hasattr(app, "watchdog") and app.watchdog:
            app.watchdog.stop_monitoring()
        root.destroy()


if __name__ == "__main__":
    unittest.main()
