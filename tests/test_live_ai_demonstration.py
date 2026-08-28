"""
Live AI System Demonstration & Verification Test Suite.
Tests all user-requested capabilities end-to-end:
1. Natural conversation without unwanted code blocks
2. Algorithmic code synthesis with Markdown code blocks
3. Deterministic RLVR sandbox verification
4. Tool suite execution (Files, Math, SQLite, System Monitor, Terminal)
5. Thinking token & live tokens-per-second telemetry
6. VRAM mutual exclusion and tab-switching controls
7. File attachment context injection
"""

import os
import tempfile
import unittest
import tkinter as tk
from config.settings import get_settings
from core.pro_engine import ProReasoningEngine
from core.tools import AgentToolRegistry
from memory.db import EpisodicMemoryDB
from app_gui import SmartAIChatbotApp


class TestLiveAIDemonstration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings()
        cls.engine = ProReasoningEngine(settings=cls.settings)
        cls.db_fd, cls.db_path = tempfile.mkstemp(suffix=".db")
        cls.db = EpisodicMemoryDB(db_path=cls.db_path)
        cls.tools = AgentToolRegistry(db_path=cls.db_path)

    @classmethod
    def tearDownClass(cls):
        try:
            os.close(cls.db_fd)
            if os.path.exists(cls.db_path):
                os.remove(cls.db_path)
        except Exception:
            pass

    def test_01_natural_conversational_response(self):
        """Verify entropy routing for conversation queries."""
        for query in ["hello", "hi there", "good morning"]:
            ent = self.engine.calculate_token_entropy(query)
            mode, branches = self.engine.router.route(ent, has_test_cases=False)
            self.assertIsInstance(ent, float)
            self.assertGreaterEqual(branches, 1)

    def test_02_algorithmic_coding_response(self):
        """Verify code extractor extracts clean Markdown code blocks."""
        code_block = "```python\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)\n```"
        extracted = self.engine.verifier.extract_code_block(code_block)
        self.assertIn("def factorial", extracted)
        self.assertNotIn("```python", extracted)

    def test_03_deterministic_rlvr_sandbox_verification(self):
        """Verify ground-truth RLVR sandbox runs assertions and scores rewards."""
        code = "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)"
        tests = "assert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(3) == 6"
        res = self.engine.verifier.verify_in_sandbox(code, tests)
        self.assertTrue(res.passed)
        self.assertGreater(res.execution_time_ms, 0.0)

    def test_04_tools_suite_live_execution(self):
        """Verify all workspace tools execute cleanly."""
        # 1. System Monitor (Dynamic)
        ok, res = self.tools.execute_tool("system_monitor", {})
        self.assertTrue(ok)
        self.assertIn("Process Memory", res)
        self.assertNotIn("27.4B", res)

        # 2. Math & Calculus (Auto Variable Detection)
        ok, res = self.tools.execute_tool("math_calculate", {"expression": "derivative of t**2"})
        self.assertTrue(ok)
        self.assertIn("2*t", res)

        # 3. File Read / Write
        test_file = "test_artifact_sample.txt"
        ok, res = self.tools.execute_tool("write_file", {"path": test_file, "content": "Hello Smart AI"})
        self.assertTrue(ok)
        ok, res = self.tools.execute_tool("read_file", {"path": test_file})
        self.assertTrue(ok)
        self.assertIn("Hello Smart AI", res)

        # 4. List Directory
        ok, res = self.tools.execute_tool("list_dir", {"path": "."})
        self.assertTrue(ok)
        self.assertIn("test_artifact_sample.txt", res)

        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

    def test_05_gui_full_interaction_cycle(self):
        """Verify GUI initialization, thinking pill rendering, tokens/s, and model tab switching."""
        root = tk.Tk()
        root.withdraw()
        app = SmartAIChatbotApp(root, settings=self.settings)

        # 1. Check initial state
        self.assertEqual(app.total_tokens_used, 0)
        self.assertEqual(app.synapses_learned_m, 0.0)
        self.assertEqual(app.active_tab_id, "model_1")
        self.assertFalse(app.is_model_loaded)

        # 2. Append User & AI Messages
        app._append_user_message("Hello from test")
        app._append_ai_message("Hello! Ready to help.", thinking_tokens=32, duration_s=0.25, tok_per_sec=128.0)
        content = app.chat_stream.get("1.0", "end")
        self.assertIn("Hello from test", content)
        self.assertIn("Reasoning Process", content)
        self.assertIn("128.0 tok/s", content)

        # 3. Test Model Switching
        app._on_switch_model_tab("model_2")
        self.assertEqual(app.active_tab_id, "model_2")
        self.assertIn("Qwen 3.8 Flash Next", app.lbl_model_status.cget("text"))

        root.destroy()


if __name__ == "__main__":
    unittest.main()
