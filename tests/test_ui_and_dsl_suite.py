"""
Unit and Integration Test Suite for UI Dialogs, Telemetry, and Interactive DSL Engine.
Verifies:
1. TensorGraphDSL evaluation across non-commutative operators (>>~fold, <#>scale, @fuse_quant, ^mask_add)
2. GlyphScript symbolic logic and invariant checking
3. Interactive RLVR sandbox runner with memory and timeout limits
4. Agent tool execution of 'dsl_evaluator'
5. Desktop UI dialogs: Multi-Branch Visualizer, Memory Explorer, Sleep Consolidation Panel, DSL Playground
"""

import os
import tempfile
import unittest
import tkinter as tk

from config.settings import get_settings
from core.dsl_engine import evaluate_tensorgraph_dsl, evaluate_glyph_script, InteractiveDSLPlayground
from core.tools import AgentToolRegistry
from app_gui import SmartAIChatbotApp


class TestUIAndDSLSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings(use_mock=True)
        cls.db_fd, cls.db_path = tempfile.mkstemp(suffix=".db")
        cls.tools = AgentToolRegistry(db_path=cls.db_path)
        cls.playground = InteractiveDSLPlayground(sandbox_timeout=3.0, max_memory_mb=512)

    @classmethod
    def tearDownClass(cls):
        try:
            os.close(cls.db_fd)
            if os.path.exists(cls.db_path):
                os.remove(cls.db_path)
        except Exception:
            pass

    def test_01_tensorgraph_dsl_operators(self):
        """Verify TensorGraphDSL sequentially applies non-commutative operators."""
        # 1. Fold & Scale: [2, 4, 6] >>~fold(1) -> [4, 6, 2] -> <#>scale(3) -> [12, 18, 6]
        res1 = evaluate_tensorgraph_dsl("[2, 4, 6] >>~fold(1) <#>scale(3)")
        self.assertEqual(res1, [12, 18, 6])

        # 2. Fuse Quant: [1, 0, -1, 3] @fuse_quant(0.5) -> [1, 0, -1, 1]
        res2 = evaluate_tensorgraph_dsl("[1, 0, -1, 3] @fuse_quant(0.5)")
        self.assertEqual(res2, [1, 0, -1, 1])

        # 3. Mask Add: [10, 20, 30] ^mask_add([1, 0, 1], 5) -> [15, 20, 35]
        res3 = evaluate_tensorgraph_dsl("[10, 20, 30] ^mask_add([1, 0, 1], 5)")
        self.assertEqual(res3, [15, 20, 35])

        # 4. Scale then Fold: [1, 2, 3, 4] <#>scale(10) >>~fold(1) -> [20, 30, 40, 10]
        res4 = evaluate_tensorgraph_dsl("[1, 2, 3, 4] <#>scale(10) >>~fold(1)")
        self.assertEqual(res4, [20, 30, 40, 10])

    def test_02_glyph_script_invariants(self):
        """Verify GlyphScript validates DAG structure and invariants."""
        valid_script = """
        RULE: DAG_MONOTONIC_FLOW
        A -> B (5)
        B -> C (3)
        INVARIANT: ALL(weight > 0)
        """
        res = evaluate_glyph_script(valid_script)
        self.assertEqual(res["status"], "VALID_GLYPH_GRAPH")
        self.assertTrue(res["invariants_passed"])
        self.assertEqual(res["nodes_count"], 3)
        self.assertEqual(res["edges_count"], 2)

        invalid_script = """
        RULE: DAG_MONOTONIC_FLOW
        A -> B (-2)
        INVARIANT: ALL(weight > 0)
        """
        res_inv = evaluate_glyph_script(invalid_script)
        self.assertEqual(res_inv["status"], "INVARIANT_VIOLATION")
        self.assertFalse(res_inv["invariants_passed"])

    def test_03_playground_sandbox_runner(self):
        """Verify interactive sandbox executes code and checks assertions."""
        # Tensorgraph execution
        res_tg = self.playground.execute_dsl("tensorgraph", "[10, 20] <#>scale(5) >>~fold(1)")
        self.assertTrue(res_tg["passed"])
        self.assertEqual(res_tg["result"], [100, 50])

        # Python sandbox execution
        code = "def add(a, b): return a + b\n"
        tests = "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n"
        res_py = self.playground.execute_dsl("python", code, tests)
        self.assertTrue(res_py["passed"])
        self.assertEqual(res_py["exit_code"], 0)

    def test_04_dsl_evaluator_tool(self):
        """Verify AgentToolRegistry dsl_evaluator tool dispatch."""
        ok, res = self.tools.execute_tool("dsl_evaluator", {
            "dsl_type": "tensorgraph",
            "expression_or_code": "[8, 16, 24] >>~fold(2) <#>scale(0.5)"
        })
        self.assertTrue(ok)
        self.assertIn("TENSORGRAPH Execution Success", res)
        self.assertIn("[12, 4, 8]", res)

    def test_05_gui_dialogs_initialization(self):
        """Verify GUI initializes all 4 major dialogs and Canvas templates."""
        try:
            root = tk.Tk()
            root.withdraw()
        except Exception:
            self.skipTest("Headless CI environment without Tkinter display server")
            return

        try:
            app = SmartAIChatbotApp(root, settings=self.settings)

            # Test Canvas DSL template loader
            app._on_canvas_load_dsl_template()
            self.assertTrue(app.show_canvas)
            canvas_text = app.txt_canvas.get("1.0", "end")
            self.assertIn("TensorGraphDSL", canvas_text)

            # Test Multi-Branch Visualizer
            app._on_open_branch_visualizer()

            # Test Memory DB Explorer
            app._on_open_memory_explorer()

            # Test Sleep Consolidation Control Panel
            app._on_open_sleep_consolidation_panel()

            # Test Interactive DSL Playground Dialog
            app._on_open_dsl_playground()

        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
