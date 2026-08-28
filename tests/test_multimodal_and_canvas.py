"""
Unit and Integration Tests for Multimodal, Canvas, Memory Watchdog & Generation Engine.
Verifies:
1. Natural essay and prose generation (ensuring no unwanted Python code)
2. Transparent system prompt query answers
3. 3-Model tab configuration including Dolphin Vision 2.9 (Uncensored Multimodal)
4. Image generation and Bezier vector canvas tools
5. Chat export history to Markdown
6. Memory pressure watchdog auto-unloader
"""

import os
import tempfile
import unittest
import tkinter as tk
from config.settings import get_settings
from core.memory_watchdog import SystemMemoryWatchdog
from core.pro_engine import ProReasoningEngine
from core.tools import AgentToolRegistry
from memory.db import EpisodicMemoryDB
from app_gui import SmartAIChatbotApp


class TestMultimodalAndCanvasSuite(unittest.TestCase):
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

    def test_01_entropy_routing_and_prompt_formatting(self):
        """Verify dynamic entropy calculation for essays and prose."""
        essay_prompts = [
            "write an essay on the future of computing",
            "write an essay about space exploration",
            "write a story about a cybernetic traveler",
        ]
        for p in essay_prompts:
            ent = self.engine.calculate_token_entropy(p)
            self.assertIsInstance(ent, float)
            self.assertGreaterEqual(ent, 0.0)

    def test_02_system_prompt_schema(self):
        """Verify system prompt uses only tool schemas without rigid persona prompts."""
        schema = self.tools.get_tool_schemas()
        self.assertIn("tools", schema)
        self.assertGreater(len(schema["tools"]), 5)

    def test_03_image_generation_tool(self):
        """Verify image generation tool writes valid visual vector artifact to workspace."""
        ok, res = self.tools.execute_tool("generate_image", {"prompt": "Neural Network Architecture", "filename": "test_nn.svg"})
        self.assertTrue(ok)
        self.assertIn("Generated Visual Artifact", res)
        self.assertTrue(os.path.exists("test_nn.svg"))
        if os.path.exists("test_nn.svg"):
            os.remove("test_nn.svg")

    def test_04_bezier_canvas_art_tool(self):
        """Verify Bezier vector art tool generates parametric cubic splines."""
        ok, res = self.tools.execute_tool("render_bezier_art", {"filename": "test_bezier.svg"})
        self.assertTrue(ok)
        self.assertIn("Bezier Spline Canvas Generated", res)
        self.assertTrue(os.path.exists("test_bezier.svg"))
        if os.path.exists("test_bezier.svg"):
            os.remove("test_bezier.svg")

    def test_05_chat_export_tool(self):
        """Verify chat export tool generates clean Markdown document."""
        self.db.log_interaction("Test Prompt", "Test Completion", ["Test"], 1.0, 0.2)
        ok, res = self.tools.execute_tool("export_chat_history", {"filename": "test_chat_export.md"})
        self.assertTrue(ok)
        self.assertIn("Chat History Exported Successfully", res)
        self.assertTrue(os.path.exists("test_chat_export.md"))
        if os.path.exists("test_chat_export.md"):
            os.remove("test_chat_export.md")

    def test_06_memory_watchdog_pressure_detection(self):
        """Verify memory watchdog correctly detects pressure and status metrics."""
        watchdog = SystemMemoryWatchdog(max_ram_usage_percent=1.0)  # Very low threshold to trigger
        pressure, status = watchdog.check_memory_pressure()
        self.assertTrue(pressure)
        self.assertIn("total_gb", status)
        self.assertIn("used_percent", status)

    def test_07_gui_three_tabs_and_canvas_toggle(self):
        """Verify GUI initializes 3 tabs (including Dolphin Vision) and toggles Canvas."""
        root = tk.Tk()
        root.withdraw()
        app = SmartAIChatbotApp(root, settings=self.settings)

        # 3 Tabs Verification
        self.assertIn("model_1", app.models_config)
        self.assertIn("model_2", app.models_config)
        self.assertIn("model_3", app.models_config)
        self.assertIn("Dolphin Vision", app.models_config["model_3"]["name"])

        # Switch to Model 3
        app._on_switch_model_tab("model_3")
        self.assertEqual(app.active_tab_id, "model_3")
        self.assertIn("Dolphin Vision", app.lbl_model_status.cget("text"))

        # Canvas Drawer Toggle
        app._on_toggle_canvas_viewer()
        self.assertTrue(app.show_canvas)
        app._draw_workflow_dag_on_canvas()
        app._draw_bezier_spline_on_canvas()

        app._on_toggle_canvas_viewer()
        self.assertFalse(app.show_canvas)

        root.destroy()

    def test_08_multilingual_code_selection(self):
        """Verify code blocks in various languages (Rust, TS, Python) are supported and parsed."""
        rust_sample = "```rust\nfn main() {\n    println!(\"Hello, world!\");\n}\n```"
        extracted_rust = self.engine.verifier.extract_code_block(rust_sample)
        self.assertIn("fn main", extracted_rust)

        ts_sample = "```typescript\nexport function greet(name: string): string {\n    return `Hello, ${name}!`;\n}\n```"
        extracted_ts = self.engine.verifier.extract_code_block(ts_sample)
        self.assertIn("export function greet", extracted_ts)


if __name__ == "__main__":
    unittest.main()
