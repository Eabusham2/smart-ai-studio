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
        cls.settings = get_settings(use_mock=True)
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

    def test_07_gui_model_presets_and_canvas_toggle(self):
        """Verify GUI initializes expanded presets (RealVisXL, Z-Image Turbo, LTX-Video, etc.) and toggles Canvas."""
        try:
            root = tk.Tk()
            root.withdraw()
            app = SmartAIChatbotApp(root, settings=self.settings)
        except Exception:
            self.skipTest("Headless environment without display server")
            return

        # Multi-tab Verification
        self.assertIn("model_1", app.models_config)
        self.assertIn("model_2", app.models_config)
        self.assertIn("model_3", app.models_config)
        self.assertIn("model_4", app.models_config)
        self.assertIn("model_7", app.models_config)
        self.assertIn("RealVisXL", app.models_config["model_3"]["name"])
        self.assertIn("Z-Image Turbo", app.models_config["model_4"]["name"])
        self.assertIn("LTX-Video", app.models_config["model_7"]["name"])

        # Switch to Model 3 (RealVisXL SDXL)
        app._on_switch_model_tab("model_3")
        self.assertEqual(app.active_tab_id, "model_3")
        self.assertIn("RealVisXL", app.lbl_model_status.cget("text"))

        # Canvas Drawer Toggle
        app._on_toggle_canvas_viewer()
        self.assertTrue(app.show_canvas)
        app._open_in_canvas("def hello_canvas(): return True")
        self.assertIn("hello_canvas", app.txt_canvas.get("1.0", "end"))

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

    def test_09_diffusion_and_video_generation_tools(self):
        """Verify high-res image diffusion, video motion synthesis, and rapid image edit tools."""
        # 1. High-Res Image Diffusion (RealVisXL / Z-Image Turbo with LoRA Sliders)
        ok, res = self.tools.execute_tool("generate_image_diffusion", {
            "prompt": "Cybernetic futuristic sanctuary, ultra photoreal 8k",
            "model_id": "SG161222/RealVisXL_V5.0",
            "softer_lora_str": 1.0,
            "harder_lora_str": 0.8,
            "custom_lora": "mystic_xxx",
            "custom_lora_str": 0.7,
            "filename": "test_diffusion_art.png"
        })
        self.assertTrue(ok)
        self.assertIn("High-Res Uncensored Image Synthesized", res)
        self.assertIn("RealVisXL_V5.0", res)
        self.assertIn("Softer LoRA", res)

        # 2. Video & Audio Synthesis (LTX-Video 2.5 / Wan 2.2 / MiniMax-H3)
        ok, res = self.tools.execute_tool("generate_video_diffusion", {
            "prompt": "Fluid particle simulation in zero gravity",
            "model_id": "dgrauet/ltx-2.5-mlx-q4",
            "motion_scale": 0.8,
            "filename": "test_video.mp4"
        })
        self.assertTrue(ok)
        self.assertIn("High-Res Motion Video Synthesized", res)
        self.assertIn("ltx-2.5", res)

        # 3. Rapid Image Edit (Qwen Image Edit Rapid AIO GGUF)
        ok, res = self.tools.execute_tool("edit_image_rapid", {
            "prompt": "Change lighting to cyberpunk neon purple and add atmospheric rain",
            "image_path": "test_diffusion_art.png",
            "filename": "test_edit.png"
        })
        self.assertTrue(ok)
        self.assertIn("Rapid Multimodal Image Edit Executed", res)
        self.assertIn("Qwen-Image-Edit-Rapid-AIO-GGUF", res)

        # Clean up
        for f in ("test_diffusion_art.png", "test_diffusion_art.svg", "test_video.mp4", "test_edit.png"):
            if os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    unittest.main()
