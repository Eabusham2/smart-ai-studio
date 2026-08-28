"""
Unit tests for Smart AI Studio GUI (app_gui.py) and Tool System (core/tools.py).
Tests window initialization, telemetry HUD gauges, tool execution (Web, Terminal, Filesystem, Memory, MCP),
and conversational messaging in a headless Tkinter environment.
"""

import os
import tempfile
import tkinter as tk
import unittest

from app_gui import SmartAIChatbotApp
from config.settings import Settings
from core.tools import AgentToolRegistry


class TestDesktopAppGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create hidden root window for headless testing
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.root.destroy()
        except Exception:
            pass

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_gui_memory.db")
        self.settings = Settings(
            database_path=self.db_path,
            device="cpu",
            mlx_model_path="prism-ml/Ternary-Bonsai-27B-mlx-2bit"
        )
        self.app = SmartAIChatbotApp(self.root, settings=self.settings)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_chatbot_ui_initialization(self):
        """Verify chat stream, telemetry badges, and session list are initialized."""
        self.assertIsNotNone(self.app.chat_stream)
        self.assertIsNotNone(self.app.txt_input)
        self.assertIsNotNone(self.app.lbl_params)
        self.assertIsNotNone(self.app.lbl_context)

        # Check HUD telemetry content
        self.assertIn("Base", self.app.lbl_params.cget("text"))
        self.assertIn("Context:", self.app.lbl_context.cget("text"))

    def test_user_and_ai_message_rendering(self):
        """Test sending user and assistant messages into the styled chat stream."""
        self.app._append_user_message("Hello from unit test!")
        content = self.app.chat_stream.get("1.0", "end")
        self.assertIn("Hello from unit test!", content)

        self.app._append_ai_message("This is a response with ```python\nprint('code')\n```")
        content_after = self.app.chat_stream.get("1.0", "end")
        self.assertIn("This is a response", content_after)
        self.assertIn("print('code')", content_after)

    def test_tool_call_pill_rendering(self):
        """Test that tool execution pills are formatted correctly in the chat stream."""
        self.app._append_tool_call("web_search", "query='Smart AI'", "Found search summary.")
        content = self.app.chat_stream.get("1.0", "end")
        self.assertIn("web_search", content)
        self.assertIn("Found search summary", content)

    def test_agent_tool_registry_executions(self):
        """Test built-in tool executions across Web, Filesystem, Terminal, Python, and MCP."""
        tools = AgentToolRegistry(db_path=self.db_path, workspace_dir=self.temp_dir.name)

        # 1. Write file
        ok, res = tools.execute_tool("write_file", {"path": "hello.txt", "content": "Hello World!"})
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir.name, "hello.txt")))

        # 2. Read file
        ok, content = tools.execute_tool("read_file", {"path": "hello.txt"})
        self.assertTrue(ok)
        self.assertIn("Hello World!", content)

        # 3. List dir
        ok, listing = tools.execute_tool("list_dir", {"path": "."})
        self.assertTrue(ok)
        self.assertIn("hello.txt", listing)

        # 4. Terminal run
        ok, out = tools.execute_tool("run_terminal", {"command": "echo 'Smart AI Studio'"})
        self.assertTrue(ok)
        self.assertIn("Smart AI Studio", out)

        # 5. Python exec
        ok, py_out = tools.execute_tool("python_sandbox", {"code": "print(2**10)"})
        self.assertTrue(ok)
        self.assertIn("1024", py_out)

        # 6. Web search
        ok, web_out = tools.execute_tool("web_search", {"query": "1.58-bit LLM"})
        self.assertTrue(ok)
        self.assertTrue(len(web_out) > 0)

        # 7. MCP server discovery
        ok, mcp_out = tools.execute_tool("mcp_list_tools", {})
        self.assertTrue(ok)
        self.assertIn("filesystem", mcp_out)

    def test_new_chat_and_session_reset(self):
        """Test creating a new chat session resets the stream and context counters."""
        self.app._append_user_message("Test message")
        self.app._on_new_chat()
        content = self.app.chat_stream.get("1.0", "end")
        self.assertEqual(self.app.total_tokens_used, 0)
        self.assertIn("Smart AI Studio", content)

    def test_dual_model_tab_switching(self):
        """Test switching between Model 1 (Ternary Bonsai) and Model 2 (Qwen 3.8 Flash Next)."""
        self.assertEqual(self.app.active_tab_id, "model_1")
        
        # Switch to Model 2
        self.app._on_switch_model_tab("model_2")
        self.assertEqual(self.app.active_tab_id, "model_2")
        self.assertIn("Qwen 3.8 Flash Next", self.app.lbl_model_status.cget("text"))
        
        # Switch back to Model 1
        self.app._on_switch_model_tab("model_1")
        self.assertEqual(self.app.active_tab_id, "model_1")
        self.assertIn("Ternary Bonsai", self.app.lbl_model_status.cget("text"))

    def test_custom_model_importer(self):
        """Test dynamically registering and switching to a custom user-imported model."""
        custom_id = "custom_test_model"
        self.app.models_config[custom_id] = {
            "name": "Llama 3.2 3B (Custom)",
            "short_name": "Llama 3.2 3B",
            "repo_id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "model_path": None,
            "precision": "3.2B Custom",
            "raw_params": 3_200_000_000,
            "base_params": "3.2B",
            "max_context": 65_536,
            "vram": "2.2 GB / 16 GB",
            "tag": "🧩 Llama 3.2 3B",
            "accent": "#c084fc"
        }
        self.app.chat_history[custom_id] = []
        self.app._on_switch_model_tab(custom_id)
        self.assertEqual(self.app.active_tab_id, custom_id)
        self.assertIn("Llama 3.2 3B", self.app.lbl_model_status.cget("text"))


if __name__ == "__main__":
    unittest.main()
