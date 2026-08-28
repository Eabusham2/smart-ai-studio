"""
Comprehensive Test Suite for Smart AI (app_gui.py) and Full Tool Registry (core/tools.py).
Validates 15+ Agentic Tools, Math / SymPy engine, File operations, SQL memory,
and Headless GUI conversation workflows.
"""

import json
import os
import tempfile
import tkinter as tk
import unittest

from app_gui import SmartAIChatbotApp
from config.settings import Settings
from core.tools import AgentToolRegistry


class TestSmartAISuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        self.db_path = os.path.join(self.temp_dir.name, "test_smart_memory.db")
        self.settings = Settings(
            database_path=self.db_path,
            device="cpu",
            mlx_model_path="prism-ml/Ternary-Bonsai-27B-mlx-2bit"
        )
        self.app = SmartAIChatbotApp(self.root, settings=self.settings)
        self.tools = AgentToolRegistry(db_path=self.db_path, workspace_dir=self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    # -------------------------------------------------------------
    # TOOL TESTS
    # -------------------------------------------------------------
    def test_file_tools_lifecycle(self):
        """Test write_file, read_file, edit_file, list_dir, and file_search."""
        # 1. Write file
        ok, res = self.tools.execute_tool("write_file", {"path": "test.py", "content": "print('hello smart ai')\n"})
        self.assertTrue(ok)
        self.assertIn("Successfully wrote", res)

        # 2. Read file
        ok, content = self.tools.execute_tool("read_file", {"path": "test.py"})
        self.assertTrue(ok)
        self.assertIn("print('hello smart ai')", content)

        # 3. Edit file
        ok, edit_res = self.tools.execute_tool("edit_file", {"path": "test.py", "target": "hello smart ai", "replacement": "hello world"})
        self.assertTrue(ok)
        self.assertIn("Successfully replaced", edit_res)

        # Read back edited
        ok, content_edited = self.tools.execute_tool("read_file", {"path": "test.py"})
        self.assertIn("print('hello world')", content_edited)

        # 4. List dir
        ok, listing = self.tools.execute_tool("list_dir", {"path": "."})
        self.assertTrue(ok)
        self.assertIn("test.py", listing)

        # 5. File search
        ok, found = self.tools.execute_tool("file_search", {"pattern": "*.py"})
        self.assertTrue(ok)
        self.assertIn("test.py", found)

    def test_math_and_calculus_tool(self):
        """Test symbolic math, algebra, derivatives, and arithmetic."""
        # Derivative: d/dx(x^3 + 2*x) = 3*x^2 + 2
        ok, diff_res = self.tools.execute_tool("math_calculate", {"expression": "diff(x**3 + 2*x, x)"})
        self.assertTrue(ok)
        self.assertIn("Calculus Result", diff_res)

        # Integral: integrate(3*x^2, x) = x^3
        ok, int_res = self.tools.execute_tool("math_calculate", {"expression": "integrate(3*x**2, x)"})
        self.assertTrue(ok)
        self.assertIn("Calculus Result", int_res)

        # Arithmetic
        ok, calc_res = self.tools.execute_tool("math_calculate", {"expression": "2**10 + 24"})
        self.assertTrue(ok)
        self.assertIn("1048", calc_res)

    def test_terminal_and_python_sandbox_tools(self):
        """Test local shell terminal command execution and Python sandbox."""
        # Terminal execution
        ok, term_out = self.tools.execute_tool("run_terminal", {"command": "echo 'Smart AI 27B Online'"})
        self.assertTrue(ok)
        self.assertIn("Smart AI 27B Online", term_out)

        # Python sandbox execution
        ok, py_out = self.tools.execute_tool("python_sandbox", {"code": "import math\nprint(math.sqrt(144))"})
        self.assertTrue(ok)
        self.assertIn("12.0", py_out)

    def test_web_search_and_system_monitor(self):
        """Test Web search and system hardware monitor."""
        # Web search
        ok, web_res = self.tools.execute_tool("web_search", {"query": "Ternary weights in LLMs"})
        self.assertTrue(ok)
        self.assertIn("Web Search", web_res)

        # System monitor
        ok, sys_res = self.tools.execute_tool("system_monitor", {})
        self.assertTrue(ok)
        self.assertIn("System Hardware & Telemetry Profile", sys_res)
        self.assertIn("AI Model Precision", sys_res)

    def test_json_csv_analyzer_tool(self):
        """Test JSON and CSV analysis tools."""
        json_file = os.path.join(self.temp_dir.name, "sample.json")
        with open(json_file, "w") as f:
            json.dump({"name": "SmartAI", "version": "2.0", "parameters": "27B"}, f)

        ok, json_res = self.tools.execute_tool("json_csv_analyzer", {"path": "sample.json"})
        self.assertTrue(ok)
        self.assertIn("JSON Analysis", json_res)
        self.assertIn("name", json_res)

    def test_mcp_discovery_and_sql_query(self):
        """Test Model Context Protocol (MCP) tool registry and SQL queries."""
        ok, mcp_out = self.tools.execute_tool("mcp_list_tools", {})
        self.assertTrue(ok)
        self.assertIn("filesystem", mcp_out)
        self.assertIn("system_terminal", mcp_out)

        # Log a record into SQLite database first
        self.app.db.log_interaction("Test Prompt", "Test Answer", ["Test Answer"], 1.0, 0.5, "Instant", 0.1, 0, "")

        # SQL query on memory.db
        ok, sql_out = self.tools.execute_tool("sql_query", {"query": "SELECT id, prompt FROM interactions LIMIT 5"})
        self.assertTrue(ok)
        self.assertIn("SQL Query Results", sql_out)

    # -------------------------------------------------------------
    # GUI & CHAT STREAM TESTS
    # -------------------------------------------------------------
    def test_gui_telemetry_hud_and_elements(self):
        """Verify Smart AI telemetry badges, buttons, and HUD are constructed."""
        self.assertIsNotNone(self.app.main_container)
        self.assertIsNotNone(self.app.lbl_params)
        self.assertIsNotNone(self.app.lbl_synapses)
        self.assertIsNotNone(self.app.lbl_context)
        self.assertIsNotNone(self.app.lbl_vram)
        self.assertIn("Base", self.app.lbl_params.cget("text"))

    def test_gui_chat_stream_and_tool_call_rendering(self):
        """Verify rendering of user messages, assistant responses, and tool pills in chat."""
        self.app._append_user_message("Calculate derivative of x^2")
        self.app._append_tool_call("math_calculate", "expr='x^2'", "2*x")
        self.app._append_ai_message("The derivative of x^2 with respect to x is 2x.")

        chat_text = self.app.chat_stream.get("1.0", "end")
        self.assertIn("Calculate derivative of x^2", chat_text)
        self.assertIn("Tool Execution: math_calculate", chat_text)
        self.assertIn("derivative of x^2 with respect to x is 2x", chat_text)

    def test_gui_new_conversation_reset(self):
        """Verify starting a new conversation clears stream and resets context tokens."""
        self.app._append_user_message("Old test prompt")
        self.app._on_new_chat()
        chat_text = self.app.chat_stream.get("1.0", "end")
        self.assertEqual(self.app.total_tokens_used, 0)
        self.assertNotIn("Old test prompt", chat_text)


if __name__ == "__main__":
    unittest.main()
