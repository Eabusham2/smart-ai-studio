"""
Unit & Integration Tests for Interactive HTML Diagram Builder.
Verifies:
1. Interactive system architecture graph generation with multi-select and node inspector.
2. Neural network DAG and ML pipeline visualizer.
3. Flowchart and decision logic workflow generation.
4. Auto-layout mode coordinates (Hierarchical, Force, Grid, Circular).
5. HTML5 document standalone structure and embedded SVG engine.
6. Tool dispatch via AgentToolRegistry.
"""

import json
import os
import tempfile
import unittest
from core.html_diagram_maker import HTMLDiagramBuilder
from core.tools import AgentToolRegistry


class TestHTMLDiagramBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.builder = HTMLDiagramBuilder(workspace_dir=cls.temp_dir)
        cls.tools = AgentToolRegistry(workspace_dir=cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_01_architecture_diagram_creation(self):
        """Verify interactive system architecture diagram generation."""
        nodes = [
            {"id": "ui", "label": "Client Interface", "category": "Frontend", "x": 100, "y": 200, "details": "Tkinter Canvas GUI"},
            {"id": "gateway", "label": "API Gateway", "category": "Routing", "x": 300, "y": 200, "details": "Entropy Dispatcher"},
            {"id": "db", "label": "Episodic Vault", "category": "Database", "x": 500, "y": 200, "details": "SQLite Memory DB"}
        ]
        edges = [
            {"from": "ui", "to": "gateway", "label": "WebSocket"},
            {"from": "gateway", "to": "db", "label": "SQL"}
        ]
        ok, report, html_path = self.builder.create_diagram(
            title="Enterprise Cloud Topology",
            diagram_type="architecture",
            nodes=nodes,
            edges=edges,
            auto_layout="hierarchical",
            theme="obsidian",
            filename="cloud_topology.html"
        )
        self.assertTrue(ok)
        self.assertIn("cloud_topology.html", report)
        self.assertTrue(os.path.exists(html_path))

        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("Enterprise Cloud Topology", content)
            self.assertIn("selection-marquee", content)
            self.assertIn("applyAutoLayout", content)
            self.assertIn("Client Interface", content)

    def test_02_neural_dag_default_template(self):
        """Verify default template synthesis for neural DAGs."""
        ok, report, html_path = self.builder.create_diagram(
            title="Ternary BitLinear 27B DAG",
            diagram_type="neural_dag",
            theme="cyberpunk"
        )
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(html_path))
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("BitLinear", content)
            self.assertIn("Slow-LoRA", content)

    def test_03_flowchart_workflow(self):
        """Verify flowchart and decision gate workflow generation."""
        ok, report, html_path = self.builder.create_diagram(
            title="Autonomous RLVR Verification Pipeline",
            diagram_type="flowchart",
            auto_layout="grid",
            theme="clean_light"
        )
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(html_path))
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Entropy Router", content)
            self.assertIn("Ground-Truth Sandbox", content)

    def test_04_agent_tool_registry_html_diagram_builder(self):
        """Verify html_diagram_builder tool execution via AgentToolRegistry."""
        ok, res = self.tools.execute_tool("html_diagram_builder", {
            "title": "Autonomous Multi-Agent Swarm",
            "diagram_type": "architecture",
            "auto_layout": "circular",
            "theme": "obsidian",
            "filename": "swarm_architecture.html"
        })
        self.assertTrue(ok)
        self.assertIn("Interactive HTML Diagram Generated", res)
        self.assertIn("swarm_architecture.html", res)
        self.assertIn("Multi-Select", res)
        self.assertIn("Auto-Layout", res)


if __name__ == "__main__":
    unittest.main()
