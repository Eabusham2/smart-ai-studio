"""
Unit & Integration Tests for Interactive Visual Maker & Generative UI Engine.
Verifies:
1. Interactive multi-series charts (Line, Bar, Scatter) with live tooltips & dynamic scale sliders.
2. Architecture flowcharts, sequence DAGs, and state machine diagrams.
3. Deep neural network layer visualizers with synaptic excitation pulses.
4. Mathematical & physics simulations with live frequency/amplitude sliders.
5. KPI & metric dashboards with progress rings.
6. Generated HTML5 standalone validity, companion SVG vector creation, and file persistence.
7. AgentToolRegistry integration for 'interactive_visual_maker' tool.
"""

import json
import os
import tempfile
import unittest
from core.tools import AgentToolRegistry
from core.visual_maker import InteractiveVisualMaker


class TestInteractiveVisualMaker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.maker = InteractiveVisualMaker(workspace_dir=cls.temp_dir)
        cls.tools = AgentToolRegistry(workspace_dir=cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_01_chart_visualization_generation(self):
        """Verify interactive chart generation creates HTML and companion SVG."""
        data = {
            "categories": ["Q1", "Q2", "Q3", "Q4"],
            "series": [
                {"name": "Revenue", "data": [100, 150, 220, 310], "color": "#38bdf8"},
                {"name": "Users", "data": [50, 80, 140, 260], "color": "#22c55e"}
            ]
        }
        ok, report, html_p, svg_p = self.maker.create_visualization(
            visual_type="chart",
            title="Quarterly Growth Metrics",
            data_or_spec=data,
            theme="obsidian",
            filename="quarterly_growth.html"
        )
        self.assertTrue(ok)
        self.assertIn("Quarterly Growth Metrics", report)
        self.assertTrue(os.path.exists(html_p))
        self.assertTrue(os.path.exists(svg_p))

        with open(html_p, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("visualCanvas", content)
            self.assertIn("slider_scale", content)
            self.assertIn("Revenue", content)

    def test_02_neural_net_visualization(self):
        """Verify deep neural network architecture layer visualizer."""
        ok, report, html_p, svg_p = self.maker.create_visualization(
            visual_type="neural_net",
            title="Ternary Bonsai 27B Attention Architecture",
            theme="cyberpunk"
        )
        self.assertTrue(ok)
        self.assertIn("NEURAL_NET", report)
        self.assertTrue(os.path.exists(html_p))
        with open(html_p, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("drawNeuralNet", content)
            self.assertIn("BitLinear", content)

    def test_03_mathematical_physics_simulation(self):
        """Verify mathematical function & physics simulation plot with sliders."""
        ok, report, html_p, svg_p = self.maker.create_visualization(
            visual_type="simulation",
            title="Damped Harmonic Oscillator Wavefunction",
            theme="obsidian"
        )
        self.assertTrue(ok)
        self.assertIn("SIMULATION", report)
        self.assertTrue(os.path.exists(html_p))
        with open(html_p, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("drawSimulation", content)
            self.assertIn("slider_amp", content)
            self.assertIn("slider_freq", content)

    def test_04_diagram_and_dashboard_generation(self):
        """Verify system flowchart diagram and multi-metric KPI dashboard."""
        # 1. Diagram
        ok_d, _, html_d, _ = self.maker.create_visualization(
            visual_type="diagram",
            title="Autonomous RLVR Pipeline Flow"
        )
        self.assertTrue(ok_d)
        self.assertTrue(os.path.exists(html_d))

        # 2. Dashboard
        ok_dash, _, html_dash, _ = self.maker.create_visualization(
            visual_type="dashboard",
            title="Executive Model Telemetry"
        )
        self.assertTrue(ok_dash)
        self.assertTrue(os.path.exists(html_dash))

    def test_05_agent_tool_registry_interactive_visual_maker(self):
        """Verify AgentToolRegistry executes interactive_visual_maker tool cleanly."""
        ok, res = self.tools.execute_tool("interactive_visual_maker", {
            "visual_type": "chart",
            "title": "Agent Reasoning Accuracy Benchmark",
            "description": "Comparative Evaluation across 13 suites",
            "theme": "obsidian",
            "filename": "benchmark_summary.html"
        })
        self.assertTrue(ok)
        self.assertIn("Interactive Visualizer Created", res)
        self.assertIn("benchmark_summary.html", res)
        self.assertIn("Interactive HTML Artifact", res)


if __name__ == "__main__":
    unittest.main()
