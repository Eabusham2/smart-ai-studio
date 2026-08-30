"""
Unit & Integration Tests for Autonomous /learn Mode and Web Crawler.
Verifies:
1. Web crawler tool crawling topics and URLs
2. Autonomous research, synthesis, and sandbox self-testing
3. Slow-LoRA EWC parametric consolidation during learning
4. Interactive progress callbacks and chat stream updates
5. Immediate cancellation via cancel_event
"""

import os
import tempfile
import threading
import unittest
from config.settings import get_settings
from core.autonomous_learner import AutonomousLearner
from core.pro_engine import ProReasoningEngine
from core.tools import AgentToolRegistry
from memory.db import EpisodicMemoryDB


class TestAutonomousLearnMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings()
        cls.db_fd, cls.db_path = tempfile.mkstemp(suffix=".db")
        cls.db = EpisodicMemoryDB(db_path=cls.db_path)
        cls.tools = AgentToolRegistry(db_path=cls.db_path)
        cls.engine = ProReasoningEngine(settings=cls.settings)
        cls.learner = AutonomousLearner(engine=cls.engine, tools=cls.tools, db=cls.db, settings=cls.settings)

    @classmethod
    def tearDownClass(cls):
        try:
            os.close(cls.db_fd)
            if os.path.exists(cls.db_path):
                os.remove(cls.db_path)
        except Exception:
            pass

    def test_01_web_crawler_tool_execution(self):
        """Verify web_crawler tool executes for topics and URLs."""
        ok, res = self.tools.execute_tool("web_crawler", {"query_or_url": "quantum computing algorithms", "max_pages": 2})
        self.assertTrue(ok)
        self.assertIn("Web Crawler Research Dossier", res)
        self.assertIn("quantum computing algorithms", res)

    def test_02_learner_research_and_synthesis(self):
        """Verify autonomous research gathering and structured knowledge synthesis."""
        research = self.learner.crawl_and_research("distributed consensus raft")
        self.assertIn("topic", research)
        self.assertIn("crawl_report", research)

        synthesis = self.learner.synthesize_knowledge("distributed consensus raft", research)
        self.assertIn("Synthesized Knowledge Base", synthesis)
        self.assertIn("def solve_", synthesis)

    def test_03_learner_self_testing_rlvr(self):
        """Verify learner formulates assertions and validates in ground-truth sandbox."""
        passed, details, reward = self.learner.self_test_and_verify("graph neural networks")
        self.assertTrue(passed)
        self.assertEqual(reward, 1.0)
        self.assertIn("Sandbox Verification", details)

    def test_04_learner_parametric_consolidation(self):
        """Verify learned traces are logged to episodic database and consolidated."""
        res = self.learner.consolidate_parameters("quantum key distribution", "Synthesis content for QKD", reward=1.0)
        self.assertIn("status", res)
        stats = self.db.get_stats()
        self.assertGreaterEqual(stats["total_interactions"], 1)

    def test_05_full_learning_session_progression(self):
        """Verify multi-cycle learning session progresses through all stages and fires callbacks."""
        stages_recorded = []
        messages_recorded = []

        def callback(stage, message, syn_delta):
            stages_recorded.append(stage)
            messages_recorded.append(message)

        res = self.learner.run_learning_session(
            topic="/learn distributed caching architectures",
            cancel_event=None,
            progress_callback=callback,
            max_cycles=1
        )
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["cycles_completed"], 1)
        self.assertGreater(res["synapses_learned_m"], 0.0)
        self.assertIn("init", stages_recorded)
        self.assertIn("crawling", stages_recorded)
        self.assertIn("synthesizing", stages_recorded)
        self.assertIn("verifying", stages_recorded)
        self.assertIn("consolidating", stages_recorded)
        self.assertIn("done", stages_recorded)

    def test_06_learning_session_cancellation(self):
        """Verify learning session halts immediately when cancel_event is set."""
        cancel_event = threading.Event()
        cancel_event.set()  # Pre-cancelled

        stages = []
        res = self.learner.run_learning_session(
            topic="deep reinforcement learning",
            cancel_event=cancel_event,
            progress_callback=lambda s, m, d: stages.append(s),
            max_cycles=2
        )
        self.assertEqual(res["status"], "cancelled")
        self.assertEqual(res["cycles_completed"], 0)


if __name__ == "__main__":
    unittest.main()
