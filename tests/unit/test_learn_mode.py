"""Tests for honest /learn semantics without loading real 27B weights in CI."""

import os
import tempfile
import threading
import unittest

from config.settings import get_settings
from core.autonomous_learner import AutonomousLearner
from memory.db import EpisodicMemoryDB


SOURCE_SENTENCE = "Raft uses a replicated log and majority quorum to commit entries safely."


class FakeTools:
    def execute_tool(self, name, args):
        if name == "web_fetch":
            return True, f"Page content: {SOURCE_SENTENCE}"
        if name == "web_crawler":
            return True, f"Web Crawler Research Dossier: {SOURCE_SENTENCE}"
        if name == "web_search":
            return True, f"Search evidence: {SOURCE_SENTENCE}"
        return False, ""


class NoResultTools:
    def execute_tool(self, name, args):
        if name == "web_search":
            return True, "No results found. The search service may be unavailable."
        if name == "web_crawler":
            return True, "Foundational concepts, API architectures, and execution rules for fake topic."
        return False, ""


class FakeArray:
    def __init__(self, size=2_000_000):
        self.size = size


class FakeMLXBackend:
    def __init__(self, adapter_path):
        self.model = object()
        self.tokenizer = object()
        self.is_mlx_available = True
        self.adapter_path = adapter_path
        self.adapters = {"lora": FakeArray()}
        self.train_calls = 0

    def compute_mlx_fisher(self, anchors):
        return {"lora": FakeArray()}

    def train_mini_batch(self, adapters, data, **kwargs):
        self.train_calls += 1
        save_path = kwargs.get("save_path")
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(b"real-test-adapter")
        self.adapters = {"lora": FakeArray()}
        return self.adapters, 0.0125


class FakeEngine:
    def __init__(self, adapter_path):
        self.mlx_backend = FakeMLXBackend(adapter_path)
        self.lora_adapter_path = adapter_path

    def solve(self, prompt, **kwargs):
        if "return ONLY one JSON object" in prompt:
            return (
                '{"claim":"Raft commits entries using a majority quorum.",'
                f'"evidence":"{SOURCE_SENTENCE}"}}',
                {"mode": "test"},
            )
        return (
            "Raft maintains a replicated log and commits entries after majority-quorum agreement.",
            {"mode": "test"},
        )


class TestAutonomousLearnMode(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "learn.db")
        self.adapter_path = os.path.join(self.temp_dir.name, "adapter.safetensors")
        self.settings = get_settings(
            use_mock=True,
            database_path=self.db_path,
            lora_adapter_path=self.adapter_path,
        )
        self.db = EpisodicMemoryDB(db_path=self.db_path)
        self.engine = FakeEngine(self.adapter_path)
        self.tools = FakeTools()
        self.learner = AutonomousLearner(
            engine=self.engine,
            tools=self.tools,
            db=self.db,
            settings=self.settings,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_topic_research_uses_real_search_only_not_fake_topic_crawler(self):
        research = self.learner.crawl_and_research("distributed consensus raft")
        self.assertEqual(research["sources_found"], 1)
        self.assertEqual(research["crawl_report"], "")
        self.assertIn(SOURCE_SENTENCE, research["search_report"])

    def test_02_direct_url_research_accepts_real_fetch_and_crawl(self):
        research = self.learner.crawl_and_research("https://example.test/raft")
        self.assertEqual(research["sources_found"], 2)
        self.assertIn(SOURCE_SENTENCE, research["crawl_report"])
        self.assertEqual(research["search_report"], "")

    def test_03_unavailable_or_fabricated_research_is_rejected(self):
        learner = AutonomousLearner(
            engine=self.engine,
            tools=NoResultTools(),
            db=self.db,
            settings=self.settings,
        )
        research = learner.crawl_and_research("fake topic")
        self.assertEqual(research["sources_found"], 0)
        self.assertEqual(learner._source_blob(research), "")
        with self.assertRaisesRegex(RuntimeError, "no usable source material"):
            learner.synthesize_knowledge("fake topic", research)

    def test_04_synthesis_is_generated_by_active_model_not_hardcoded_code(self):
        research = self.learner.crawl_and_research("distributed consensus raft")
        synthesis = self.learner.synthesize_knowledge("distributed consensus raft", research)
        self.assertIn("replicated log", synthesis)
        self.assertNotIn("def solve_", synthesis)
        self.assertNotIn("return True", synthesis)

    def test_05_self_test_requires_verbatim_source_evidence(self):
        research = self.learner.crawl_and_research("distributed consensus raft")
        synthesis = self.learner.synthesize_knowledge("distributed consensus raft", research)
        passed, details, reward = self.learner.self_test_and_verify(
            "distributed consensus raft", research, synthesis
        )
        self.assertTrue(passed)
        self.assertEqual(reward, 1.0)
        self.assertIn("Source-grounded verification passed", details)

    def test_06_parametric_consolidation_updates_same_active_mlx_backend(self):
        result = self.learner.consolidate_parameters(
            "distributed consensus raft",
            "Raft maintains a replicated log and majority quorum.",
            reward=1.0,
        )
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["parameter_drift_l2"], 0.0)
        self.assertGreater(result["trainable_parameters_m"], 0.0)
        self.assertEqual(self.engine.mlx_backend.train_calls, 1)
        self.assertTrue(os.path.exists(self.adapter_path))
        stats = self.db.get_stats()
        self.assertGreaterEqual(stats["total_interactions"], 1)
        self.assertEqual(stats["unconsolidated_verified"], 0)

    def test_07_full_learning_session_reports_measured_parameter_change(self):
        stages = []
        res = self.learner.run_learning_session(
            topic="/learn distributed consensus raft",
            progress_callback=lambda s, m, d: stages.append((s, m, d)),
            max_cycles=1,
        )
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["cycles_completed"], 1)
        self.assertGreater(res["synapses_learned_m"], 0.0)
        self.assertGreater(res["parameter_drift_l2"], 0.0)
        stage_names = [x[0] for x in stages]
        for expected in ("init", "crawling", "synthesizing", "verifying", "consolidating", "done"):
            self.assertIn(expected, stage_names)
        consolidated = next(x for x in stages if x[0] == "consolidating")
        self.assertIn("||ΔW||", consolidated[1])

    def test_08_learning_session_cancellation(self):
        cancel_event = threading.Event()
        cancel_event.set()
        stages = []
        res = self.learner.run_learning_session(
            topic="deep reinforcement learning",
            cancel_event=cancel_event,
            progress_callback=lambda s, m, d: stages.append(s),
            max_cycles=2,
        )
        self.assertEqual(res["status"], "cancelled")
        self.assertEqual(res["cycles_completed"], 0)
        self.assertIn("stopped", stages)


if __name__ == "__main__":
    unittest.main()
