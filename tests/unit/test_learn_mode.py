"""Tests for honest /learn semantics without loading real 27B weights in CI."""

import os
import tempfile
import threading
from pathlib import Path
import unittest

from config.settings import get_settings
from core.autonomous_learner import AutonomousLearner
from memory.db import EpisodicMemoryDB
from tests.unit.context_backend_fixture import TransactionalBackend


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



class FakeEngine:
    def __init__(self, adapter_path):
        self.mlx_backend = TransactionalBackend(os.path.dirname(adapter_path))
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
        self.assertTrue(os.path.isfile(result["adapter_saved_to"]))
        stats = self.db.get_stats()
        self.assertGreaterEqual(stats["total_interactions"], 1)
        self.assertEqual(stats["unconsolidated_verified"], 0)

    def test_attachment_preview_reaches_content_after_whitespace_and_reports_media_kinds(self):
        folder = Path(self.temp_dir.name) / "mixed"
        folder.mkdir()
        (folder / "record.txt").write_text(" " * 6144 + SOURCE_SENTENCE)
        (folder / "second.txt").write_text("Another documented fact.")
        (folder / "example.png").touch()
        result = self.learner.research_from_source(folder, "records")
        assert result["sources_found"] == 2 and result["media_kinds"] == ["image"]
        assert SOURCE_SENTENCE in result["source_text"] and "second.txt" in result["source_text"]
        assert len(result["source_text"]) <= 4096 and result["stream"]["status"] == "completed"

    def test_verified_note_is_supervised_and_its_source_witness_is_bound_to_that_note(self):
        from unittest.mock import patch
        topic = "distributed consensus raft"
        research = self.learner.crawl_and_research(topic)
        note = self.learner.synthesize_knowledge(topic, research)
        assert self.learner.self_test_and_verify(topic, research, note)[0]
        service = self.learner._consolidation_service()
        with patch.object(service, "train_pairs", wraps=service.train_pairs) as train:
            self.learner.consolidate_parameters(topic, note, reward=1.0)
            rows, = train.call_args.args
            assert rows[0]["completion"] == note != SOURCE_SENTENCE
            assert train.call_args.kwargs["evidence"] == [SOURCE_SENTENCE]
            # Even an unconsumed verifier witness must not cross lesson/topic boundaries.
            assert self.learner.self_test_and_verify(topic, research, note)[0]
            direct_note = "The new archive label is green."
            self.learner.consolidate_parameters("archive labels", direct_note, reward=1.0)
            assert train.call_args.args[0][0]["completion"] == direct_note
            assert train.call_args.kwargs["evidence"] == [direct_note]

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

        # /learn is an explicit active-learning workflow layered on top of the
        # always-on chat consolidator: one verified Learn update, then one distinct
        # self-generated RSI update on the already-updated model.
        self.assertEqual(self.engine.mlx_backend.train_calls, 2)

        rsi_batch = self.engine.mlx_backend.received_batches[-1]
        self.assertEqual(len(rsi_batch), 1)
        self.assertNotIn("quorum", rsi_batch[0]["prompt"])
        self.assertNotIn("reward", rsi_batch[0])

        stage_names = [x[0] for x in stages]
        for expected in ("init", "crawling", "synthesizing", "verifying", "rsi", "consolidating", "done"):
            self.assertIn(expected, stage_names)
        consolidated = next(x for x in stages if x[0] == "consolidating")
        self.assertIn("Learn + RSI", consolidated[1])
        self.assertIn("||ΔW||", consolidated[1])

        # RSI training deliberately does not persist a second source question,
        # correctness/reward label, or benchmark-style interaction row.
        stats = self.db.get_stats()
        self.assertEqual(stats["total_interactions"], 1)

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
