"""Tests for honest /learn semantics without loading real 27B weights in CI."""

import os
import tempfile
import threading
from pathlib import Path
import unittest

from config.settings import get_settings
from core.autonomous_learner import ManagedAutonomousLearner
from memory.db import EpisodicMemoryDB
from tests.unit.context_backend_fixture import TransactionalBackend


from tests.unit.test_learn_mode import SOURCE_SENTENCE, FakeEngine as LegacyTestEngine, FakeTools, NoResultTools


class FakeEngine(LegacyTestEngine):
    def __init__(self, adapter_path):
        self.mlx_backend = TransactionalBackend(os.path.dirname(adapter_path))
        self.lora_adapter_path = adapter_path


class TestManagedAutonomousLearnMode(unittest.TestCase):
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
        self.learner = ManagedAutonomousLearner(
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
        learner = ManagedAutonomousLearner(
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
        assert result["stream"]["status"] == "completed"

    def test_learn_skips_invalid_documents_reports_errors_and_restores_attachment_context(self):
        from core.test_time_learning import OnlineLearning
        folder = Path(self.temp_dir.name) / "lessons"
        folder.mkdir()
        (folder / "broken.docx").write_bytes(b"bad archive")
        valid = folder / "notes.txt"
        valid.write_bytes(b"Topic " * 1200 + b"cafe\xff. " + SOURCE_SENTENCE.encode() + b" filler" * 2000)
        messages = []
        result = self.learner.research_from_source(folder, "raft", progress_callback=lambda *args: messages.append(args))
        assert result["extraction_errors"] and "broken.docx" in result["extraction_errors"][0]
        assert any(stage == "source_warnings" for stage, _, _ in messages)
        assert len(result["source_text"]) == 16000
        assert SOURCE_SENTENCE in self.learner._lesson_source(result, "raft")
        assert "cafe\ufffd." in result["source_text"]
        assert result["stream"]["source_chars"] == len(valid.read_bytes().decode("utf-8", "replace"))
        with self.assertRaises(UnicodeDecodeError):
            OnlineLearning(self.learner._consolidation_service()).run_request(
                {"method": "ingest", "source": str(valid), "chunk_chars": 30000})

    def test_legacy_learn_resumes_preexisting_strict_source_checkpoint(self):
        from core.test_time_learning import OnlineLearning
        source = Path(self.temp_dir.name) / "earlier.txt"
        source.write_text("A readable source statement. " * 250)
        service = self.learner._consolidation_service()
        first = OnlineLearning(service).run_request(
            {"method": "ingest", "source": str(source), "learning_rate": self.settings.consolidation_lr})
        updates = service.engine.train_calls
        result = self.learner.research_from_source(source, "earlier")
        assert result["stream"]["source_chars"] == first["source_chars"]
        assert result["stream"]["chunks_this_run"] == 0 and service.engine.train_calls == updates

    def test_unreadable_attachment_exposes_the_extraction_reason(self):
        path = Path(self.temp_dir.name) / "corrupt.docx"
        path.write_bytes(b"bad archive")
        with self.assertRaisesRegex(RuntimeError, "corrupt.docx: BadZipFile"):
            self.learner.research_from_source(path, "attachments")

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
