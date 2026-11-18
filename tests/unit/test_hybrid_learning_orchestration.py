"""Hybrid lesson verification, backend dispatch, and durable retirement contracts."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from core.autonomous_learner import HybridAutonomousLearner, make_autonomous_learner
from core.online_consolidator import HybridAwakeOnlineConsolidator, make_online_consolidator
from tests.unit.context_backend_fixture import TransactionalBackend
from tests.unit.test_native_context_learning import tiny_controller


class RecordingBackend(TransactionalBackend):
    def train_mini_batch(self, *, learning_mode, **kwargs):
        assert learning_mode == "hybrid"
        return super().train_mini_batch(**kwargs)


def exchange(i):
    return [{"role": "user", "content": f"The archive label for sample {i} is cedar{i}."},
            {"role": "assistant", "content": f"I recorded sample {i} as cedar{i}."}]


def test_hybrid_commits_one_checkpoint_and_retires_only_after_full_recall(tmp_path):
    backend = RecordingBackend(tmp_path)
    service = make_online_consolidator(backend, implementation="hybrid", session_id="same-chat")
    assert type(service) is HybridAwakeOnlineConsolidator
    history = exchange(1)
    assert service.consolidate_chunk_sync(history)
    first = backend.learning_state.current_manifest()
    assert first["metadata"]["learning_implementation"] == "hybrid"
    assert backend.train_calls == 1 and service.active_history(history) == []

    backend.regress = True
    assert not service.consolidate_chunk_sync(exchange(2))
    assert backend.learning_state.current_manifest() == first
    assert service.active_history(history) == [] and service.active_history(exchange(2))
    assert backend.train_calls == 2 and service.consolidation_count == 1

    backend.regress = False
    backend.stream_generate_tokens = lambda *args, **kwargs: iter(["unknown"])
    assert not service.consolidate_chunk_sync(exchange(2))
    assert service.active_history(exchange(2))
    assert backend.train_calls == 3


def test_hybrid_canonicalizes_evidence_and_binds_it_to_one_lesson(tmp_path):
    source = "Die Straße trägt den Namen der Weide."
    engine = SimpleNamespace(active_backend="controller", controller_backend=RecordingBackend(tmp_path),
        solve=Mock(return_value=(json.dumps({"claim": "A street name", "evidence": source.upper()}), {})))
    settings = SimpleNamespace(learning_implementation="hybrid", ewc_lambda=0, consolidation_lr=.001)
    learner = make_autonomous_learner(engine=engine, tools=object(), db=Mock(), settings=settings)
    assert type(learner) is HybridAutonomousLearner
    # Retain the complete original excerpt, including text after the managed
    # short-report threshold. One verification call suffices for this source.
    research = {"source_text": "material " * 800 + source}
    note = "The street takes its name from a willow."
    assert learner._lesson_source(research, "street") == research["source_text"]
    assert learner.self_test_and_verify("street", research, note)[0]
    assert engine.solve.call_count == 1 and learner._verified_lesson[2] == source
    assert learner.consolidate_parameters("different topic", note, 2.0)["status"] == "skipped"
    assert learner.consolidate_parameters("street", note, float("inf"))["status"] == "skipped"
    service = learner._consolidation_service()
    with patch.object(service, "train_pairs", wraps=service.train_pairs) as train:
        assert learner.consolidate_parameters("street", note, 2.0)["learning_implementation"] == "hybrid"
        assert train.call_args.kwargs["evidence"] == [source]
    assert learner.consolidate_parameters("street", note, 2.0)["status"] == "skipped"
    assert learner.consolidate_rsi_parameters("Independently reasoned revision")["mode"] == "rsi"
    assert engine.controller_backend.train_calls == 2
    assert len(engine.controller_backend.received_batches[-1]) == 1
    learner.db.log_interaction.assert_called_once()


def test_hybrid_long_report_uses_full_managed_source_selection(tmp_path):
    quote = "This final passage contains the target evidence."
    source = "opening " * 2304 + quote
    def solve(prompt, **kwargs):
        return json.dumps({"evidence": quote if quote in prompt else ""}), {}
    engine = SimpleNamespace(active_backend="controller", controller_backend=RecordingBackend(tmp_path), solve=Mock(side_effect=solve))
    learner = HybridAutonomousLearner(engine=engine, tools=object(), db=Mock(), settings=SimpleNamespace())
    research = {"search_report": source}
    assert learner._lesson_source(research, "evidence") == quote
    assert research["source_selection"]["scanned_characters"] == len(source) and engine.solve.call_count > 1
    # Subsequent synthesis/RSI reads the same selected witness.
    calls = engine.solve.call_count
    assert learner._lesson_source(research, "evidence") == quote and engine.solve.call_count == calls


def test_hybrid_keeps_unavailable_backend_read_only(tmp_path):
    backend = TransactionalBackend(tmp_path)
    service = HybridAwakeOnlineConsolidator(backend)
    with pytest.raises(RuntimeError, match="does not expose a hybrid"):
        service.train_pairs([{"prompt": "color", "completion": "slate"}])
    assert backend.train_calls == 0 and not backend.learning_state.current_manifest()


@pytest.mark.parametrize("tiny_controller", [96], indirect=True)
def test_hybrid_serving_model_commits_original_then_reward_updates(tiny_controller):
    """Exercise both optimizers and their shared checkpoint chain on real CPU PEFT."""
    from core.test_time_learning import OnlineLearning
    import torch

    backend = tiny_controller
    backend.unload_model()
    backend.learning_implementation = "hybrid"
    assert backend.load_model()
    service = HybridAwakeOnlineConsolidator(backend, lambda_ewc=0)
    update = service.train_pairs([{"prompt": "vault color", "completion": "slate"}], mode="rsi",
                                 steps=1, learning_rate=.001)
    assert update["training_metrics"]["training_algorithm"] == "original_completion_adamw"
    assert update["parameter_drift_l2"] > 0 and Path(update["adapter_saved_to"]).is_dir()

    with torch.random.fork_rng():
        torch.manual_seed(91)
        result = OnlineLearning(service).run_request({"method": "reinforce", "prompt": "vault color",
            "rollouts": 8, "max_tokens": 2, "learning_rate": .002, "mode": "verifier",
            "verifier": lambda answer: float(any(x in answer for x in ("slate", "red", "color")))})
    assert result["updates"] == 1 and service.consolidation_count == 2
    policy = result["rounds"][0]["update"]
    assert policy["training_metrics"]["training_algorithm"] == "managed_reward_policy"
    assert policy["reward_logprob_contrast_gain"] > 0
    manifest = backend.learning_state.current_manifest()
    assert manifest["parent"] == update["generation"]
    backend.unload_model()
    assert backend.load_model() and backend.learning_state.current_manifest() == manifest
