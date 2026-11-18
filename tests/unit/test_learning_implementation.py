"""Public selection and behavior contracts for the two production learning loops."""

import threading
from types import SimpleNamespace

import pytest

from core.autonomous_learner import AutonomousLearner, ManagedAutonomousLearner, make_autonomous_learner
from core.context_memory import ChatArchive
from core.learning_implementation import resolve_learning_mode
from core.learning_sources import extract_complete_learning_text, extract_learning_text
from core.online_consolidator import AwakeOnlineConsolidator, ManagedAwakeOnlineConsolidator, make_online_consolidator
from tests.unit.test_learn_mode import FakeEngine, FakeTools, SOURCE_SENTENCE


def test_original_public_constructors_and_explicit_selection(tmp_path):
    backend = FakeEngine(str(tmp_path / "adapter.safetensors"))
    args = dict(engine=backend, tools=FakeTools(), db=object(), settings=SimpleNamespace(learning_implementation="managed"))
    assert type(make_autonomous_learner(**args)) is ManagedAutonomousLearner
    assert type(make_autonomous_learner(**args, implementation="legacy")) is AutonomousLearner
    assert type(make_online_consolidator(backend)) is AwakeOnlineConsolidator
    assert type(make_online_consolidator(backend, implementation="managed")) is ManagedAwakeOnlineConsolidator
    with pytest.raises(ValueError, match="implementation"):
        make_online_consolidator(backend, implementation="unknown")
    owner = SimpleNamespace(learning_state=object())
    assert resolve_learning_mode(owner, None, {}) == "managed"
    assert resolve_learning_mode(owner, "legacy", {}) == "legacy"
    owner.learning_implementation = "legacy"
    assert resolve_learning_mode(owner, None, {}) == "legacy"
    assert resolve_learning_mode(owner, None, {"row_weights": [1.0]}) == "managed"
    with pytest.raises(ValueError, match="managed"):
        resolve_learning_mode(owner, "legacy", {"validation_callback": None})


def test_standalone_managed_learner_selects_managed_engine_without_changing_settings(monkeypatch):
    import core.autonomous_learner as learner_module

    monkeypatch.setattr(learner_module, "ProReasoningEngine", lambda settings: SimpleNamespace(settings=settings))
    settings = SimpleNamespace(learning_implementation="legacy")
    learner = make_autonomous_learner(tools=object(), db=object(), settings=settings, implementation="managed")
    assert learner.engine.settings.learning_implementation == "managed"
    assert settings.learning_implementation == "legacy"


def test_original_attachment_limits_formatting_and_casefold_reward(tmp_path):
    path = tmp_path / "long.csv"
    path.write_text("\n".join(f"line {i}, record" for i in range(2300)))
    old = extract_learning_text(path)
    new = extract_complete_learning_text(path)
    assert "===== SOURCE FILE: long.csv =====" in old["text"]
    assert "line 1999" in old["text"] and "line 2000" not in old["text"]
    assert "line 2299" in new["text"]
    engine = FakeEngine(str(tmp_path / "lora.safetensors"))
    marked = []
    db = SimpleNamespace(log_interaction=lambda **kw: 5, mark_consolidated=marked.extend)
    args = dict(engine=engine, tools=FakeTools(), db=db, settings=SimpleNamespace())
    original = make_autonomous_learner(**args)
    managed = make_autonomous_learner(**args, implementation="managed")
    source = {"source_text": SOURCE_SENTENCE.upper()}
    assert original.self_test_and_verify("raft", source, "Learned note")[0]
    assert not managed.self_test_and_verify("raft", source, "Learned note")[0]
    excerpt = original._lesson_source({"source_text": "a" * 16000 + "suffix"}, "raft")
    assert excerpt == "a" * 16000
    assert original.consolidate_parameters("raft", "Learned note", reward=1.1)["ewc_active"]
    assert managed.consolidate_parameters("raft", "Learned note", reward=1.1)["status"] == "skipped"
    assert marked == [5]


def test_original_mlx_cleanup_reaches_the_model_owner(tmp_path, monkeypatch):
    from core import training_memory

    cleanup_threads = []
    def cleanup(backend):
        cleanup_threads.append(threading.get_ident())
        return {"after_mb": 16.0, "released_mb": 2.0}

    monkeypatch.setattr(training_memory, "release_training_memory", cleanup)
    engine = FakeEngine(str(tmp_path / "adapter.safetensors"))
    backend = engine.mlx_backend
    backend._mlx_owner = object()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as executor:
        backend.run_on_mlx_owner = lambda fn, *args: executor.submit(fn, *args).result(timeout=5)
        owner_thread = backend.run_on_mlx_owner(threading.get_ident)
        # This third-party fixture trains ordinary Python values.
        backend.is_mlx_available = False
        learner = make_autonomous_learner(engine=engine, tools=FakeTools(), db=object(), settings=SimpleNamespace())
        result = learner.consolidate_rsi_parameters("Self-generated trace")
        assert result["ram_released_mb"] == 2.0
        service = AwakeOnlineConsolidator(backend)
        assert service.consolidate_chunk_sync([{"role": "user", "content": "A fact"},
                                               {"role": "assistant", "content": "Remember it"}])
        assert cleanup_threads == [owner_thread, owner_thread]


def test_original_mlx_watermark_returns_while_training_and_never_retires_archive(tmp_path):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    class Backend:
        model = tokenizer = object()
        is_mlx_available = True
        def train_mini_batch(self, adapters, data, lambda_ewc, steps, save_path):
            started.set()
            assert release.wait(5)
            return {}, 0.125

        def release_training_memory(self):
            finished.set()

    service = AwakeOnlineConsolidator(Backend(), max_context=2, evict_ratio=0.5)
    turns = [{"role": role, "content": role * 3} for _ in range(3) for role in ("user", "assistant")]
    try:
        retained, triggered = service.check_and_prune(turns)
        assert triggered and retained == turns[4:]
        assert started.wait(5) and service.is_consolidating
    finally:
        release.set()
    assert finished.wait(5)
    archive = ChatArchive(tmp_path / "history.sqlite3", "tab")
    for turn in turns:
        archive.append(turn)
    retained, triggered = service.check_and_prune(archive)
    assert retained is archive and not triggered
    assert len(retained) == len(turns)
