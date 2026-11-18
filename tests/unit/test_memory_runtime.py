"""Exercise scope lifecycle against real state files and pinned bank generations."""
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from core.learning_state import LearningStateStore
from core.memory_runtime import install_memory_runtime
from core.memory_scopes import LearningScope


def commit(store, value):
    with store.begin() as transaction:
        (transaction.directory / "weights.bin").write_bytes(value)
        return transaction.commit("weights.bin")


class ScopedEngine:
    def __init__(self, store):
        self.store = store
        self._model_lock = threading.RLock()
        self._loaded = False
        self.awake_consolidator = SimpleNamespace(engine=None, is_consolidating=False)
        self.seen = []
        self.fail_scope = None

    @property
    def is_model_loaded(self):
        return self._loaded

    def load_model(self, model_name, model_path=None, backend=None, model_info=None):
        self._loaded = False
        self.seen.append(self.learning_scope)
        if self.learning_scope == self.fail_scope:
            return {"status": "error", "error": "fixture native load failed"}
        state = self.store.for_scope(self.learning_scope)
        path = state.current_checkpoint()
        self.awake_consolidator.engine = SimpleNamespace(
            learning_state=state, model=object(), weights=Path(path).read_bytes() if path else b"base")
        self._loaded = True
        return {"status": "loaded", "backend": backend, "path": model_path}

    def solve(self, prompt, **kwargs):
        return self.awake_consolidator.engine.weights.decode(), {"prompt": prompt}


install_memory_runtime(ScopedEngine)


@pytest.fixture
def engine(tmp_path):
    store = LearningStateStore({"model": "synthetic", "tokenizer": "fixed"}, state_root=tmp_path / "state")
    commit(store, b"persistent")
    engine = ScopedEngine(store)
    engine.load_model("test", "/local/model", backend="cpu", model_info={"architecture": "fixture"})
    return engine


def test_request_forks_restores_and_removes_only_its_own_state(engine):
    pointer = (engine.store.directory / "current.json").read_bytes()
    with engine.use_learning_scope() as backend:
        assert backend.weights == b"persistent"
        assert backend.learning_state.scope.kind == "request"
        directory = backend.learning_state.directory
        commit(backend.learning_state, b"temporary")
        backend.weights = b"temporary"
        assert engine.solve("answer")[0] == "temporary"
    assert engine.solve("answer")[0] == "persistent"
    assert engine.learning_scope == LearningScope()
    assert not directory.exists()
    assert (engine.store.directory / "current.json").read_bytes() == pointer


def test_scope_isolation_and_restore_after_failed_load(engine):
    project = LearningScope("project", "alpha")
    engine.set_learning_scope(project)
    assert engine.solve("anything")[0] == "base"
    commit(engine.awake_consolidator.engine.learning_state, b"alpha")
    engine.set_learning_scope(LearningScope())
    assert engine.solve("anything")[0] == "persistent"
    engine.fail_scope = LearningScope("project", "broken")
    with pytest.raises(RuntimeError, match="native load failed"):
        engine.set_learning_scope(engine.fail_scope)
    assert engine.learning_scope == LearningScope()
    assert engine.solve("anything")[0] == "persistent"
    engine.set_learning_scope(project)
    assert engine.solve("anything")[0] == "alpha"


def test_exception_restores_parent_before_cleaning_request(engine):
    with pytest.raises(ValueError, match="cancelled study"):
        with engine.use_learning_scope() as backend:
            request_dir = backend.learning_state.directory
            raise ValueError("cancelled study")
    assert engine.solve("answer")[0] == "persistent"
    assert not request_dir.exists()


def test_changed_base_is_not_reported_as_a_restored_parent(engine, tmp_path, monkeypatch):
    original_load = engine.load_model

    def changed_load(**kwargs):
        engine.store = LearningStateStore({"model": "changed", "tokenizer": "changed"},
                                          state_root=tmp_path / "state")
        return original_load(**kwargs)

    monkeypatch.setattr(engine, "load_model", changed_load)
    with pytest.raises(RuntimeError, match="previous model identity"):
        with engine.use_learning_scope():
            pytest.fail("The changed model must never become an active request")


def test_failed_parent_restore_keeps_the_request_checkpoint(engine):
    with pytest.raises(RuntimeError, match="native load failed"):
        with engine.use_learning_scope() as backend:
            request_scope = engine.learning_scope
            request_dir = backend.learning_state.directory
            engine.fail_scope = LearningScope()
    # Recovery reloads the temporary model when the parent is unavailable. Its
    # checkpoint must survive; the caller receives an explicit restore error.
    assert request_dir.exists()
    assert engine.learning_scope == request_scope
    engine.fail_scope = None
    engine.set_learning_scope(LearningScope())


def test_pinned_document_routes_weights_and_restores_parent(engine):
    bank = engine.adapter_memory_bank()
    doc = engine.store.for_scope(bank.document_scope("fern"))
    commit(doc, b"fern-memory")
    bank.register("fern", doc, title="Fern project", source="fern.txt", content="fern colors")
    commit(doc, b"new-unselected-memory")
    text, metadata = engine.solve_with_memory("What color is the fern?")
    assert text == "fern-memory"
    assert metadata["memory_selection"]["key"] == "fern"
    assert "pin" not in metadata["memory_selection"]
    assert engine.solve("base question")[0] == "persistent"
    assert engine.solve_with_memory("unmatchedxyz")[1]["memory_selection"] is None


def test_other_thread_cannot_observe_temporary_request_weights(engine):
    started, finished = threading.Event(), threading.Event()
    observations = []

    def answer():
        started.set()
        observations.append(engine.solve("concurrent")[0])
        finished.set()

    with engine.use_learning_scope() as backend:
        backend.weights = b"temporary"
        worker = threading.Thread(target=answer)
        worker.start()
        assert started.wait(1)
        assert not finished.wait(0.03)
    worker.join(timeout=2)
    assert finished.is_set() and observations == ["persistent"]


def test_workflow_default_request_isolated_but_ingest_persists(engine, monkeypatch):
    import core.test_time_learning as workflows

    class Work:
        def __init__(self, service):
            self.store = service.engine.learning_state

        def run_request(self, request, **kwargs):
            commit(self.store, request["method"].encode())
            return {"status": "completed", "method": request["method"]}

    monkeypatch.setattr(workflows, "OnlineLearning", Work)
    result = engine.run_online_learning({"method": "few_shot"})
    assert result["state_persisted"] is False and result["scope"]["kind"] == "request"
    assert result["total_wall_time_seconds"] >= 0
    assert Path(engine.store.current_checkpoint()).read_bytes() == b"persistent"
    assert engine.run_online_learning({"method": "ingest"})["state_persisted"] is True
    assert Path(engine.store.current_checkpoint()).read_bytes() == b"ingest"


@pytest.mark.parametrize("status", ["completed", "paused"])
def test_document_registration_waits_for_full_study(engine, monkeypatch, tmp_path, status):
    import core.test_time_learning as workflows
    source = tmp_path / "study.txt"
    source.write_text("project orchid grows green")

    class Work:
        def __init__(self, service):
            self.store = service.engine.learning_state

        def run_request(self, request, **kwargs):
            commit(self.store, b"orchid")
            return {"status": status, "trained_chunks": 1}

    monkeypatch.setattr(workflows, "OnlineLearning", Work)
    result = engine.run_online_learning({"method": "bank_ingest", "source": str(source),
                                         "key": "orchid", "title": "Orchid"})
    assert result["status"] == status
    entries = engine.adapter_memory_bank().list_entries()
    assert len(entries) == (1 if status == "completed" else 0)
    assert engine.solve("parent")[0] == "persistent"


def test_changed_document_cannot_be_catalogued(engine, monkeypatch, tmp_path):
    import core.test_time_learning as workflows
    source = tmp_path / "study.txt"
    source.write_text("first version")

    class Work:
        def __init__(self, service):
            self.store = service.engine.learning_state

        def run_request(self, request, **kwargs):
            commit(self.store, b"first version")
            source.write_text("second version")
            return {"status": "completed", "trained_chunks": 1}

    monkeypatch.setattr(workflows, "OnlineLearning", Work)
    with pytest.raises(RuntimeError, match="changed"):
        engine.run_online_learning({"method": "bank_ingest", "source": str(source),
                                    "key": "file", "title": "File"})
    assert not engine.adapter_memory_bank().list_entries()
    assert engine.solve("parent")[0] == "persistent"
