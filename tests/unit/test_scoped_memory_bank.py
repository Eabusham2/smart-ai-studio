import hashlib
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import zipfile

import pytest

from core.adapter_memory_bank import AdapterMemoryBank
from core.context_memory import ChatArchive, RetirementLedger
from core.learning_state import (
    bind_learning_state,
    LearningStateStore,
    validate_checkpoint,
    write_checkpoint_manifest,
)
from core.memory_scopes import LearningScope

from tests.unit.test_native_context_learning import tiny_controller


def commit(store, weights):
    with store.begin() as txn:
        (txn.directory / "adapter.bin").write_bytes(weights)
        return txn.commit("adapter.bin")


def test_archive_open_page_and_cold_projection_only_decode_requested_messages(tmp_path, monkeypatch):
    path = tmp_path / "chat.sqlite3"
    arc = ChatArchive(path, "tab")
    for i in range(300):
        arc.append({"content": str(i)})
    ledger = RetirementLedger(tmp_path / "state")
    ledger.stage("first", None, [turn.turn_id for turn in arc[:298]])
    original = ChatArchive._decode
    decoded = []

    def decode(row):
        decoded.append(row[0])
        return original(row)

    monkeypatch.setattr(ChatArchive, "_decode", staticmethod(decode))
    fresh = ChatArchive(path, "tab")
    assert len(fresh) == 300 and not decoded
    visible = fresh.active_view(ledger, "first")
    assert [turn["content"] for turn in visible] == ["298", "299"]
    assert decoded == [t.turn_id for t in visible]
    decoded.clear()
    assert [int(t["content"]) for t in fresh[-4::-70]] == list(range(300))[-4::-70]
    assert len(decoded) == 5
    decoded.clear()
    pages = fresh.iter_pages(page_size=7)
    assert [t["content"] for t in next(pages)] == list(map(str, range(7)))
    assert len(decoded) == 7
    pages.close()
    arc.clear()
    arc.append({"content": "new"})
    assert fresh.active_view(ledger, "first") == [{"content": "new"}]
    assert fresh == [{"content": "new"}] and [{"content": "new"}] == fresh


def test_scoped_binding_preserves_legacy_identity_and_rejects_cross_scope_restore(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"base")
    backend = SimpleNamespace(model_path=str(model))
    options = {"state_root": tmp_path / "memory"}
    legacy = bind_learning_state(backend, **options)
    commit(legacy, b"v1")
    assert "scope" not in legacy.current_manifest()
    named = LearningScope("session", "../private?")
    backend.learning_scope = named
    session = bind_learning_state(backend, **options)
    assert session.directory.is_relative_to(options["state_root"])
    assert ".." not in session.directory.parts and session.current_manifest() is None
    assert session is bind_learning_state(backend, **options)
    commit(session, b"session")
    assert session.identity == legacy.identity
    assert session.current_manifest()["scope"] == named.record
    for kind in ["session", "document", "project"]:
        other = session.for_scope(LearningScope(kind, "other session"))
        assert other.current_manifest() is None
    with pytest.raises(RuntimeError, match="scope"):
        validate_checkpoint(backend, legacy.current_checkpoint(), allow_legacy=True)
    exported = tmp_path / "user.bin"
    exported.write_bytes(b"export")
    write_checkpoint_manifest(backend, exported)
    restored = bind_learning_state(backend, scope=LearningScope(), **options)
    assert Path(restored.current_checkpoint()).read_bytes() == b"v1"
    with pytest.raises(RuntimeError, match="scope"):
        validate_checkpoint(backend, exported)


def test_request_cleanup_fences_old_transactions_and_preserves_main_state(tmp_path):
    main = LearningStateStore({"source": "m"}, state_root=tmp_path)
    commit(main, b"main")
    pointer = (main.directory / "current.json").read_bytes()
    request = main.for_scope(LearningScope.request())
    commit(request, b"request")
    with request.begin() as txn:
        (txn.directory / "adapter.bin").write_bytes(b"rejected")
        stop = Event()
        stop.set()
        with pytest.raises(InterruptedError):
            txn.commit("adapter.bin", cancel_event=stop)
    assert Path(request.current_checkpoint()).read_bytes() == b"request"
    staged = request.begin()
    (staged.directory / "adapter.bin").write_bytes(b"stale")
    request.discard_request()
    assert not request.directory.exists()
    replacement = main.for_scope(request.scope)
    with pytest.raises(RuntimeError, match="ended"):
        staged.commit("adapter.bin")
    staged.abort()
    with pytest.raises(RuntimeError, match="ended"):
        request.discard_request()
    assert replacement.current_checkpoint() is None
    assert (main.directory / "current.json").read_bytes() == pointer
    with pytest.raises(ValueError, match="Only request"):
        main.discard_request()


def test_bank_pins_rank_reload_verify_export_and_collect(tmp_path):
    owner = LearningStateStore({"source": "model", "tokenizer": "v1"}, state_root=tmp_path,
                               scope=LearningScope("project", "project 1"))
    bank = AdapterMemoryBank(owner)
    source = tmp_path / "colors.txt"
    source.write_text("vermilion " + "x" * 100_000 + " end")
    doc = owner.for_scope(bank.document_scope("color"))
    first = commit(doc, b"original")
    entry = bank.register("color", doc, title="Pigment study", source="colors.txt", content=source,
                          expected_generation=first["generation"])
    assert entry["content_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    other = owner.for_scope(bank.document_scope("paper"))
    commit(other, b"paper")
    bank.register("paper", other, title="Paper", source="papers.txt", content="sheet texture", keywords=["study"])
    for i in range(3):
        commit(doc, bytes([i]))
    resumed = AdapterMemoryBank(owner.for_scope(owner.scope))
    entries = resumed.select("vermilion study", top_k=2)
    assert [e["key"] for e in entries] == ["color", "paper"]
    assert resumed.select("xyzzy") == []
    assert len(resumed.list_entries(limit=1, offset=1)) == 1
    loaded = owner.for_scope(resumed.activation_scope(entries[0]))
    path = resumed.resolve(entries[0], loaded)
    assert Path(path).read_bytes() == b"original"
    assert validate_checkpoint(SimpleNamespace(learning_state=loaded), path)["generation"] == first["generation"]
    with pytest.raises(RuntimeError, match="read-only"):
        loaded.begin()
    with pytest.raises(ValueError, match="model"):
        resumed.resolve(entries[0], doc)
    wrong_model = LearningStateStore({"source": "model", "tokenizer": "v2"}, state_root=tmp_path, scope=loaded.scope)
    with pytest.raises(ValueError, match="tokenizer"):
        resumed.resolve(entries[0], wrong_model)
    with pytest.raises(RuntimeError, match="pinned"):
        wrong_model.current_checkpoint()
    destination = resumed.export("color", tmp_path / "color.zip")
    with zipfile.ZipFile(destination) as archive:
        assert archive.read("artifacts/adapter.bin") == b"original"
        assert json.loads(archive.read("entry.json"))["source"] == "colors.txt"
    with pytest.raises(FileExistsError):
        resumed.export("color", destination)
    outsider = AdapterMemoryBank(owner.for_scope(LearningScope("project", "2")))
    assert not outsider.list_entries() and not outsider.delete("color")
    with pytest.raises(KeyError):
        outsider.export("color", tmp_path / "wrong.zip")
    with pytest.raises(ValueError, match="owner"):
        outsider.register("color", doc, title="T", source="p", content="red")
    assert Path(path).exists()
    assert resumed.delete("color")
    with pytest.raises(RuntimeError, match="pinned"):
        loaded.current_checkpoint()
    commit(doc, b"following update")
    assert not Path(path).exists()


def test_bank_cannot_register_stale_generation_or_activate_corrupt_parameters(tmp_path):
    owner = LearningStateStore({"source": "model"}, state_root=tmp_path)
    bank = AdapterMemoryBank(owner)
    doc = owner.for_scope(bank.document_scope("notes"))
    old = commit(doc, b"old")
    commit(doc, b"new")
    with pytest.raises(RuntimeError, match="unavailable"):
        bank.register("notes", doc, expected_generation=old["generation"], title="A", source="n", content="text")
    entry = bank.register("notes", doc, title="A", source="n", content="text")
    Path(doc.current_checkpoint()).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="integrity"):
        bank.activation_scope(entry)


def test_seed_request_copies_verified_weights_and_retirement_lineage(tmp_path):
    parent = LearningStateStore({"source": "base"}, state_root=tmp_path, scope=LearningScope("project", "one"))
    archive = ChatArchive(tmp_path / "history", "a")
    archive.append({"content": "retired"})
    archive.append({"content": "still live"})
    ledger = RetirementLedger(parent.directory)
    ledger.stage("h1", None, [archive[0].turn_id])
    from core.learning_recall import RecallBank
    RecallBank(parent.directory).stage("h1", None, [["question", "digest", 20]])
    with parent.begin(metadata={"memory": {"retirement_head": "h1"}}) as txn:
        (txn.directory / "adapter").mkdir()
        (txn.directory / "adapter" / "weights.bin").write_bytes(b"weights")
        (txn.directory / "optimizer.bin").write_bytes(b"adam")
        txn.commit("adapter")
    pointer = (parent.directory / "current.json").read_bytes()
    request = parent.for_scope(LearningScope.request())
    cutoff = Event()
    cutoff.set()
    with pytest.raises(InterruptedError):
        request.seed_from(parent, cancel_event=cutoff)
    assert not request.current_manifest()
    with pytest.raises(RuntimeError, match="source generation changed"):
        request.seed_from(parent, expected_generation=None)
    fork = request.seed_from(parent, expected_generation=parent.current_manifest()["generation"])
    assert fork["scope"] == request.scope.record
    assert fork["metadata"]["request_seed"]["generation"] == parent.current_manifest()["generation"]
    assert (Path(request.current_checkpoint()) / "weights.bin").read_bytes() == b"weights"
    assert fork["artifacts"] == parent.current_manifest()["artifacts"]
    pages = archive.iter_active_pages(RetirementLedger(request.directory), "h1", page_size=1)
    assert list(pages) == [[{"content": "still live"}]]
    assert RecallBank(request.directory).sample("h1", "test") == [["question", "digest", 20]]
    with pytest.raises(RuntimeError, match="already"):
        request.seed_from(parent)
    commit(request, b"temporary")
    request.discard_request()
    assert (parent.directory / "current.json").read_bytes() == pointer
    assert (Path(parent.current_checkpoint()) / "weights.bin").read_bytes() == b"weights"


def test_exclusive_request_reservation_does_not_own_another_callers_empty_stage(tmp_path):
    base = LearningStateStore({"source": "m"}, state_root=tmp_path)
    scope = LearningScope.request()
    owner = base.for_scope(scope, require_new=True)
    with owner.begin() as pending:
        with pytest.raises(FileExistsError):
            base.for_scope(scope, require_new=True)
        assert pending.directory.is_dir()
    owner.discard_request()
    assert base.for_scope(scope, require_new=True).current_manifest() is None


def test_stream_close_releases_scope_lock_without_exposing_temporary_weights(tmp_path):
    import threading
    from tests.unit.test_memory_runtime import ScopedEngine
    from core.memory_runtime import install_memory_runtime

    # Use a fresh unwrapped test engine and the production stream wrapper.
    class StreamEngine:
        __init__ = ScopedEngine.__init__.__wrapped__
        is_model_loaded = ScopedEngine.is_model_loaded
        load_model = ScopedEngine.load_model.__wrapped__
        solve = ScopedEngine.solve.__wrapped__

        def stream_solve(self, prompt):
            yield self.awake_consolidator.engine.weights
            yield self.awake_consolidator.engine.weights

    install_memory_runtime(StreamEngine)
    base = LearningStateStore({"source": "m"}, state_root=tmp_path)
    commit(base, b"durable")
    engine = StreamEngine(base)
    engine.load_model("test", "/local/model", backend="cpu")
    ready, switched = threading.Event(), threading.Event()
    stream = engine.stream_solve("stream")
    assert next(stream) == b"durable"

    def switch():
        ready.set()
        engine.set_learning_scope(LearningScope("session", "separate"))
        switched.set()

    worker = threading.Thread(target=switch)
    worker.start()
    try:
        assert ready.wait(2) and not switched.wait(.05)
    finally:
        stream.close()
        worker.join(2)
    assert switched.is_set()
    assert list(engine.stream_solve("stream")) == [b"base", b"base"]


def test_real_peft_pinned_ancestor_seeds_request_with_optimizer_and_ewc(tiny_controller):
    backend = tiny_controller
    main = backend.learning_state
    bank = AdapterMemoryBank(main)

    def load(scope):
        backend.unload_model()
        backend.learning_scope = scope
        assert backend.load_model()
        assert backend.learning_state.scope == scope

    def train(answer):
        return backend.train_mini_batch(
            {}, [{"prompt": "What is the vault color?", "completion": answer}],
            steps=1, learning_rate=0.01, lambda_ewc=1., anchor_texts=["vault color slate"],
            validation_callback=lambda candidate: True,
        )[0]

    load(bank.document_scope("vault"))
    doc = backend.learning_state
    train("slate")
    first_score = backend.score_learning_probe("What is the vault color?", "slate")
    data = bank.register("vault", doc, title="Vault", source="vault.txt", content="slate")
    train("red")
    train("red")
    latest = doc.current_manifest()["generation"]
    frozen = bank.activation_scope(data)
    load(frozen)
    assert backend.score_learning_probe("What is the vault color?", "slate") == pytest.approx(first_score)
    assert Path(bank.resolve(data, backend.learning_state)) == Path(backend.adapter_path)
    request = main.for_scope(LearningScope.request(), require_new=True)
    request.seed_from(backend.learning_state)
    load(request.scope)
    assert backend.score_learning_probe("What is the vault color?", "slate") == pytest.approx(first_score)
    metrics = train("red")
    assert metrics["optimizer_resumed"] and metrics["ewc_used"]
    load(LearningScope())
    request.discard_request()
    assert main.current_manifest() is None
    assert doc.current_manifest()["generation"] == latest
    assert main.for_scope(frozen).current_manifest()["generation"] == data["generation"]
