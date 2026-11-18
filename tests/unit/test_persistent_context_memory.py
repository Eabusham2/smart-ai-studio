import json
import threading
from types import SimpleNamespace

import pytest

from core.awake_auto_hook import install_awake_auto_learning
from core.context_memory import ChatArchive, oldest_completed_chunk
from core.media_orchestrator import MediaController
from core.online_consolidator import AwakeOnlineConsolidator, REPLAY_LIMIT
from tests.unit.context_backend_fixture import TransactionalBackend


@pytest.fixture(autouse=True)
def skip_allocator_cleanup(monkeypatch):
    monkeypatch.setattr("core.online_consolidator.release_training_memory",
                        lambda backend: {"after_mb": 0, "released_mb": 0})


def archive_at(tmp_path):
    return ChatArchive(str(tmp_path / "app.sqlite3"), "model_1")


def add_pair(history, index):
    history.append({"role": "user", "content": f"The archive label for sample {index} is cedar{index}."})
    history.append({"role": "assistant", "content": f"I recorded sample {index} as cedar{index}."})


def test_receipts_match_checkpoint_on_restart_and_model_switch(tmp_path):
    archive = archive_at(tmp_path)
    for i in range(5):
        add_pair(archive, i)
    visible = list(archive)
    service = AwakeOnlineConsolidator(TransactionalBackend(tmp_path))
    assert service.sleep(archive)["status"] == "success"
    first_remaining = service.active_history(archive)
    assert 0 < len(first_remaining) < len(archive)
    assert archive == visible
    resumed_archive = archive_at(tmp_path)
    resumed = AwakeOnlineConsolidator(TransactionalBackend(tmp_path))
    assert resumed.active_history(resumed_archive) == first_remaining
    # A genuinely new chat with identical text has unique IDs.
    new_chat = ChatArchive(str(tmp_path / "app.sqlite3"), "other_chat")
    add_pair(new_chat, 0)
    assert resumed.active_history(new_chat) == new_chat
    other_model = AwakeOnlineConsolidator(TransactionalBackend(tmp_path, identity="different-weights"))
    assert other_model.active_history(resumed_archive) == visible

    while resumed.sleep(resumed_archive)["status"] != "idle":
        pass
    assert resumed.active_history(resumed_archive) == []
    batches = resumed.engine.received_batches
    assert all(row["prompt"] != visible[0]["content"] for batch in batches for row in batch)
    assert len(resumed.engine.learning_state.current_manifest()["metadata"]["memory"]["replay"]) <= REPLAY_LIMIT
    assert "earlier statement" in json.dumps(batches[-1])
    assert [m["content"] for m in archive_at(tmp_path)] == [m["content"] for m in visible]


@pytest.mark.parametrize("reason", ["cancel", "regression"])
def test_rejected_candidate_cannot_hide_archived_turns_or_replace_checkpoint(tmp_path, reason):
    archive = archive_at(tmp_path)
    for i in range(6):
        add_pair(archive, i)
    backend = TransactionalBackend(tmp_path)
    service = AwakeOnlineConsolidator(backend)
    service.sleep(archive)
    pointer = (backend.learning_state.directory / "current.json").read_bytes()
    active = service.active_history(archive)
    before = backend.weight
    stop = threading.Event()
    if reason == "cancel":
        backend.before_validation = stop.set
    else:
        backend.regress = True
    with pytest.raises(RuntimeError, match="cancelled|regressed"):
        service.sleep(archive, stop)
    assert backend.weight == before
    assert (backend.learning_state.directory / "current.json").read_bytes() == pointer
    assert AwakeOnlineConsolidator(TransactionalBackend(tmp_path)).active_history(archive) == active


def test_system_tools_and_incomplete_sequences_remain_in_order(tmp_path):
    archive = archive_at(tmp_path)
    archive.append({"role": "system", "content": "Policy"})
    archive.append({"role": "user", "content": "Check a result"})
    archive.append({"role": "assistant", "content": "", "tool_calls": [{"id": "abc"}]})
    archive.append({"role": "tool", "tool_call_id": "abc", "content": "27"})
    archive.append({"role": "assistant", "content": "The result was 27"})
    add_pair(archive, 0)
    add_pair(archive, 1)
    archive.append({"role": "user", "content": "unfinished"})
    selected, retained = oldest_completed_chunk(archive)
    assert selected == archive[5:7]
    assert retained == archive[:5] + archive[7:]
    service = AwakeOnlineConsolidator(TransactionalBackend(tmp_path))
    assert service.consolidate_chunk_sync(selected)
    assert service.active_history(archive) == retained
    assert not any("Policy" in str(row) or "result was 27" in str(row)
                   for row in service.engine.received_batches[0])


def test_plain_api_resubmission_after_checkpoint_filters_identical_positions(tmp_path):
    history = []
    for _ in range(4):
        add_pair(history, 1)
    service = AwakeOnlineConsolidator(TransactionalBackend(tmp_path), session_id="named-api-chat")
    active = service.active_history(history)
    first, retained = oldest_completed_chunk(active, 0.1)
    assert service.consolidate_chunk_sync(first)
    assert [t.turn_id for t in service.active_history([dict(t) for t in history])] == [
        t.turn_id for t in retained]
    resumed = AwakeOnlineConsolidator(TransactionalBackend(tmp_path), session_id="named-api-chat")
    assert resumed.active_history(history) == retained
    other = AwakeOnlineConsolidator(TransactionalBackend(tmp_path))
    assert other.active_history(history) == history


@pytest.mark.parametrize("best_work", [False, True])
def test_gui_request_copies_keep_retirement_through_both_real_routes(tmp_path, best_work):
    class Engine:
        active_backend = "controller"
        is_model_loaded = True
        _gui_force_single_stream = True

        def __init__(self):
            self.settings = SimpleNamespace(max_new_tokens=64)
            self.controller_backend = TransactionalBackend(tmp_path)
            self.awake_consolidator = AwakeOnlineConsolidator(self.controller_backend)
            self.router = SimpleNamespace(route=lambda *args, **kwargs: ("Instant", 1))
            self._context_budget_tokens = 155
            self.received = []

        def calculate_token_entropy(self, prompt):
            return 0.1

        def _format_prompt_with_history(self, prompt, history):
            # Use a counting stub insensitive to the two guides' different sizes.
            return " ".join(m["content"] for m in (history or []) if m["role"] != "system") + " " + prompt

        def stream_solve(self, prompt, history=None, **kwargs):
            self.received.append(list(history))
            yield "ok"

        def solve(self, prompt, **kwargs):
            return "ok", {}

    install_awake_auto_learning(Engine)
    engine = Engine()
    archive = archive_at(tmp_path)
    for i in range(10):
        add_pair(archive, i)
    tools = SimpleNamespace(list_available_tools=lambda: [], execute_tool=lambda *args: (False, ""))
    app = SimpleNamespace(engine=engine, models_config={}, tools=tools, active_tab_id="model_1",
                          cancel_event=threading.Event())
    controller = MediaController(app)
    assert "".join(controller.stream_solve("continue", archive, best_work_enabled=best_work)) == "ok"
    first = engine.received[-1]
    assert all(turn["content"] != archive[0]["content"] for turn in first)
    assert len(archive) == 20 and any(turn["role"] == "system" for turn in first)
    calls = engine.controller_backend.train_calls
    assert "".join(controller.stream_solve("again", archive_at(tmp_path), best_work_enabled=best_work)) == "ok"
    assert engine.controller_backend.train_calls == calls
    assert engine.received[-1] == first


def test_projection_follows_rollback_and_never_uses_a_newer_models_receipt(tmp_path):
    archive = archive_at(tmp_path)
    for i in range(8):
        add_pair(archive, i)
    backend = TransactionalBackend(tmp_path)
    service = AwakeOnlineConsolidator(backend)
    service.sleep(archive)
    pointer = backend.learning_state.directory / "current.json"
    first_checkpoint = pointer.read_bytes()
    first_active = service.active_history(archive)
    stale = AwakeOnlineConsolidator(TransactionalBackend(tmp_path))
    service.sleep(archive)
    assert len(service.active_history(archive)) < len(first_active)
    with pytest.raises(RuntimeError, match="Reload"):
        stale.active_history(archive)
    # Both committed generations remain. Selecting the older checkpoint must
    # restore its input, including turns retired only by the second update.
    pointer.write_bytes(first_checkpoint)
    restored = AwakeOnlineConsolidator(TransactionalBackend(tmp_path))
    assert restored.active_history(archive) == first_active


def test_archive_cache_scans_only_new_or_still_active_turns(tmp_path, monkeypatch):
    from core.context_memory import RetirementLedger

    archive = archive_at(tmp_path)
    for i in range(20):
        add_pair(archive, i)
    service = AwakeOnlineConsolidator(TransactionalBackend(tmp_path))
    service.sleep(archive)
    active = service.active_history(archive)
    assert len(active) < len(archive)
    original = RetirementLedger.active_turns
    visited = []

    def observe(ledger, turns, head):
        rows = list(turns)
        visited.append(len(rows))
        return original(ledger, rows, head)

    monkeypatch.setattr(RetirementLedger, "active_turns", observe)
    add_pair(archive, 22)
    assert len(service.active_history(archive)) == len(active) + 2
    assert visited == []
    service.sleep(archive)
    visited.clear()
    assert len(service.active_history(archive)) < len(active)
    assert visited == [len(active) + 2]


def test_sleep_panel_dispatches_to_selected_backend_and_reports_only_committed_result(tmp_path, monkeypatch):
    import app_gui
    from collections import defaultdict

    widgets = []

    class Widget:
        def __init__(self, *args, **kwargs):
            self.options = kwargs
            self.log = []
            widgets.append(self)

        def configure(self, **kwargs):
            self.options.update(kwargs)

        def insert(self, position, text):
            self.log.append(text)

        def after(self, delay, callback):
            callback()

        def winfo_exists(self):
            return True

        def __getattr__(self, name):
            if name in {"title", "geometry", "minsize", "transient", "pack", "see"}:
                return lambda *args, **kwargs: None
            raise AttributeError(name)

    for name in ("Toplevel", "Label", "Frame", "Text", "Button"):
        monkeypatch.setattr(app_gui.tk, name, Widget)
    monkeypatch.setattr(app_gui.threading, "Thread",
                        lambda target, **kwargs: SimpleNamespace(start=target))
    backend = TransactionalBackend(tmp_path)
    service = AwakeOnlineConsolidator(backend)
    archive = archive_at(tmp_path)
    add_pair(archive, 0)
    app = SimpleNamespace(engine=SimpleNamespace(awake_consolidator=service), active_tab_id="model_1",
                          models_config={"model_1": {}}, chat_history={"model_1": archive},
                          is_generating=False, root=Widget(), C=defaultdict(str), _make_badge=Widget,
                          cancel_event=threading.Event(), multimodal=SimpleNamespace(busy=False),
                          _update_telemetry=lambda: None, _check_and_run_next_queue=lambda: None)
    app._set_generating_state = lambda state: setattr(app, "is_generating", state)
    app_gui.SmartAIChatbotApp._on_open_sleep_consolidation_panel(app)
    button = next(widget for widget in widgets if "command" in widget.options)
    button.options["command"]()
    assert backend.train_calls == 1
    assert not app.is_generating
    assert len(archive) == 2 and service.active_history(archive) == []
    logs = "".join(line for widget in widgets for line in widget.log)
    assert backend.learning_state.current_checkpoint() in logs
    button.options["command"]()
    assert backend.train_calls == 1
    assert any("No pending complete exchanges" in line for widget in widgets for line in widget.log)


@pytest.mark.parametrize("incompatible", [False, True])
def test_legacy_gui_import_is_scoped_to_the_original_model_and_keeps_source(tmp_path, incompatible):
    from app_gui import SmartAIChatbotApp

    source = tmp_path / "old-shared.safetensors"
    source.write_bytes(b"legacy checkpoint fixture")
    info = {"name": "Original base", "repo_id": "local/original", "precision": "original"}
    calls = []
    engine = SimpleNamespace(mlx_backend=SimpleNamespace(last_restored_adapter_path=None))

    def load(name, **kwargs):
        calls.append((name, kwargs))
        selected = kwargs["model_info"].get("legacy_adapter_path")
        if incompatible and selected:
            return {"status": "error", "error": "tensor shape mismatch"}
        engine.mlx_backend.last_restored_adapter_path = selected
        return {"status": "loaded"}

    engine.load_model = load
    app = SimpleNamespace(engine=engine)
    result = SmartAIChatbotApp._reload_legacy_mlx(app, info, "local/original", str(source), threading.Event())
    assert result["status"] == "loaded"
    assert source.read_bytes() == b"legacy checkpoint fixture"
    assert "legacy_adapter_path" not in info
    assert all(name == "Original base" and call["backend"] == "mlx"
               and call["model_path"] == "local/original" for name, call in calls)
    assert calls[0][1]["model_info"]["legacy_adapter_path"] == str(source)
    if incompatible:
        assert len(calls) == 2 and calls[-1][1]["model_info"] == info
        assert "tensor shape mismatch" in result["legacy_import_note"]
    else:
        assert len(calls) == 1
        assert engine.mlx_backend.last_restored_adapter_path == str(source)
        assert "next validated learning update" in result["legacy_import_note"]


@pytest.mark.parametrize("measured,active", [({"ewc_lambda": 400.0}, False), ({"fisher_tensors": 2}, True)])
def test_ewc_status_requires_trainer_evidence(tmp_path, measured, active):
    backend = TransactionalBackend(tmp_path)
    backend.training_metrics = measured
    service = AwakeOnlineConsolidator(backend)
    archive = archive_at(tmp_path)
    add_pair(archive, 0)
    assert service.sleep(archive)["ewc_active"] is active
