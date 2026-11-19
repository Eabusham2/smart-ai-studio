"""Real Tk paging and learning-panel worker handoff, using synthetic replies."""
import threading
import time
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace

import pytest

from app_gui import SmartAIChatbotApp
from config.settings import Settings
from core.learning_ui import show_online_learning


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTAI_DATA_DIR", str(tmp_path))
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("A desktop display is required")
    root.withdraw()
    app = SmartAIChatbotApp(root, settings=Settings(use_mock=True, auto_download=False, learning_implementation="managed",
                                                  database_path=str(tmp_path / "chat.sqlite3")))
    app._check_and_run_next_queue = lambda: None
    try:
        yield app
    finally:
        app.cancel_event.set()
        worker = getattr(app, "_generation_worker", None)
        if worker is not None:
            worker.join(timeout=2)
            assert not worker.is_alive()
        if app.watchdog:
            app.watchdog.stop_monitoring()
        app.engine.unload_model()
        # Child widgets own some of these Tcl callback commands. Cancel their
        # schedules without deleting commands before the owning widgets die.
        for pending in root.tk.splitlist(root.tk.call("after", "info")):
            root.tk.call("after", "cancel", pending)
        root.destroy()
        import gc
        gc.collect()


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


def _wait_for_idle(app, timeout_ms=3000):
    """Give each operation its own deadline and release every test callback."""
    poll_id = None

    def poll():
        nonlocal poll_id
        worker = getattr(app, "_generation_worker", None)
        if not app.is_generating and (worker is None or not worker.is_alive()):
            app.root.quit()
        else:
            poll_id = app.root.after(10, poll)

    poll_id = app.root.after(10, poll)
    deadline_id = app.root.after(timeout_ms, app.root.quit)
    try:
        app.root.mainloop()
    finally:
        # A completed operation must not leave a quit callback that interrupts
        # a later operation. Cancel the poll too if this operation timed out.
        app.root.after_cancel(poll_id)
        app.root.after_cancel(deadline_id)
    assert not app.is_generating
    worker = getattr(app, "_generation_worker", None)
    assert worker is None or not worker.is_alive()


@pytest.mark.parametrize("switch_delay", [0.0, 0.6], ids=["normal", "delayed-worker"])
def test_implementation_control_preserves_both_transcripts_and_original_default_actions(app, monkeypatch, switch_delay):
    from core.memory_preferences import load_learning_implementation
    original_switch = app.engine.set_learning_implementation

    def switch(*args, **kwargs):
        # Each change fits its deadline, while the full sequence deliberately
        # outlives the first deadline to expose callbacks leaked between waits.
        if switch_delay:
            time.sleep(switch_delay)
        return original_switch(*args, **kwargs)

    monkeypatch.setattr(app.engine, "set_learning_implementation", switch)
    app.chat_history[app.active_tab_id].append({"role": "user", "content": "Managed conversation"})
    archived = app.chat_history[app.active_tab_id]
    show_online_learning(app)
    modal = next(child for child in app.root.winfo_children() if isinstance(child, tk.Toplevel))
    chooser = next(widget for widget in descendants(modal)
                   if isinstance(widget, ttk.Combobox) and tuple(widget.cget("values")) == ("legacy", "managed", "hybrid"))
    apply = next(widget for widget in descendants(modal)
                 if isinstance(widget, ttk.Button) and widget.cget("text") == "Apply implementation")

    def select(mode):
        chooser.set(mode)
        apply.invoke()
        _wait_for_idle(app)
        assert app.settings.learning_implementation == mode
        assert app.engine.awake_consolidator.learning_implementation == mode
        assert load_learning_implementation(app.settings.database_path) == mode

    select("legacy")
    assert app.chat_history[app.active_tab_id] == []
    app.chat_history[app.active_tab_id].append({"role": "user", "content": "Original conversation"})
    original = app.chat_history[app.active_tab_id]
    app.models_config["added_in_original_mode"] = dict(app.models_config[app.active_tab_id])
    select("managed")
    assert app.chat_history[app.active_tab_id] is archived
    assert archived[0]["content"] == "Managed conversation"
    assert "added_in_original_mode" in app.chat_history
    select("legacy")
    assert app.chat_history[app.active_tab_id] is original
    assert original[0]["content"] == "Original conversation"
    select("hybrid")
    combined = app.chat_history[app.active_tab_id]
    assert combined is not archived and list(combined) == []
    assert combined.tab.startswith("hybrid:")
    combined.append({"role": "user", "content": "Hybrid conversation"})
    select("managed")
    assert app.chat_history[app.active_tab_id] is archived
    assert archived[0]["content"] == "Managed conversation"
    select("legacy")
    assert app.chat_history[app.active_tab_id] is original
    select("hybrid")
    assert app.chat_history[app.active_tab_id] is combined
    assert combined[0]["content"] == "Hybrid conversation"


def test_switching_models_in_legacy_reopens_managed_text_for_the_destination_scope(tmp_path, monkeypatch):
    from core.memory_scopes import LearningScope
    from tests.unit.test_implementation_switch import SwitchEngine

    engine = SwitchEngine(tmp_path / "weights", "managed")
    engine.settings.database_path = str(tmp_path / "transcripts.sqlite3")
    project = LearningScope("project", "A only")
    engine.learning_scope = project
    engine.load_model("A", "fixture/A", backend="fixture")

    # Exercise the ordinary GUI history callback with real SQLite transcripts
    # and model-specific state selection, without requiring a window server.
    app = SmartAIChatbotApp.__new__(SmartAIChatbotApp)
    app.settings, app.engine = engine.settings, engine
    app.models_config = {"A": {}, "B": {}, "audio": {"model_type": "audio"}}
    app.active_tab_id = "A"
    app.tools, app.db = object(), object()
    app._show_transcript_page = lambda: None
    monkeypatch.setattr("app_gui.make_autonomous_learner", lambda **kwargs: object())
    app.chat_history = {tab: app._open_tab_history(tab) for tab in app.models_config}
    private = {"role": "user", "content": "Project-only turn"}
    app.chat_history["B"].append(private)
    media = app.chat_history["audio"]

    def choose(mode):
        receipt = engine.set_learning_implementation(mode)
        app._learning_implementation_changed(receipt["previous_implementation"])

    choose("legacy")
    original = app.chat_history["B"]
    original.append({"role": "user", "content": "Live legacy turn"})
    engine.load_model("B", "fixture/B", backend="fixture")
    choose("managed")
    assert engine.learning_scope == LearningScope()
    assert app.chat_history["B"].tab == "B" and list(app.chat_history["B"]) == []
    assert app.chat_history["audio"] is media
    app.chat_history["B"].append({"role": "user", "content": "Model-wide turn"})

    choose("legacy")
    assert app.chat_history["B"] is original
    engine.load_model("A", "fixture/A", backend="fixture")
    choose("managed")
    assert engine.learning_scope == project
    assert list(app.chat_history["B"]) == [private]


def test_visible_chat_is_paged_without_deleting_archived_turns(app):
    archive = app.chat_history[app.active_tab_id]
    archive.clear()
    for number in range(140):
        archive.append({"role": "user", "content": f"Saved message {number}."})
    app._show_transcript_page()
    visible = app.chat_stream.get("1.0", "end")
    assert "Saved message 0." not in visible and "Saved message 139." in visible
    assert app._transcript_page_start == 40 and len(archive) == 140
    app._show_transcript_page(40)
    assert "Saved message 0." in app.chat_stream.get("1.0", "end")
    app._append_user_message("Newest message")
    assert len(archive) == 141
    assert app.chat_stream.get("1.0", "end").count("Newest message") == 1
    assert app._transcript_page_end == 141


@pytest.mark.parametrize("task,method", [("Read a document", "ingest"), ("Discover a better solution", "discover")])
def test_learning_panel_runs_model_work_off_tk_and_returns_to_idle(app, task, method):
    main_thread = threading.get_ident()
    observed = []

    def run(request, **kwargs):
        observed.append((threading.get_ident(), request))
        kwargs["progress"]({"status": "running", "next": 1})
        return {"status": "completed", "trained_chunks": 1}

    app.engine.run_online_learning = run
    show_online_learning(app)
    modal = next(child for child in app.root.winfo_children() if isinstance(child, tk.Toplevel))
    chooser = next(widget for widget in descendants(modal)
                   if isinstance(widget, ttk.Combobox) and task in widget.cget("values"))
    chooser.set(task)
    chooser.event_generate("<<ComboboxSelected>>")
    run_button = next(widget for widget in descendants(modal)
                      if isinstance(widget, ttk.Button) and widget.cget("text") == "Run learning task")
    run_button.invoke()

    _wait_for_idle(app)
    assert observed and observed[0][0] != main_thread
    assert observed[0][1]["method"] == method
    assert not app.is_generating and not app._sleep_consolidating
    output = [widget.get("1.0", "end") for widget in descendants(modal) if isinstance(widget, tk.Text)]
    assert any('"trained_chunks": 1' in text and '"completed"' in text for text in output)


@pytest.mark.parametrize("unload_flag", [
    "_model_unload_in_progress", "_model_unload_pending",
    "_strata_unload_pending", "_native_unload_pending",
])
@pytest.mark.parametrize("source_label", [
    "Verified episodes (original Sleep/EWC)", "Current chat exchanges",
])
def test_open_managed_sleep_refuses_pending_or_running_unload(app, monkeypatch, unload_flag, source_label):
    service = app.engine.awake_consolidator
    trained = []
    # A failed cleanup can retain a backend that still passes readiness checks.
    monkeypatch.setattr(service, "_real_training_ready", lambda: True)
    monkeypatch.setattr(service, "pending_verified_episodes", lambda db: 1)
    monkeypatch.setattr(service, "pending_exchanges", lambda history: 1)

    def train(*args, **kwargs):
        trained.append((args, kwargs))
        return {"status": "idle"}

    monkeypatch.setattr(service, "sleep_verified_episodes", train)
    monkeypatch.setattr(service, "sleep", train)
    app._on_open_sleep_consolidation_panel()
    modal = next(w for w in app.root.winfo_children() if isinstance(w, tk.Toplevel))
    chooser = next(w for w in descendants(modal) if isinstance(w, ttk.Combobox))
    trigger = next(w for w in descendants(modal) if isinstance(w, tk.Button)
                   and "Trigger Sleep" in w.cget("text"))
    chooser.set(source_label)

    # Set the cleanup state after opening the panel: its real button must recheck.
    setattr(app, unload_flag, True)
    previous_worker = getattr(app, "_generation_worker", None)
    trigger.invoke()
    assert getattr(app, "_generation_worker", None) is previous_worker
    assert not trained
    assert not app.is_generating and not getattr(app, "_sleep_consolidating", False)
    assert getattr(app, unload_flag)
    assert str(chooser.cget("state")) == "readonly"
    assert str(trigger.cget("state")) == "normal"
    text = "".join(w.get("1.0", "end") for w in descendants(modal) if isinstance(w, tk.Text))
    assert "Finish unloading the current model" in text and "Retry unload" in text


def test_sleep_keeps_verified_episode_queue_and_chat_modes_separate(app, monkeypatch):
    service = app.engine.awake_consolidator
    observed = []
    ui_thread = threading.get_ident()
    monkeypatch.setattr(service, '_real_training_ready', lambda: True)
    monkeypatch.setattr(service, 'pending_verified_episodes', lambda db: 1)
    monkeypatch.setattr(service, 'pending_exchanges', lambda history: 2)

    def verified(database, **kwargs):
        observed.append(('verified', database, threading.get_ident()))
        return {'status': 'success', 'memories_consolidated': 1, 'parameter_drift_l2': .01,
                'execution_time_seconds': .1, 'adapter_saved_to': 'verified-checkpoint'}

    def chat(history, **kwargs):
        observed.append(('chat', history, threading.get_ident()))
        return {'status': 'retained', 'adapter_saved_to': 'chat-checkpoint'}

    monkeypatch.setattr(service, 'sleep_verified_episodes', verified)
    monkeypatch.setattr(service, 'sleep', chat)
    app.chat_history[app.active_tab_id].clear()
    app._on_open_sleep_consolidation_panel()
    modal = next(w for w in app.root.winfo_children() if isinstance(w, tk.Toplevel))
    chooser = next(w for w in descendants(modal) if isinstance(w, ttk.Combobox))
    trigger = next(w for w in descendants(modal) if isinstance(w, tk.Button)
                   and 'Trigger Sleep' in w.cget('text'))
    assert chooser.get() == 'Verified episodes (original Sleep/EWC)'

    for mode in ('Verified episodes (original Sleep/EWC)', 'Current chat exchanges'):
        chooser.set(mode)
        chooser.event_generate('<<ComboboxSelected>>')
        expected = len(observed) + 1
        trigger.invoke()
        _wait_for_idle(app, timeout_ms=4000)
        assert len(observed) == expected and not app.is_generating
        assert str(chooser.cget('state')) == 'readonly'

    assert observed[0][0] == 'verified' and observed[0][1] is app.learner.db
    assert observed[1][0] == 'chat' and observed[1][1] is app.chat_history[app.active_tab_id]
    assert all(item[2] != ui_thread for item in observed)
    text = ''.join(w.get('1.0', 'end') for w in descendants(modal) if isinstance(w, tk.Text))
    assert 'verified-checkpoint' in text and 'chat-checkpoint' in text
    assert "this model and learning implementation's queue" in text and 'remain in active context' in text

    from core.memory_scopes import LearningScope
    app.engine.learning_scope = LearningScope('project', 'different-project')
    trigger.invoke()
    assert len(observed) == 2
    text = ''.join(w.get('1.0', 'end') for w in descendants(modal) if isinstance(w, tk.Text))
    assert 'reopen this panel' in text


@pytest.mark.parametrize('adapter_format', ['gguf', 'bitnet', 'peft'])
@pytest.mark.parametrize('incompatible', [False, True])
def test_gui_restores_previous_native_and_controller_learning(app, tmp_path, monkeypatch, adapter_format, incompatible):
    from core.memory_scopes import LearningScope

    selected = tmp_path / ('peft' if adapter_format == 'peft' else 'previous.gguf')
    if adapter_format == 'peft':
        selected.mkdir()
        artifact = selected / 'adapter_config.json'
    else:
        artifact = selected
    artifact.write_bytes(b'original learned state')
    tab = app.active_tab_id
    app.models_config[tab] = dict(app.models_config[tab], name='Original base', short_name='Original',
                                 model_type='text', model_path=str(tmp_path / 'base'), repo_id='')
    info = dict(app.models_config[tab])
    app.is_model_loaded = False
    app.engine.learning_scope = LearningScope('model')
    calls = []
    backend = SimpleNamespace(model=None, is_loaded=False, adapter_path=None)
    backend.unload_model = lambda: None
    attr = {'gguf': 'gguf_backend', 'bitnet': 'bitnet_backend', 'peft': 'controller_backend'}[adapter_format]
    monkeypatch.setattr(app.engine, attr, backend)
    monkeypatch.setattr(app, '_model_is_cached', lambda _info: True)
    monkeypatch.setattr(app, '_show_model_loading_popup', lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(app, '_sync_memory_watchdog', lambda *args, **kwargs: None)

    def load(name, **kwargs):
        calls.append((name, kwargs))
        path = kwargs['model_info'].get('legacy_adapter_path')
        if path and not incompatible:
            backend.model, backend.is_loaded, backend.adapter_path = object(), True, path
            return {'status': 'loaded', 'trainable': True,
                    'learning_migration': {'status': 'legacy_adapter_loaded', 'format': adapter_format, 'path': path}}
        return {'status': 'error', 'error': 'shape mismatch' if path else 'Legacy state requires selection',
                'learning_migration': {'status': 'legacy_adapter_required', 'format': adapter_format, 'path': str(selected)}}

    monkeypatch.setattr(app.engine, 'load_model', load)
    monkeypatch.setattr(app.engine, 'unload_model', lambda: {'status': 'unloaded'})
    app._on_toggle_load_unload(_skip_download=True)
    assert not app.is_model_loaded
    assert app._pending_legacy_learning()['path'] == str(selected)
    app._on_open_memory_explorer()
    modal = next(w for w in app.root.winfo_children() if isinstance(w, tk.Toplevel))
    button = next(w for w in descendants(modal) if isinstance(w, tk.Button)
                  and w.cget('text') == 'Restore learned state…')
    button.invoke()
    assert len(calls) == 2
    assert all(name == 'Original base' and 'backend' not in kwargs
               and kwargs['model_path'] == info['model_path'] for name, kwargs in calls)
    assert calls[1][1]['model_info'] == dict(info, legacy_adapter_path=str(selected))
    assert app.models_config[tab] == info and artifact.read_bytes() == b'original learned state'
    text = app.chat_stream.get('1.0', 'end')
    if incompatible:
        assert not app.is_model_loaded and app._pending_legacy_learning()
        assert 'shape mismatch' in text and 'Loaded the previous learned state' not in text
    else:
        assert app.is_model_loaded and app._pending_legacy_learning() is None
        assert 'Loaded the previous learned state' in text


def test_legacy_recovery_cannot_follow_a_changed_model_or_scope(app, tmp_path, monkeypatch):
    from core.memory_scopes import LearningScope

    tab = app.active_tab_id
    info = app.models_config[tab]
    pending = {'status': 'legacy_adapter_required', 'format': 'gguf', 'path': str(tmp_path / 'old.gguf'),
               'model_name': info['name'], 'model_path': str(info.get('model_path') or info.get('repo_id'))}
    app._legacy_learning_migrations = {tab: pending}
    app.engine.learning_scope = LearningScope('model')
    app.is_model_loaded = False
    app._on_open_memory_explorer()
    modal = next(w for w in app.root.winfo_children() if isinstance(w, tk.Toplevel))
    button = next(w for w in descendants(modal) if isinstance(w, tk.Button)
                  and w.cget('text') == 'Restore learned state…')
    notices, loads = [], []
    monkeypatch.setattr('app_gui.messagebox.showinfo', lambda *args, **kwargs: notices.append(args))
    monkeypatch.setattr(app, '_on_toggle_load_unload', lambda **kwargs: loads.append(kwargs))
    app.engine.learning_scope = LearningScope('project', 'another')
    button.invoke()
    assert notices and not loads and app._pending_legacy_learning() is None
    app.engine.learning_scope = LearningScope('model')
    assert app._pending_legacy_learning() == pending
    app.models_config[tab] = dict(info, model_path=str(tmp_path / 'different-model'))
    button.invoke()
    assert len(notices) == 2 and not loads and app._pending_legacy_learning() is None


def test_implementation_control_restores_its_scope_fields_and_keeps_both_startup_selections(app, monkeypatch):
    from core.memory_preferences import (
        load_learning_implementation, load_memory_preference, save_memory_preference,
    )
    from core.memory_scopes import LearningScope

    # This hybrid preference was saved in another desktop session. The running
    # engine has never entered hybrid and has no in-memory scope hint for it.
    db = app.settings.database_path
    managed, hybrid = LearningScope("project", "managed M"), LearningScope("session", "hybrid H")
    save_memory_preference(db, hybrid, implementation="hybrid")
    app.engine.load_model("fixture", backend="mock")
    app.engine.awake_consolidator.engine = None  # A mock load has no resident MLX adapter to swap.
    show_online_learning(app)
    modal = next(child for child in app.root.winfo_children() if isinstance(child, tk.Toplevel))
    controls = list(descendants(modal))
    chooser = next(w for w in controls if isinstance(w, ttk.Combobox) and tuple(w.cget("values")) == ("legacy", "managed", "hybrid"))
    scope = next(w for w in controls if isinstance(w, ttk.Combobox) and tuple(w.cget("values")) == ("model", "session", "project"))
    key = next(w for w in controls if isinstance(w, ttk.Entry) and not isinstance(w, ttk.Combobox))

    def apply(label):
        next(w for w in controls if isinstance(w, ttk.Button) and w.cget("text") == label).invoke()
        _wait_for_idle(app)

    scope.set(managed.kind)
    key.insert(0, managed.key)
    apply("Apply memory")
    assert load_memory_preference(db) == managed, [w.get("1.0", "end") for w in controls if isinstance(w, tk.Text)]
    for target, expected in (("hybrid", hybrid), ("legacy", LearningScope()), ("managed", managed)):
        chooser.set(target)
        apply("Apply implementation")
        assert load_learning_implementation(db) == target
        assert app.engine.learning_scope == expected
        assert (scope.get(), key.get()) == (expected.kind, expected.key)
    assert load_memory_preference(db, implementation=load_learning_implementation(db)) == managed
    assert load_memory_preference(db, implementation="hybrid") == hybrid

    def interrupted(*args, **kwargs):
        raise InterruptedError("selection cancelled")

    monkeypatch.setattr(app.engine, "set_learning_implementation", interrupted)
    chooser.set("hybrid")
    apply("Apply implementation")
    assert chooser.get() == "managed" and load_learning_implementation(db) == "managed"
    assert (scope.get(), key.get()) == (managed.kind, managed.key)
    assert load_memory_preference(db, implementation="hybrid") == hybrid
