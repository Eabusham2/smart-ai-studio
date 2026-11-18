"""Real Tk paging and learning-panel worker handoff, using synthetic replies."""
import threading
import tkinter as tk
from tkinter import ttk

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
    app = SmartAIChatbotApp(root, settings=Settings(use_mock=True, auto_download=False,
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
        root.destroy()
        import gc
        gc.collect()


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


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

    def poll():
        if observed and not app.is_generating:
            app.root.quit()
        else:
            app.root.after(10, poll)

    app.root.after(10, poll)
    app.root.after(3000, app.root.quit)
    app.root.mainloop()
    assert observed and observed[0][0] != main_thread
    assert observed[0][1]["method"] == method
    assert not app.is_generating and not app._sleep_consolidating
    output = [widget.get("1.0", "end") for widget in descendants(modal) if isinstance(widget, tk.Text)]
    assert any('"trained_chunks": 1' in text and '"completed"' in text for text in output)


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
        def poll():
            if len(observed) == expected and not app.is_generating:
                app.root.quit()
            else:
                app.root.after(10, poll)
        app.root.after(10, poll)
        app.root.after(4000, app.root.quit)
        app.root.mainloop()
        assert len(observed) == expected and not app.is_generating
        assert str(chooser.cget('state')) == 'readonly'

    assert observed[0][0] == 'verified' and observed[0][1] is app.learner.db
    assert observed[1][0] == 'chat' and observed[1][1] is app.chat_history[app.active_tab_id]
    assert all(item[2] != ui_thread for item in observed)
    text = ''.join(w.get('1.0', 'end') for w in descendants(modal) if isinstance(w, tk.Text))
    assert 'verified-checkpoint' in text and 'chat-checkpoint' in text
    assert 'shared queue' in text and 'remain in active context' in text

    from core.memory_scopes import LearningScope
    app.engine.learning_scope = LearningScope('project', 'different-project')
    trigger.invoke()
    assert len(observed) == 2
    text = ''.join(w.get('1.0', 'end') for w in descendants(modal) if isinstance(w, tk.Text))
    assert 'reopen this panel' in text
