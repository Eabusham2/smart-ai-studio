"""The real desktop binds all episodic consumers to its selected model."""
from pathlib import Path
import sqlite3
import threading
import time
import tkinter as tk
from tkinter import ttk

import pytest

import app_gui
from config.settings import get_settings
from memory.db import EpisodicMemoryDB


@pytest.fixture
def desktop(tmp_path, monkeypatch):
    monkeypatch.setenv('SMARTAI_DATA_DIR', str(tmp_path))
    monkeypatch.setattr(app_gui, 'APP_PREFS_FILE', str(tmp_path / 'preferences.json'))
    base = str(tmp_path / 'memory.db')
    old = EpisodicMemoryDB(base)
    old.log_interaction('Earlier mixed-owner history', 'Preserve this old response')
    settings = get_settings(backend='mock', use_mock=True, database_path=base, learning_implementation='hybrid')
    root = tk.Tk()
    app = app_gui.SmartAIChatbotApp(root, settings=settings)
    monkeypatch.setattr(app, '_model_is_cached', lambda info: False)
    root.geometry('1100x900')
    root.update()
    yield app, root, old
    root.destroy()


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


@pytest.mark.parametrize('messages', [('The next request',), ('First queued request', 'Second queued request')])
def test_stop_then_send_waits_for_the_old_worker_to_release_its_request(desktop, monkeypatch, messages):
    app, root, _ = desktop
    release = threading.Event()
    old_worker = threading.Thread(target=lambda: release.wait(5), daemon=True)
    started, handed_off = [], []
    monkeypatch.setattr(app, '_process_message_thread', lambda *args: started.append(args))
    monkeypatch.setattr(app, '_run_queued_task', lambda *args: handed_off.append((old_worker.is_alive(), args)))
    app.is_model_loaded = True
    app._set_generating_state(True)
    app._generation_worker = old_worker
    old_worker.start()
    try:
        app._on_stop_generation()
        assert not app.is_generating and app.cancel_event.is_set()
        # The previous request's completion callback can run before its outer
        # telemetry/SQLite wrapper has released the worker, with no next item.
        app._check_and_run_next_queue()
        assert app.prompt_queue == []
        stopped_epoch = app._generation_epoch
        app._placeholder_active = False
        app.txt_input.configure(state='normal')
        pending_waits = []
        for message in messages:
            app.txt_input.delete('1.0', 'end')
            app.txt_input.insert('1.0', message)
            app._on_send_message()
            pending_waits.append(app._queue_worker_wait_id)
        assert None not in pending_waits and len(set(pending_waits)) == 1
        assert not started
        assert app._generation_worker is old_worker
        assert app.cancel_event.is_set() and app._generation_epoch == stopped_epoch
        queued = [(message, message, None, None) for message in messages]
        assert app.prompt_queue == queued
        root.update()
        assert not handed_off and app.prompt_queue == queued
        release.set()
        old_worker.join(timeout=2)
        deadline = time.monotonic() + 2
        while not handed_off and time.monotonic() < deadline:
            root.update()
            time.sleep(.01)
        assert handed_off == [(False, queued[0])]
        assert app.prompt_queue == queued[1:]
    finally:
        release.set()
        old_worker.join(timeout=2)


def test_model_switch_routes_sql_tools_learner_and_transcript_to_same_owner(desktop):
    app, root, old = desktop
    normal_db = app.db
    normal_db.log_interaction('Normal episodic secret', 'normal answer', verified_reward=1)
    app.chat_history['model_1'].append({'role': 'user', 'content': 'Normal transcript only'})
    assert app.tools.db_path == normal_db.db_path and app.learner.db is normal_db
    assert 'Earlier mixed-owner' not in app.tools.execute_tool('read_chat_history', {})[1]
    app._on_switch_model_tab('model_2')
    root.update()
    crack_db = app.db
    assert crack_db.db_path != normal_db.db_path
    assert app.tools.db_path == crack_db.db_path and app.learner.db is crack_db
    assert app.learner.tools is app.tools
    assert 'Normal episodic secret' not in app.tools.execute_tool('read_chat_history', {})[1]
    assert 'media_generate' in {tool['name'] for tool in app.tools.list_available_tools()}
    crack_db.log_interaction('CRACK episodic secret', 'crack answer', verified_reward=1)
    app.chat_history['model_2'].append({'role': 'user', 'content': 'CRACK transcript only'})
    app._on_switch_model_tab('model_1')
    root.update()
    assert app.db.db_path == normal_db.db_path
    reply = app.tools.execute_tool('read_chat_history', {})[1]
    assert 'Normal episodic secret' in reply and 'CRACK episodic secret' not in reply
    visible = app.chat_stream.get('1.0', 'end')
    assert 'Normal transcript only' in visible and 'CRACK transcript only' not in visible
    assert old.get_stats()['total_interactions'] == 1


def test_implementation_rebind_keeps_each_models_original_episodes(desktop):
    app, root, _ = desktop
    hybrid_path = app.db.db_path
    app.db.log_interaction('Hybrid-owned episode', 'answer')
    app.settings.learning_implementation = 'legacy'
    app._learning_implementation_changed('hybrid')
    assert app.db.db_path != hybrid_path
    assert app.learner.learning_implementation == 'legacy'
    assert app.learner.db is app.db and app.tools.db_path == app.db.db_path
    assert 'Hybrid-owned' not in app.tools.execute_tool('read_chat_history', {})[1]
    app.settings.learning_implementation = 'hybrid'
    app._learning_implementation_changed('legacy')
    assert app.db.db_path == hybrid_path
    assert 'Hybrid-owned' in app.tools.execute_tool('read_chat_history', {})[1]


@pytest.mark.parametrize('raises', [False, True])
def test_failed_unload_keeps_original_model_memory_and_exposes_retry(desktop, monkeypatch, raises):
    app, root, _ = desktop
    origin_db, origin_tools, origin_tab = app.db, app.tools, app.active_tab_id
    def fail():
        if raises:
            raise RuntimeError('owned worker did not stop')
        return {'status': 'error', 'unloaded': False, 'error': 'owned worker did not stop'}
    monkeypatch.setattr(app.engine, 'unload_model', fail)
    app._on_switch_model_tab('model_2')
    root.update()
    assert app.active_tab_id == origin_tab
    assert app.db is origin_db and app.tools is origin_tools
    assert app._model_unload_pending and not app.is_model_loaded
    assert app.btn_load_unload.cget('text') == 'Retry unload'
    assert app.btn_send.cget('state') == 'disabled'
    monkeypatch.setattr(app.engine, 'unload_model', lambda: {'status': 'unloaded'})
    app.btn_load_unload.invoke()
    assert not app._model_unload_pending
    app._on_switch_model_tab('model_2')
    assert app.active_tab_id == 'model_2' and app.db.db_path != origin_db.db_path


def test_open_explorer_and_export_keep_captured_owner_after_model_switch(desktop, tmp_path, monkeypatch):
    app, root, _ = desktop
    app.db.log_interaction('Normal visible record', 'normal response')
    app._on_open_memory_explorer()
    root.update()
    modal = next(child for child in root.winfo_children() if isinstance(child, tk.Toplevel))
    tree = next(widget for widget in descendants(modal) if isinstance(widget, ttk.Treeview))
    app._on_switch_model_tab('model_2')
    app.db.log_interaction('CRACK hidden record', 'crack response')
    refresh = next(widget for widget in descendants(modal) if getattr(widget, '_command', None)
                   and 'Refresh' in str(widget.cget('text')))
    refresh.invoke()
    root.update()
    values = str([tree.item(row)['values'] for row in tree.get_children()])
    assert 'Normal visible record' in values and 'CRACK hidden record' not in values
    output = tmp_path / 'captured-model.md'
    monkeypatch.setattr(app_gui.filedialog, 'asksaveasfilename', lambda **kw: str(output))
    monkeypatch.setattr(app_gui.messagebox, 'showinfo', lambda *a, **kw: None)
    export = next(widget for widget in descendants(modal) if getattr(widget, '_command', None)
                  and 'Export Markdown' in str(widget.cget('text')))
    export.invoke()
    assert 'Normal visible record' in output.read_text() and 'CRACK hidden record' not in output.read_text()
    # sqlite.Row has no .get; selecting a row must produce the detail body.
    tree.selection_set(tree.get_children()[0])
    tree.event_generate('<<TreeviewSelect>>')
    root.update()
    assert any('normal response' in widget.get('1.0', 'end') for widget in descendants(modal)
               if isinstance(widget, tk.Text))


def test_earlier_shared_history_is_readonly_and_does_not_rebind_active_memory(desktop, monkeypatch):
    app, root, old = desktop
    current_db, current_tools, current_learner = app.db, app.tools, app.learner
    connections = []
    original = sqlite3.connect
    def observed(path, *args, **kwargs):
        connections.append((path, kwargs))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(sqlite3, 'connect', observed)
    app._on_open_memory_explorer(legacy_shared=True)
    root.update()
    modal = next(child for child in root.winfo_children() if isinstance(child, tk.Toplevel))
    tree = next(widget for widget in descendants(modal) if isinstance(widget, ttk.Treeview))
    assert 'Earlier mixed-owner' in str([tree.item(row)['values'] for row in tree.get_children()])
    assert all(options.get('uri') and str(path).endswith('?mode=ro') for path, options in connections)
    assert (app.db, app.tools, app.learner) == (current_db, current_tools, current_learner)
    assert not any('Learning & saved memory' in str(widget.cget('text')) for widget in descendants(modal)
                   if getattr(widget, '_command', None))
