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
        if app.watchdog:
            app.watchdog.stop_monitoring()
        app.engine.unload_model()
        root.destroy()


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


def test_learning_panel_runs_model_work_off_tk_and_returns_to_idle(app):
    main_thread = threading.get_ident()
    observed = []

    def run(request, **kwargs):
        observed.append((threading.get_ident(), request))
        kwargs["progress"]({"status": "running", "next": 1})
        return {"status": "completed", "trained_chunks": 1}

    app.engine.run_online_learning = run
    show_online_learning(app)
    modal = next(child for child in app.root.winfo_children() if isinstance(child, tk.Toplevel))
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
    assert observed[0][1]["method"] == "ingest"
    assert not app.is_generating and not app._sleep_consolidating
    output = [widget.get("1.0", "end") for widget in descendants(modal) if isinstance(widget, tk.Text)]
    assert any('"trained_chunks": 1' in text and '"completed"' in text for text in output)
