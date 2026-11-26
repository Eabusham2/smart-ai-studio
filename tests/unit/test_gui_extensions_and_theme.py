"""Real Tk controls and theme regressions, without downloading/loading a model."""
from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
import tkinter as tk

import pytest

from core import gui_extensions
from tests.unit.test_topbar_click_targets import _assert_click_target, _resize_on_screen


@pytest.fixture
def app(monkeypatch, tmp_path):
    import app_gui
    from config.settings import get_settings

    monkeypatch.setattr(app_gui, "APP_PREFS_FILE", str(tmp_path / "prefs.json"))
    monkeypatch.setattr(app_gui, "CUSTOM_MODELS_FILE", str(tmp_path / "models.json"))
    monkeypatch.setattr(app_gui, "get_portable_data_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_gui, "GUI_EXTENSION_FAILURES", [])
    settings = get_settings(backend="mock")
    settings.database_path = str(tmp_path / "memory.sqlite")
    root = tk.Tk()
    instance = app_gui.SmartAIChatbotApp(root, settings=settings)
    _resize_on_screen(root, 1120, 900)
    try:
        yield instance
    finally:
        instance.watchdog.stop_monitoring()
        root.destroy()


def _descendants(widget):
    yield widget
    for child in widget.winfo_children():
        yield from _descendants(child)


def _fake_module(tmp_path):
    class App:
        def _build_single_top_bar(self):
            return "base"
        def _on_switch_model_tab(self, tab):
            return tab
        def _apply_theme_colors(self):
            pass
        def _on_close_app(self):
            pass
        def _set_generating_state(self, value):
            pass
        def _process_message_thread(self, *args):
            pass
    return SimpleNamespace(SmartAIChatbotApp=App, tk=tk, messagebox=SimpleNamespace(),
                           get_portable_data_dir=lambda: str(tmp_path))


def test_installers_bind_explicit_module_and_are_idempotent(monkeypatch, tmp_path):
    actual = _fake_module(tmp_path)
    decoy = _fake_module(tmp_path)
    monkeypatch.setitem(sys.modules, "app_gui", decoy)
    assert gui_extensions.install_gui_extensions(actual) == []
    cls = actual.SmartAIChatbotApp
    for _, _, _, flag in gui_extensions.EXTENSIONS:
        assert getattr(cls, flag)
        assert not getattr(decoy.SmartAIChatbotApp, flag, False)
    methods = dict(vars(cls))
    assert gui_extensions.install_gui_extensions(actual) == []
    assert all(vars(cls)[name] is value for name, value in methods.items())


def test_legacy_direct_installer_resolves_main_module(monkeypatch, tmp_path):
    module = _fake_module(tmp_path)
    module.__file__ = str(tmp_path / "app_gui.py")
    monkeypatch.delitem(sys.modules, "app_gui", raising=False)
    monkeypatch.setitem(sys.modules, "__main__", module)
    from core.gui_eval_panel import install_gui_eval_panel
    from core.gui_generation_cap import install_gui_generation_cap
    install_gui_eval_panel()
    install_gui_generation_cap()
    assert module.SmartAIChatbotApp._eval_panel_installed
    assert module.SmartAIChatbotApp._generation_cap_ui_installed


def test_partial_registration_rolls_back_and_records_actionable_failure(monkeypatch, tmp_path, caplog):
    module = _fake_module(tmp_path)
    cls = module.SmartAIChatbotApp
    original = cls._build_single_top_bar
    def broken(target):
        target.SmartAIChatbotApp._build_single_top_bar = lambda self: "partial"
        target.SmartAIChatbotApp._temporary_partial_install = True
        raise RuntimeError("missing dependency fixture")
    monkeypatch.setattr(gui_extensions, "EXTENSIONS", (("Eval", "bad", "install", "_eval_panel_installed"),))
    monkeypatch.setattr(gui_extensions.importlib, "import_module", lambda _: SimpleNamespace(install=broken))
    failures = gui_extensions.install_gui_extensions(module)
    assert failures == [{"feature": "Eval", "message": "RuntimeError: missing dependency fixture"}]
    assert cls._build_single_top_bar is original
    assert not hasattr(cls, "_temporary_partial_install")
    assert "missing dependency fixture" in caplog.text
    assert "RuntimeError" in Path(module.GUI_EXTENSION_LOG).read_text()


def test_packaged_script_entry_registers_all_extensions(tmp_path):
    """Exercise __main__ just like launch_studio.sh, rather than importing app_gui."""
    source = Path(__file__).resolve().parents[2] / "app_gui.py"
    code = """
import runpy
import tkinter as tk
import config.paths
config.paths.terminate_existing_app_instances = lambda: None
def check_and_close(root):
    root.update()
    def walk(widget):
        yield widget
        for child in widget.winfo_children():
            yield from walk(child)
    labels = [str(w.cget('text')) for w in walk(root) if 'text' in w.keys() and w.winfo_ismapped()]
    for text in ('Eval', 'Context Limit', 'Apply', 'Auto speed ✓ On', 'Retune', 'Cancel', 'Speedups…'):
        assert text in labels, (text, labels)
    root.destroy()
tk.Tk.mainloop = check_and_close
state = runpy.run_path(%r, run_name='__main__')
assert state['GUI_EXTENSION_FAILURES'] == []
print('all packaged controls visible')
""" % str(source)
    result = subprocess.run([sys.executable, "-c", code], cwd=source.parent,
                            env={**os.environ, "SMARTAI_DATA_DIR": str(tmp_path),
                                 "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all packaged controls visible" in result.stdout


@pytest.mark.parametrize("width,height", [(760, 600), (760, 900), (1280, 720), (1280, 900)])
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_all_persistent_controls_visible_clickable_and_inside_window(app, width, height, theme):
    root = app.root
    # Include the conditional Reset action by representing an installed model.
    app._model_is_cached = lambda _info: True
    app._update_model_action_buttons()
    app.current_theme = theme
    app._apply_theme_colors()
    _resize_on_screen(root, width, height)
    names = (
        "btn_model_menu", "lbl_workspace", "chk_pro_branches", "chk_memory_limit",
        "entry_memory_limit", "btn_eval", "btn_load_unload", "btn_reset_reinstall",
        "btn_export_chat", "btn_branch_vis", "btn_memory_exp", "btn_sleep_panel",
        "btn_dsl_play", "btn_toggle_canvas", "btn_toggle_resources", "btn_zoom",
        "btn_theme_toggle", "ent_generation_cap", "btn_apply_generation_cap",
        "chk_auto_speed", "btn_speed_tune", "btn_speed_cancel", "btn_lossless_speedups",
        "lbl_generation_cap_reason", "lbl_speed_status",
        "history_older_button", "history_latest_button", "history_newer_button",
        "btn_attach", "txt_input", "btn_send", "chat_stream",
    )
    for name in names:
        widget = getattr(app, name)
        _assert_click_target(root, widget)
        assert widget.winfo_rooty() + widget.winfo_height() <= root.winfo_rooty() + root.winfo_height()
    # Import remains an actual menu action, rather than a hidden dead button.
    menu = root.nametowidget(app.btn_model_menu.cget("menu"))
    labels = [menu.entrycget(i, "label") for i in range(int(menu.index("end")) + 1)
              if menu.type(i) not in {"separator", "tearoff"}]
    assert any("Import" in label for label in labels)


@pytest.mark.parametrize("model", ["model_1", "model_2"])
@pytest.mark.parametrize("loaded", [False, True])
def test_both_bonsai_presets_expose_eval_before_and_after_load(app, model, loaded):
    app.active_tab_id = model
    app.is_model_loaded = loaded
    app._sync_eval_button_visibility()
    app.root.update()
    _assert_click_target(app.root, app.btn_eval)
    app.btn_eval.invoke()
    app.root.update()
    assert app._eval_modal.winfo_ismapped()
    for name in ("_eval_mem_check", "_eval_mem_entry", "_eval_deepswe_check", "_eval_start_button",
                 "_eval_pause_button", "_eval_cancel_button", "_eval_log_widget"):
        assert getattr(app, name).winfo_ismapped(), name
    assert app._eval_proc is None  # Opening never runs a benchmark or loads weights.


def test_every_model_type_obeys_eval_visibility_rule(app):
    for model, info in app.models_config.items():
        app.active_tab_id = model
        app._sync_eval_button_visibility()
        app.root.update()
        assert bool(app.btn_eval.winfo_ismapped()) is (info.get("model_type", "text") == "text"), model


def test_eval_controls_wrap_at_minimum_dialog_width(app):
    app._open_eval_window()
    window = app._eval_modal
    for width in (1000, 760, 1000):
        _resize_on_screen(window, width, 720)
        for name in ("_eval_mem_check", "_eval_mem_entry", "_eval_deepswe_check",
                     "_eval_start_button", "_eval_pause_button", "_eval_cancel_button"):
            _assert_click_target(window, getattr(app, name))


def test_canvas_actions_wrap_and_remain_accessible_during_resize(app):
    app._on_toggle_canvas_viewer()
    for width in (1280, 760, 1050):
        _resize_on_screen(app.root, width, 900)
        assert app.canvas_panel.winfo_rootx() + app.canvas_panel.winfo_width() <= app.root.winfo_rootx() + app.root.winfo_width()
        controls = app.canvas_toolbar.winfo_children()
        assert len(controls) == 9  # Title plus all eight original actions.
        for widget in controls:
            _assert_click_target(app.root, widget)
        _assert_click_target(app.root, app.txt_canvas)


def test_wrapping_controls_wait_for_real_width_and_restore_labels_when_widened(app):
    from core.responsive_toolbar import install_wrapping_controls
    frame = tk.Frame(app.root)
    label = tk.Label(frame, text="A long status label should return to one line when the window grows.")
    label.pack(side="left")
    app.root.update_idletasks()
    natural_width, natural_height = label.winfo_reqwidth(), label.winfo_reqheight()
    schedule = install_wrapping_controls(frame)
    app.root.update()
    assert label.winfo_pixels(label.cget("wraplength")) == 0  # The unmapped 1px frame is not a real viewport.
    frame.place(x=10, y=10, width=150, height=140)
    app.root.update()
    assert label.winfo_reqheight() > natural_height
    frame.place_configure(width=natural_width + 40)
    schedule()
    app.root.update()
    assert label.winfo_pixels(label.cget("wraplength")) == 0
    assert label.winfo_reqwidth() == natural_width
    assert label.winfo_reqheight() == natural_height
    frame.destroy()


def _actions(window):
    from tkinter import ttk
    classes = (tk.Button, tk.Checkbutton, tk.Entry, tk.Menubutton, tk.Scale, tk.Text,
               ttk.Button, ttk.Checkbutton, ttk.Combobox, ttk.Entry, ttk.Spinbox, ttk.Treeview)
    return [widget for widget in _descendants(window) if isinstance(widget, classes)]


@pytest.mark.parametrize("kind,expected", [
    ("memory", {"🔄 Refresh", "Learning & saved memory…", "📥 Export Markdown"}),
    ("learning", {"Apply implementation", "Apply memory", "Choose source…", "Run learning task", "Cancel", "Close"}),
    ("branches", {"Apply passes"}),
    ("sleep", {"▶ Trigger Sleep Consolidation Now"}),
    ("dsl", {"Send to AI Canvas", "▶ Run in Sandbox"}),
    ("telemetry", {"🧹 Trim Cache", "Close"}),
    ("import", {"Cancel", "➕ Import & Select"}),
])
def test_all_dialog_actions_are_visible_at_minimum_and_regular_sizes(app, kind, expected):
    before = set(app.root.winfo_children())
    if kind == "learning":
        from core.learning_ui import show_online_learning
        show_online_learning(app)
    else:
        method = {"memory": "_on_open_memory_explorer", "branches": "_on_open_branch_visualizer",
                  "sleep": "_on_open_sleep_consolidation_panel", "dsl": "_on_open_dsl_playground",
                  "telemetry": "_on_show_memory_telemetry_dialog", "import": "_on_import_custom_model_dialog"}[kind]
        getattr(app, method)()
    window = next(w for w in app.root.winfo_children() if w not in before and isinstance(w, tk.Toplevel))
    controls = _actions(window)
    labels = {str(widget.cget("text")) for widget in controls if "text" in widget.keys()}
    assert expected <= labels
    window.update()
    # Opening a dialog must already place it on screen; a later test resize
    # must not conceal a production startup-position or initial-layout bug.
    assert window.winfo_rootx() >= 0 and window.winfo_rooty() >= 0
    assert window.winfo_rootx() + window.winfo_width() <= window.winfo_screenwidth()
    assert window.winfo_rooty() + window.winfo_height() <= window.winfo_screenheight()
    minimum = window.minsize()
    for width, height in ((max(minimum[0], 640), max(minimum[1], 560)), (1000, 800)):
        _resize_on_screen(window, width, height)
        for widget in controls:
            _assert_click_target(window, widget)
            assert widget.winfo_rooty() + widget.winfo_height() <= window.winfo_rooty() + window.winfo_height(), (kind, widget)
        app._on_toggle_theme()
        window.update()
        for widget in controls:
            _assert_click_target(window, widget)


def test_speedups_scroll_exposes_every_option_and_keeps_save_visible(app):
    app._open_lossless_speedups()
    window = app._speedups_dialog
    canvas = next(w for w in _descendants(window) if isinstance(w, tk.Canvas))
    body = canvas.nametowidget(canvas.itemcget(canvas.find_all()[0], "window"))
    controls = _actions(window)
    from core.gui_lossless_speedups import _BOOLEAN_FIELDS, _PATH_FIELDS
    assert len([w for w in controls if isinstance(w, tk.Checkbutton)]) == len(_BOOLEAN_FIELDS)
    assert len([w for w in controls if isinstance(w, tk.Entry)]) == len(_PATH_FIELDS) + 2
    for width in (780, 420):
        _resize_on_screen(window, width, 620)
        for widget in controls:
            if str(widget).startswith(str(body) + "."):
                y = widget.winfo_rooty() - body.winfo_rooty()
                canvas.yview_moveto(max(0.0, (y - canvas.winfo_height() / 2) / body.winfo_height()))
                window.update()
            _assert_click_target(window, widget)
            _assert_click_target(window, app._speedups_save_button)


def _mark_busy(app, kind):
    if kind == "worker":
        app._generation_worker = SimpleNamespace(is_alive=lambda: True)
    elif kind == "consolidation":
        app.engine.awake_consolidator = SimpleNamespace(is_consolidating=True)
    elif kind == "media":
        app.multimodal = SimpleNamespace(busy=True)
    elif kind == "evaluation":
        app._eval_proc = SimpleNamespace(poll=lambda: None)
    else:
        setattr(app, kind, True)


@pytest.mark.parametrize("busy", ["is_generating", "worker", "_sleep_consolidating", "consolidation", "media",
                                   "_model_unload_in_progress", "_model_unload_pending",
                                   "_strata_unload_pending", "_native_unload_pending"])
def test_open_eval_window_cannot_start_over_another_model_operation(app, monkeypatch, busy):
    import core.gui_eval_panel as panel
    notices = []
    monkeypatch.setattr(panel.subprocess, "Popen", lambda *a, **k: pytest.fail("Eval launched while model busy"))
    monkeypatch.setattr("app_gui.messagebox.showinfo", lambda *a, **k: notices.append(a))
    monkeypatch.setattr(app.engine, "unload_model", lambda *a, **k: pytest.fail("Busy model was unloaded by Eval"))
    app._open_eval_window()
    _mark_busy(app, busy)
    app._eval_start_button.invoke()
    assert notices and notices[-1][0] == "Model Busy"
    assert app._eval_proc is None


@pytest.mark.parametrize("busy", ["worker", "_sleep_consolidating", "_model_unload_in_progress",
                                   "_model_unload_pending", "_strata_unload_pending", "_native_unload_pending", "evaluation"])
def test_open_learning_window_cannot_overlap_model_worker_or_evaluation(app, monkeypatch, busy):
    from core.learning_ui import show_online_learning
    from tkinter import ttk
    notices = []
    monkeypatch.setattr("tkinter.messagebox.showinfo", lambda *a, **k: notices.append(a))
    monkeypatch.setattr(app.engine, "run_online_learning", lambda *a, **k: pytest.fail("Overlapping learning task"))
    show_online_learning(app)
    window = next(w for w in app.root.winfo_children() if isinstance(w, tk.Toplevel))
    run = next(w for w in _descendants(window) if isinstance(w, ttk.Button) and w.cget("text") == "Run learning task")
    _mark_busy(app, busy)
    run.invoke()
    assert notices and notices[-1][0] == "Learning"
    assert not app.is_generating


@pytest.mark.parametrize("failure", ["exception", "error", "busy", "contradiction", "unconfirmed", "still_loaded"])
def test_eval_requires_confirmed_unload_and_preserves_retry_owner(app, monkeypatch, failure):
    import core.gui_eval_panel as panel
    from core.gui_speed_tuning import speed_busy_reason
    notices, watchdog_calls = [], []
    owner = SimpleNamespace(runtime="fixture", is_mlx_available=True, model=object())
    app.settings.use_mock = False
    app.engine.mlx_backend = owner
    app.engine.active_backend = "mlx"
    app.is_model_loaded = True
    def unload_failure():
        if failure == "exception":
            raise RuntimeError("fixture cleanup failed")
        if failure in {"error", "busy"}:
            return {"status": failure, "unloaded": False, "error": "fixture cleanup failed"}
        if failure == "contradiction":
            return {"status": "unloaded", "unloaded": False}
        if failure == "unconfirmed":
            return None
        return {"status": "unloaded"}  # Contradicts the engine's live loaded flag.
    monkeypatch.setattr(app.engine, "unload_model", unload_failure)
    monkeypatch.setattr(app, "_sync_memory_watchdog", lambda **kwargs: watchdog_calls.append(kwargs))
    monkeypatch.setattr(panel.subprocess, "Popen", lambda *a, **k: pytest.fail("Second model launched before unload"))
    monkeypatch.setattr("app_gui.messagebox.showerror", lambda *a, **k: notices.append(a))
    app._open_eval_window()
    app._eval_start_button.invoke()
    assert app._eval_proc is None
    assert app._model_unload_pending
    assert not app.is_model_loaded
    assert app.engine.mlx_backend is owner
    assert app.engine.is_model_loaded
    assert watchdog_calls == []
    assert app.btn_load_unload.cget("text") == "Retry unload"
    assert app.btn_load_unload.cget("state") == "normal"
    assert "Retry unload" in app._eval_state_label.cget("text")
    assert notices and notices[0][0] == "Model Unload Failed"
    assert "unloading" in speed_busy_reason(app)

    # The pending flag blocks new work, but must never block the user's retry.
    def successful_retry():
        app.engine.mlx_backend = None
        return {"status": "unloaded"}
    monkeypatch.setattr(app.engine, "unload_model", successful_retry)
    app.btn_load_unload.invoke()
    assert not app._model_unload_pending
    assert app.engine.mlx_backend is None
    assert app._eval_proc is None


def test_eval_confirmed_unload_reaches_launch_only_after_releasing_the_model(app, monkeypatch):
    import core.gui_eval_panel as panel
    app.settings.use_mock = False
    app.engine.active_backend = "mlx"
    app.engine.mlx_backend = SimpleNamespace(is_mlx_available=True, model=object())
    app.is_model_loaded = True
    calls, notices = [], []
    def unload():
        calls.append("unload")
        app.engine.mlx_backend = None
        return {"status": "unloaded"}
    def launch(*args, **kwargs):
        assert not app.is_model_loaded and not app.engine.is_model_loaded
        assert not app._model_unload_pending
        calls.append("launch")
        raise OSError("fixture stops before spawning an actual benchmark")
    monkeypatch.setattr(app.engine, "unload_model", unload)
    monkeypatch.setattr(panel.subprocess, "Popen", launch)
    monkeypatch.setattr("app_gui.messagebox.showerror", lambda *a, **k: notices.append(a))
    app._open_eval_window()
    app._eval_start_button.invoke()
    assert calls == ["unload", "launch"]
    assert notices[0][0] == "Eval Launch Failed"
    assert "fixture stops" in notices[0][1]


def test_resource_attachment_and_queue_controls_are_reachable(app, tmp_path):
    _resize_on_screen(app.root, 760, 900)
    app._on_toggle_resource_viewer()
    app.root.update()
    controls = _actions(app.res_drawer)
    assert len([w for w in controls if isinstance(w, tk.Scale)]) == 3
    for widget in controls:
        _assert_click_target(app.root, widget)
    app._on_toggle_resource_viewer()
    attached = tmp_path / "example.txt"
    attached.write_text("sample document")
    app._set_attachment_path(str(attached))
    app.prompt_queue.append(("chat", "queued fixture", None, None))
    app._update_queue_ui()
    app.root.update()
    for panel in (app.attachment_bar, app.queue_bar):
        assert panel.winfo_ismapped()
        for widget in _actions(panel):
            _assert_click_target(app.root, widget)
    app._on_toggle_theme()
    assert app.lbl_attached_file.cget("fg") == app.C["accent_cyan"]
    app.btn_clear_queue.invoke()
    assert app.prompt_queue == []
    app._on_remove_attachment()
    assert not app.attachment_bar.winfo_ismapped()


def test_branch_empty_state_never_fabricates_benchmark_or_ram_numbers(app):
    from tkinter import ttk
    app.last_metadata = None
    app._on_open_branch_visualizer()
    window = next(w for w in app.root.winfo_children() if isinstance(w, tk.Toplevel))
    labels = " ".join(str(w.cget("text")) for w in _descendants(window) if "text" in w.keys())
    assert "No completed run for this model" in labels
    assert "RSS: unavailable" in labels
    assert "10.9" not in labels and "1536" not in labels
    tree = next(w for w in _descendants(window) if isinstance(w, ttk.Treeview))
    assert tree.get_children() == ()


@pytest.mark.parametrize("switch_at", [None, "worker", "callback"])
def test_context_extension_cannot_publish_metadata_into_another_model(tmp_path, switch_at):
    from core.gui_generation_cap import install_gui_generation_cap
    module = _fake_module(tmp_path)
    databases = [tmp_path / "a.sqlite", tmp_path / "b.sqlite"]
    for path in databases:
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE interactions (id INTEGER PRIMARY KEY, prompt TEXT, mode TEXT, raw_branches TEXT, "
                       "verified_reward REAL, surprise_score REAL, entropy REAL, winning_branch INTEGER, winning_temp REAL)")
            db.execute("INSERT INTO interactions(id,prompt,mode) VALUES(1, 'same question', 'Instant Stream (N=1)')")
    def switch(instance):
        instance.active_tab_id = "b"
        instance._generation_epoch += 1
        instance.db.db_path = str(databases[1])
    def original(instance, *_args, **_kwargs):
        if switch_at == "worker":
            switch(instance)
        return "response"
    module.SmartAIChatbotApp._process_message_thread = original
    install_gui_generation_cap(module)
    instance = module.SmartAIChatbotApp()
    callbacks, telemetry = [], []
    instance.active_tab_id = "a"
    instance._generation_epoch = 2
    instance.chat_stream = object()
    instance.last_metadata = None
    instance.db = SimpleNamespace(db_path=str(databases[0]))
    instance.root = SimpleNamespace(after=lambda delay, callback: callbacks.append(callback))
    instance._update_telemetry = telemetry.append
    instance.engine = SimpleNamespace(mlx_backend=SimpleNamespace(last_tok_per_sec=21.0),
                                     _last_stream_pro_meta={"branch_count": 2, "mode": "Pro Search", "verified_reward": 1.0})
    assert instance._process_message_thread("same question", "same question") == "response"
    if switch_at == "callback":
        switch(instance)
    for callback in callbacks:
        callback()
    assert telemetry == ([21.0] if switch_at is None else [])
    assert (instance.last_metadata is not None) is (switch_at is None)
    for index, path in enumerate(databases):
        with sqlite3.connect(path) as db:
            mode = db.execute("SELECT mode FROM interactions").fetchone()[0]
        assert mode == ("Pro Search" if index == 0 and switch_at != "worker" else "Instant Stream (N=1)")


def test_runtime_visibility_failure_is_reported_in_the_window(app, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("broken layout fixture")
    monkeypatch.setattr(app, "_responsive_toolbar_set_visible", broken)
    app._on_switch_model_tab(app.active_tab_id)
    app.root.update()
    assert app._gui_failure_notice.winfo_ismapped()
    assert "Eval visibility" in app._gui_failure_notice.cget("text")
    assert "broken layout fixture" in app._gui_failure_notice.cget("text")


def test_repeated_theme_switch_covers_hidden_panels_dialogs_tags_and_hover(app):
    import app_gui
    from tkinter import ttk

    # Capture existing controls, including all initially hidden panes, before
    # opening dialogs. No widget rebuilding or content loss is acceptable.
    app.current_theme = "dark"
    app._apply_theme_colors()
    palette_identity = app.C
    app._open_eval_window()
    app._open_lossless_speedups()
    app._on_open_memory_explorer()
    app._on_show_memory_telemetry_dialog()
    app.root.update()
    widgets = list(_descendants(app.root))
    originals = {widget: {option: str(widget.cget(option)) for option in ("background", "foreground")
                          if option in widget.keys()} for widget in widgets}
    app.chat_stream.configure(state="normal")
    app.chat_stream.insert("end", "theme-check content", "think_body")
    app.chat_stream.configure(state="disabled")
    original_text = app.chat_stream.get("1.0", "end")
    send_state = app.btn_send.cget("state")
    dark_only = set(app_gui._THEMES["dark"].values()) - set(app_gui._THEMES["light"].values())
    for theme in ("light", "dark", "light", "dark"):
        app.current_theme = theme
        app._apply_theme_colors()
        app.root.update()
        assert app.C is palette_identity
        assert app.chat_stream.get("1.0", "end") == original_text
        assert app.btn_send.cget("state") == send_state
        assert app.entry_memory_limit.cget("bg") == app.C["bg_input_inner"]
        assert app.chk_memory_limit.cget("bg") == app.C["bg_hud"]
        assert app.chk_pro_branches.cget("fg") == app.C["btn_fg"]
        assert app.lbl_workspace.master.cget("bg") == app.C["bg_hud"]
        assert app.btn_zoom.cget("bg") == app.C["btn_bg"]
        assert app._eval_log_widget.cget("bg") == app.C["code_bg"]
        assert app.chat_stream.tag_cget("think_body", "background") == app.C["bg_thinking"]
        assert ttk.Style().lookup("Treeview", "background") == app.C["bg_app"]
        for widget, options in originals.items():
            assert widget.winfo_exists()
            if theme == "light":
                for option, color in options.items():
                    if color in dark_only:
                        assert str(widget.cget(option)) not in dark_only, (widget, option, widget.cget(option))
        app.btn_zoom.event_generate("<Enter>")
        assert app.btn_zoom.cget("bg") == app.C["btn_hover"]
        app.btn_zoom.event_generate("<Leave>")
        assert app.btn_zoom.cget("bg") == app.C["btn_bg"]


def test_solid_button_exposes_real_disabled_state_and_preserves_it_on_theme_switch(app):
    calls = []
    app.btn_eval.configure(command=lambda: calls.append("Eval"), state="disabled")
    assert app.btn_eval.cget("state") == "disabled"
    assert app.btn_eval["state"] == "disabled"
    app.btn_eval.invoke()
    app._on_toggle_theme()
    assert app.btn_eval.cget("state") == "disabled"
    app.btn_eval.invoke()
    assert calls == []
    app.btn_eval.configure(state="normal")
    app.btn_eval.invoke()
    assert calls == ["Eval"]
