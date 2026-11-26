"""Speedup form and idle-apply contracts with fake Tk and engine; no display/model."""
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gui_lossless_speedups_test", ROOT / "core/gui_lossless_speedups.py")
gui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gui)


class Variable:
    def __init__(self, value):
        self.value = value
    def get(self):
        return self.value
    def set(self, value):
        self.value = value


class Widget:
    def __init__(self, parent=None, *args, **options):
        self.parent = parent
        self.args = args
        self.options = options
        self.bindings = {}
        self.children = []
        self.destroyed = False
        self.pack_options = {}
        self.lifted = False
        if hasattr(parent, "children"):
            parent.children.append(self)
    def configure(self, **options):
        self.options.update(options)
    def cget(self, name):
        return self.options[name]
    def pack(self, **options):
        self.pack_options = options
    def bind(self, event, callback):
        self.bindings[event] = callback
    def title(self, value):
        self.title_value = value
    def transient(self, parent):
        self.transient_parent = parent
    def geometry(self, value):
        self.geometry_value = value
    def minsize(self, *values):
        self.minimum = values
    def protocol(self, name, callback):
        self.bindings[name] = callback
    def winfo_screenheight(self):
        return 768
    def winfo_screenwidth(self):
        return 1280
    def winfo_exists(self):
        return not self.destroyed
    def destroy(self):
        self.destroyed = True
    def lift(self):
        self.lifted = True
    def yview(self, *args):
        pass
    def set(self, *args):
        pass
    def create_window(self, *args, **kwargs):
        return 1
    def itemconfigure(self, *args, **kwargs):
        pass
    def bbox(self, *args):
        return (0, 0, 700, 1000)


def make_app(monkeypatch, tmp_path, *, existing_bar=True, build=True):
    class Engine:
        def __init__(self):
            self.lossless_speed_options = dict(gui.DEFAULT_OPTIONS)
            self.configure_calls = []
            self.result = {"status": "configured", "message": "Prompt options applied."}
        def configure_lossless_speedups(self, options):
            self.configure_calls.append(options)
            if self.result["status"] != "busy":
                self.lossless_speed_options = dict(options)
            return self.result
        def load_model(self, *args, **kwargs):
            raise AssertionError("GUI settings must never load a model")
        def stream_generate(self, *args, **kwargs):
            raise AssertionError("GUI settings must never execute inference")

    class App:
        def __init__(self):
            self.root = Widget()
            self.main_container = Widget()
            self.engine = Engine()
            self.C = {key: key for key in ("bg_hud", "text_main", "btn_bg", "btn_fg", "btn_hover", "text_muted")}
            self.is_generating = False
            self.is_model_loaded = False
            self.build_calls = 0
        def _build_single_top_bar(self):
            self.build_calls += 1
            self.options_seen_at_build = dict(self.engine.lossless_speed_options)
            if existing_bar:
                self.speed_tuning_bar = Widget(self.main_container)
                self.lbl_speed_status = Widget(self.speed_tuning_bar)
        def _apply_theme_colors(self):
            return "base-theme"
        def _on_close_app(self):
            self.closed = True
        def _on_toggle_load_unload(self):
            self.options_seen_at_load_handler = dict(self.engine.lossless_speed_options)
            return "load handler"

    chooser = SimpleNamespace(askdirectory=lambda **kwargs: "/local/cache", askopenfilename=lambda **kwargs: "/local/artifact")
    module = SimpleNamespace(SmartAIChatbotApp=App, APP_PREFS_FILE=str(tmp_path / "prefs.json"),
                             _FONT_TINY="small", _FONT_TINY_BOLD="bold", filedialog=chooser,
                             tk=SimpleNamespace(Frame=Widget, Button=Widget, Label=Widget, Checkbutton=Widget,
                                                Toplevel=Widget, Canvas=Widget, Scrollbar=Widget,
                                                Entry=Widget, OptionMenu=Widget, BooleanVar=Variable, StringVar=Variable))
    monkeypatch.setitem(sys.modules, "app_gui", module)
    gui.install_gui_lossless_speedups()
    app = App()
    if build:
        app._build_single_top_bar()
    return app, module


def test_saved_options_loaded_before_ui_and_model_handler(monkeypatch, tmp_path):
    (tmp_path / "prefs.json").write_text(json.dumps({"lossless_speedups": {
        "enabled": False, "persistent_cache": True, "speculative_mode": "lookup"}}))
    app, _ = make_app(monkeypatch, tmp_path)
    assert app.options_seen_at_build["enabled"] is False
    assert app.options_seen_at_build["persistent_cache"] is True
    assert app.engine.configure_calls == []
    assert app._on_toggle_load_unload() == "load handler"
    assert app.options_seen_at_load_handler["speculative_mode"] == "lookup"
    # Later load requests retain current runtime choices rather than re-reading
    # a stale preference file over options just applied in this session.
    app.engine.lossless_speed_options["enabled"] = True
    app._on_toggle_load_unload()
    assert app.options_seen_at_load_handler["enabled"] is True


def test_direct_load_handler_initializes_preferences_even_before_build(monkeypatch, tmp_path):
    (tmp_path / "prefs.json").write_text('{"lossless_speedups":{"metal_warmup":true}}')
    app, _ = make_app(monkeypatch, tmp_path, build=False)
    app._on_toggle_load_unload()
    assert app.options_seen_at_load_handler["metal_warmup"] is True
    assert app.build_calls == 0
    assert not app.engine.configure_calls


def test_button_is_visible_with_existing_hud_and_dialog_is_singleton(monkeypatch, tmp_path):
    app, module = make_app(monkeypatch, tmp_path)
    assert app.btn_lossless_speedups.cget("text") == "Speedups…"
    assert app.btn_lossless_speedups.parent is app.speed_tuning_bar
    assert app.btn_lossless_speedups.pack_options["before"] is app.lbl_speed_status
    original_build = module.SmartAIChatbotApp._build_single_top_bar
    gui.install_gui_lossless_speedups()
    assert module.SmartAIChatbotApp._build_single_top_bar is original_build
    app.btn_lossless_speedups.cget("command")()
    dialog = app._speedups_dialog
    app.btn_lossless_speedups.cget("command")()
    assert app._speedups_dialog is dialog and dialog.lifted
    assert set(app._speedups_variables) == set(gui.DEFAULT_OPTIONS)
    assert app._speedups_variables["persistent_cache"].get() is False
    assert app._speedups_variables["metal_warmup"].get() is False
    width, height, left, top = map(int, re.fullmatch(r"(\d+)x(\d+)\+(\d+)\+(\d+)", dialog.geometry_value).groups())
    assert height <= 668
    assert left + width <= dialog.winfo_screenwidth()
    assert top + height <= dialog.winfo_screenheight()
    assert not app.engine.configure_calls


def test_button_has_fallback_row_and_uses_theme_colors(monkeypatch, tmp_path):
    app, _ = make_app(monkeypatch, tmp_path, existing_bar=False)
    assert app.btn_lossless_speedups.parent is app._lossless_speedups_bar
    app._open_lossless_speedups()
    app.C.update(btn_bg="new-button", bg_hud="new-hud")
    assert app._apply_theme_colors() == "base-theme"
    assert app.btn_lossless_speedups.cget("bg") == "new-button"
    assert app._speedups_dialog.cget("bg") == "new-hud"


def test_dialog_saves_all_controls_and_preserves_other_preferences(monkeypatch, tmp_path):
    preferences = tmp_path / "prefs.json"
    preferences.write_text('{"speed_tuning":{"enabled":false},"memory_limit":{"limit_gb":12}}')
    app, _ = make_app(monkeypatch, tmp_path)
    app._open_lossless_speedups()
    values = app._speedups_variables
    for key, _ in gui._BOOLEAN_FIELDS:
        values[key].set(key in {"enabled", "branch_prefill", "persistent_cache", "metal_warmup"})
    values["speculative_mode"].set("dflash2")
    values["max_draft_tokens"].set("8")
    values["native_branch_count"].set("16")
    for key, _, _ in gui._PATH_FIELDS:
        values[key].set(" /local/" + key + " ")
    app.engine.result = {"status": "reload_required", "message": "Options saved. Reload the model to apply native drafting changes."}
    app._speedups_save_button.cget("command")()
    assert len(app.engine.configure_calls) == 1
    written = json.loads(preferences.read_text())
    assert written["memory_limit"]["limit_gb"] == 12
    assert written["speed_tuning"]["enabled"] is False
    options = written["lossless_speedups"]
    assert options == app.engine.configure_calls[0]
    assert options["speculative_mode"] == "dflash2"
    assert options["max_draft_tokens"] == 8
    assert options["native_branch_count"] == 16
    assert options["cache_dir"] == "/local/cache_dir"
    assert options["prompt_reuse"] is False
    assert "Reload" in app._speedups_dialog_status.cget("text")


def test_included_native_runtime_requires_explicit_selection_and_save(monkeypatch, tmp_path):
    bundled = {"speculative_server_path": "/app/runtime/native/llama-server",
               "speculative_runtime_receipt_path": "/app/runtime/native/llama-server.smartai.json"}
    monkeypatch.setattr(gui, "bundled_runtime_options", lambda: bundled)
    app, _ = make_app(monkeypatch, tmp_path)
    app._open_lossless_speedups()
    assert app._speedups_variables["speculative_server_path"].get() == ""
    app._speedups_bundled_native_button.cget("command")()
    assert app._speedups_variables["speculative_server_path"].get() == gui.INCLUDED_RUNTIME
    assert app._speedups_variables["speculative_runtime_receipt_path"].get() == ""
    assert not app.engine.configure_calls
    assert app.engine.lossless_speed_options["speculative_server_path"] is None
    assert app._speedups_variables["speculative_mode"].get() == "auto"
    assert "Bonsai 2.5 keeps its own runtime" in app._speedups_dialog_status.cget("text")
    app._speedups_save_button.cget("command")()
    assert app.engine.configure_calls[0]["speculative_server_path"] == gui.INCLUDED_RUNTIME
    assert app.engine.configure_calls[0]["speculative_runtime_receipt_path"] is None


def test_included_native_button_preserves_custom_server_or_receipt(monkeypatch, tmp_path):
    bundled = {"speculative_server_path": "/app/runtime/native/llama-server",
               "speculative_runtime_receipt_path": "/app/runtime/native/llama-server.smartai.json"}
    monkeypatch.setattr(gui, "bundled_runtime_options", lambda: bundled)
    app, _ = make_app(monkeypatch, tmp_path)
    app._open_lossless_speedups()
    app._speedups_variables["speculative_server_path"].set("/my/custom/server")
    app._speedups_bundled_native_button.cget("command")()
    assert app._speedups_variables["speculative_server_path"].get() == "/my/custom/server"
    assert app._speedups_variables["speculative_runtime_receipt_path"].get() == ""
    assert "custom native path" in app._speedups_dialog_status.cget("text")
    assert not app.engine.configure_calls


@pytest.mark.parametrize("busy", ["loading", "generation", "media", "tuning", "sleep", "learning", "evaluation", "closing"])
def test_open_dialog_rechecks_busy_state_at_save_without_writing(monkeypatch, tmp_path, busy):
    app, _ = make_app(monkeypatch, tmp_path)
    app._open_lossless_speedups()
    if busy == "loading":
        app._model_load_in_progress = True
    elif busy == "generation":
        app.is_generating = True
    elif busy == "media":
        app.multimodal = SimpleNamespace(busy=True)
    elif busy == "tuning":
        app.engine.speed_tuning_active = True
    elif busy == "sleep":
        app._sleep_consolidating = True
    elif busy == "learning":
        app.engine.awake_consolidator = SimpleNamespace(is_consolidating=True)
    elif busy == "evaluation":
        app._eval_proc = SimpleNamespace(poll=lambda: None)
    else:
        app._speed_close_requested = True
    app._speedups_save_button.cget("command")()
    assert not app.engine.configure_calls
    assert not (tmp_path / "prefs.json").exists()
    assert app._speedups_dialog_status.cget("text")


def test_backend_busy_result_is_never_persisted(monkeypatch, tmp_path):
    app, _ = make_app(monkeypatch, tmp_path)
    app._open_lossless_speedups()
    app.engine.result = {"status": "busy", "message": "Runtime lock is busy."}
    app._speedups_save_button.cget("command")()
    assert len(app.engine.configure_calls) == 1
    assert not (tmp_path / "prefs.json").exists()
    assert "Runtime lock" in app._speedups_dialog_status.cget("text")


@pytest.mark.parametrize("value", ["0", "65", "-1", "1.5", "1e1", "bad", "\u0661"])
def test_invalid_draft_count_keeps_engine_and_disk_unchanged(monkeypatch, tmp_path, value):
    app, _ = make_app(monkeypatch, tmp_path)
    app._open_lossless_speedups()
    app._speedups_variables["max_draft_tokens"].set(value)
    app._speedups_save_button.cget("command")()
    assert not app.engine.configure_calls
    assert not (tmp_path / "prefs.json").exists()
    assert "not saved" in app._speedups_dialog_status.cget("text")


def test_save_failure_is_honest_and_retry_can_persist(monkeypatch, tmp_path):
    app, _ = make_app(monkeypatch, tmp_path)
    app._open_lossless_speedups()
    real_save = gui.save_preferences
    def fail(*args):
        raise OSError("read-only folder")
    monkeypatch.setattr(gui, "save_preferences", fail)
    app._speedups_save_button.cget("command")()
    assert "applied for this session" in app._speedups_dialog_status.cget("text")
    assert "could not be saved" in app._speedups_dialog_status.cget("text")
    monkeypatch.setattr(gui, "save_preferences", real_save)
    app._speedups_save_button.cget("command")()
    assert (tmp_path / "prefs.json").exists()
    assert "Saved" in app._speedups_dialog_status.cget("text")


def test_close_destroys_dialog_and_preserves_application_close(monkeypatch, tmp_path):
    app, _ = make_app(monkeypatch, tmp_path)
    app._open_lossless_speedups()
    dialog = app._speedups_dialog
    app._on_close_app()
    assert dialog.destroyed
    assert app._speedups_dialog is None
    assert app.closed


@pytest.mark.parametrize("value", ["", "0", "65", "-1", "1.5", "bad"])
def test_invalid_native_branch_count_does_not_change_runtime(monkeypatch, tmp_path, value):
    app, _ = make_app(monkeypatch, tmp_path)
    app._open_lossless_speedups()
    app._speedups_variables["native_branch_count"].set(value)
    app._speedups_save_button.cget("command")()
    assert not app.engine.configure_calls
    assert not (tmp_path / "prefs.json").exists()
    assert "Native branches" in app._speedups_dialog_status.cget("text")
