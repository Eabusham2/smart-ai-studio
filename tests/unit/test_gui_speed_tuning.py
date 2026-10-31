"""Preference and UI concurrency checks without importing neural runtimes."""
import ast
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core import gui_speed_tuning as gui


def test_speed_status_shows_measured_test_gain_and_decode_rate():
    result = {
        "status": "improved", "message": "Selected execution settings", "gain_pct": 12.5,
        "selected": {"decode_tokens_per_second": 47.125}, "cached": True,
    }
    assert gui.format_speed_status(result) == "Saved profile · Synthetic speed test: +12.5% · 47.1 tokens/s decode."
    assert gui.format_speed_status({
        "status": "baseline", "baseline": {"decode_tokens_per_second": 0.04},
    }) == "Baseline retained after speed test · 0.04 tokens/s decode."


@pytest.mark.parametrize("bad", [None, True, "fast", float("nan"), float("inf"), -1, 10**400, {}, []])
def test_speed_status_ignores_invalid_metrics(bad):
    result = {
        "status": "improved", "message": "Working settings retained.",
        "gain_pct": bad, "selected": {"decode_tokens_per_second": bad},
    }
    assert gui.format_speed_status(result) == "Working settings retained."
    result["gain_pct"] = 5.0
    assert gui.format_speed_status(result) == "Synthetic speed test: +5.0%."


def test_speed_status_keeps_unsupported_and_cancel_details():
    for status in ("unsupported", "cancelled", "memory_pressure", "error"):
        assert gui.format_speed_status({"status": status, "message": "Specific runtime detail"}) == "Specific runtime detail"
    assert gui.format_speed_status({"status": "baseline", "selected": None, "baseline": []}) == "baseline"
    assert gui.format_speed_status(None) == "Speed tuning is ready."


def test_speed_preference_merges_existing_settings(tmp_path):
    target = tmp_path / "prefs.json"
    original = {
        "memory_limit": {"enabled": True, "limit_gb": 12.5},
        "custom_setting": ["keep"],
        "speed_tuning": {"future": "keep"},
    }
    target.write_text(json.dumps(original))
    gui.save_auto_speed_preference(str(target), False)
    saved = json.loads(target.read_text())
    assert saved["memory_limit"] == original["memory_limit"]
    assert saved["custom_setting"] == ["keep"]
    assert saved["speed_tuning"] == {"enabled": False, "future": "keep"}
    assert gui.load_auto_speed_preference(str(target), default=True) is False


def test_bad_preference_is_not_overwritten_and_strings_are_not_bools(tmp_path):
    target = tmp_path / "prefs.json"
    target.write_text('{"speed_tuning":{"enabled":"false"}}')
    assert gui.load_auto_speed_preference(str(target), default=False) is False
    target.write_text("partial old preferences")
    assert gui.load_auto_speed_preference(str(target), default=True) is True
    with pytest.raises(ValueError):
        gui.save_auto_speed_preference(str(target), True)
    assert target.read_text() == "partial old preferences"


@pytest.mark.parametrize("case", ["chat", "media", "eval", "sleep", "awake", "tune", "load"])
def test_busy_work_is_rejected(case):
    app = SimpleNamespace(engine=SimpleNamespace())
    if case == "chat":
        app.is_generating = True
    elif case == "media":
        app.multimodal = SimpleNamespace(busy=True)
    elif case == "eval":
        app._eval_proc = SimpleNamespace(poll=lambda: None)
    elif case == "sleep":
        app._sleep_consolidating = True
    elif case == "awake":
        app.engine.awake_consolidator = SimpleNamespace(is_consolidating=True)
    elif case == "load":
        app._model_load_in_progress = True
    else:
        app.engine.speed_tuning_active = True
    assert gui.speed_busy_reason(app)


class Widget:
    def __init__(self, master=None, **kwargs):
        self.master = master
        self.options = {"state": "normal", **kwargs}
        self.bindings = {}

    def configure(self, **kwargs):
        self.options.update(kwargs)

    def cget(self, key):
        return self.options[key]

    def pack(self, **_kwargs):
        pass

    def bind(self, event, callback):
        self.bindings[event] = callback


class Variable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Root:
    def __init__(self):
        self.callbacks = {}
        self.next_id = 0

    def after(self, _delay, callback):
        self.next_id += 1
        self.callbacks[self.next_id] = callback
        return self.next_id

    def after_cancel(self, ident):
        self.callbacks.pop(ident, None)

    def winfo_exists(self):
        return True

    def tick(self):
        ident = min(self.callbacks)
        self.callbacks.pop(ident)()


def make_app(monkeypatch, tmp_path):
    pending = []

    class Thread:
        def __init__(self, target, **_kwargs):
            self.target = target

        def start(self):
            pending.append(self.target)

    class Engine:
        auto_speed_enabled = True
        speed_tuning_active = False
        speed_status = {"message": "Ready"}

        def configure_auto_speed(self, *, enabled, force, progress, cancelled):
            self.last_request = (enabled, force)
            progress("Checking native speed")
            self.speed_status = {
                "status": "cancelled" if cancelled() else "improved",
                "message": "Cancelled safely" if cancelled() else "Measured improvement",
            }
            self.auto_speed_enabled = enabled
            return self.speed_status

        def cancel_speed_tuning(self):
            self.cancel_requested = True

    class App:
        def __init__(self):
            self.root = Root()
            self.main_container = Widget()
            self.C = {key: "black" for key in (
                "bg_hud", "text_main", "btn_bg", "btn_fg", "btn_hover", "text_muted",
            )}
            self.engine = Engine()
            self.models_config = {"model": {"model_type": "text"}}
            self.active_tab_id = "model"
            self.is_model_loaded = True
            self.is_generating = False
            self._memory_limit_enabled = True
            self._memory_limit_gb = 9.0
            self.sent = 0
            self.queued_calls = []

        def _build_single_top_bar(self):
            self.btn_send = Widget(state="normal")
            self.btn_reset_reinstall = Widget(state="disabled")
            self.btn_load_unload = Widget(state="normal")

        def _apply_theme_colors(self):
            return "theme"

        def _on_close_app(self):
            self.closed = True

        def _on_send_message(self):
            self.sent += 1

        def _run_queued_task(self, *args, **kwargs):
            self.queued_calls.append((args, kwargs))

    module = SimpleNamespace(
        SmartAIChatbotApp=App,
        tk=SimpleNamespace(Frame=Widget, Label=Widget, Button=Widget, Checkbutton=Widget, BooleanVar=Variable),
        APP_PREFS_FILE=str(tmp_path / "prefs.json"), _FONT_TINY="small", _FONT_TINY_BOLD="bold",
    )
    monkeypatch.setitem(sys.modules, "app_gui", module)
    monkeypatch.setattr(gui.threading, "Thread", Thread)
    gui.install_gui_speed_tuning()
    app = App()
    app._build_single_top_bar()
    return app, pending, module


def test_cancel_keeps_ui_responsive_and_restores_disabled_states(monkeypatch, tmp_path):
    app, pending, _module = make_app(monkeypatch, tmp_path)
    assert app.engine.speed_memory_limit_gb == 9.0
    app.btn_speed_tune.cget("command")()
    assert app._speed_ui_running
    assert app.btn_send.cget("state") == "disabled"
    app._on_send_message()  # Keyboard/direct-handler paths are guarded too.
    assert app.sent == 0
    app.btn_speed_cancel.cget("command")()
    assert app.engine.cancel_requested
    pending.pop()()
    app.root.tick()
    assert app._speed_ui_running is False
    assert app.btn_send.cget("state") == "normal"
    assert app.btn_reset_reinstall.cget("state") == "disabled"
    assert app.lbl_speed_status.cget("text") == "Cancelled safely"
    app._on_send_message()
    assert app.sent == 1


def test_saved_off_is_loaded_before_model_load_and_memory_cap_can_be_off(monkeypatch, tmp_path):
    (tmp_path / "prefs.json").write_text('{"speed_tuning":{"enabled":false}}')
    app, _pending, _module = make_app(monkeypatch, tmp_path)
    assert app.auto_speed_var.get() is False
    assert app.engine.auto_speed_enabled is False
    app._memory_limit_enabled = False
    app.root.tick()
    assert app.engine.speed_memory_limit_gb is None
    assert app.lbl_speed_status.cget("text") == "Auto speed is off."


def test_ui_rejects_tuning_during_eval(monkeypatch, tmp_path):
    app, pending, _module = make_app(monkeypatch, tmp_path)
    app._eval_proc = SimpleNamespace(poll=lambda: None)
    app.btn_speed_tune.cget("command")()
    assert not pending
    assert "evaluation" in app.lbl_speed_status.cget("text")


def test_queued_learning_dispatch_is_preserved_across_speed_test(monkeypatch, tmp_path):
    app, pending, _module = make_app(monkeypatch, tmp_path)
    app.btn_speed_tune.cget("command")()
    # The main chat queue has already removed this item before scheduling its
    # dispatch. Starting Retune in that interval must not consume the item.
    args = ("/learn source", "learn source", "/local/document.pdf", "")
    app._run_queued_task(*args)
    assert app.queued_calls == []
    pending.pop()()
    app.root.tick()  # Complete the test on the Tk poll.
    app.root.tick()  # Resume the exact queued request once.
    app.root.tick()
    assert app.queued_calls == [(args, {})]


def test_close_does_not_resume_deferred_chat_or_start_another_test(monkeypatch, tmp_path):
    app, pending, _module = make_app(monkeypatch, tmp_path)
    app.btn_speed_tune.cget("command")()
    app._run_queued_task("queued", "queued", None, None)
    app._on_close_app()
    pending.pop()()
    app.root.tick()
    app.root.tick()
    app.btn_speed_tune.cget("command")()
    app._on_send_message()
    assert not pending
    assert app.sent == 0
    assert app.queued_calls == []


@pytest.mark.parametrize("busy", ["load", "manual_tune", "automatic_tune"])
def test_eval_direct_start_cannot_race_model_loading_or_tuning(busy):
    source = Path(__file__).parents[2] / "core" / "gui_eval_panel.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    install = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "install_gui_eval_panel")
    method = next(node for node in install.body if isinstance(node, ast.FunctionDef) and node.name == "_start_eval")
    active_info = Mock(side_effect=AssertionError("Eval must not inspect or load another runtime while busy"))
    messagebox = SimpleNamespace(showinfo=Mock())
    namespace = {"_eval_alive": lambda app: False, "_active_text_info": active_info, "messagebox": messagebox}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    app = SimpleNamespace(
        _model_load_in_progress=busy == "load", _speed_ui_running=busy == "manual_tune",
        engine=SimpleNamespace(speed_tuning_active=busy == "automatic_tune"),
    )
    namespace["_start_eval"](app)
    active_info.assert_not_called()
    messagebox.showinfo.assert_called_once()


@pytest.mark.parametrize("handler", ["_start_eval", "_open_eval_window"])
def test_strata_training_eval_is_rejected_before_unloading_or_starting_a_process(handler):
    source = Path(__file__).parents[2] / "core" / "gui_eval_panel.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    install = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "install_gui_eval_panel")
    method = next(node for node in install.body if isinstance(node, ast.FunctionDef) and node.name == handler)
    messagebox = SimpleNamespace(showinfo=Mock())
    namespace = {
        "_eval_alive": lambda app: False,
        "_active_text_info": lambda app: {"controller_runtime": "strata", "model_type": "text"},
        "messagebox": messagebox,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    app = SimpleNamespace(engine=SimpleNamespace(unload_model=Mock()))
    namespace[handler](app)
    app.engine.unload_model.assert_not_called()
    messagebox.showinfo.assert_called_once()
    assert messagebox.showinfo.call_args.args[0] == "Inference Only"


def test_cancel_is_available_during_strata_setup_before_auto_speed_starts(monkeypatch, tmp_path):
    import threading

    app, _pending, _module = make_app(monkeypatch, tmp_path)
    app._model_load_in_progress = True
    app._model_load_cancel_event = threading.Event()
    app.engine.speed_tuning_active = False
    app.root.tick()
    assert app.btn_speed_cancel.cget("state") == "normal"
    app.btn_speed_cancel.cget("command")()
    assert app._model_load_cancel_event.is_set()
    assert app.engine.cancel_requested


@pytest.mark.parametrize("cleanup_succeeds", [True, False])
def test_close_uses_owned_async_unload_and_keeps_retry_available_on_failure(monkeypatch, tmp_path, cleanup_succeeds):
    app, _pending, _module = make_app(monkeypatch, tmp_path)
    app.engine.controller_backend = SimpleNamespace(runtime="strata", _manager=object())
    attempts = []

    def unload():
        attempts.append(True)
        app._model_load_in_progress = True

    app._unload_strata_async = unload
    app._on_close_app()
    assert attempts == [True]
    assert not getattr(app, "closed", False)
    app._model_load_in_progress = False
    app._strata_unload_pending = not cleanup_succeeds
    if cleanup_succeeds:
        app.engine.controller_backend = None
    app.root.tick()  # Ordinary status poll.
    app.root.tick()  # Deferred close after worker completion.
    assert attempts == [True]
    assert bool(getattr(app, "closed", False)) is cleanup_succeeds
    if not cleanup_succeeds:
        assert not app._speed_close_requested
        assert not app._speed_ui_closed
        assert "Retry unload" in app.lbl_speed_status.cget("text")


def test_automatic_load_restores_controls_using_loaded_state(monkeypatch, tmp_path):
    app, _pending, _module = make_app(monkeypatch, tmp_path)
    app.is_model_loaded = False
    app.btn_send.configure(state="disabled")
    app._model_load_in_progress = True
    app.engine.speed_tuning_active = True
    app.engine.speed_status = {"status": "tuning", "message": "Measuring native settings"}
    app.root.tick()
    assert app.btn_load_unload.cget("state") == "disabled"
    assert app.btn_speed_cancel.cget("state") == "normal"
    app.btn_speed_cancel.cget("command")()
    assert app.engine.cancel_requested
    app.engine.speed_tuning_active = False
    app._model_load_in_progress = False
    app.is_model_loaded = True
    app._update_input_lock_state = lambda: app.btn_send.configure(state="normal")
    app.root.tick()
    assert app.btn_send.cget("state") == "normal"
    assert app.btn_load_unload.cget("state") == "normal"


def test_close_during_automatic_tuning_requests_cancel_without_blocking(monkeypatch, tmp_path):
    app, _pending, _module = make_app(monkeypatch, tmp_path)
    app.engine.speed_tuning_active = True
    app._model_load_in_progress = True
    app._on_close_app()
    assert app.engine.cancel_requested
    assert not getattr(app, "closed", False)
    app.engine.speed_tuning_active = False
    app._model_load_in_progress = False
    # Both scheduled operations execute on Tk: the normal status poll, then
    # deferred close after the load worker has yielded its model lock.
    app.root.tick()
    app.root.tick()
    assert app.closed
    assert app._speed_ui_closed


@pytest.mark.parametrize("cached,local", [(True, False), (False, True), (False, False)])
def test_text_tab_switch_uses_async_loader_without_implicit_download(cached, local, tmp_path):
    # Execute the real switch handler independently of Tk/display and GPU
    # imports. A synchronous neural load here would freeze cancellation.
    source = Path(__file__).parents[2] / "app_gui.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SmartAIChatbotApp")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_on_switch_model_tab")
    namespace = {
        "os": os, "is_model_cached_locally": lambda _repo: cached,
        "format_parameter_count": lambda value: str(value),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    local_path = tmp_path / "model.gguf"
    if local:
        local_path.write_bytes(b"test")
    app = SimpleNamespace(
        multimodal=SimpleNamespace(busy=False),
        models_config={"m": {
            "name": "model", "short_name": "m", "model_type": "text", "repo_id": "hf/model",
            "model_path": str(local_path) if local else None,
        }},
        active_tab_id="old", chat_history={"m": []}, chat_stream=Mock(),
        engine=SimpleNamespace(unload_model=Mock(), load_model=Mock()),
        audio_engine=SimpleNamespace(unload_model=Mock()),
        media_engine=SimpleNamespace(unload_model=Mock()),
        is_model_loaded=True, synapses_learned_count=0,
        _model_is_cached=lambda _info: cached,
        lbl_model_status=Mock(), lbl_params=Mock(),
        C={"accent_cyan": "cyan", "accent_yellow": "yellow"},
    )
    for name in (
        "_send_welcome_messages", "_scroll_chat_to_bottom", "_sync_memory_watchdog",
        "_update_memory_hud_badge", "_update_model_action_buttons",
        "_update_resource_view_metrics", "_update_input_lock_state", "_on_toggle_load_unload",
    ):
        setattr(app, name, Mock())
    namespace["_on_switch_model_tab"](app, "m")
    app.engine.load_model.assert_not_called()
    if cached or local:
        app._on_toggle_load_unload.assert_called_once_with(_skip_download=True)
    else:
        app._on_toggle_load_unload.assert_not_called()
        assert app.is_model_loaded is False
