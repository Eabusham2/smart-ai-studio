"""API dialog, serving lease and desktop watchdog lifecycle without model weights."""
import queue
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.local_api_ui import LocalAPIUI, api_serving, install_local_api_ui


class FakeServer:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.is_running = False
        self.active_requests = 0
        self.base_url = "http://127.0.0.1:1234/v1"
    def start(self):
        self.is_running = True
    def stop(self, timeout=2):
        self.is_running = False
        return not self.active_requests


def app_fixture():
    app = SimpleNamespace(engine=SimpleNamespace(), models_config={"model_1": {"name": "Bonsai"}},
                          active_tab_id="model_1", root=SimpleNamespace(after=Mock()),
                          _event_queue=queue.Queue(), btn_local_api=Mock())
    adapter = SimpleNamespace(model_info=Mock(return_value={"id": "smart-ai-studio", "name": "Bonsai"}),
                              generate=Mock(), count_tokens=Mock())
    ui = LocalAPIUI(app, server_factory=FakeServer, adapter_factory=lambda *a, **k: adapter)
    app._local_api_ui = ui
    return app, ui, adapter


def finish_worker(app, ui):
    ui._worker.join(timeout=3)
    assert not ui._worker.is_alive()
    while not app._event_queue.empty():
        kind, callback = app._event_queue.get_nowait()
        assert kind == "ui_callback"
        callback()


def test_start_reserves_loaded_model_stop_keeps_it_loaded():
    app, ui, _ = app_fixture()
    assert ui.start(port="1234")
    assert api_serving(app)
    assert ui.server.kwargs["host"] == "127.0.0.1"
    assert not ui.start()
    ui.stop()
    finish_worker(app, ui)
    assert not api_serving(app)
    assert ui.server is None
    assert "still loaded" in ui._status


def test_unloaded_busy_and_queued_desktop_work_cannot_start_server():
    app, ui, adapter = app_fixture()
    adapter.model_info.return_value = None
    assert not ui.start()
    adapter.model_info.return_value = {"name": "Bonsai"}
    for flag in ("is_generating", "_model_load_in_progress", "_sleep_consolidating", "_speed_ui_running"):
        setattr(app, flag, True)
        assert not ui.start()
        setattr(app, flag, False)
    app.prompt_queue = ["pending"]
    assert not ui.start()
    assert ui.server is None


def test_close_waits_until_generator_has_released_model():
    app, ui, _ = app_fixture()
    ui.start()
    server = ui.server
    server.active_requests = 1
    assert not ui.shutdown_ready()
    finish_worker(app, ui)
    assert api_serving(app)
    assert ui.server is server
    server.active_requests = 0
    ui._finish_stop(server, True)
    assert ui.shutdown_ready()


def test_keyboard_model_switch_and_shutdown_handlers_respect_serving_lease():
    class App:
        def __init__(self):
            self.root = SimpleNamespace(after=Mock())
            self.calls = []
            self._local_api_ui = SimpleNamespace(reserves_model=True, open=Mock(), _render=Mock(),
                                                  shutdown_ready=Mock(return_value=False))
        def _on_send_message(self):
            self.calls.append("send")
        def _on_switch_model_tab(self, tab):
            self.calls.append(tab)
        def _on_close_app(self):
            self.calls.append("close")
    module = SimpleNamespace(SmartAIChatbotApp=App, tk=object())
    install_local_api_ui(module)
    app = App()
    app._on_send_message()
    app._on_switch_model_tab("model_2")
    app._on_close_app()
    assert app.calls == []
    assert app._local_api_ui.open.call_count == 2
    assert app.root.after.called
    app._local_api_ui.reserves_model = False
    app._local_api_ui.shutdown_ready.return_value = True
    app._on_send_message()
    app._on_switch_model_tab("model_2")
    app._on_close_app()
    assert app.calls == ["send", "model_2", "close"]


def test_serving_blocks_speed_tuning_and_app_updates():
    from core.app_update_ui import AppUpdateUI
    from core.gui_speed_tuning import speed_busy_reason
    app, ui, _ = app_fixture()
    ui.start()
    assert "API server" in speed_busy_reason(app)
    updater = AppUpdateUI.__new__(AppUpdateUI)
    updater.app = app
    assert updater._application_busy()


def add_desktop_watchdog(app, *, running=True):
    from app_gui import SmartAIChatbotApp
    app.is_model_loaded = True
    app._memory_limit_enabled = running
    app._memory_limit_gb = 7.5
    app._memory_watchdog_epoch = 1
    app._owned_runtime_pids = Mock(return_value=[1234])
    app._save_memory_limit_preferences = Mock()
    watchdog = app.watchdog = SimpleNamespace(_running=running, max_process_ram_gb=7.5)
    watchdog.start_monitoring = Mock(side_effect=lambda: setattr(watchdog, "_running", True))
    watchdog.stop_monitoring = Mock(side_effect=lambda: setattr(watchdog, "_running", False))
    app._sync_memory_watchdog = SmartAIChatbotApp._sync_memory_watchdog.__get__(app)
    return watchdog


@pytest.mark.parametrize("desktop_enabled", [True, False])
def test_api_defaults_and_desktop_watchdog_are_independent(desktop_enabled):
    app, ui, adapter = app_fixture()
    watchdog = add_desktop_watchdog(app, running=desktop_enabled)
    ui.adapter_factory = Mock(return_value=adapter)
    assert ui.start()
    args = ui.adapter_factory.call_args.kwargs
    assert args["memory_limit_defaults"] == {"enabled": False, "limit_gb": 14.5}
    assert args["owned_processes_provider"] is app._owned_runtime_pids
    assert not watchdog._running
    app._sync_memory_watchdog()
    assert not watchdog._running
    server = ui.server
    server.active_requests = 1
    ui.stop()
    finish_worker(app, ui)
    assert not watchdog._running and api_serving(app)
    server.active_requests = 0
    ui._finish_stop(server, True)
    assert watchdog._running is desktop_enabled
    assert app._memory_limit_gb == watchdog.max_process_ram_gb == 7.5
    assert app._memory_limit_enabled is desktop_enabled
    app._save_memory_limit_preferences.assert_not_called()


def test_failed_listener_bind_restores_desktop_watchdog():
    from core.local_api import LocalModelAPIServer
    app, ui, _ = app_fixture()
    watchdog = add_desktop_watchdog(app)
    ui.server_factory = LocalModelAPIServer
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        assert not ui.start(host="127.0.0.1", port=occupied.getsockname()[1])
    assert not api_serving(app) and watchdog._running
    watchdog.stop_monitoring.assert_called_once()
    watchdog.start_monitoring.assert_called_once()
    app._save_memory_limit_preferences.assert_not_called()


def test_stop_error_keeps_desktop_watchdog_paused_until_retry():
    app, ui, _ = app_fixture()
    watchdog = add_desktop_watchdog(app)
    assert ui.start()
    server = ui.server
    server.is_running = False
    server.stop = Mock(side_effect=[RuntimeError("pending cleanup"), True])
    ui.stop()
    finish_worker(app, ui)
    assert api_serving(app) and not watchdog._running and not ui.start()
    ui.stop()
    finish_worker(app, ui)
    assert not api_serving(app) and watchdog._running


def test_queued_desktop_pressure_cannot_stop_or_unload_api_model():
    class App:
        _on_close_app = Mock()
        def _on_memory_pressure_emergency(self, status):
            self.unload(status)
    install_local_api_ui(SimpleNamespace(SmartAIChatbotApp=App, tk=object()))
    app = App()
    app.unload = Mock()
    app._local_api_ui = SimpleNamespace(reserves_model=True)
    status = {"process_rss_gb": 9.0}
    app._on_memory_pressure_emergency(status)
    app.unload.assert_not_called()
    app._local_api_ui.reserves_model = False
    app._on_memory_pressure_emergency(status)
    app.unload.assert_called_once_with(status)


@pytest.mark.parametrize("invalid", ["", "word", "0", "-2", "0.49", "nan", "inf", True])
def test_invalid_memory_limit_keeps_desktop_and_listener_untouched(invalid):
    app, ui, _ = app_fixture()
    watchdog = add_desktop_watchdog(app)
    ui.server_factory = Mock()
    assert not ui.start(memory_limit_enabled=True, memory_limit_gb=invalid)
    assert "0.5 GiB" in ui._status and not api_serving(app)
    ui.server_factory.assert_not_called()
    watchdog.stop_monitoring.assert_not_called()


def test_dialog_keeps_memory_controls_across_reopen_and_disables_them_while_serving():
    import tkinter as tk
    app, ui, adapter = app_fixture()
    try:
        app.root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"Tk display unavailable: {error}")
    try:
        app.root.withdraw()
        ui.adapter_factory = Mock(return_value=adapter)
        ui.open()
        assert not ui._memory_enabled_var.get() and ui._memory_gb_var.get() == "14.5"
        assert ui._memory_entry.instate(["disabled"])
        ui._memory_toggle.invoke()
        assert ui._memory_entry.instate(["!disabled"])
        ui._memory_gb_var.set("6.25")
        ui._dialog.destroy()
        ui.open()
        assert ui._memory_enabled_var.get() and ui._memory_gb_var.get() == "6.25"
        ui._start_button.invoke()
        assert ui.adapter_factory.call_args.kwargs["memory_limit_defaults"] == {"enabled": True, "limit_gb": 6.25}
        assert ui._memory_toggle.instate(["disabled"]) and ui._memory_entry.instate(["disabled"])
        ui._dialog.destroy()
        ui.open()
        assert ui._memory_toggle.instate(["disabled"]) and ui._memory_entry.instate(["disabled"])
        ui.stop()
        finish_worker(app, ui)
        assert ui._memory_toggle.instate(["!disabled"]) and ui._memory_entry.instate(["!disabled"])
        assert ui._memory_enabled_var.get() and ui._memory_gb_var.get() == "6.25"
    finally:
        app.root.destroy()
