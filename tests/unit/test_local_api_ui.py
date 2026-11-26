"""Serving lease and shutdown behavior without a model or display."""
import queue
from types import SimpleNamespace
from unittest.mock import Mock

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
