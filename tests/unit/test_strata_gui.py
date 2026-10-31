"""Exercise the real GUI methods with queued workers and lightweight widgets."""
from __future__ import annotations

import ast
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import re
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


class Widget:
    def __init__(self, text=""):
        self.options = {"state": "normal", "text": text}
        self.text = text
        self.marks = set()

    def configure(self, **kwargs):
        self.options.update(kwargs)

    def cget(self, name):
        return self.options.get(name)

    def get(self, *_args):
        return self.text

    def insert(self, _position, text, *_tags):
        self.text += text

    def delete(self, *_args):
        self.text = ""

    def mark_set(self, name, _position):
        self.marks.add(name)

    def mark_names(self):
        return tuple(self.marks)

    def mark_gravity(self, *_args):
        pass

    def focus_set(self):
        pass


class Root:
    def __init__(self):
        self.callbacks = []

    def after(self, _delay, callback):
        self.callbacks.append(callback)

    def drain(self):
        while self.callbacks:
            self.callbacks.pop(0)()

    def winfo_viewable(self):
        return False


@pytest.fixture
def app():
    """Compile real methods without importing the desktop's neural dependencies."""
    pending = []

    class Thread:
        def __init__(self, target, args=(), **_kwargs):
            self.target, self.args = target, args

        def start(self):
            pending.append(lambda: self.target(*self.args))

    source = Path(__file__).resolve().parents[2] / "app_gui.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    app_class = next(node for node in parsed.body if isinstance(node, ast.ClassDef)
                     and node.name == "SmartAIChatbotApp")
    methods = {
        "_model_is_cached", "_update_model_action_buttons", "_update_input_lock_state",
        "_on_toggle_load_unload", "_unload_strata_async", "_on_switch_model_tab",
        "_on_memory_pressure_emergency", "_on_send_message", "_run_queued_task",
        "_process_message_thread", "_update_telemetry",
    }
    nodes = [node for node in app_class.body if isinstance(node, ast.FunctionDef)
             and node.name in methods]
    namespace = {
        "os": os, "math": math, "json": json, "time": time, "re": re,
        "datetime": datetime,
        "threading": SimpleNamespace(Thread=Thread, Event=threading.Event),
        "format_parameter_count": str, "format_added_synapses": str,
        "is_model_cached_locally": Mock(return_value=False),
        "SystemMemoryWatchdog": SimpleNamespace(reclaim_process_memory=Mock()),
    }
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    harness = type("GuiHarness", (), {name: namespace[name] for name in methods})
    obj = harness()
    obj.pending = pending
    obj.root = Root()
    obj._event_queue = queue.Queue()
    obj.cancel_event = threading.Event()
    obj.models_config = {
        "strata": {"name": "Strata Q2", "short_name": "Q2", "model_key": "q2_0",
                   "controller_runtime": "strata", "model_type": "text", "inference_only": True,
                   "storage_note": "Large download", "model_path": "strata://q2_0"},
        "native": {"name": "Native", "short_name": "Native", "repo_id": "example/not-cached",
                   "model_type": "text"},
    }
    obj.active_tab_id = "strata"
    obj.is_model_loaded = True
    obj.is_generating = False
    obj._memory_limit_enabled = False
    obj._memory_limit_gb = 8.0
    obj.C = {name: name for name in ("accent_yellow", "accent_red", "accent_green", "accent_cyan", "text_main")}
    obj.chat_history = {"strata": []}
    obj.total_tokens_used = 777
    obj.max_context_window = 32768
    obj.synapses_learned_count = 0
    obj._placeholder_active = False
    obj.attached_file_path = None
    obj.prompt_queue = []
    for name in ("txt_input", "btn_send", "chat_stream", "lbl_context", "lbl_tps", "lbl_params",
                 "lbl_model_status", "btn_load_unload", "btn_reset_reinstall"):
        setattr(obj, name, Widget())
    obj.lbl_tps.configure(text="⚡ 999.9 tok/s")
    backend = SimpleNamespace(runtime="strata", is_loaded=True, cancel_current_request=Mock(),
                              last_usage={}, last_generation={}, n_ctx=lambda: 32768, _manager=object())
    obj.backend = backend
    obj.engine = SimpleNamespace(controller_backend=backend, _last_stream_pro_meta=None,
                                 solve=Mock(side_effect=AssertionError("unexpected second inference")))

    def unload():
        obj.engine.controller_backend = None
        return {"status": "unloaded", "unloaded": True}

    obj.engine.unload_model = Mock(side_effect=unload)
    obj.engine.load_model = Mock(return_value={"status": "loaded"})
    obj.audio_engine = SimpleNamespace(unload_model=Mock())
    obj.media_engine = SimpleNamespace(unload_model=Mock())
    obj.multimodal = SimpleNamespace(busy=False, handle_command=Mock(return_value=None),
                                     stream_solve=Mock(return_value=iter(())))
    obj.tools = SimpleNamespace(execute_tool=Mock(side_effect=AssertionError("unexpected tool")))
    obj.db = SimpleNamespace(log_interaction=Mock())
    for name in ("_sync_memory_watchdog", "_update_memory_hud_badge", "_update_resource_view_metrics",
                 "_scroll_chat_to_bottom", "_send_welcome_messages", "_on_download_hf_model",
                 "_on_reset_and_reinstall_single_confirm", "_show_strata_model_info",
                 "_append_ai_message", "_append_user_message", "_recompute_input_height",
                 "_set_generating_state", "_check_and_run_next_queue", "_prepare_memory_limit_for_model"):
        setattr(obj, name, Mock())
    obj._show_model_loading_popup = Mock(return_value=(None, Mock()))
    obj._render_styled_markdown = lambda text: obj.chat_stream.insert("end", text)
    obj._model_is_cached = lambda info: info.get("controller_runtime") == "strata"
    return obj


def finish_worker(app):
    app.pending.pop(0)()
    kind, callback = app._event_queue.get_nowait()
    assert kind == "model_load_complete"
    callback()


def test_cache_hint_on_unsupported_platform_is_false(app, monkeypatch):
    from core import strata_runtime

    def unsupported(*_args):
        raise strata_runtime.StrataRuntimeError("No engine for this platform")

    monkeypatch.setattr(strata_runtime, "StrataRuntime", unsupported)
    assert type(app)._model_is_cached(app, app.models_config["strata"]) is False


def test_unload_cancels_transport_then_waits_for_worker_and_ui_completion(app):
    done = Mock()
    assert app._unload_strata_async(done) is True
    app.backend.cancel_current_request.assert_called_once()
    assert app.cancel_event.is_set()
    assert app._model_load_in_progress and app._model_unload_in_progress
    app.engine.unload_model.assert_not_called()
    assert app.is_model_loaded
    app.pending.pop(0)()
    assert app.is_model_loaded
    kind, callback = app._event_queue.get_nowait()
    assert kind == "model_load_complete"
    callback()
    assert not app.is_model_loaded and not app._strata_unload_pending
    assert not app._model_load_in_progress
    assert app.btn_send.cget("state") == "disabled"
    done.assert_called_once()


def test_failed_unload_keeps_retry_without_switching_and_retry_works(app):
    app.engine.unload_model.side_effect = None
    app.engine.unload_model.return_value = {"status": "error", "unloaded": False, "error": "still stopping"}
    app._on_switch_model_tab("native")
    assert app.active_tab_id == "strata"
    finish_worker(app)
    assert app.active_tab_id == "strata"
    assert app._strata_unload_pending and not app.is_model_loaded
    assert app.btn_load_unload.cget("text") == "Retry unload"
    assert app.btn_send.cget("state") == "disabled"
    app.engine.unload_model.return_value = {"status": "unloaded", "unloaded": True}
    app._on_toggle_load_unload()
    assert app.pending
    finish_worker(app)
    assert not app._strata_unload_pending
    app.engine.load_model.assert_not_called()


def test_switch_waits_for_owned_stop_and_does_not_download_uncached_target(app):
    app._on_switch_model_tab("native")
    assert app.active_tab_id == "strata"
    app.engine.unload_model.assert_not_called()
    finish_worker(app)
    assert app.active_tab_id == "native"
    assert not app.is_model_loaded
    app.engine.load_model.assert_not_called()
    app._on_download_hf_model.assert_not_called()


def test_unload_does_not_stop_replacement_backend(app):
    app._unload_strata_async()
    replacement = SimpleNamespace(runtime="other")
    app.engine.controller_backend = replacement
    finish_worker(app)
    app.engine.unload_model.assert_not_called()
    assert app.engine.controller_backend is replacement


@pytest.mark.parametrize("enabled,rss,expected", [(True, 8.0, True), (True, 7.9, False), (False, 9.0, False)])
def test_cap_breach_cancels_before_async_unload_even_during_generation(app, enabled, rss, expected):
    app._memory_limit_enabled = enabled
    app.is_generating = True
    app._on_memory_pressure_emergency({"process_rss_gb": rss, "used_percent": 70.0, "free_gb": 4.0})
    assert bool(app.pending) is expected
    assert app.cancel_event.is_set() is expected
    assert app.backend.cancel_current_request.called is expected
    app.engine.unload_model.assert_not_called()
    assert app._memory_limit_gb == 8.0


@pytest.mark.parametrize("retained_manager", [True, False])
def test_failed_load_cleanup_retains_only_needed_retry_action(app, retained_manager):
    app.is_model_loaded = False
    app.backend._manager = object() if retained_manager else None
    app.engine.load_model.return_value = {"status": "error", "error": "startup failed"}
    app._on_toggle_load_unload()
    app.engine.load_model.assert_not_called()
    assert app._model_load_in_progress
    finish_worker(app)
    assert app._strata_unload_pending is retained_manager
    assert not app.is_model_loaded
    assert app.btn_send.cget("state") == "disabled"
    assert (app.btn_load_unload.cget("text") == "Retry unload") is retained_manager


@pytest.mark.parametrize("cancel_when", ["never", "before", "during"])
def test_empty_or_cancelled_stream_does_not_retry_fabricate_or_save_answer(app, cancel_when):
    def stream(*_args, **_kwargs):
        if cancel_when == "during":
            yield "unfinished partial answer"
            app.cancel_event.set()
        return
        yield  # keep this a generator for the empty-response case

    app.multimodal.stream_solve = stream
    if cancel_when == "before":
        app.cancel_event.set()
    app._process_message_thread("Hello", "Hello")
    app.root.drain()
    app.engine.solve.assert_not_called()
    app.db.log_interaction.assert_not_called()
    assert app.chat_history["strata"] == []
    assert "I am ready" not in app.chat_stream.text
    assert ("without response text" if cancel_when == "never" else "stopped") in app.chat_stream.text
    assert app.lbl_tps.cget("text") == "⚡ Speed: unavailable"
    app._append_ai_message.assert_not_called()


def completed_request(app, *, total=43, rate=12.5):
    app.backend.last_usage = {"prompt_tokens": 34, "completion_tokens": 9, "total_tokens": total}
    app.backend.last_generation = {
        "status": "completed", "usage": dict(app.backend.last_usage),
        "timings": {"predicted_per_second": rate},
    }


def test_answer_uses_actual_last_request_usage_and_native_decode_timing(app):
    completed_request(app)
    app.multimodal.stream_solve = lambda *_args, **_kwargs: iter(["A real answer."])
    app._process_message_thread("Hello", "Hello")
    app.root.drain()
    app.engine.solve.assert_not_called()
    assert app.chat_history["strata"] == [{"role": "assistant", "content": "A real answer."}]
    assert app.total_tokens_used == 43
    assert app.lbl_context.cget("text") == "📊 Last request: 43 / 32,768 tokens"
    assert app.lbl_tps.cget("text") == "⚡ 12.5 tok/s decode (last request)"
    assert app.db.log_interaction.call_args.kwargs["completion"] == "A real answer."


def test_pro_throughput_uses_matching_pro_wall_time_without_summing_context(app):
    completed_request(app, total=61, rate=999.0)
    app.engine._last_stream_pro_meta = {
        "branch_count": 3, "usage": {"completion_tokens": 120, "total_tokens": 2200},
        "measured_wall_time_s": 4.0,
    }
    app._update_telemetry(500.0)
    assert app.total_tokens_used == 61
    assert app.lbl_context.cget("text") == "📊 Last request: 61 / 32,768 tokens"
    assert app.lbl_tps.cget("text") == "⚡ 30.0 tok/s (Pro requests)"


@pytest.mark.parametrize("bad", [None, True, "12", float("nan"), float("inf"), -1, 10**400, {}, []])
def test_invalid_or_missing_native_timing_does_not_reuse_previous_badge(app, bad):
    completed_request(app, rate=bad)
    app._update_telemetry(999.9)
    assert app.lbl_tps.cget("text") == "⚡ Speed: unavailable"


@pytest.mark.parametrize("bad", [None, True, "4", float("nan"), float("inf"), 0, -1, 10**400])
def test_invalid_pro_wall_time_does_not_fall_back_to_unrelated_native_rate(app, bad):
    completed_request(app, rate=999.0)
    app.engine._last_stream_pro_meta = {
        "branch_count": 3, "usage": {"completion_tokens": 120}, "measured_wall_time_s": bad,
    }
    app._update_telemetry(500.0)
    assert app.lbl_tps.cget("text") == "⚡ Speed: unavailable"


def test_missing_usage_clears_speed_even_when_a_timing_field_remains(app):
    completed_request(app, rate=999.0)
    app.backend.last_generation["usage"] = {}
    app._update_telemetry(500.0)
    assert app.total_tokens_used == 0
    assert app.lbl_context.cget("text") == "📊 Last request: unavailable"
    assert app.lbl_tps.cget("text") == "⚡ Speed: unavailable"


@pytest.mark.parametrize("status", ["cancelled", "error", "streaming"])
def test_unfinished_request_clears_usage_and_speed(app, status):
    completed_request(app)
    app.backend.last_generation["status"] = status
    app._update_telemetry(999.9)
    assert app.total_tokens_used == 0
    assert app.lbl_context.cget("text") == "📊 Last request: unavailable"
    assert app.lbl_tps.cget("text") == "⚡ Speed: unavailable"


@pytest.mark.parametrize("method", ["_on_send_message", "_run_queued_task"])
def test_strata_send_does_not_add_word_estimate_to_reported_usage(app, method):
    completed_request(app, total=43)
    if method == "_on_send_message":
        app.txt_input.text = "A question with several words"
        app._on_send_message()
    else:
        app._run_queued_task("A question with several words", "Question")
    assert app.total_tokens_used == 43
    assert len(app.pending) == 1


def test_native_telemetry_preserves_existing_counter_and_speed_behavior(app):
    app.active_tab_id = "native"
    app.engine.controller_backend = None
    app._update_telemetry(5.0)
    assert app.total_tokens_used == 777
    assert app.lbl_context.cget("text").startswith("📊 Context: 777 /")
    assert app.lbl_tps.cget("text") == "⚡ 5.0 tok/s"
