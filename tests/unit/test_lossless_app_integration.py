"""Production wrapper order and owned-GUI lifecycle, with no model execution.

The native adapter is only constructed (its constructor is model-free). Every
generation/loader entry point is replaced before use. GUI methods come from the
existing AST harness, so importing the desktop or neural runtimes is unnecessary.
"""
import copy
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.awake_auto_hook import install_awake_auto_learning
from core.engines.prism_speculative_engine import PrismSpeculativeReasoningBackend
from core.full_context_hardening import install as install_full_context
from core.lossless_speed_control import DEFAULT_OPTIONS
from core.pro_runtime_hardening import install_pro_runtime_hardening
from core.speed_control import install_speed_control
from core.temperature_policy import install_chat
from tests.unit.test_strata_gui import app as gui_app


@pytest.fixture
def pro_module(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "core/pro_engine.py"
    spec = importlib.util.spec_from_file_location("_lossless_pro_integration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Match core/__init__.py rather than testing an unwrapped approximation.
    install_chat(module)
    install_awake_auto_learning(module.ProReasoningEngine)
    install_full_context(module.ProReasoningEngine)
    install_pro_runtime_hardening(module.ProReasoningEngine)
    install_speed_control(module.ProReasoningEngine)
    from core.memory_watchdog import SystemMemoryWatchdog
    monkeypatch.setattr(SystemMemoryWatchdog, "get_detailed_memory_breakdown",
                        lambda **kwargs: {"total_allocated_gb": 0.0})
    return module


def engine_for(module, *, loaded=True, branches=1):
    engine = module.ProReasoningEngine.__new__(module.ProReasoningEngine)
    engine.settings = SimpleNamespace(use_mock=False, auto_download=False, max_new_tokens=128,
                                     search_temperature=0.75, search_top_p=0.92,
                                     mlx_model_path="unused-fixture")
    engine.mlx_backend = engine.bitnet_backend = engine.controller_backend = None
    engine.model = engine.base_model = engine.tokenizer = None
    engine.active_backend = "gguf" if loaded else None
    engine.active_model_name = "Native fixture" if loaded else None
    engine.active_model_path = "fixture.gguf" if loaded else None
    engine.active_input_modalities = {"text"}
    engine._model_lock = threading.RLock()
    engine._speed_cancel_event = threading.Event()
    engine._speed_loading_thread = None
    engine.auto_speed_enabled = True
    engine.speed_tuning_active = False
    engine.speed_status = {"status": "idle"}
    engine.lossless_speed_options = {**DEFAULT_OPTIONS, "native_branch_count": branches}
    engine._context_budget_tokens = 4096
    engine.speed_memory_limit_gb = None
    engine.awake_consolidator = SimpleNamespace(
        consolidate_chunk_sync=Mock(side_effect=AssertionError("Immutable target must not consolidate")))
    engine.speculative_engine = SimpleNamespace(
        target_model=None, draft_model=None, tokenizer=None,
        get_telemetry=Mock(side_effect=AssertionError("Stale legacy speculation telemetry")))
    engine.router = SimpleNamespace(
        route=Mock(side_effect=AssertionError("Native runtime must not infer routing entropy")),
        estimate_prompt_entropy_heuristic=Mock(side_effect=AssertionError("Heuristic routing changes the budget")))
    engine.verifier = SimpleNamespace(
        consensus_voting=lambda candidates: (candidates[0], len(candidates), 1.0),
        extract_code_block=lambda text: None)
    engine._generate_fallback_branches = Mock(side_effect=AssertionError("Unexpected fallback"))
    backend = PrismSpeculativeReasoningBackend("fixture.gguf", "fixture-server", mode="none", n_ctx=4096)
    backend.model = object() if loaded else None
    backend.is_loaded = loaded
    backend.tokenizer = SimpleNamespace(encode=lambda prompt: list(range(len(prompt))))
    engine.gguf_backend = backend if loaded else None
    return engine, backend


def recorded_stream(backend, *, empty=False, cancel_at=None, event=None, fail_at=None):
    calls, closed = [], []
    def stream(prompt, max_tokens, temperature, top_p, image_bytes=None, cancelled=None, **kwargs):
        index = len(calls)
        calls.append({"prompt": prompt, "max_tokens": max_tokens,
                      "temperature": temperature, "top_p": top_p, "cancelled": cancelled})
        try:
            if fail_at == index:
                raise RuntimeError("synthetic transport error")
            if cancel_at == index:
                if event is not None:
                    event.set()
                backend._cancelled.set()
                backend.last_generation = {"status": "cancelled", "usage": None}
                return
            backend.last_generation = {"status": "completed", "usage": {
                "prompt_tokens": 20, "completion_tokens": index + 1, "total_tokens": 21 + index}}
            if not empty:
                yield f"answer {index}"
        finally:
            closed.append(index)
    backend.stream_generate_tokens = stream
    return calls, closed


@pytest.mark.parametrize("cancelled", [False, True])
def test_full_wrapper_native_empty_or_cancelled_chat_never_retries(pro_module, cancelled):
    engine, backend = engine_for(pro_module)
    event = threading.Event()
    calls, closed = recorded_stream(backend, empty=True, cancel_at=0 if cancelled else None, event=event)
    engine.solve = Mock(side_effect=AssertionError("Empty native stream must not restart"))
    assert list(engine.stream_solve("hello", cancel_event=event)) == []
    assert len(calls) == 1 and closed == [0]
    assert calls[0]["temperature"] == 0.65 and calls[0]["top_p"] == 0.92
    assert callable(calls[0]["cancelled"])
    engine.solve.assert_not_called()
    engine.router.route.assert_not_called()
    engine.awake_consolidator.consolidate_chunk_sync.assert_not_called()
    assert engine.settings.max_new_tokens == 128


def test_preexisting_cancel_avoids_tokenization_generation_and_entropy(pro_module):
    engine, backend = engine_for(pro_module)
    backend.tokenizer.encode = Mock(side_effect=AssertionError("Cancelled prompt tokenized"))
    backend.stream_generate_tokens = Mock(side_effect=AssertionError("Cancelled request restarted"))
    event = threading.Event()
    event.set()
    assert list(engine.stream_solve("cancelled", cancel_event=event)) == []
    backend.tokenizer.encode.assert_not_called()
    backend.stream_generate_tokens.assert_not_called()


def test_native_explicit_pro_preserves_ladder_full_history_and_actual_usage(pro_module):
    engine, backend = engine_for(pro_module, branches=3)
    calls, closed = recorded_stream(backend)
    history = [{"role": role, "content": f"message {i} " + "retained " * 100}
               for i, role in enumerate(("user", "assistant", "user", "assistant"))]
    before = copy.deepcopy(history)
    text = "".join(engine.stream_solve("continue", history=history))
    assert text == "answer 0"
    assert history == before
    assert len(calls) == 4 and closed == [0, 1, 2, 3]
    assert [call["temperature"] for call in calls] == pro_module.get_ladder_temperatures(3)
    assert all(call["top_p"] == 0.92 for call in calls)
    assert all(f"message {i}" in call["prompt"] for i in range(4) for call in calls)
    # The actual adapter counts exact tokens inside the newly started request;
    # the app forwards its total Context and the untouched complete prompt.
    assert all(call["max_tokens"] == 4096 for call in calls)
    assert backend.context_budget_tokens == 4096
    assert backend.last_branch_usage == {"prompt_tokens": 80, "completion_tokens": 10, "total_tokens": 90}
    metadata = engine._last_stream_pro_meta
    assert metadata["entropy"] is None and metadata["branch_count"] == 4
    assert metadata["tokens_generated"] == 10 and metadata["backend"] == "prism_speculative"
    assert engine.settings.max_new_tokens == 128
    assert "immutable" in engine._last_context_consolidation_note
    engine.router.route.assert_not_called()
    engine.router.estimate_prompt_entropy_heuristic.assert_not_called()
    engine.awake_consolidator.consolidate_chunk_sync.assert_not_called()


def test_native_direct_pro_requires_explicit_budget_and_never_uses_heuristic(pro_module):
    engine, backend = engine_for(pro_module)
    calls, _ = recorded_stream(backend)
    with pytest.raises(ValueError, match="explicit thinking branch count"):
        engine.solve("question")
    assert calls == []
    answer, metadata = engine.solve("question", force_branch_count=1)
    assert answer == "answer 0" and metadata["entropy"] is None
    engine.router.estimate_prompt_entropy_heuristic.assert_not_called()


@pytest.mark.parametrize("cancel_at", [0, 1, 3])
def test_native_pro_cancellation_never_runs_remaining_or_fallback_branches(pro_module, cancel_at):
    engine, backend = engine_for(pro_module, branches=3)
    event = threading.Event()
    calls, _ = recorded_stream(backend, cancel_at=cancel_at, event=event)
    assert list(engine.stream_solve("question", cancel_event=event)) == []
    assert len(calls) == cancel_at + 1
    assert engine._last_stream_pro_meta["cancelled"] is True
    assert not backend.last_branch_usage
    engine._generate_fallback_branches.assert_not_called()


def test_close_of_outer_chat_stream_closes_inner_native_generator(pro_module):
    engine, backend = engine_for(pro_module)
    calls, closed = recorded_stream(backend)
    stream = engine.stream_solve("question")
    assert next(stream) == "answer 0"
    stream.close()
    assert len(calls) == 1 and closed == [0]
    assert engine.settings.max_new_tokens == 128


def test_retained_but_unloaded_native_owner_cannot_fall_back_or_generate(pro_module):
    engine, backend = engine_for(pro_module)
    backend.is_loaded = False
    backend.model = None
    backend.stream_generate_tokens = Mock(side_effect=AssertionError("Unloaded owner generated"))
    with pytest.raises(RuntimeError, match="Load the selected native runtime"):
        list(engine.stream_solve("must not restart"))
    backend.stream_generate_tokens.assert_not_called()
    engine._generate_fallback_branches.assert_not_called()


def test_native_unload_during_search_keeps_cancellation_after_owner_detaches(pro_module):
    engine, backend = engine_for(pro_module, branches=3)
    calls = []
    def stream(*args, **kwargs):
        calls.append(kwargs)
        assert engine.unload_model()["status"] == "unloaded"
        backend.last_generation = {"status": "cancelled", "usage": None}
        return
        yield
    backend.stream_generate_tokens = stream
    answer, metadata = engine.solve("question", force_branch_count=3)
    assert answer == "" and metadata["cancelled"] is True
    assert len(calls) == 1 and engine.gguf_backend is None
    engine._generate_fallback_branches.assert_not_called()


def test_failed_native_owner_stop_blocks_replacement_before_artifact_resolution(pro_module, monkeypatch):
    engine, backend = engine_for(pro_module)
    backend.unload_model = Mock(side_effect=RuntimeError("owned process remains alive"))
    resolve = Mock(side_effect=AssertionError("Replacement resolution started beside live owner"))
    monkeypatch.setattr(pro_module, "resolve_gguf_artifacts", resolve)
    result = engine.load_model("replacement", model_path="replacement.gguf", model_info={"backend_family": "gguf"})
    assert result["status"] == "error" and result["requested_model"] == "replacement"
    assert engine.gguf_backend is backend and backend.model is not None
    assert engine.active_model_name == "Native fixture"
    resolve.assert_not_called()


def test_cancelled_native_load_forwards_live_callback_and_never_tunes(pro_module, monkeypatch):
    import core.engines.prism_speculative_engine as native_module
    engine, _ = engine_for(pro_module, loaded=False)
    engine.lossless_speed_options["speculative_server_path"] = "fixture-server"
    external = threading.Event()
    engine._model_load_cancelled = external.is_set
    created = []
    class Loader(PrismSpeculativeReasoningBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)
        def load_model(self, *, cancelled):
            assert not cancelled()
            external.set()
            assert cancelled()
            self.last_error = "setup cancelled before any model load"
            return False
        configure_auto_speed = Mock(side_effect=AssertionError("Cancelled setup tuned"))
    monkeypatch.setattr(native_module, "PrismSpeculativeReasoningBackend", Loader)
    monkeypatch.setattr(pro_module, "resolve_gguf_artifacts", lambda path, info: (path, None))
    result = engine.load_model("native fixture", model_path="fixture.gguf", model_info={"backend_family": "gguf"})
    assert result["status"] == "cancelled" and len(created) == 1
    assert created[0].model is None and created[0]._process is None
    Loader.configure_auto_speed.assert_not_called()
    assert engine._speed_loading_thread is None


@pytest.fixture
def native_gui(gui_app):
    app = gui_app
    app.active_tab_id = "native"
    app.models_config["native"] = {"name": "Native fixture", "short_name": "Native", "model_type": "text",
                                   "model_path": "fixture.gguf"}
    app._model_is_cached = lambda info: True
    app.engine.controller_backend = None
    app.backend = SimpleNamespace(runtime="prism_speculative", model=object(), is_loaded=True,
                                  _process=object(), cancel_generation=Mock())
    app.engine.gguf_backend = app.backend
    app.engine.speed_tuning_active = False
    app.engine.cancel_speed_tuning = Mock()
    app._update_telemetry = Mock()
    app.chat_history["native"] = []
    def unload():
        app.engine.gguf_backend = None
        app.backend._process = None
        return {"status": "unloaded", "unloaded": True}
    app.engine.unload_model = Mock(side_effect=unload)
    return app


def finish_gui_worker(app):
    app.pending.pop(0)()
    kind, callback = app._event_queue.get_nowait()
    assert kind == "model_load_complete"
    callback()


def test_native_gui_unload_retains_failed_owner_and_retry_never_reloads(native_gui):
    app = native_gui
    owner = app.backend
    stop = app.engine.unload_model.side_effect
    app.engine.unload_model.side_effect = None
    app.engine.unload_model.return_value = {"status": "error", "error": "owned process still alive"}
    app._on_toggle_load_unload()
    owner.cancel_generation.assert_called_once()
    assert app.is_model_loaded and not app.engine.unload_model.called
    finish_gui_worker(app)
    assert not app.is_model_loaded and app._native_unload_pending
    assert app.engine.gguf_backend is owner and owner._process is not None
    assert app.btn_load_unload.cget("text") == "Retry unload"
    assert app.btn_send.cget("state") == "disabled"
    app.engine.unload_model.side_effect = stop
    app._on_toggle_load_unload()
    finish_gui_worker(app)
    assert not app._native_unload_pending and app.engine.gguf_backend is None
    assert app.btn_load_unload.cget("text") == "⚡ Load"
    app.engine.load_model.assert_not_called()


def test_native_gui_stale_unload_does_not_touch_replacement(native_gui):
    app = native_gui
    app._unload_strata_async()
    replacement = SimpleNamespace(runtime="prism_speculative", _process=object())
    app.engine.gguf_backend = replacement
    finish_gui_worker(app)
    assert app.engine.gguf_backend is replacement
    app.engine.unload_model.assert_not_called()
    app.engine.load_model.assert_not_called()


@pytest.mark.parametrize("cancel_when", ["never", "before", "during"])
def test_native_gui_empty_or_cancelled_answer_never_retries_or_saves(native_gui, cancel_when):
    app = native_gui
    def stream(*args, **kwargs):
        if cancel_when == "during":
            yield "unfinished partial answer"
            app.cancel_event.set()
        return
        yield
    app.multimodal.stream_solve = stream
    if cancel_when == "before":
        app.cancel_event.set()
    app._process_message_thread("Hello", "Hello")
    app.root.drain()
    app.engine.solve.assert_not_called()
    app.db.log_interaction.assert_not_called()
    assert app.chat_history["native"] == []
    assert "I am ready" not in app.chat_stream.text


def test_native_gui_uses_actual_current_counts_and_decoder_rate(native_gui):
    app = native_gui
    app.backend.last_generation = {"status": "completed", "usage": {
        "prompt_tokens": 34, "completion_tokens": 9, "total_tokens": 43},
        "timings": {"predicted_per_second": 12.5}}
    app.backend.n_ctx = 32768
    app.backend.get_speculative_telemetry = lambda: {"decode_tokens_per_second": 12.5}
    type(app)._update_telemetry(app)
    assert app.total_tokens_used == 43
    assert "43" in app.lbl_context.cget("text") and "777" not in app.lbl_context.cget("text")
    assert "12.5" in app.lbl_tps.cget("text")
    app.cancel_event.set()
    type(app)._update_telemetry(app)
    assert app.total_tokens_used == 0
    assert "unavailable" in app.lbl_context.cget("text").lower()
    assert "unavailable" in app.lbl_tps.cget("text").lower()


def test_native_gui_pro_rate_uses_matching_whole_search_usage(native_gui):
    app = native_gui
    app.backend.last_generation = {"status": "completed", "usage": {
        "prompt_tokens": 34, "completion_tokens": 9, "total_tokens": 43},
        "timings": {"predicted_per_second": 999.0}}
    app.backend.n_ctx = 32768
    app.backend.get_speculative_telemetry = lambda: {"decode_tokens_per_second": 999.0}
    app.engine._last_stream_pro_meta = {"branch_count": 4, "usage": {"completion_tokens": 40},
                                      "measured_wall_time_s": 8.0}
    type(app)._update_telemetry(app)
    assert "5.0" in app.lbl_tps.cget("text") and "999" not in app.lbl_tps.cget("text")
    assert app.total_tokens_used == 43  # Context is one request, not summed branches.


@pytest.mark.parametrize("cleanup_succeeds", [True, False])
def test_native_gui_cancellation_after_successful_setup_cleans_or_preserves_retry(native_gui, cleanup_succeeds):
    app = native_gui
    app.is_model_loaded = False
    def load(*args, **kwargs):
        app._model_load_cancel_event.set()
        assert app.engine._model_load_cancelled()
        return {"status": "loaded"}
    app.engine.load_model.side_effect = load
    if not cleanup_succeeds:
        app.engine.unload_model.side_effect = None
        app.engine.unload_model.return_value = {"status": "error", "error": "cleanup still active"}
    app._on_toggle_load_unload()
    # The harness root is hidden, so native loading executes synchronously.
    assert app.engine.unload_model.call_count == 1
    assert not app.is_model_loaded
    assert app._native_unload_pending is not cleanup_succeeds
    assert app.engine._model_load_cancelled is None
    assert not app._model_load_in_progress
    assert (app.btn_load_unload.cget("text") == "Retry unload") is not cleanup_succeeds


def install_close_guard(app, monkeypatch):
    from core.gui_speed_tuning import install_gui_speed_tuning
    cls = type(app)
    cls._build_single_top_bar = lambda self: None
    cls._apply_theme_colors = lambda self: None
    cls._on_close_app = lambda self: setattr(self, "closed", True)
    monkeypatch.setitem(sys.modules, "app_gui", SimpleNamespace(SmartAIChatbotApp=cls, tk=SimpleNamespace()))
    install_gui_speed_tuning()
    app._speed_cancel_event = threading.Event()
    app._speed_ui_running = False
    app._speed_ui_closed = False
    app._speed_close_requested = False
    app.closed = False


@pytest.mark.parametrize("cleanup_succeeds", [True, False])
def test_native_gui_close_waits_for_owner_and_never_restarts_setup(native_gui, monkeypatch, cleanup_succeeds):
    app = native_gui
    install_close_guard(app, monkeypatch)
    if not cleanup_succeeds:
        app.engine.unload_model.side_effect = None
        app.engine.unload_model.return_value = {"status": "error", "error": "process did not stop"}
    app._on_close_app()
    assert not app.closed and len(app.pending) == 1
    finish_gui_worker(app)
    app.root.callbacks.pop(0)()
    assert app.closed is cleanup_succeeds
    assert not app.pending
    app.engine.load_model.assert_not_called()
    assert app.engine.unload_model.call_count == 1
    if not cleanup_succeeds:
        assert app._native_unload_pending and not app._speed_close_requested
        assert app.btn_load_unload.cget("text") == "Retry unload"


def test_close_during_pending_native_switch_suppresses_replacement_load(native_gui, monkeypatch):
    app = native_gui
    install_close_guard(app, monkeypatch)
    app._on_switch_model_tab("strata")
    assert app.active_tab_id == "native" and len(app.pending) == 1
    app._on_close_app()
    assert not app.closed and len(app.pending) == 1
    finish_gui_worker(app)
    assert app.active_tab_id == "native"  # Suppress the queued switch callback.
    app.root.callbacks.pop(0)()
    assert app.closed and not app.pending
    app.engine.load_model.assert_not_called()
