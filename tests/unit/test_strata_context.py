"""Strata owns exact token accounting; context edits must match its loaded server."""
import copy
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from config.strata_models import get_strata_preset
from core.awake_auto_hook import _remaining_generation_tokens, install_awake_auto_learning
from core.gui_generation_cap import install_gui_generation_cap


def backend(context=32768):
    return SimpleNamespace(
        runtime="strata", model=object(), is_loaded=True, n_ctx=lambda: context,
        context_budget_tokens=context, tokenizer=None,
        count_tokens=Mock(side_effect=AssertionError("An estimate must not govern Strata generation")),
    )


def test_strata_native_server_counts_long_prompt_and_owns_output_room():
    model = backend()
    engine = SimpleNamespace(
        active_backend="controller", controller_backend=model,
        _context_budget_tokens=32768,
    )
    # More UTF-8 bytes than the context need not mean more native tokens.
    prompt = "repeat 42 " * 20_000
    assert _remaining_generation_tokens(engine, prompt) == 32768
    assert engine._effective_context_budget_tokens == 32768
    model.count_tokens.assert_not_called()


@pytest.mark.parametrize("requested", [8192, 65536])
def test_strata_external_context_mismatch_requires_reload(requested):
    engine = SimpleNamespace(
        active_backend="controller", controller_backend=backend(),
        _context_budget_tokens=requested,
    )
    with pytest.raises(RuntimeError, match="Unload the model"):
        _remaining_generation_tokens(engine, "a short prompt")
    assert engine._context_budget_tokens == requested


def test_native_token_count_and_overflow_behavior_are_preserved():
    tokenizer = SimpleNamespace(encode=lambda prompt: list(range(len(prompt))))
    native = SimpleNamespace(model=object(), tokenizer=tokenizer, n_ctx=lambda: 4096)
    engine = SimpleNamespace(active_backend="mlx", mlx_backend=native, _context_budget_tokens=2048)
    assert _remaining_generation_tokens(engine, "x" * 1000) == 1048
    with pytest.raises(RuntimeError, match="refusing truncation"):
        _remaining_generation_tokens(engine, "x" * 3000)


@pytest.mark.parametrize("branch_count", [1, 8])
@pytest.mark.parametrize("explicit_context", [True, False])
def test_strata_chat_keeps_full_history_and_never_runs_weight_consolidation(branch_count, explicit_context, monkeypatch):
    monkeypatch.setattr("core.awake_auto_hook.get_auto_context_window_size", lambda: 65536)
    class Engine:
        def __init__(self):
            self.controller_backend = backend()
            # A stray old consolidator/tokenizer must not enable inference-only training.
            self.controller_backend.tokenizer = SimpleNamespace(encode=Mock(return_value=list(range(32768))))
            self.active_backend = "controller"
            self._context_budget_tokens = 32768 if explicit_context else 0
            self.settings = SimpleNamespace(max_new_tokens=321)
            self.awake_consolidator = SimpleNamespace(consolidate_chunk_sync=Mock())
            self.router = SimpleNamespace(route=lambda *args, **kwargs: ("test", branch_count))
            self.calculate_token_entropy = lambda prompt: 0.2
            self._last_context_consolidation_note = "Old native weight update"

        def _format_prompt_with_history(self, prompt, history):
            return "".join(turn["content"] for turn in history) + prompt

        def stream_solve(self, prompt, history=None, **kwargs):
            self.sent = (prompt, copy.deepcopy(history), self.settings.max_new_tokens)
            yield "answer"

        def solve(self, prompt, history=None, **kwargs):
            self.sent = (prompt, copy.deepcopy(history), self.settings.max_new_tokens)
            return "answer", {"branch_count": branch_count}

    install_awake_auto_learning(Engine)
    engine = Engine()
    history = [{"role": role, "content": " keep whitespace \n" + "data " * 20_000}
               for role in ("user", "assistant", "user", "assistant")]
    before = copy.deepcopy(history)
    assert "".join(engine.stream_solve("continue", history=history)) == "answer"
    assert history == before
    assert engine.sent == ("continue", before, 32768)
    assert engine.settings.max_new_tokens == 321
    assert engine._context_budget_tokens == 32768
    engine.awake_consolidator.consolidate_chunk_sync.assert_not_called()
    engine.controller_backend.tokenizer.encode.assert_not_called()
    assert "inference only" in engine._last_context_consolidation_note
    assert "history is kept" in engine._last_context_consolidation_note


class Variable:
    def __init__(self, value):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class Widget:
    def __init__(self, master=None, **kwargs):
        self.options = kwargs
        self.bindings = {}

    def configure(self, **kwargs):
        self.options.update(kwargs)

    def pack(self, **kwargs):
        pass

    def bind(self, name, callback):
        self.bindings[name] = callback


@pytest.fixture
def app(monkeypatch):
    class App:
        _build_single_top_bar = lambda self: None
        _set_generating_state = lambda self, value: "original-result"
        _process_message_thread = lambda self, *args, **kwargs: None
        _apply_theme_colors = lambda self: None

    module = SimpleNamespace(
        SmartAIChatbotApp=App,
        tk=SimpleNamespace(Frame=Widget, Label=Widget, Entry=Widget, Button=Widget, StringVar=Variable),
        _FONT_TINY_BOLD="font", _FONT_TINY="font",
    )
    monkeypatch.setitem(sys.modules, "app_gui", module)
    install_gui_generation_cap()
    app = App()
    app.C = {key: "color" for key in (
        "bg_hud", "text_main", "bg_input_inner", "border", "btn_bg", "btn_fg", "text_muted",
    )}
    app.main_container = object()
    app.active_tab_id = "strata_q2_0"
    app.models_config = {"strata_q2_0": get_strata_preset("q2_0"),
                         "native": {"max_context": 131072}}
    model = backend()
    app.engine = SimpleNamespace(
        active_backend="controller", controller_backend=model,
        mlx_backend=SimpleNamespace(context_budget_tokens=12345, last_generation_cap_reason="Old native reason"),
        _context_budget_tokens=32768,
        _last_context_consolidation_note="Old native consolidated weights",
    )
    app.is_model_loaded = True
    app._build_single_top_bar()
    return app


def apply_context(app, value):
    app.generation_cap_var.set(value)
    app.btn_apply_generation_cap.options["command"]()


def test_loaded_strata_context_edit_is_rejected_without_mutating_selected_window(app):
    apply_context(app, "65536")
    assert app.engine._context_budget_tokens == 32768
    assert app.engine.controller_backend.context_budget_tokens == 32768
    assert app.generation_cap_var.get() == "32768"
    assert "Unload the model" in app.lbl_generation_cap_reason.options["text"]
    assert app.engine.mlx_backend.context_budget_tokens == 12345


def test_unloaded_strata_accepts_explicit_context_for_next_load(app):
    app.is_model_loaded = False
    app.engine.controller_backend.is_loaded = False
    apply_context(app, "65,536")
    assert app.engine._context_budget_tokens == 65536
    assert app.engine.controller_backend.context_budget_tokens == 65536
    assert app.generation_cap_var.get() == "65536"
    assert "inference only" in app.lbl_generation_cap_reason.options["text"]
    assert "consolidated into weights" not in app.lbl_generation_cap_reason.options["text"]


@pytest.mark.parametrize("case", ["load", "chat", "media", "sleep", "awake", "eval", "tune", "closing"])
def test_strata_context_edit_waits_for_active_work(app, case):
    app.engine.controller_backend.is_loaded = False
    if case == "load":
        app._model_load_in_progress = True
    elif case == "chat":
        app.is_generating = True
    elif case == "media":
        app.multimodal = SimpleNamespace(busy=True)
    elif case == "sleep":
        app._sleep_consolidating = True
    elif case == "awake":
        app.engine.awake_consolidator = SimpleNamespace(is_consolidating=True)
    elif case == "eval":
        app._eval_proc = SimpleNamespace(poll=lambda: None)
    elif case == "tune":
        app.engine.speed_tuning_active = True
    else:
        app._speed_close_requested = True
    apply_context(app, "65536")
    assert app.engine._context_budget_tokens == 32768
    assert app.generation_cap_var.get() == "32768"


@pytest.mark.parametrize("requested", ["invalid", "1023", "0", "262145"])
def test_invalid_strata_context_keeps_previous_selection_without_clamping(app, requested):
    app.engine.controller_backend.is_loaded = False
    apply_context(app, requested)
    assert app.engine._context_budget_tokens == 32768
    assert app.generation_cap_var.get() == "32768"


def test_strata_note_refresh_overrides_stale_native_consolidation_message(app):
    assert app._set_generating_state(False) == "original-result"
    note = app.lbl_generation_cap_reason.options["text"]
    assert "Strata inference only" in note
    assert "Old native" not in note
    app.active_tab_id = "native"
    app._refresh_context_model_note()
    assert "Old dialogue consolidates at 80%" in app.lbl_generation_cap_reason.options["text"]
    assert "Strata" not in app.lbl_generation_cap_reason.options["text"]


def test_strata_initial_default_does_not_use_native_ram_context_heuristic(app, monkeypatch):
    monkeypatch.setattr("core.gui_generation_cap.get_auto_context_window_size", lambda: 131072)
    app.engine._context_budget_tokens = 0
    app.engine.controller_backend.is_loaded = False
    app._refresh_context_model_note()
    assert app.engine._context_budget_tokens == 32768
    assert app.generation_cap_var.get() == "32768"


def test_native_context_apply_retains_existing_behavior(app):
    app.active_tab_id = "native"
    app.engine.active_backend = "mlx"
    app.engine.controller_backend.is_loaded = False
    apply_context(app, "65536")
    assert app.engine._context_budget_tokens == 65536
    assert app.engine.mlx_backend.context_budget_tokens == 65536
    assert "consolidated into weights" in app.lbl_generation_cap_reason.options["text"]
