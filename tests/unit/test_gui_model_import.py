"""Real import/install GUI callbacks with queued workers, without a Tk display."""
from __future__ import annotations

import ast
from collections import defaultdict
import os
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


class Widgets:
    """Record UI operations and reject accidental calls from the worker lane."""
    def __init__(self):
        self.items = []
        self.worker = False
        self.violations = []

    def create(self, kind, *args, **kwargs):
        item = Widget(self, kind, **kwargs)
        self.items.append(item)
        return item

    def of(self, kind):
        return [w for w in self.items if w.kind == kind]

    def button(self, text):
        return next(w for w in self.of("Button") if w.options.get("text") == text)


class Widget:
    def __init__(self, owner, kind, **kwargs):
        self.owner, self.kind = owner, kind
        self.options = {"state": "normal", **kwargs}
        self.value = str(kwargs.get("value", ""))
        self.alive = True
        self.protocols = {}
        self.bindings = {}

    def _ui(self):
        if self.owner.worker:
            self.owner.violations.append(self.kind)
            raise AssertionError("A worker touched a Tk widget")

    def configure(self, **kwargs):
        self._ui()
        self.options.update(kwargs)

    def cget(self, name):
        self._ui()
        return self.options.get(name)

    def get(self):
        self._ui()
        return self.value

    def set(self, value):
        self._ui()
        self.value = value

    def insert(self, _index, value):
        self._ui()
        self.value += str(value)

    def delete(self, *_args):
        self._ui()
        self.value = ""

    def bind(self, event, callback):
        self._ui()
        self.bindings[event] = callback

    def protocol(self, name, callback):
        self._ui()
        self.protocols[name] = callback

    def destroy(self):
        self._ui()
        self.alive = False

    def winfo_exists(self):
        self._ui()
        return self.alive

    def invoke(self):
        self._ui()
        if self.options["state"] != "disabled":
            return self.options["command"]()

    def pack(self, *_args, **_kwargs):
        self._ui()

    title = geometry = resizable = minsize = transient = grab_set = focus_set = pack


@pytest.fixture
def gui(tmp_path):
    widgets, workers = Widgets(), []

    class Thread:
        def __init__(self, target, args=(), **_kwargs):
            self.target, self.args = target, args

        def start(self):
            workers.append(lambda: self.target(*self.args))

    source = Path(__file__).resolve().parents[2] / "app_gui.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == "SmartAIChatbotApp")
    names = {
        "_on_fetch_custom_hf_model_dialog", "_on_import_custom_model_dialog",
        "_on_download_hf_model", "_remove_hf_fetched_model",
        "_on_reset_and_reinstall_single_confirm", "_model_is_cached",
        "_update_model_action_buttons", "_on_download_hf_completed", "_on_download_hf_failed",
    }
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    tk = SimpleNamespace(**{kind: lambda *a, _kind=kind, **k: widgets.create(_kind, *a, **k)
                            for kind in ("Toplevel", "Frame", "Label", "Entry", "Button", "Radiobutton", "StringVar")})
    env = {
        "os": os, "time": time, "tk": tk,
        "ttk": SimpleNamespace(Combobox=lambda *a, **k: widgets.create("Combobox", *a, **k)),
        "threading": SimpleNamespace(Thread=Thread, Event=threading.Event),
        "messagebox": SimpleNamespace(askyesno=Mock(return_value=True), showerror=Mock(), showinfo=Mock()),
        "filedialog": SimpleNamespace(askdirectory=Mock(), askopenfilename=Mock()),
        "inspect_mlx_model_folder": Mock(return_value={"valid": False}),
        "admit_hf_model": Mock(), "admit_local_model": Mock(),
        "download_imported_model": Mock(), "download_model_from_hf": Mock(),
        "purge_local_model_cache": Mock(return_value=True),
        "is_model_cached_locally": Mock(return_value=True),
    }
    for name in ("_FONT_H2", "_FONT_SMALL", "_FONT_TINY", "_FONT_MAIN", "_FONT_TINY_BOLD", "_FONT_BOLD"):
        env[name] = name
    def make_entry(metadata, **kwargs):
        return {**metadata, "name": kwargs["display_name"], "short_name": kwargs["display_name"],
                "model_path": kwargs["local_path"], "repo_id": kwargs.get("repo_id"),
                "source_origin": "hf_fetch" if kwargs.get("fetched") else "local",
                "runtime_preference": kwargs.get("runtime_preference")}
    env["make_imported_model_entry"] = Mock(side_effect=make_entry)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), env)
    app = type("ImportGuiHarness", (), {name: env[name] for name in names})()
    app.env, app.widgets, app.workers = env, widgets, workers
    app.root = widgets.create("Root")
    app._event_queue = queue.Queue()
    app.cancel_event = threading.Event()
    app.C = defaultdict(lambda: "color")
    app.models_config = {"model_1": {"name": "Other", "short_name": "Other", "model_type": "text", "model_path": str(tmp_path)}}
    app.active_tab_id = "model_1"
    app.is_model_loaded = False
    app.multimodal = SimpleNamespace(busy=False)
    app.chat_history, app.tab_buttons = {}, {}
    app.thinking_cache, app.thinking_expanded, app.thinking_meta = {}, {}, {}
    app.engine = SimpleNamespace(unload_model=Mock(return_value={"status": "unloaded"}))
    app.audio_engine = SimpleNamespace(unload_model=Mock())
    app.media_engine = SimpleNamespace(unload_model=Mock())
    for name in ("btn_model_menu", "btn_load_unload", "btn_reset_reinstall", "lbl_model_status"):
        setattr(app, name, widgets.create(name))
    for name in ("_save_custom_models", "_refresh_model_menu", "_on_switch_model_tab", "_append_ai_message",
                 "_sync_memory_watchdog", "_show_strata_model_info", "_on_toggle_load_unload"):
        setattr(app, name, Mock())
    app._unload_strata_async = Mock(return_value=False)
    app._on_download_hf_completed = Mock(wraps=app._on_download_hf_completed)
    app._on_download_hf_failed = Mock(wraps=app._on_download_hf_failed)
    return app


def run_worker(gui):
    gui.widgets.worker = True
    try:
        gui.workers.pop(0)()
    finally:
        gui.widgets.worker = False
    assert gui.widgets.violations == []


def drain(gui):
    while not gui._event_queue.empty():
        kind, callback = gui._event_queue.get_nowait()
        assert kind == "model_load_complete"
        callback()


def selected_entry(tmp_path, *, inspected=False):
    return {"name": "Chosen", "short_name": "Chosen", "repo_id": "owner/chosen",
            "source_repo_id": "owner/chosen", "source_origin": "hf_fetch", "source_kind": "hf_fetch",
            "repo_revision": "a" * 40, "source_revision": "a" * 40, "path": str(tmp_path),
            "model_path": str(tmp_path), "gguf_file": "q/chosen-00001-of-00002.gguf",
            "gguf_files": ["q/chosen-00001-of-00002.gguf", "q/chosen-00002-of-00002.gguf"],
            "download_required_files": ["q/chosen-00001-of-00002.gguf", "q/chosen-00002-of-00002.gguf"],
            "download_allow_patterns": ["q/chosen-00001-of-00002.gguf", "q/chosen-00002-of-00002.gguf"],
            "requires_local_validation": not inspected, "runtime_preference": "native", "model_type": "text",
            "controller_runtime": "gguf", "backend_family": "gguf", "precision": "GGUF Q4_0", "ternary": False}


def setup_transfer(gui, tmp_path, *, inspected=False):
    meta = selected_entry(tmp_path, inspected=inspected)
    gui.models_config["custom_chosen"] = meta
    gui.active_tab_id = "custom_chosen"
    gui.env["admit_hf_model"].return_value = True, "verified", dict(meta)
    gui.env["admit_local_model"].return_value = True, "verified", dict(meta)
    gui.env["download_imported_model"].return_value = {
        "status": "success", "local_dir": str(tmp_path),
        "metadata": {**meta, "requires_local_validation": False}}
    return meta


def open_fetch(gui, tmp_path):
    setup_transfer(gui, tmp_path)
    gui._on_fetch_custom_hf_model_dialog()
    gui.widgets.of("Entry")[-1].set("owner/chosen")
    choice, runtime = gui.widgets.of("Combobox")[-2:]
    choice.set("q/chosen-00002-of-00002.gguf")
    runtime.set("Native")
    return gui.widgets.of("Toplevel")[-1], choice, runtime


def test_fetch_captures_selection_before_worker_and_only_updates_ui_from_queue(gui, tmp_path):
    _, file_field, runtime_field = open_fetch(gui, tmp_path)
    gui.widgets.button("⬇️ Fetch & Add").invoke()
    gui.widgets.of("Entry")[-1].set("other/repository")
    file_field.set("other-Q8.gguf")
    runtime_field.set("Strata")
    run_worker(gui)
    gui.env["admit_hf_model"].assert_called_once_with("owner/chosen", gguf_file="q/chosen-00002-of-00002.gguf")
    gui._save_custom_models.assert_not_called()
    drain(gui)
    gui._save_custom_models.assert_called_once()
    call = gui.env["make_imported_model_entry"].call_args
    assert call.kwargs["runtime_preference"] == "native"
    assert call.kwargs["repo_id"] == "owner/chosen"
    assert call.args[0]["source_revision"] == "a" * 40


@pytest.mark.parametrize("close_after_worker", [False, True])
def test_cancelled_fetch_dialog_never_registers_queued_success(gui, tmp_path, close_after_worker):
    dialog, _, _ = open_fetch(gui, tmp_path)
    before = dict(gui.models_config)
    gui.widgets.button("⬇️ Fetch & Add").invoke()
    if close_after_worker:
        run_worker(gui)
    dialog.protocols["WM_DELETE_WINDOW"]()
    if not close_after_worker:
        run_worker(gui)
    drain(gui)
    assert gui.models_config == before
    gui._save_custom_models.assert_not_called()
    gui._on_switch_model_tab.assert_not_called()
    gui.env["make_imported_model_entry"].assert_not_called()
    assert gui.env["download_imported_model"].call_args.kwargs["cancel_event"].is_set()


def test_cancelled_inspection_dialog_does_not_register_or_switch(gui, tmp_path):
    setup_transfer(gui, tmp_path)
    before = dict(gui.models_config)
    gui._on_import_custom_model_dialog()
    gui.widgets.of("Entry")[1].set("owner/chosen")
    gui.widgets.button("➕ Import & Select").invoke()
    gui.widgets.of("Toplevel")[-1].protocols["WM_DELETE_WINDOW"]()
    run_worker(gui)
    drain(gui)
    assert gui.models_config == before
    gui._save_custom_models.assert_not_called()
    gui._on_switch_model_tab.assert_not_called()


def test_finished_download_after_switch_updates_only_the_selected_entry(gui, tmp_path):
    selected = setup_transfer(gui, tmp_path)
    gui._on_download_hf_model()
    gui.active_tab_id = "model_1"
    gui._update_model_action_buttons()
    run_worker(gui)
    drain(gui)
    assert gui.models_config["custom_chosen"] is selected
    assert selected["requires_local_validation"] is False
    assert gui.active_tab_id == "model_1"
    gui._on_download_hf_completed.assert_not_called()
    gui._on_download_hf_failed.assert_not_called()
    assert not gui._hf_download_in_progress


@pytest.mark.parametrize("replace", [False, True])
def test_finished_download_never_resurrects_a_removed_or_replaced_entry(gui, tmp_path, replace):
    setup_transfer(gui, tmp_path)
    gui._on_download_hf_model()
    replacement = {"name": "A different selection"}
    if replace:
        gui.models_config["custom_chosen"] = replacement
    else:
        gui.models_config.pop("custom_chosen")
    gui.active_tab_id = "model_1"
    run_worker(gui)
    drain(gui)
    assert gui.models_config.get("custom_chosen") is (replacement if replace else None)
    gui._save_custom_models.assert_not_called()
    assert not gui._hf_download_in_progress and not gui._hf_download_events


def test_verified_model_reinstall_keeps_immutable_revision_and_all_shards(gui, tmp_path):
    meta = setup_transfer(gui, tmp_path, inspected=True)
    gui._on_download_hf_model()
    run_worker(gui)
    call = gui.env["download_imported_model"].call_args
    assert call.args[0] == "owner/chosen"
    assert call.args[1]["repo_revision"] == "a" * 40
    assert call.args[1]["download_required_files"] == meta["gguf_files"]
    gui.env["download_model_from_hf"].assert_not_called()
    drain(gui)


@pytest.mark.parametrize("cancel_after_worker", [False, True])
def test_download_cancel_wins_over_queued_success_and_restores_buttons(gui, tmp_path, cancel_after_worker):
    meta = setup_transfer(gui, tmp_path)
    original = dict(meta)
    gui._on_download_hf_model()
    assert gui.btn_reset_reinstall.cget("text") == "Cancel download"
    if cancel_after_worker:
        run_worker(gui)
    gui.btn_reset_reinstall.options["command"]()
    assert gui.btn_reset_reinstall.cget("state") == "disabled"
    if not cancel_after_worker:
        run_worker(gui)
    drain(gui)
    assert meta == original
    gui._save_custom_models.assert_not_called()
    gui._on_download_hf_completed.assert_not_called()
    assert not gui._hf_download_events and not gui._hf_download_in_progress
    assert gui.btn_reset_reinstall.cget("state") == "normal"
    assert gui.btn_reset_reinstall.cget("text") == "📥 Grab"


def test_switching_tabs_restores_button_state_while_other_download_is_pending(gui, tmp_path):
    setup_transfer(gui, tmp_path)
    gui._on_download_hf_model()
    event = gui._hf_download_events["custom_chosen"]
    gui.btn_reset_reinstall.options["command"]()
    gui.active_tab_id = "model_1"
    gui._update_model_action_buttons()
    assert event.is_set()
    assert gui.btn_reset_reinstall.cget("state") == "normal"
    assert gui.btn_reset_reinstall.cget("text") != "Cancelling…"


def test_cache_hint_requires_the_exact_chosen_group_not_another_cached_quant(gui, tmp_path):
    meta = selected_entry(tmp_path, inspected=True)
    assert gui._model_is_cached(meta) is False
    first = tmp_path / meta["gguf_files"][0]
    first.parent.mkdir()
    first.write_bytes(b"first shard")
    assert gui._model_is_cached(meta) is False
    (tmp_path / meta["gguf_files"][1]).write_bytes(b"second shard")
    assert gui._model_is_cached(meta) is True
    gui.env["is_model_cached_locally"].assert_not_called()


def test_reset_keeps_files_and_entry_when_unload_fails(gui, tmp_path):
    meta = setup_transfer(gui, tmp_path, inspected=True)
    gui.is_model_loaded = True
    gui.engine.unload_model.return_value = {"status": "error", "error": "still running"}
    gui._on_reset_and_reinstall_single_confirm()
    assert gui.models_config["custom_chosen"] is meta and gui.is_model_loaded
    gui.env["purge_local_model_cache"].assert_not_called()
    assert not gui.workers


def test_reset_preserves_cache_shared_with_a_strata_source(gui, tmp_path):
    setup_transfer(gui, tmp_path, inspected=True)
    gui.models_config["another"] = {"source_kind": "strata_local", "source_repo_id": "owner/chosen"}
    gui._on_reset_and_reinstall_single_confirm()
    gui.env["purge_local_model_cache"].assert_not_called()
    assert gui.workers


def test_reset_and_remove_cannot_delete_a_pending_download(gui, tmp_path):
    meta = setup_transfer(gui, tmp_path)
    gui._hf_download_in_progress = {"custom_chosen"}
    gui._on_reset_and_reinstall_single_confirm()
    gui._remove_hf_fetched_model("custom_chosen")
    gui.env["purge_local_model_cache"].assert_not_called()
    gui.env["messagebox"].askyesno.assert_not_called()
    assert gui.models_config["custom_chosen"] is meta


def test_remove_preserves_shared_repo_cache_and_other_selection(gui, tmp_path):
    setup_transfer(gui, tmp_path, inspected=True)
    gui.models_config["another"] = {"source_repo_id": "owner/chosen", "controller_runtime": "strata"}
    gui.active_tab_id = "model_1"
    gui._remove_hf_fetched_model("custom_chosen")
    assert "custom_chosen" not in gui.models_config and "another" in gui.models_config
    gui.env["purge_local_model_cache"].assert_not_called()
    gui.engine.unload_model.assert_not_called()


def test_exact_native_selection_does_not_attach_an_unrequested_projector(tmp_path):
    from core.controller_runtime import resolve_gguf_artifacts
    from tests.unit.test_model_admission import write_gguf
    model = write_gguf(tmp_path / "chosen.gguf")
    write_gguf(tmp_path / "mmproj-unrelated-1.gguf")
    write_gguf(tmp_path / "mmproj-unrelated-2.gguf")
    selected, projector = resolve_gguf_artifacts(str(tmp_path), {
        "gguf_file": model.name, "gguf_files": [model.name], "mmproj_file": None,
        "mmproj_choices": ["mmproj-unrelated-1.gguf", "mmproj-unrelated-2.gguf"],
    })
    assert selected == str(model) and projector is None
