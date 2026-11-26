"""Real import/install GUI callbacks with queued workers, without a Tk display."""
from __future__ import annotations

import ast
from collections import defaultdict
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from core.context_memory import ChatArchive


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
        self.visible = False
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
        self.visible = True

    def pack_forget(self):
        self._ui()
        self.visible = False

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
        "_open_tab_history",
    }
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    tk = SimpleNamespace(**{kind: lambda *a, _kind=kind, **k: widgets.create(_kind, *a, **k)
                            for kind in ("Toplevel", "Frame", "Label", "Entry", "Button", "Radiobutton", "StringVar")})
    env = {
        "os": os, "time": time, "tk": tk, "ChatArchive": ChatArchive,
        "fit_window_to_screen": Mock(),
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
    app.settings = SimpleNamespace(database_path=str(tmp_path / "chat.db"))
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
    assert gui.btn_load_unload.cget("text") == "Cancel download"
    if cancel_after_worker:
        run_worker(gui)
    gui.btn_load_unload.options["command"]()
    assert gui.btn_load_unload.cget("state") == "disabled"
    if not cancel_after_worker:
        run_worker(gui)
    drain(gui)
    assert meta == original
    gui._save_custom_models.assert_not_called()
    gui._on_download_hf_completed.assert_not_called()
    assert not gui._hf_download_events and not gui._hf_download_in_progress
    assert gui.btn_load_unload.cget("state") == "normal"
    assert gui.btn_load_unload.cget("text") == "⬇️ Install"


def test_switching_tabs_restores_button_state_while_other_download_is_pending(gui, tmp_path):
    setup_transfer(gui, tmp_path)
    gui._on_download_hf_model()
    event = gui._hf_download_events["custom_chosen"]
    gui.btn_load_unload.options["command"]()
    gui.active_tab_id = "model_1"
    gui._update_model_action_buttons()
    assert event.is_set()
    assert gui.btn_load_unload.cget("state") == "normal"
    assert gui.btn_load_unload.cget("text") != "Cancelling…"


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


@pytest.mark.parametrize("repo", ["SG161222/RealVisXL_V5.0", "mlx-community/flux2-klein-4b-4bit",
                                  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "mlx-community/stable-audio-3-small-music"])
def test_media_install_changes_to_load_and_stays_cached_after_restart(gui, tmp_path, monkeypatch, repo):
    from config.media_models import media_download_metadata
    info = {"name": "Selected media", "short_name": "Selected media", "repo_id": repo,
            **media_download_metadata(repo)}
    gui.models_config["media"] = info
    gui.active_tab_id = "media"
    snapshot = tmp_path / info["repo_revision"]
    monkeypatch.setattr("core.hf_transfer.snapshot_download", lambda *args, **_selection: str(snapshot))
    gui._update_model_action_buttons()
    assert gui.btn_load_unload.cget("text") == "⬇️ Install"
    assert not gui.btn_reset_reinstall.visible
    def complete(repo_id, **kwargs):
        assert repo_id == repo and kwargs["revision"] == info["repo_revision"]
        assert kwargs["allow_patterns"] == kwargs["required_files"] == info["download_required_files"]
        for name in kwargs["required_files"]:
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith(".index.json"):
                shards = [Path(candidate).name for candidate in kwargs["required_files"]
                          if Path(candidate).parent == Path(name).parent
                          and candidate.endswith((".safetensors", ".bin"))]
                assert shards, "The selected index needs its real shard filenames"
                path.write_text(json.dumps({"weight_map": {f"tensor_{i}": shard
                                                           for i, shard in enumerate(shards)}}))
            else:
                path.write_bytes(b"complete pinned cache fixture")
        return {"status": "success", "local_dir": str(snapshot)}
    gui.env["download_model_from_hf"].side_effect = complete
    gui.btn_load_unload.invoke()
    assert gui.btn_load_unload.cget("text") == "Cancel download"
    assert not gui.btn_reset_reinstall.visible
    run_worker(gui)
    drain(gui)
    assert gui.btn_load_unload.cget("text") == "⚡ Load"
    assert gui.btn_reset_reinstall.visible
    restarted = {"repo_id": repo, **media_download_metadata(repo)}
    assert not restarted.get("path") and gui._model_is_cached(restarted)
    (snapshot / info["download_required_files"][-1]).write_bytes(b"")
    assert not gui._model_is_cached(restarted)


def test_single_install_action_uses_the_responsive_layout_visibility_api(gui):
    gui._responsive_toolbar_set_visible = Mock()
    gui.env["is_model_cached_locally"].return_value = False
    gui.models_config["model_1"] = {"repo_id": "owner/model", "short_name": "Model"}
    gui._update_model_action_buttons()
    gui._responsive_toolbar_set_visible.assert_called_once_with(gui.btn_reset_reinstall, False, before=None)
    assert gui.btn_load_unload.cget("text") == "⬇️ Install"
    gui.is_model_loaded = True
    gui._update_model_action_buttons()
    assert gui.btn_load_unload.cget("text") == "⏏ Unload"


def test_single_action_can_cancel_runtime_setup_then_restores_load(gui):
    gui._model_load_in_progress = True
    gui._model_loading_tab_id = gui.active_tab_id
    gui._model_load_cancel_event = threading.Event()
    gui._update_model_action_buttons()
    assert gui.btn_load_unload.cget("text") == "Cancel setup"
    assert not gui.btn_reset_reinstall.visible
    gui.btn_load_unload.invoke()
    assert gui._model_load_cancel_event.is_set()
    assert gui.btn_load_unload.cget("state") == "disabled"
    gui._model_load_in_progress = False
    gui._update_model_action_buttons()
    assert gui.btn_load_unload.cget("text") == "⚡ Load"
    assert gui.btn_load_unload.cget("state") == "normal"


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


_RESTORED_MEDIA_REPOS = (
    "lesliemore/z-image-turbo-nsfw-v2-GGUF",
    "Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF",
    "freeguyfroverrrr/Wan-2.2-Remix-GGUF",
    "dgrauet/ltx-2.5-mlx-q4",
    "pipenetwork/MiniMax-H3-MLX-4bit",
)


@pytest.fixture
def real_media_graph(tmp_path, monkeypatch, gui):
    """Actual Hub snapshots for a component graph; replace only network I/O."""
    from core import hf_downloader, media_assets
    from config.media_components import RESTORED_MEDIA_MODELS
    # Import at fixture execution to avoid the GUI/Bonsai test module cycle.
    # Reuse its SDK-alias isolation and actual tree/blob/cache implementation.
    from tests.unit.test_bonsai25_download import real_hub_selection
    hub = real_hub_selection.__wrapped__(tmp_path, monkeypatch)
    original_download = hf_downloader.download_model_from_hf
    attempts, transfers, false_success = [], [], {}
    inventories = defaultdict(set)
    for preset in RESTORED_MEDIA_MODELS:
        for part in [preset, *preset["options"].get("download_components", [])]:
            inventories[(part["repo_id"], part["revision"])].update(part["required_files"])

    def download(repo_id, **kwargs):
        attempts.append(repo_id)
        if repo_id in false_success:
            return {"status": "success", "local_dir": false_success[repo_id]}
        hub.prepare({"repo_id": repo_id, "repo_revision": kwargs["revision"],
                     "download_required_files": sorted(inventories[(repo_id, kwargs["revision"])])})
        before = len(hub.downloaded)
        result = original_download(repo_id, **kwargs)
        transfers.extend((repo_id, name) for name in hub.downloaded[before:])
        return result

    monkeypatch.setattr(hf_downloader, "download_model_from_hf", download)
    gui.env["download_model_from_hf"] = download
    # Exercise the original Mac presets on the platform they actually support.
    monkeypatch.setattr(media_assets.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(media_assets.platform, "machine", lambda: "arm64")
    return SimpleNamespace(cache=hub.cache, attempts=attempts, transfers=transfers,
                           false_success=false_success, hub=hub, original_download=original_download)


def _media_selection(gui, repo):
    from config.media_models import media_download_metadata
    info = {"repo_id": repo, **media_download_metadata(repo)}
    gui.models_config["restored_media"] = info
    gui.active_tab_id = "restored_media"
    return info


def _media_snapshot_files(info):
    selections = [(info["repo_id"], Path(info["path"]), info["download_required_files"])]
    selections.extend((part["repo_id"], Path(info["component_paths"][part["role"]]), part["required_files"])
                      for part in info.get("download_components", []))
    return [(repo, directory, name) for repo, directory, names in selections for name in names]


def _cache_file_state(info, exclude=None):
    return {(repo, name): ((directory / name).read_bytes(), (directory / name).stat().st_mtime_ns)
            for repo, directory, name in _media_snapshot_files(info) if (repo, name) != exclude}


@pytest.mark.parametrize("repo", _RESTORED_MEDIA_REPOS)
def test_restored_presets_install_exact_graph_and_restart_with_real_hub_cache(gui, real_media_graph, monkeypatch, repo):
    info = _media_selection(gui, repo)
    gui._update_model_action_buttons()
    assert gui.btn_load_unload.cget("text") == "⬇️ Install"
    gui.btn_load_unload.invoke()
    run_worker(gui)
    drain(gui)
    assert not gui._on_download_hf_failed.called, gui._on_download_hf_failed.call_args
    assert gui.btn_load_unload.cget("text") == "⚡ Load"
    assert gui._on_download_hf_completed.call_count == 1
    graph = _media_snapshot_files(info)
    assert graph and all((directory / name).is_file() for _, directory, name in graph)
    for _, directory, _ in graph:
        assert not (directory / "README.md").exists()
        assert not (directory.parent.parent / "refs" / "main").exists()
    expected_files = {(source, name) for source, _, name in graph}
    assert set(real_media_graph.transfers) <= expected_files
    assert set(real_media_graph.attempts) == {source for source, _, _ in graph}
    before = _cache_file_state(info)
    real_media_graph.attempts.clear()
    real_media_graph.transfers.clear()
    gui._on_download_hf_model()
    run_worker(gui)
    drain(gui)
    assert real_media_graph.transfers == []
    assert not info.get("download_components") or real_media_graph.attempts == []
    assert _cache_file_state(info) == before

    # A new process recreates the catalog without recorded local paths. Actual
    # Hub local-only lookup must find every pinned selection without refs/main.
    child = subprocess.run([sys.executable, "-c", """
import json, sys
from config.media_models import media_download_metadata
from core.media_assets import resolve_media_model_assets
info = {'repo_id': sys.argv[1], **media_download_metadata(sys.argv[1])}
assert not info.get('path') and not info.get('component_paths')
ready = resolve_media_model_assets(info)
assert ready is not None, info['repo_id']
print(json.dumps({'path': ready['path'], 'components': ready['component_paths']}))
""", repo], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=30,
                           env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    assert child.returncode == 0, child.stdout + child.stderr
    report = json.loads(child.stdout.strip().splitlines()[-1])
    assert report == {"path": info["path"], "components": info.get("component_paths", {})}
    assert _cache_file_state(info) == before


@pytest.mark.parametrize("repo", _RESTORED_MEDIA_REPOS)
@pytest.mark.parametrize("empty", [False, True])
def test_incomplete_media_graph_cannot_claim_ready_and_retry_reuses_completed_bytes(
        gui, real_media_graph, repo, empty):
    info = _media_selection(gui, repo)
    gui._on_download_hf_model()
    run_worker(gui)
    drain(gui)
    assert not gui._on_download_hf_failed.called, gui._on_download_hf_failed.call_args
    source, directory, filename = next((source, directory, filename)
                                        for source, directory, filename in reversed(_media_snapshot_files(info))
                                        if filename.endswith((".gguf", ".safetensors")))
    path = directory / filename
    blob = path.resolve()
    before = _cache_file_state(info, exclude=(source, filename))
    blob.write_bytes(b"") if empty else blob.unlink()
    gui._update_model_action_buttons()
    assert gui.btn_load_unload.cget("text") == "⬇️ Install"
    gui._append_ai_message.reset_mock()
    gui._on_download_hf_completed.reset_mock()
    # Even a transfer backend claiming success cannot bypass the same actual
    # local-cache readiness checks used after a normal download.
    real_media_graph.false_success[source] = str(directory)
    gui.btn_load_unload.invoke()
    run_worker(gui)
    drain(gui)
    assert gui.btn_load_unload.cget("text") == "⬇️ Install"
    assert gui._on_download_hf_failed.called
    assert not any("Download Complete" in call.args[0] for call in gui._append_ai_message.call_args_list)
    assert _cache_file_state(info, exclude=(source, filename)) == before

    real_media_graph.false_success.clear()
    real_media_graph.attempts.clear()
    real_media_graph.transfers.clear()
    gui._on_download_hf_failed.reset_mock()
    gui.btn_load_unload.invoke()
    run_worker(gui)
    drain(gui)
    assert not gui._on_download_hf_failed.called, gui._on_download_hf_failed.call_args
    assert gui.btn_load_unload.cget("text") == "⚡ Load"
    assert real_media_graph.attempts == [source]
    assert real_media_graph.transfers == [(source, filename)]
    assert _cache_file_state(info, exclude=(source, filename)) == before


def test_completed_media_graph_is_remembered_after_switching_tabs(gui, real_media_graph):
    info = _media_selection(gui, "freeguyfroverrrr/Wan-2.2-Remix-GGUF")
    gui._on_download_hf_model()
    gui.active_tab_id = "model_1"
    run_worker(gui)
    drain(gui)
    assert info.get("path") and set(info["component_paths"]) == {"t5xxl", "vae"}
    assert gui.active_tab_id == "model_1"
    gui._on_download_hf_completed.assert_not_called()
    gui._on_download_hf_failed.assert_not_called()
    gui.active_tab_id = "restored_media"
    gui._update_model_action_buttons()
    assert gui.btn_load_unload.cget("text") == "⚡ Load"
    assert not gui._hf_download_in_progress and not gui._hf_download_events


@pytest.mark.parametrize("revision", [None, "main"])
def test_empty_blob_repair_uses_resolved_commit_even_for_a_moving_branch(
        gui, real_media_graph, monkeypatch, revision):
    from huggingface_hub import file_download
    info = _media_selection(gui, _RESTORED_MEDIA_REPOS[0])
    gui._on_download_hf_model()
    run_worker(gui)
    drain(gui)
    assert not gui._on_download_hf_failed.called, gui._on_download_hf_failed.call_args
    primary = Path(info["path"]) / info["download_required_files"][0]
    primary.resolve().write_bytes(b"")
    real_media_graph.hub.prepare(info)
    metadata = file_download.get_hf_file_metadata
    urls = []

    def record(url, **kwargs):
        urls.append(url)
        return metadata(url, **kwargs)

    monkeypatch.setattr(file_download, "get_hf_file_metadata", record)
    result = real_media_graph.original_download(
        info["repo_id"], revision=revision, required_files=info["download_required_files"],
        allow_patterns=info["download_allow_patterns"],
    )
    assert result["status"] == "success", result
    assert result["local_dir"] == info["path"] and primary.stat().st_size > 0
    assert urls and "/resolve/" + info["repo_revision"] + "/" in urls[-1]


def test_empty_blob_repair_rejects_snapshot_from_a_different_explicit_commit(tmp_path, monkeypatch):
    from core import hf_downloader, hf_transfer
    wrong = tmp_path / "models--owner--model" / "snapshots" / ("b" * 40)
    wrong.mkdir(parents=True)
    (wrong / "selected.gguf").write_bytes(b"")
    monkeypatch.setattr(hf_transfer, "snapshot_download", lambda **_kwargs: str(wrong))
    result = hf_downloader.download_model_from_hf(
        "owner/model", revision="a" * 40,
        required_files=["selected.gguf"], allow_patterns=["selected.gguf"],
    )
    assert result["status"] == "error" and "does not match" in result["error"]
    assert (wrong / "selected.gguf").stat().st_size == 0
