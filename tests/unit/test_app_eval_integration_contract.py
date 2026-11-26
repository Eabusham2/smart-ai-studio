"""Source contracts for the additive desktop Eval integration."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.unit.test_mlx_adapter_persistence import DummyMLX, FakeModel, restore_fixture

ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


@pytest.fixture
def apple_mlx_platform(monkeypatch):
    from core import cross_platform_mlx, pro_engine
    apple = SimpleNamespace(system=lambda: "Darwin", machine=lambda: "arm64")
    monkeypatch.setattr(pro_engine, "platform", apple)
    monkeypatch.setattr(cross_platform_mlx, "platform", apple)


def test_eval_button_is_text_only_and_cancel_is_confirmed():
    src = _src("core/gui_eval_panel.py")
    assert 'text="Eval"' in src
    assert 'model_type", "text"' in src
    assert "btn.pack_forget()" in src
    assert 'messagebox.askyesno(' in src
    assert '"Cancel Evaluation"' in src
    assert "pause.flag" in src
    assert "CTRL_BREAK_EVENT" in src
    assert "signal.SIGINT" in src


def test_eval_window_has_live_output_and_process_tree_memory_limit():
    src = _src("core/gui_eval_panel.py")
    runner = _src("eval/app_eval_runner.py")
    assert "os.read(fd, 4096)" in src
    assert "getincrementaldecoder" in src
    assert "children(recursive=True)" in src
    assert "memory_limit_gb" in src
    assert "SmartAI-EvalMemoryGuard" in runner
    assert "root.children(recursive=True)" in runner
    assert "memory_limit_enabled" in runner


def test_app_eval_reuses_production_backends_and_fails_closed_on_fake_training():
    bridge = _src("eval/app_cross_platform_bridge.py")
    assert "ProReasoningEngine" in bridge
    assert "self.pro.load_model(" in bridge
    assert "backend.stream_generate_tokens(" in bridge
    assert "backend.generate_branches(" in bridge
    assert "training_ready" in bridge
    assert "train_mini_batch" in bridge
    assert "zero parameter drift" in bridge
    assert "no persisted adapter/model artifact" in bridge
    assert "_backend_context_limit" in bridge
    assert "_clamp_output_to_backend" in bridge


def test_eval_state_is_isolated_but_native_runtime_cache_remains_shared():
    runner = _src("eval/app_eval_runner.py")
    bridge = _src("eval/app_cross_platform_bridge.py")
    gguf = _src("core/engines/gguf_engine.py")
    bitnet = _src("core/engines/bitnet_cpp_engine.py")
    controller = _src("core/controller_runtime.py")

    assert 'os.environ["SMARTAI_DATA_DIR"]' not in runner
    assert "SMARTAI_GGUF_ADAPTER_ROOT" in runner
    assert "SMARTAI_BITNET_TRAINING_ROOT" in runner
    assert "SMARTAI_CONTROLLER_ADAPTER_ROOT" in runner
    assert 'backend_state", "mlx", "adapters.safetensors' in bridge
    assert "SMARTAI_GGUF_ADAPTER_ROOT" in gguf
    assert "SMARTAI_BITNET_TRAINING_ROOT" in bitnet
    assert "SMARTAI_CONTROLLER_ADAPTER_ROOT" in controller


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("implementation", [None, "legacy", "managed", "hybrid"])
def test_mlx_eval_never_reads_or_writes_desktop_weights(tmp_path, monkeypatch, request, resume, implementation,
                                                        apple_mlx_platform):
    from config.settings import Settings
    from core import pro_engine
    from core.learning_state import LearningStateStore, model_identity
    from eval import app_cross_platform_bridge as bridge, phase4_pro_rsi as phase4
    # Let the real eval modules finish importing before substituting the codec.
    request.getfixturevalue("restore_fixture")
    import mlx.core as mx

    monkeypatch.chdir(tmp_path)
    desktop_data = tmp_path / "desktop"
    monkeypatch.setenv("SMARTAI_DATA_DIR", str(desktop_data))
    source = tmp_path / "local-mlx"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"fixed base")
    info = {"name": "Local MLX", "repo_id": "owner/eval-fixture", "model_path": str(source), "controller_runtime": "mlx_lm"}
    identity = model_identity(str(source), info)

    def commit(store, value):
        with store.begin() as candidate:
            (candidate.directory / "adapter.safetensors").write_bytes(value)
            candidate.commit("adapter.safetensors")

    desktop = LearningStateStore(identity, state_root=desktop_data / "learned_memory")
    commit(desktop, b"desktop-weights")
    desktop_manifest = desktop.current_manifest()
    info["legacy_adapter_path"] = desktop.current_checkpoint()
    run = tmp_path / "eval"
    export = run / "backend_state" / "mlx" / "adapters.safetensors"
    state_root = export.parent / "learned_memory"
    if implementation == "legacy":
        state_root /= "legacy-implementation"
    elif implementation == "hybrid":
        state_root /= "hybrid-implementation"
    run_store = LearningStateStore(identity, state_root=state_root)
    if resume:
        commit(run_store, b"eval-resume-weights")
    run_manifest = run_store.current_manifest()
    config_path = tmp_path / "run.json"
    config = {"model_info": info, "run_dir": str(run)}
    if implementation is not None:
        config["learning_implementation"] = implementation
    config_path.write_text(json.dumps(config))
    monkeypatch.setenv(bridge.APP_EVAL_CONFIG_ENV, str(config_path))

    # Use the production Pro loader, MLX restore wrapper and eval export routine.
    # CPU fixtures stand in for the Metal model and its safetensors codec.
    class EvalMLX(DummyMLX):
        def __init__(self, model_path, adapter_path=None, model_info=None):
            super().__init__(tmp_path, adapter_path or "")
            del self.learning_state_root
            self.model_path = model_path
            self.model_info = model_info or {}
            self.model = None
            self.tokenizer = SimpleNamespace()
            self.is_mlx_available = False

        def unload_model(self):
            self.model = None
            self.is_mlx_available = False

    monkeypatch.setattr(pro_engine, "MLXReasoningBackend", EvalMLX)
    monkeypatch.setattr(bridge, "get_settings", lambda: Settings(
        use_mock=False, auto_download=False, backend="mlx", device="cpu"))
    read = []
    decode = mx.load

    def load_weights(path, **kwargs):
        read.append(path)
        return decode(path, **kwargs)

    monkeypatch.setattr(mx, "load", load_weights)
    monkeypatch.setattr(FakeModel, "trainable_parameters", lambda self: {"eval.weight": 1}, raising=False)
    monkeypatch.setattr(phase4, "MLX_AVAILABLE", True)
    monkeypatch.setattr(phase4, "mlx", SimpleNamespace(utils=SimpleNamespace(tree_flatten=dict.items)), raising=False)
    monkeypatch.setattr(phase4, "mx", SimpleNamespace(
        save_safetensors=lambda path, weights: Path(path).write_bytes(b"updated-eval-export")), raising=False)
    # Undo the benchmark bridge's module hooks when this isolated run finishes.
    for name in ("_generate_branches_same_model", "_normalized_entropy", "_pro_backend", "_assert_same_model",
                 "_restore_rsi_adapter", "_run_phase3_consolidation", "RSI_ADAPTER_PATH"):
        monkeypatch.setattr(phase4, name, getattr(phase4, name))

    class Suite:
        _fast_generate = None

    bridge.install(SimpleNamespace(_model_context_limit=lambda engine: None), phase4, Suite)
    suite = Suite()
    try:
        assert suite.engine.settings.mlx_model_path == str(source)
        assert suite.engine.model_info["model_path"] == str(source)
        assert suite.engine.backend.learning_state.root == run_store.root
        assert suite.engine.backend.adapter_path == run_store.current_checkpoint()
        assert read == ([run_store.current_checkpoint()] if resume else [])
        assert phase4.RSI_ADAPTER_PATH == str(export)
        assert phase4._save_rsi_adapter(suite)
        assert export.read_bytes() == b"updated-eval-export"
        assert desktop.current_manifest() == desktop_manifest
        assert run_store.current_manifest() == run_manifest
    finally:
        suite.engine.unload_model()


@pytest.mark.parametrize("crack", [False, True])
@pytest.mark.parametrize("auto_download", [False, True])
def test_pro_load_uses_exact_cached_mlx_path_and_refuses_missing_companions(
        tmp_path, monkeypatch, crack, auto_download, apple_mlx_platform):
    import importlib
    import inspect
    import threading
    import huggingface_hub as hub_api
    from huggingface_hub import constants, file_download
    from config.bonsai_models import bonsai_model_info
    from core import hf_transfer, pro_engine

    # Transfer-fault tests intentionally substitute Hub implementations. Exercise
    # the installed Hub API here and restore every policy alias on teardown.
    snapshots = importlib.import_module("huggingface_hub._snapshot_download")
    for module, names in ((hub_api, ("hf_hub_download", "snapshot_download")),
                          (file_download, ("hf_hub_download",)),
                          (snapshots, ("snapshot_download", "hf_hub_download"))):
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))
    for name in ("_hub", "_original_snapshot", "_original_file"):
        monkeypatch.setattr(hf_transfer, name, None)
    importlib.reload(file_download)
    importlib.reload(snapshots)
    hub_api.hf_hub_download = file_download.hf_hub_download
    hub_api.snapshot_download = snapshots.snapshot_download

    info = bonsai_model_info(crack=crack)
    hub = tmp_path / "shared cache with spaces"
    snapshot = hub / ("models--" + info["repo_id"].replace("/", "--")) / "snapshots" / info["repo_revision"]
    for relative in info["download_required_files"]:
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}" if relative.endswith(".json") else b"tiny selected artifact")
    # A commit-pinned, filtered install intentionally has neither refs/main nor
    # every optional repository file. No real model allocation/network occurs.
    assert not (snapshot.parent.parent / "refs/main").exists()
    assert not (snapshot / "README.md").exists()
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(hub))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    allocated = []

    class TinyMLX:
        def __init__(self, model_path, adapter_path=None, model_info=None):
            self.model_path = model_path
            self.model_info = model_info
            self.model = None
            self.is_mlx_available = True
            allocated.append(self)

        def load_model(self):
            self.model = object()
            return True

    monkeypatch.setattr(pro_engine, "MLXReasoningBackend", TinyMLX)
    engine = pro_engine.ProReasoningEngine.__new__(pro_engine.ProReasoningEngine)
    engine._model_lock = threading.Lock()
    engine.unload_model = lambda: {"status": "unloaded"}
    engine.settings = SimpleNamespace(use_mock=False, auto_download=auto_download, learning_implementation="legacy")
    engine.lora_adapter_path = str(tmp_path / "eval adapter.safetensors")
    load = inspect.unwrap(pro_engine.ProReasoningEngine.load_model)
    result = load(engine, info["name"], model_info=info)
    assert result["status"] == "loaded", result
    assert result["path"] == engine.active_model_path == str(snapshot)
    assert allocated[-1].model_path == str(snapshot)
    assert allocated[-1].model_info["repo_revision"] == info["repo_revision"]
    assert allocated[-1].model_info["mtp_file"] == info["mtp_file"]
    assert allocated[-1].model_info["mtp_sha256"] == info["mtp_sha256"]

    (snapshot / "runtime/mtp_artifact.py").unlink()
    result = load(engine, info["name"], model_info={**info, "path": str(snapshot)})
    assert result["status"] == "not_downloaded", result
    assert len(allocated) == 1, "An incomplete pinned model reached the MLX loader"


def test_canonical_suite_order_and_standalone_packaging_are_preserved():
    suite = _src("master_4000_eval_suite.py")
    build = _src("build_app.py")
    bridge = suite.index("app_cross_platform_bridge.install(master_runtime")
    integrity = suite.index("install_stage_integrity_telemetry(phase4_pro_rsi")
    legacy = suite.index("install_rsi_legacy_training_hardening(phase4_pro_rsi")
    real_data = suite.index("real_benchmark_runtime.install(BenchmarkDatasetProvider")
    context = suite.index("unified_context_budget.install(")
    assert bridge < integrity < legacy < real_data < context
    assert build.count('"eval"') >= 3
    assert '"master_4000_eval_suite.py"' in build
    assert '"run_studio_complete.py"' in build


def test_bonsai25_is_the_default_eval_model_and_current_model_is_reported():
    runtime = _src("run_studio_complete.py")
    report = _src("eval/_master_4000_base.py")
    assert 'get_bundled_model_path(' in runtime
    assert '"prism-ml/Ternary-Bonsai-2.5-27B-mlx-2bit"' in runtime
    assert "_eval_target_model_label" in report
    assert "prism-ml/Ternary-Bonsai-2.5-27B-mlx-2bit" in report


def test_top_app_controls_explicitly_include_context_limit_and_memory_watcher():
    app = _src("app_gui.py")
    context = _src("core/gui_generation_cap.py")
    eval_ui = _src("core/gui_eval_panel.py")

    # One existing app Context control + one existing top-app memory control.
    # Eval gets its own memory watcher but must not clone the normal app controls.
    from core.gui_extensions import EXTENSIONS
    assert app.count("install_gui_extensions(sys.modules[__name__])") == 1
    assert sum(row[2] == "install_gui_generation_cap" for row in EXTENSIONS) == 1
    assert context.count('text="Context Limit"') == 1
    assert app.count('text="Mem Limit"') == 1
    assert 'text="Context Limit"' not in eval_ui
    assert 'text="Mem Limit"' not in eval_ui

    assert "self.memory_limit_var = tk.BooleanVar" in app
    assert "self.entry_memory_limit.pack(" in app
    assert 'text="GiB"' in app
    assert "tk.Toplevel(self.root)" in eval_ui


def test_app_eval_uses_canonical_suite_and_restores_saved_console():
    panel = _src("core/gui_eval_panel.py")
    runner = _src("eval/app_eval_runner.py")

    # The app subprocess imports the canonical suite; there is no copied Eval runner.
    assert "import master_4000_eval_suite as suite" in runner
    assert '"-m", "eval.app_eval_runner"' in panel

    # Raw realtime output is persisted per run and the newest saved console is
    # restored into the Eval window after a relaunch/reopen.
    assert 'run_dir / "live_console.log"' in panel
    assert 'with logfile.open("a", encoding="utf-8")' in panel
    assert 'base.glob("*/live_console.log")' in panel
    assert "max(logs, key=lambda p: p.stat().st_mtime)" in panel
    assert "handle.read(64 * 1024)" in panel
    assert "restored = _restore_saved_eval_log(self)" in panel
