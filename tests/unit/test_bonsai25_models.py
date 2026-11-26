"""Model-free coverage of the new model selection and actual MTP wiring."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from config.bonsai_models import (
    BONSAI_MLX_REPO, BONSAI_MLX_REVISION, BONSAI_GGUF_REPO, CRACK_GGUF_REPO,
    CRACK_MLX_REPO, GGUF_MTP_HASHES, MLX_MTP_FILE, MMPROJ_SHA256, bonsai_model_info,
)
from core.engines.bonsai25_engine import Bonsai25ReasoningBackend, PUBLISHER_CONTRACT
from core.mlx_mtp import MLXMTP


def test_official_and_crack_presets_select_mtp_in_both_formats():
    for crack, mlx_repo, gguf_repo in ((False, BONSAI_MLX_REPO, BONSAI_GGUF_REPO),
                                       (True, CRACK_MLX_REPO, CRACK_GGUF_REPO)):
        mlx = bonsai_model_info(crack=crack)
        native = bonsai_model_info(crack=crack, is_apple_silicon=False)
        assert mlx["repo_id"] == mlx_repo
        assert native["repo_id"] == gguf_repo
        assert mlx["repo_revision"] and native["repo_revision"]
        assert mlx["mtp_file"] in mlx["download_required_files"]
        assert "runtime/mtp_artifact.py" in mlx["download_required_files"]
        assert native["gguf_file"].endswith("-mtp.gguf")
        assert native["gguf_sha256"] == GGUF_MTP_HASHES[native["gguf_file"]]
        assert native["mmproj_repo_id"] == BONSAI_GGUF_REPO
        assert native["input_modalities"] == ["text", "image", "video"]


def test_settings_match_current_model_defaults_without_mutating_other_presets():
    from config.settings import MODEL_PRESETS, Settings
    settings = Settings(backend="mlx", device="cpu")
    assert settings.mlx_model_path == BONSAI_MLX_REPO
    assert settings.gguf_model_path == BONSAI_GGUF_REPO
    assert MODEL_PRESETS["model_1"]["default_repo_id"] == BONSAI_MLX_REPO
    assert MODEL_PRESETS["model_2"]["default_repo_id"] == CRACK_MLX_REPO
    assert MODEL_PRESETS["model_3"]["default_repo_id"] == "SG161222/RealVisXL_V5.0"


def _mock_mlx_pack(tmp_path, monkeypatch, *, symlink=False):
    runtime = tmp_path / "pack" / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "mtp_artifact.py").write_text(
        "def load_mtp_head(model, sidecar):\n"
        "    return {'target': model, 'sidecar': str(sidecar)}\n")
    head = tmp_path / "pack" / MLX_MTP_FILE
    head.parent.mkdir()
    raw = b"synthetic MTP tensor fixture; no MLX tensors or model execution"
    if symlink:
        blob = tmp_path / "shared-blob"
        blob.write_bytes(raw)
        head.symlink_to(blob)
    else:
        head.write_bytes(raw)
    generation = ModuleType("mlx_vlm.generate")
    generation.generate_step = lambda draft_model=None, draft_kind=None, draft_block_size=None: None
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", generation)
    info = {"mtp_file": MLX_MTP_FILE, "mtp_sha256": hashlib.sha256(raw).hexdigest()}
    return tmp_path / "pack", info


@pytest.mark.parametrize("symlink", [False, True])
def test_verified_sidecar_is_loaded_and_passed_to_native_mtp_api(tmp_path, monkeypatch, symlink):
    pack, info = _mock_mlx_pack(tmp_path, monkeypatch, symlink=symlink)
    target = object()
    mtp = MLXMTP()
    result = mtp.configure(target, pack, info, mode="auto")
    assert result["mode"] == "mtp"
    options = mtp.generation_kwargs()
    assert options["draft_model"]["target"] is target
    assert options["draft_kind"] == "mtp"
    assert options["draft_block_size"] == 2
    mtp.configure(target, pack, info, mode="mtp", max_draft_tokens=3)
    assert mtp.generation_kwargs()["draft_block_size"] == 4
    assert mtp.telemetry()["speedup"] is None
    mtp.configure(target, pack, info, mode="none")
    assert mtp.generation_kwargs() == {}
    assert mtp.head is None


def test_wrong_head_and_old_dflash_cannot_be_enabled(tmp_path, monkeypatch):
    pack, info = _mock_mlx_pack(tmp_path, monkeypatch)
    (pack / MLX_MTP_FILE).write_bytes(b"different head")
    mtp = MLXMTP()
    with pytest.raises(ValueError, match="checksum"):
        mtp.configure(object(), pack, info)
    assert mtp.head is None
    with pytest.raises(ValueError, match="DFlash2"):
        mtp.configure(object(), pack, info, mode="dflash2")
    with pytest.raises(ValueError, match="no verified MTP"):
        mtp.configure(object(), pack, {}, mode="mtp")
    assert mtp.configure(object(), pack, {}, mode="auto")["mode"] == "none"


def test_mlx_rejects_runtime_without_actual_mtp_api(tmp_path, monkeypatch):
    pack, info = _mock_mlx_pack(tmp_path, monkeypatch)
    sys.modules["mlx_vlm.generate"].generate_step = lambda: None
    with pytest.raises(RuntimeError, match="mlx-vlm 0.6.3"):
        MLXMTP().configure(object(), pack, info)


@pytest.mark.parametrize("mode", ["auto", "lookup", "dflash2"])
def test_mlx_without_a_sidecar_ignores_native_only_draft_preferences(mode):
    mtp = MLXMTP()
    result = mtp.configure(object(), None, {}, mode=mode)
    assert result["mode"] == "none"
    assert result["requested_mode"] == mode
    assert mtp.generation_kwargs() == {}


def test_saving_speed_preferences_on_an_unloaded_model_does_not_load_mtp():
    from core.lossless_speed_control import apply_loaded_options
    backend = SimpleNamespace(
        model=None,
        configure_lossless_speedups=lambda **options: options,
        configure_speculation=lambda **options: pytest.fail("Saving preferences must not load a draft head"),
    )
    engine = SimpleNamespace(active_backend="mlx", mlx_backend=backend)
    result = apply_loaded_options(engine, {"speculative_mode": "auto"})
    assert result["speculation"] is None


def test_mlx_speed_configuration_ignores_saved_native_draft_path():
    from core.lossless_speed_control import apply_loaded_options
    configured = []
    def configure(**options):
        configured.append(options)
        return {"mode": "mtp"}
    backend = SimpleNamespace(model=object(), configure_speculation=configure,
                              configure_lossless_speedups=lambda **options: options)
    engine = SimpleNamespace(active_backend="mlx", mlx_backend=backend)
    result = apply_loaded_options(engine, {"speculative_mode": "auto", "max_draft_tokens": 3,
                                          "draft_model_path": "/legacy/dflash.gguf"})
    assert configured == [{"mode": "auto", "max_draft_tokens": 3, "draft_model_path": None}]
    assert result["speculation"]["mode"] == "mtp"


def test_formatted_api_prompt_is_preserved_with_its_output_limit():
    from core.mlx_engine import MLXReasoningBackend
    class Formatted(str):
        _smartai_formatted_chat = True
        _smartai_api_max_tokens = 11
    backend = MLXReasoningBackend(model_info={})
    prompt = Formatted("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
    try:
        assert backend._prepare_vlm_prompt(prompt) is prompt
    finally:
        backend.unload_model()


def _native_backend(tmp_path, monkeypatch, *, crack=False, mode="auto"):
    info = bonsai_model_info(crack=crack, is_apple_silicon=False)
    model, projector = tmp_path / "selected.gguf", tmp_path / "projector.gguf"
    model.write_bytes(b"synthetic target fixture")
    projector.write_bytes(b"synthetic projector fixture")
    def digest(path, cancelled=None):
        return info["gguf_sha256"] if path == str(model) else MMPROJ_SHA256
    monkeypatch.setattr("core.engines.bonsai25_engine.sha256_file", digest)
    monkeypatch.setattr("core.engines.bonsai25_engine.subprocess.run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout="--spec-type draft-mtp --spec-draft-n-max --jinja", stderr=""))
    return Bonsai25ReasoningBackend(str(model), str(tmp_path / "llama-server"),
        model_sha256=info["gguf_sha256"], mmproj_path=str(projector), mode=mode)


@pytest.mark.parametrize("crack", [False, True])
def test_native_target_verified_mtp_launch_preserves_selected_model_projector_and_context(tmp_path, monkeypatch, crack):
    backend = _native_backend(tmp_path, monkeypatch, crack=crack)
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: pytest.fail("preflight must not start a model"))
    assert backend.preflight()["plan"]["runtime_contract"] == PUBLISHER_CONTRACT
    command = backend._build_command(9000)
    assert command[command.index("-m") + 1] == backend.model_path
    assert command[command.index("--mmproj") + 1] == backend.mmproj_path
    assert command[command.index("--spec-type") + 1] == "draft-mtp"
    assert command[command.index("-c") + 1] == "32768"
    assert command[command.index("-np") + 1] == "1"
    assert backend.get_speculative_telemetry()["runtime_contract"] == PUBLISHER_CONTRACT
    assert not backend.get_speculative_telemetry()["output_equivalence_measured"]


def test_native_wrong_model_and_incompatible_drafter_are_rejected(tmp_path, monkeypatch):
    backend = _native_backend(tmp_path, monkeypatch)
    backend._selected_model_sha256 = "incorrect hash"
    with pytest.raises(ValueError, match="checksum"):
        backend.preflight()
    backend.mode = "dflash2"
    with pytest.raises(ValueError, match="not compatible"):
        backend.preflight()


def test_native_off_and_cancelled_request_recovery(tmp_path, monkeypatch):
    backend = _native_backend(tmp_path, monkeypatch, mode="none")
    assert backend.preflight()["plan"]["active_mode"] == "none"
    backend.is_loaded = True
    backend._process = SimpleNamespace(poll=lambda: None)
    backend._cancelled.set()
    backend.prepare_request()
    assert not backend._cancelled.is_set()
    payload = backend._completion_payload("user_data = 1", 16, 0.7, 0.8)
    assert "\nuser" not in payload["stop"]


@pytest.mark.parametrize("bonsai25, mode, enabled", [
    (True, "auto", True), (True, "mtp", True), (True, "none", True),
    (True, "mtp", False), (False, "dflash2", True),
])
@pytest.mark.parametrize("included_runtime", [False, True])
def test_model_load_keeps_legacy_native_paths_out_of_bonsai25(tmp_path, monkeypatch, bonsai25, mode, enabled, included_runtime):
    from core import pro_engine
    from core import bundled_native
    from core.engines.prism_speculative_engine import PrismSpeculativeReasoningBackend
    info = bonsai_model_info(is_apple_silicon=False) if bonsai25 else {"model_type": "text"}
    model_path = str(tmp_path / "model.gguf")
    monkeypatch.setattr(pro_engine, "resolve_gguf_artifacts", lambda *args: (model_path, None))
    monkeypatch.setattr(PrismSpeculativeReasoningBackend, "load_model", lambda self, **kwargs: True)
    monkeypatch.setattr(PrismSpeculativeReasoningBackend, "training_ready", lambda self: False)
    engine = pro_engine.ProReasoningEngine.__new__(pro_engine.ProReasoningEngine)
    engine._model_lock = threading.Lock()
    engine.unload_model = lambda: None
    engine.settings = SimpleNamespace(use_mock=False, auto_download=False, learning_implementation="legacy")
    engine.lossless_speed_options = {
        "enabled": enabled, "speculative_mode": mode, "max_draft_tokens": 3,
        "speculative_server_path": str(tmp_path / "old-server"),
        "draft_model_path": str(tmp_path / "old-dflash.gguf"),
        "speculative_runtime_receipt_path": str(tmp_path / "old-receipt.json"),
    }
    if included_runtime:
        engine.lossless_speed_options.update(speculative_server_path=bundled_native.INCLUDED_RUNTIME,
                                             speculative_runtime_receipt_path=None)
        resolved = {"speculative_server_path": str(tmp_path / "moved app/runtime/native/llama-server"),
                    "speculative_runtime_receipt_path": str(tmp_path / "moved app/runtime/native/llama-server.smartai.json")}
        def discover(*args):
            if bonsai25:
                pytest.fail("Bonsai 2.5 must not inspect the older included runtime")
            return resolved
        monkeypatch.setattr(bundled_native, "bundled_runtime_options", discover)
    engine._context_budget_tokens = 65536
    # Exercise load routing without unrelated memory/tuning lifecycle wrappers.
    load = inspect.unwrap(pro_engine.ProReasoningEngine.load_model)
    result = load(engine, "Selected GGUF", model_path=model_path, backend="gguf", model_info=info)
    assert result["status"] == "loaded"
    backend = engine.gguf_backend
    assert backend.mode == (mode if enabled else "none")
    assert backend.max_draft_tokens == 3
    assert backend.n_ctx == 65536
    if bonsai25:
        assert isinstance(backend, Bonsai25ReasoningBackend)
        assert backend._managed_server is True
        assert backend.draft_model_path is None
        assert backend.runtime_receipt_path is None
    else:
        assert type(backend) is PrismSpeculativeReasoningBackend
        selected = resolved if included_runtime else engine.lossless_speed_options
        assert backend.server_path == selected["speculative_server_path"]
        assert backend.draft_model_path == engine.lossless_speed_options["draft_model_path"]
        assert backend.runtime_receipt_path == selected["speculative_runtime_receipt_path"]


def test_snapshot_lookup_uses_exact_model_and_projector_revisions(monkeypatch, tmp_path):
    from core.controller_runtime import resolve_gguf_artifacts
    info = bonsai_model_info(is_apple_silicon=False)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / info["gguf_file"]).write_bytes(b"fixture")
    (model_dir / info["mmproj_file"]).write_bytes(b"fixture")
    calls = []
    def resolve(identifier, revision=None):
        calls.append((identifier, revision))
        return str(model_dir)
    monkeypatch.setattr("core.controller_runtime.resolve_local_snapshot", resolve)
    resolve_gguf_artifacts(info["repo_id"], info)
    assert calls == [(info["repo_id"], info["repo_revision"]),
                     (info["mmproj_repo_id"], info["mmproj_repo_revision"])]
