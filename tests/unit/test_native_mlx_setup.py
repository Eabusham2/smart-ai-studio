"""Native backend installation and explicit original-checkpoint choices."""
import os
from pathlib import Path
from unittest.mock import Mock
import subprocess
import venv

import pytest

import mlx_setup
import bootstrap_runtime
from config.bonsai_models import cross_platform_mlx_presets, BONSAI_MLX_REPO, CRACK_MLX_REPO


def test_automatic_backend_uses_supported_gpu_and_driver(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert mlx_setup.select_backend(system="Windows", machine="amd64", devices=[]) == "cpu"
    assert mlx_setup.select_backend(system="Linux", machine="x86_64", devices=[(8.9, (570, 20, 0))]) == "cuda12"
    assert mlx_setup.select_backend(system="Windows", machine="amd64", devices=[(12.0, (580, 10, 0))]) == "cuda13"
    assert mlx_setup.select_backend(system="Linux", machine="x86_64", devices=[(6.1, (590, 0, 0))]) == "cpu"
    assert mlx_setup.select_backend(system="Windows", machine="arm64", devices=[(12.1, (590, 0, 0))]) == "cpu"
    with pytest.raises(RuntimeError, match="Windows ARM64"):
        mlx_setup.select_backend("cuda13", system="Windows", machine="arm64")



def test_automatic_backend_honors_hidden_and_selected_devices(monkeypatch):
    devices = [(6.1, (590, 0, 0)), (8.9, (590, 0, 0))]
    def selected():
        return mlx_setup.select_backend(system="Linux", machine="x86_64", devices=devices)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert selected() == "cpu"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert selected() == "cuda13"
    for hidden in ("", "-1", "10", "GPU-unresolved"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", hidden)
        assert selected() == "cpu"

def test_windows_plan_installs_the_backend_distribution_explicitly():
    for backend, package in mlx_setup.BACKENDS.items():
        plan = mlx_setup.packages(backend)
        assert f"{package}==0.32.5" in plan
        assert "mlx==0.32.5" in plan
        assert not any("[cuda" in item or "[cpu" in item for item in plan)


def test_failed_download_does_not_uninstall_existing_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(mlx_setup, "installed_backends", lambda python: {"mlx-cpu": "0.32.5"})
    monkeypatch.setattr(mlx_setup, "select_backend", lambda requested: "cuda12")
    calls = []
    def fail_download(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(mlx_setup.subprocess, "run", fail_download)
    with pytest.raises(subprocess.CalledProcessError):
        mlx_setup.install(tmp_path / "python", "cuda12")
    assert len(calls) == 1 and "download" in calls[0]
    assert all("uninstall" not in command for command in calls)


def test_check_never_installs_and_reuses_locked_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(mlx_setup, "installed_backends", lambda python: {"mlx-cpu": "0.32.5"})
    monkeypatch.setattr(mlx_setup, "select_backend", lambda requested: requested)
    monkeypatch.setattr(mlx_setup.subprocess, "run", lambda *args, **kwargs: pytest.fail("check must not run pip"))
    monkeypatch.setattr(mlx_setup, "probe", lambda python, backend: {"backend": backend, "model_loaded": False})
    assert mlx_setup.install(tmp_path / "python", check_only=True)["backend"] == "cpu"


def test_setup_runs_in_the_selected_virtual_environment(tmp_path, monkeypatch):
    env = tmp_path / "Studio Python"
    venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(env)
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    monkeypatch.setattr(mlx_setup, "select_backend", lambda requested: "cpu")

    def read_prefix(interpreter, backend):
        return subprocess.check_output(
            [interpreter, "-I", "-c", "import sys; print(sys.prefix)"], text=True
        ).strip()

    monkeypatch.setattr(mlx_setup, "probe", read_prefix)
    assert Path(mlx_setup.install(python, "cpu", check_only=True)).resolve() == env.resolve()


def test_failed_native_probe_never_marks_setup_ready(monkeypatch, tmp_path):
    (tmp_path / "requirements.txt").write_text("example==1\n")
    runtime = tmp_path / "runtime"; runtime.mkdir()
    (runtime / ".smartai-deps-ready").write_text("previous installation\n")
    monkeypatch.setattr(bootstrap_runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(bootstrap_runtime.subprocess, "run", Mock())
    monkeypatch.setattr(mlx_setup, "install", Mock(side_effect=RuntimeError("GPU probe failed")))
    with pytest.raises(RuntimeError, match="GPU probe failed"):
        bootstrap_runtime.pip_install(runtime, tmp_path)
    assert not (runtime / ".smartai-deps-ready").exists()


def test_explicit_mlx_presets_keep_pinned_original_weights_and_mtp():
    presets = cross_platform_mlx_presets()
    assert {row["repo_id"] for row in presets.values()} == {BONSAI_MLX_REPO, CRACK_MLX_REPO}
    for row in presets.values():
        assert row["mlx_runtime"] == "native"
        assert row["controller_runtime"] == "mlx_repo_vlm"
        assert row["bonsai25_mtp"] and row["mtp_file"].endswith(".safetensors")
        assert row["gguf_file"] is None
        assert "runtime/vision_artifact.py" in row["download_required_files"]


def test_import_keeps_explicit_tensorfold_choice_and_artifact_identity(monkeypatch):
    monkeypatch.setattr("core.cross_platform_mlx.platform.system", lambda: "Windows")
    from core.model_import_entry import make_imported_model_entry
    meta = {"controller_runtime": "mlx_lm", "task": "text", "repo_id": "owner/original",
            "source_revision": "a" * 40, "input_modalities": ["text"], "max_context": 4096}
    imported = make_imported_model_entry(meta, repo_id="owner/original", runtime_preference="tensorfold")
    assert imported["mlx_runtime"] == "tensorfold" and imported["inference_only"]
    assert imported["repo_id"] == "owner/original" and imported["source_revision"] == "a" * 40
    native = make_imported_model_entry(imported, runtime_preference="native")
    assert native["mlx_runtime"] == "native" and not native.get("inference_only")
    automatic = make_imported_model_entry(imported, runtime_preference="auto")
    assert "mlx_runtime" not in automatic and not automatic.get("inference_only")
    with pytest.raises(ValueError, match="publisher-packed"):
        make_imported_model_entry({**meta, "controller_runtime": "mlx_repo_vlm"},
                                  runtime_preference="tensorfold")


def test_native_cpu_telemetry_does_not_claim_an_available_torch_gpu(monkeypatch):
    import sys
    from types import SimpleNamespace
    from core import cross_platform_mlx
    from core.platform import PlatformRouter
    monkeypatch.setattr(cross_platform_mlx, "native_mlx_installed", lambda: True)
    mx = SimpleNamespace(default_device=lambda: "cpu", gpu="gpu")
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                           get_device_name=lambda index: "AMD GPU (Torch HIP)")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    info = PlatformRouter().get_platform_info()
    assert info["backend"] == "mlx" and info["device"] == "cpu"
    assert info["device_name"] == "CPU (MLX)"
