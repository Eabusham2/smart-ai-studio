import pytest
from core import cross_platform_mlx as runtime


def test_mac_keeps_native(monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "arm64")
    monkeypatch.delenv("SMARTAI_MLX_RUNTIME", raising=False)
    assert runtime.select_mlx_runtime("mlx_repo_vlm", {"bonsai25_mtp": True}) == "native"


def test_linux_native_and_unported_bonsai(monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime, "native_mlx_installed", lambda: True)
    monkeypatch.delenv("SMARTAI_MLX_RUNTIME", raising=False)
    assert runtime.select_mlx_runtime("mlx_lm", {}) == "native"
    with pytest.raises(RuntimeError, match="Hadamard"):
        runtime.select_mlx_runtime("mlx_repo_vlm", {"bonsai25_mtp": True})


def test_windows_keeps_mlx_weights(monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(runtime, "tensorfold_executable", lambda: "tensorfold-native.exe")
    monkeypatch.delenv("SMARTAI_MLX_RUNTIME", raising=False)
    assert runtime.select_mlx_runtime("mlx_lm", {}) == "tensorfold"
    with pytest.raises(RuntimeError, match="standard text"):
        runtime.select_mlx_runtime("mlx_repo_vlm", {"bonsai25_mtp": True})
    with pytest.raises(RuntimeError, match="Windows uses TensorFold"):
        runtime.select_mlx_runtime("mlx_lm", {"mlx_runtime": "native"})


def test_missing_windows_engine_explained(monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(runtime, "tensorfold_executable", lambda: None)
    monkeypatch.delenv("SMARTAI_MLX_RUNTIME", raising=False)
    with pytest.raises(RuntimeError, match="SMARTAI_TENSORFOLD_EXECUTABLE"):
        runtime.select_mlx_runtime("mlx_lm", {})
