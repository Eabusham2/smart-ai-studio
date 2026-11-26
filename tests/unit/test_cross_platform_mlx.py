import sys

import pytest

from core import cross_platform_mlx as runtime


def test_mac_keeps_native(monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "arm64")
    monkeypatch.delenv("SMARTAI_MLX_RUNTIME", raising=False)
    assert runtime.select_mlx_runtime("mlx_repo_vlm", {"bonsai25_mtp": True}) == "native"


@pytest.mark.parametrize("system", ["Linux", "Windows"])
def test_native_routes_standard_and_both_bonsai_presets(system, monkeypatch):
    from config.bonsai_models import bonsai_model_info

    monkeypatch.setattr(runtime.platform, "system", lambda: system)
    monkeypatch.setattr(runtime.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(runtime, "native_mlx_error", lambda: None)
    monkeypatch.delenv("SMARTAI_MLX_RUNTIME", raising=False)
    assert runtime.select_mlx_runtime("mlx_lm", {}) == "native"
    for crack in (False, True):
        assert runtime.select_mlx_runtime("mlx_repo_vlm", bonsai_model_info(crack=crack)) == "native"
    with pytest.raises(RuntimeError, match="older JANG"):
        runtime.select_mlx_runtime("jang_vlm", {})


def test_native_reports_load_failure_even_with_tensorfold_installed(monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(runtime, "native_mlx_error", lambda: "missing backend DLL")
    monkeypatch.setattr(runtime, "tensorfold_executable", lambda: "tensorfold-native.exe")
    monkeypatch.delenv("SMARTAI_MLX_RUNTIME", raising=False)
    with pytest.raises(RuntimeError, match="missing backend DLL"):
        runtime.select_mlx_runtime("mlx_lm", {})

    monkeypatch.setenv("SMARTAI_MLX_RUNTIME", "tensorfold")
    assert runtime.select_mlx_runtime("mlx_lm", {}) == "tensorfold"
    with pytest.raises(RuntimeError, match="standard text"):
        runtime.select_mlx_runtime("mlx_repo_vlm", {"bonsai25_mtp": True})
    monkeypatch.setattr(runtime, "tensorfold_executable", lambda: None)
    with pytest.raises(RuntimeError, match="SMARTAI_TENSORFOLD_EXECUTABLE"):
        runtime.select_mlx_runtime("mlx_lm", {})


def test_windows_loader_scopes_resource_to_mlx_utils(tmp_path, monkeypatch):
    import mlx_import_compat as compat

    pkg = tmp_path / "mlx_lm"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from . import utils\n")
    (pkg / "utils.py").write_text("import resource\nresource.setrlimit(resource.RLIMIT_NOFILE, (2048, 4096))\n")
    saved_resource = sys.modules.get("resource")
    for key in ("mlx_lm", "mlx_lm.utils"):
        monkeypatch.delitem(sys.modules, key, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(compat.sys, "platform", "win32")
    calls = []
    monkeypatch.setattr(compat._WindowsFileLimits, "setrlimit", lambda *args: calls.append(args))
    try:
        assert compat.load_mlx_lm().utils.resource is compat._WindowsFileLimits
        assert calls == [(compat._WindowsFileLimits.RLIMIT_NOFILE, (2048, 4096))]
        assert sys.modules.get("resource") is saved_resource
        assert not any(isinstance(entry, compat._WindowsUtilsFinder) for entry in sys.meta_path)
    finally:
        sys.modules.pop("mlx_lm.utils", None)
        sys.modules.pop("mlx_lm", None)
