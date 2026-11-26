import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core import cross_platform_mlx as runtime


def test_mac_keeps_native(monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "arm64")
    monkeypatch.delenv("SMARTAI_MLX_RUNTIME", raising=False)
    assert runtime.select_mlx_runtime("mlx_repo_vlm", {"bonsai25_mtp": True}) == "native"


@pytest.mark.parametrize("machine,available", [("arm64", True), ("aarch64", False),
                                               ("ARM64", False), ("x86_64", False)])
def test_darwin_keeps_original_native_architecture_admission(monkeypatch, machine, available):
    monkeypatch.setattr(runtime, "platform", SimpleNamespace(system=lambda: "Darwin", machine=lambda: machine))
    assert runtime.native_mlx_platform() is available


def test_mac_loader_keeps_native_import_without_portability_helper(tmp_path, monkeypatch):
    import builtins
    from core import _mlx_engine_base as base
    from core.memory_watchdog import SystemMemoryWatchdog

    monkeypatch.setattr(base, "platform", SimpleNamespace(system=lambda: "Darwin", machine=lambda: "arm64"))
    monkeypatch.setattr(runtime, "native_mlx_platform", Mock(side_effect=AssertionError("portability admission")))
    original_import = builtins.__import__

    def import_without_windows_helper(name, *args, **kwargs):
        if name == "mlx_import_compat":
            raise AssertionError("Mac imports must not depend on the Windows helper")
        return original_import(name, *args, **kwargs)

    mx, nn = SimpleNamespace(), SimpleNamespace()
    model, tokenizer = object(), object()
    lm = SimpleNamespace(load=Mock(return_value=(model, tokenizer)))
    for name, module in {"mlx": SimpleNamespace(core=mx, nn=nn), "mlx.core": mx,
                         "mlx.nn": nn, "mlx_lm": lm}.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(builtins, "__import__", import_without_windows_helper)
    monkeypatch.setattr(SystemMemoryWatchdog, "adjust_dynamic_metal_headroom", Mock())
    backend = base.MLXReasoningBackend(str(tmp_path))
    assert backend.load_model()
    assert (backend.model, backend.tokenizer) == (model, tokenizer)
    lm.load.assert_called_once_with(str(tmp_path), adapter_path=None)


@pytest.mark.parametrize("machine", ["arm64", "x86_64", "aarch64"])
def test_mac_publisher_load_bypasses_portable_runtime_and_projection_changes(tmp_path, monkeypatch, machine):
    import inspect
    from core import mlx_engine as engine, mlx_repo_config_compat as compat
    from core.memory_watchdog import SystemMemoryWatchdog

    monkeypatch.setattr(engine, "platform", SimpleNamespace(system=lambda: "Darwin", machine=lambda: machine))
    monkeypatch.setattr(runtime, "select_mlx_runtime", Mock(side_effect=AssertionError("portable routing")))
    monkeypatch.setattr(runtime, "native_mlx_platform", Mock(side_effect=AssertionError("portable admission")))
    monkeypatch.setattr(compat, "enable_portable_qwen35_verification", Mock(side_effect=AssertionError("portable projections")))
    monkeypatch.setattr(SystemMemoryWatchdog, "adjust_dynamic_metal_headroom", Mock())
    monkeypatch.setenv("SMARTAI_MLX_RUNTIME", "tensorfold")
    loader = SimpleNamespace(load_model=lambda: True, model=object(), processor=object(), tokenizer=object())
    factory = Mock(return_value=loader)
    monkeypatch.setattr(engine, "UniversalControllerBackend", factory)
    backend = SimpleNamespace(
        model_path=str(tmp_path), adapter_path=None,
        model_info={"controller_runtime": "mlx_repo_vlm", "bonsai25_mtp": True},
        invalidate_prompt_reuse=Mock(), configure_speculation=Mock(), _lossless_speedups={"metal_warmup": False},
    )
    resolve = inspect.unwrap(engine.MLXReasoningBackend._resolve_mlx_source)
    backend._resolve_mlx_source = lambda: resolve(backend)
    assert resolve(backend) == (str(tmp_path.absolute()) if machine == "arm64" else None)
    load = inspect.unwrap(engine.MLXReasoningBackend.load_model)
    assert load(backend) is (machine == "arm64")
    if machine == "arm64":
        assert backend.model is loader.model and backend.tokenizer is loader.tokenizer
        factory.assert_called_once()
    else:
        factory.assert_not_called()
        assert backend.last_error == "This MLX checkpoint requires an Apple Silicon Mac and its compatible MLX runtime."


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


def test_metal_hardware_probe_never_allocates_an_array_on_the_calling_thread(monkeypatch):
    from core.platform import PlatformRouter
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "arm64")
    mx = SimpleNamespace(default_device=lambda: "gpu", gpu="gpu",
                         array=Mock(side_effect=AssertionError("unexpected Metal work")))
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace())
    # The Mac package intentionally does not ship the Windows import shim.
    monkeypatch.setitem(sys.modules, "mlx_import_compat", None)
    assert runtime.native_mlx_error() is None
    assert runtime.native_mlx_installed()
    info = PlatformRouter().get_platform_info()
    assert info["mlx_available"] and info["device"] == "mps"
    mx.array.assert_not_called()
