"""
Unit tests for Cross-Platform compatibility (macOS/Apple Silicon, Windows, Linux/Unix).
"""

import json
import os
import platform
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from config.settings import (
    Settings,
    detect_optimal_backend,
    detect_optimal_device,
    detect_system_platform,
    get_settings,
)
from core.platform import PlatformRouter, detect_hardware, get_platform_router
from core.verifier import GroundTruthVerifier


@pytest.mark.parametrize(("machine", "device", "backend"), [
    ("arm64", "mps", "mlx"), ("aarch64", "cpu", "torch"), ("x86_64", "cpu", "torch"),
])
def test_darwin_settings_original_default_device_and_backend(monkeypatch, machine, device, backend):
    from config import settings
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("BACKEND", raising=False)
    monkeypatch.setattr(settings, "platform", SimpleNamespace(system=lambda: "Darwin", machine=lambda: machine))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False), backends=SimpleNamespace()))
    core = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=core))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mlx_import_compat", None)
    assert detect_optimal_device() == device
    assert detect_optimal_backend() == backend


@pytest.mark.parametrize("failure", [ImportError, OSError, RuntimeError])
def test_darwin_settings_preserve_native_import_errors(monkeypatch, failure):
    import builtins
    from config import settings
    monkeypatch.delenv("BACKEND", raising=False)
    monkeypatch.setattr(settings, "platform", SimpleNamespace(system=lambda: "Darwin", machine=lambda: "arm64"))
    original_import = builtins.__import__
    def fail_native_import(name, *args, **kwargs):
        if name == "mlx.core":
            raise failure("simulated native MLX import")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fail_native_import)
    if failure is ImportError:
        assert detect_optimal_backend() == "torch"
    else:
        with pytest.raises(failure, match="simulated native MLX import"):
            detect_optimal_backend()


class TestPlatformCompatibility(unittest.TestCase):
    def test_platform_detection(self):
        detected = detect_system_platform()
        self.assertIn(detected, ["macos", "windows", "linux"])

    def test_device_detection(self):
        device = detect_optimal_device()
        self.assertIn(device, ["cuda", "mps", "cpu"])

    def test_backend_detection(self):
        backend = detect_optimal_backend()
        self.assertIn(backend, ["mlx", "torch", "mock"])

    def test_platform_router_telemetry(self):
        router = get_platform_router()
        info = router.get_platform_info()
        self.assertIn("os_family", info)
        self.assertIn("backend", info)
        self.assertIn("device", info)
        self.assertIn("python_version", info)

    def test_hardware_detection_helper(self):
        hw = detect_hardware()
        self.assertIsInstance(hw, dict)
        self.assertIn(hw["os_family"], ["macos", "windows", "linux"])

    def test_model_loader_mock_and_fallback_safety(self):
        router = PlatformRouter(override_backend="none")
        backend, model, tokenizer = router.route_model_loader("prism-ml/Ternary-Bonsai-27B-mlx-2bit")
        self.assertIn(backend, ["mlx", "torch", "none"])

    def test_live_settings_configuration(self):
        settings = Settings(
            live_mode=True,
            base_model_path="prism-ml/Ternary-Bonsai-27B-mlx-2bit",
            kv_bits=4
        )
        self.assertTrue(settings.live_mode)
        self.assertEqual(settings.kv_bits, 4)
        self.assertEqual(settings.base_model_path, "prism-ml/Ternary-Bonsai-27B-mlx-2bit")

    def test_verifier_strict_resource_bounds(self):
        verifier = GroundTruthVerifier(sandbox_timeout=4.0, max_memory_mb=512)
        self.assertEqual(verifier.sandbox_timeout, 4.0)
        self.assertEqual(verifier.max_memory_mb, 512)

    def test_python_executable_sandbox(self):
        """Verifies sandbox executes reliably with sys.executable across all platforms."""
        verifier = GroundTruthVerifier(sandbox_timeout=3.0, max_memory_mb=512)
        code = "def is_positive(x):\n    return x > 0"
        assertions = "assert is_positive(10) == True\nassert is_positive(-5) == False"
        res = verifier.verify_in_sandbox(code, assertions)
        self.assertTrue(res.passed)
        self.assertEqual(res.verifier_type, "subprocess_sandbox")

    def test_windows_and_posix_path_handling(self):
        settings = get_settings()
        db_path = os.path.normpath(settings.database_path)
        self.assertTrue(len(db_path) > 0)


@pytest.mark.parametrize("quantization", [None, {"bits": 4, "group_size": 64}])
def test_portable_mlx_failure_retries_only_transformers_checkpoints(tmp_path, monkeypatch, quantization):
    # A neutral directory name makes the actual checkpoint metadata decisive.
    config = {"model_type": "llama"}
    if quantization:
        config["quantization"] = quantization
    (tmp_path / "config.json").write_text(json.dumps(config))
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (tmp_path / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + bytes(4))
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    router = PlatformRouter()
    router._cached_platform_info = {"backend": "mlx", "device": "cpu", "os_family": "linux",
                                    "torch_available": True, "cuda_available": False, "mps_available": False}
    mlx_load = Mock(side_effect=ValueError("MLX cannot load this model"))
    monkeypatch.setattr("mlx_import_compat.load_mlx_lm", lambda: SimpleNamespace(load=mlx_load))
    model, tokenizer = Mock(), Mock()
    model.to.return_value = model
    model_factory = Mock(from_pretrained=Mock(return_value=model))
    tokenizer_factory = Mock(from_pretrained=Mock(return_value=tokenizer))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=model_factory, AutoTokenizer=tokenizer_factory))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bfloat16", float32="float32"))
    result = router.route_model_loader(str(tmp_path))
    mlx_load.assert_called_once_with(str(tmp_path))
    if quantization:
        assert result == ("none", None, None)
        assert "MLX cannot load" in router.last_error
        model_factory.from_pretrained.assert_not_called()
        tokenizer_factory.from_pretrained.assert_not_called()
    else:
        assert result == ("torch", model, tokenizer) and not router.last_error
        model.to.assert_called_once_with("cpu")
        model_factory.from_pretrained.assert_called_once_with(
            str(tmp_path), torch_dtype="float32", device_map=None, trust_remote_code=True)


@pytest.mark.parametrize("machine,override,mps,backend,device,name", [
    ("arm64", None, True, "mlx", "mps", "Apple Silicon Metal (MPS)"),
    ("aarch64", "torch", False, "torch", "mps", "CPU"),
    ("x86_64", None, False, "torch", "cpu", "CPU"),
])
def test_darwin_platform_report_preserves_master_contract(monkeypatch, machine, override, mps, backend, device, name):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: machine)
    mx = SimpleNamespace()  # The original Mac report requires imports only.
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))))
    router = PlatformRouter(override)
    assert router.get_platform_info() == {
        "os_family": "macos", "architecture": machine, "backend": backend, "device": device,
        "device_name": name, "mlx_available": machine != "x86_64", "torch_available": True,
        "cuda_available": False, "mps_available": mps, "gpu_count": 0,
        "python_version": sys.version.split()[0], "executable": sys.executable,
    }


def test_darwin_platform_probe_propagates_broken_native_library(monkeypatch):
    import builtins
    original = builtins.__import__
    def broken_import(name, *args, **kwargs):
        if name == "mlx.core":
            raise OSError("broken native library")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(builtins, "__import__", broken_import)
    with pytest.raises(OSError, match="broken native library"):
        PlatformRouter().get_platform_info()


def test_darwin_public_mlx_loader_preserves_original_transformers_fallback(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    router = PlatformRouter()
    router._cached_platform_info = {"backend": "mlx", "device": "mps", "os_family": "macos"}
    mx = SimpleNamespace()
    mlx_load = Mock(side_effect=ValueError("MLX cannot load this checkpoint"))
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(load=mlx_load))
    monkeypatch.setattr("core.platform.run_hf_operation", lambda operation: operation())
    model, tokenizer = Mock(), object()
    model.to.return_value = model
    model_factory = Mock(from_pretrained=Mock(return_value=model))
    tokenizer_factory = Mock(from_pretrained=Mock(return_value=tokenizer))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=model_factory, AutoTokenizer=tokenizer_factory))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bfloat16", float32="float32"))
    source = "owner/original-checkpoint"
    assert router.route_model_loader(source) == ("torch", model, tokenizer)
    mlx_load.assert_called_once_with(source)
    tokenizer_factory.from_pretrained.assert_called_once_with(source, trust_remote_code=True)
    model_factory.from_pretrained.assert_called_once_with(
        source, torch_dtype="bfloat16", device_map=None, trust_remote_code=True)
    model.to.assert_called_once_with("mps")


if __name__ == "__main__":
    unittest.main()
