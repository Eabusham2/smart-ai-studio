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
def test_mac_mlx_failure_retries_only_transformers_checkpoints(tmp_path, monkeypatch, quantization):
    # A neutral directory name makes the actual checkpoint metadata decisive.
    config = {"model_type": "llama"}
    if quantization:
        config["quantization"] = quantization
    (tmp_path / "config.json").write_text(json.dumps(config))
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (tmp_path / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + bytes(4))
    router = PlatformRouter()
    router._cached_platform_info = {"backend": "mlx", "device": "mps", "os_family": "macos",
                                    "torch_available": True, "cuda_available": False, "mps_available": True}
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
        model.to.assert_called_once_with("mps")
        model_factory.from_pretrained.assert_called_once_with(
            str(tmp_path), torch_dtype="bfloat16", device_map=None, trust_remote_code=True)


if __name__ == "__main__":
    unittest.main()
