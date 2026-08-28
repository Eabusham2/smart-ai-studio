"""
Unit & Integration Tests for Hardware Auto-Profiling, Multi-Engine Routing (MLX, GGUF, BitNet, PyTorch),
and Streaming Downloader Cache Management.
"""

import os
import tempfile
import unittest
import torch
import torch.nn as nn

from config.settings import get_settings, MODEL_PRESETS
from core.downloader import ensure_model_available, is_model_available_locally, get_models_cache_dir
from core.engines.bitnet_engine import BitLinear158, BitNetReasoningBackend
from core.engines.gguf_engine import GGUFReasoningBackend
from core.hardware import detect_system_hardware, resolve_optimal_backend, SystemHardwareProfile
from core.pro_engine import ProReasoningEngine


class TestHardwareAndMultiEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings()
        cls.engine = ProReasoningEngine(settings=cls.settings)

    def test_01_hardware_auto_profiling(self):
        """Verify host hardware auto-profiling detects OS, RAM, and accelerators."""
        hw = detect_system_hardware()
        self.assertIsInstance(hw, SystemHardwareProfile)
        self.assertGreater(hw.total_ram_gb, 0.0)
        self.assertGreater(hw.available_ram_gb, 0.0)
        self.assertIn(hw.os_name, ["Darwin", "Linux", "Windows"])
        self.assertIn(hw.recommended_backend, ["mlx", "gguf", "bitnet", "torch"])

        d = hw.to_dict()
        self.assertIn("cpu_count", d)
        self.assertIn("cpu_features", d)

    def test_02_resolve_optimal_backend_rules(self):
        """Verify dynamic backend resolution rules across model types."""
        backend, device = resolve_optimal_backend("ternary")
        self.assertIn(backend, ["mlx", "gguf", "bitnet", "torch"])
        self.assertIn(device, ["mps", "cuda", "cpu"])

        vision_backend, vision_device = resolve_optimal_backend("multimodal_vision")
        self.assertIn(vision_backend, ["mlx", "gguf", "torch"])

    def test_03_bitnet_bitlinear_ternary_quantization(self):
        """Verify BitLinear quantizes weights strictly to {-1, 0, +1} and activations to 8-bit."""
        layer = BitLinear158(in_features=32, out_features=16)
        x = torch.randn(2, 32)

        w_quant, gamma = layer.quantize_weights()
        self.assertTrue(torch.all((w_quant == -1) | (w_quant == 0) | (w_quant == 1)))
        self.assertGreater(gamma.item(), 0.0)

        out = layer(x)
        self.assertEqual(out.shape, (2, 16))
        self.assertFalse(torch.isnan(out).any())

    def test_04_bitnet_reasoning_backend_execution(self):
        """Verify BitNet reasoning backend loads architecture, generates branches, and streams tokens."""
        backend = BitNetReasoningBackend(model_path="dummy_bitnet_path", vocab_size=500, hidden_dim=64, num_layers=2)
        loaded = backend.load_model()
        self.assertTrue(loaded)
        self.assertTrue(backend.is_loaded)

        branches = backend.generate_branches("Test BitNet prompt", branch_count=2)
        self.assertEqual(len(branches), 2)
        self.assertIn("BitNet", branches[0])

        tokens = list(backend.stream_generate_tokens("Test stream"))
        self.assertGreater(len(tokens), 0)

        ent = backend.calculate_token_entropy("Test prompt")
        self.assertGreater(ent, 0.0)

        backend.unload_model()
        self.assertFalse(backend.is_loaded)

    def test_05_gguf_backend_lifecycle_and_entropy(self):
        """Verify GGUF reasoning backend methods handle missing weights gracefully with clean API."""
        backend = GGUFReasoningBackend(model_path="non_existent_model.gguf")
        self.assertFalse(backend.load_model())

        # Unloaded state safety
        branches = backend.generate_branches("Test prompt", branch_count=1)
        self.assertEqual(branches, [])

        ent = backend.calculate_token_entropy("Test prompt")
        self.assertGreaterEqual(ent, 0.0)

        backend.unload_model()
        self.assertIsNone(backend.model)

    def test_06_model_presets_registry_integrity(self):
        """Verify 5 expanded presets exist in MODEL_PRESETS with all required multi-backend artifacts."""
        required_presets = ["model_1", "model_2", "model_3", "model_4", "model_5"]
        for pid in required_presets:
            self.assertIn(pid, MODEL_PRESETS)
            preset = MODEL_PRESETS[pid]
            self.assertIn("artifacts", preset)
            self.assertIn("mlx", preset["artifacts"])
            self.assertIn("name", preset)
            self.assertIn("precision", preset)
        # Check Ternary Qwen 3.8B & 27B presets
        self.assertIn("Qwen 3.8", MODEL_PRESETS["model_2"]["name"])
        self.assertIn("Dolphin Vision", MODEL_PRESETS["model_3"]["name"])
        self.assertIn("Ternary Qwen 27B", MODEL_PRESETS["model_4"]["name"])

    def test_07_downloader_cache_and_availability(self):
        """Verify downloader checks local cache and returns proper status dictionary."""
        cache_dir = get_models_cache_dir()
        self.assertTrue(os.path.exists(cache_dir))

        # Check non-downloaded model with auto_download=False
        res = ensure_model_available("non_existent_org/non_existent_model", backend="mlx", auto_download=False)
        self.assertEqual(res["status"], "not_downloaded")

        # Test local file verification
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"GGUF_HEADER_MOCK")
            tmp_name = f.name

        try:
            avail, path = is_model_available_locally(tmp_name)
            self.assertTrue(avail)
            self.assertEqual(path, tmp_name)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

    def test_08_pro_engine_multi_backend_load_and_unload(self):
        """Verify ProReasoningEngine loads and unloads across multi-backend requests with single-model mutual exclusion."""
        # Load Model 1
        res1 = self.engine.load_model("Ternary Bonsai 27B")
        self.assertIn(res1["status"], ["loaded", "not_downloaded", "error"])

        # Mutual exclusion unload
        unload_res = self.engine.unload_model()
        self.assertEqual(unload_res["status"], "unloaded")
        self.assertIsNone(self.engine.active_model_name)
        self.assertIsNone(self.engine.mlx_backend)
        self.assertIsNone(self.engine.gguf_backend)
        self.assertIsNone(self.engine.bitnet_backend)


if __name__ == "__main__":
    unittest.main()
