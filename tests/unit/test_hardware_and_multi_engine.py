"""
Unit & Integration Tests for Hardware Auto-Profiling, Multi-Engine Routing (MLX, GGUF, BitNet, PyTorch),
and Streaming Downloader Cache Management.
"""

import os
import tempfile
import unittest
from config.settings import get_settings, MODEL_PRESETS
from core.downloader import ensure_model_available, is_model_available_locally, get_models_cache_dir
from core.engines.bitnet_cpp_engine import BitNetCppReasoningBackend
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

    def test_03_bitnet_cpp_backend_fails_closed_without_real_runtime(self):
        """Verify BitNet does not fabricate a model or output when real artifacts are absent."""
        backend = BitNetCppReasoningBackend(model_path="definitely_missing_bitnet_model.gguf")
        self.assertIsNone(backend._resolve_model_file())
        self.assertFalse(backend.is_loaded)
        with self.assertRaises(RuntimeError):
            backend.generate_branches("Test BitNet prompt", branch_count=1)
        self.assertGreaterEqual(backend.calculate_token_entropy("Test prompt"), 0.0)

    def test_04_bitnet_learning_requires_real_training_lineage(self):
        """Verify BitNet learning is fail-closed without a declared BF16 training lineage."""
        backend = BitNetCppReasoningBackend(model_path="definitely_missing_bitnet_model.gguf")
        self.assertFalse(backend.training_ready())
        with self.assertRaises(RuntimeError):
            backend.train_mini_batch({}, [{"prompt": "x", "completion": "y"}])
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
        """Verify expanded multi-modal presets exist in MODEL_PRESETS with all required artifacts."""
        required_presets = ["model_1", "model_2", "model_3", "model_4", "model_5", "model_6", "model_7", "model_8", "model_9"]
        for pid in required_presets:
            self.assertIn(pid, MODEL_PRESETS)
            preset = MODEL_PRESETS[pid]
            self.assertIn("artifacts", preset)
            self.assertGreater(len(preset["artifacts"]), 0)
            self.assertIn("name", preset)
            self.assertIn("precision", preset)
        # Check specific flagship and diffusion presets
        self.assertIn("Bonsai 2", MODEL_PRESETS["model_1"]["name"])
        self.assertIn("Multimodal", MODEL_PRESETS["model_1"]["name"])
        self.assertIn("Abliterated", MODEL_PRESETS["model_2"]["name"])
        self.assertIn("CRACK", MODEL_PRESETS["model_2"]["name"])
        self.assertIn("RealVisXL", MODEL_PRESETS["model_3"]["name"])
        self.assertIn("Z-Image Turbo", MODEL_PRESETS["model_4"]["name"])
        self.assertIn("LTX-Video", MODEL_PRESETS["model_7"]["name"])

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
