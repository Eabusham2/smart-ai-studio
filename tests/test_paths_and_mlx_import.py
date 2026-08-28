"""
Unit and Integration Tests for Portable Paths and MLX Model Folder Importer.
Verifies:
1. Portable data directory resolution in standard and .app modes
2. MLX model folder metadata inspection (config.json, weights, architecture, quantization)
3. Dynamic model registration and persistent custom_models.json storage
"""

import json
import os
import tempfile
import unittest
import tkinter as tk

from config.paths import get_portable_data_dir, get_custom_models_file, inspect_mlx_model_folder
from config.settings import get_settings
from app_gui import SmartAIChatbotApp


class TestPortablePathsAndMLXImport(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        import gc
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_01_portable_data_dir_resolution(self):
        """Verify get_portable_data_dir returns a valid, writable path."""
        pdir = get_portable_data_dir()
        self.assertTrue(os.path.exists(pdir))
        self.assertTrue(os.access(pdir, os.W_OK))

        cm_file = get_custom_models_file()
        self.assertTrue(cm_file.endswith("custom_models.json"))

    def test_02_inspect_valid_mlx_model_folder(self):
        """Verify inspecting an MLX model directory extracts architecture, scale, and quantization."""
        model_dir = os.path.join(self.temp_dir.name, "Qwen2.5-Coder-7B-MLX-4bit")
        os.makedirs(model_dir, exist_ok=True)

        config_data = {
            "model_type": "qwen2",
            "hidden_size": 3584,
            "num_hidden_layers": 28,
            "vocab_size": 152064,
            "max_position_embeddings": 32768,
            "quantization": {
                "bits": 4,
                "group_size": 64
            }
        }
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config_data, f)

        # Create mock safetensors weight file
        with open(os.path.join(model_dir, "model.safetensors"), "w") as f:
            f.write("mock binary safetensors")

        info = inspect_mlx_model_folder(model_dir)
        self.assertTrue(info["valid"])
        self.assertEqual(info["name"], "Qwen2.5-Coder-7B-MLX-4bit")
        self.assertEqual(info["model_type"], "Qwen2")
        self.assertIn("4-bit MLX", info["precision"])
        self.assertEqual(info["context_window"], 32768)
        self.assertTrue(info["has_weights"])

    def test_03_gui_import_mlx_folder_lifecycle(self):
        """Verify importing an MLX folder into the GUI registers it and updates active state."""
        root = tk.Tk()
        root.withdraw()

        settings = get_settings(backend="mock")
        app = SmartAIChatbotApp(root, settings=settings)

        custom_id = "custom_test_mlx_folder"
        app.models_config[custom_id] = {
            "name": "Local Qwen Coder (Custom)",
            "short_name": "Local Qwen Coder",
            "repo_id": None,
            "model_path": "/tmp/mock_mlx_model",
            "precision": "4-bit MLX",
            "raw_params": 7_000_000_000,
            "base_params": "7B",
            "max_context": 32_768,
            "vram": "4.2 GB / 16 GB",
            "tag": "🧩 Local Qwen Coder",
            "accent": "#c084fc"
        }
        app.chat_history[custom_id] = []
        app._on_switch_model_tab(custom_id)

        self.assertEqual(app.active_tab_id, custom_id)
        self.assertIn("Local Qwen Coder", app.lbl_model_status.cget("text"))

        root.destroy()


if __name__ == "__main__":
    unittest.main()
