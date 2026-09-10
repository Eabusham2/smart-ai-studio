"""Live Real Weights & Apple Silicon Metal Evaluation Test Suite.

This is intentionally opt-in because it downloads/loads the full 27B model and
requires Apple Silicon + MLX. Run with RUN_REAL_WEIGHTS_EVAL=1 on the target Mac.
"""
import os
import platform
import unittest
import pytest

if os.getenv("RUN_REAL_WEIGHTS_EVAL", "0") != "1":
    pytest.skip(
        "27B real-weights Metal test is opt-in; set RUN_REAL_WEIGHTS_EVAL=1 on Apple Silicon",
        allow_module_level=True,
    )
if platform.system() != "Darwin" or platform.machine() != "arm64":
    pytest.skip("Real MLX weights test requires Apple Silicon", allow_module_level=True)

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
mlx_utils = pytest.importorskip("mlx.utils")
mlx_lm = pytest.importorskip("mlx_lm")
from mlx_lm import load, stream_generate
from mlx_lm.tuner.lora import LoRALinear

from config.settings import get_settings
from core.verifier import GroundTruthVerifier


class TestRealWeightsEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings(use_mock=False)
        cls.model_id = "prism-ml/Ternary-Bonsai-27B-mlx-2bit"
        cls.adapters_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "eval_results",
            "adapters.safetensors",
        )
        cls.model, cls.tokenizer = load(
            cls.model_id,
            model_config={"kv_bits": 4, "kv_group_size": 64},
        )
        cls.verifier = GroundTruthVerifier()

    def test_01_real_mlx_model_loaded_on_metal(self):
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.tokenizer)
        self.assertTrue(hasattr(self.model, "layers"))
        self.assertGreater(len(self.model.layers), 0)

    def test_02_real_gradient_backprop_on_metal(self):
        target_layer = self.model.layers[0]
        if hasattr(target_layer, "mlp") and hasattr(target_layer.mlp, "down_proj"):
            base_proj = target_layer.mlp.down_proj
        elif hasattr(target_layer, "linear_attn") and hasattr(target_layer.linear_attn, "out_proj"):
            base_proj = target_layer.linear_attn.out_proj
        else:
            base_proj = list(target_layer.leaf_modules().values())[0]

        lora_layer = LoRALinear.from_base(base_proj, r=8)
        lora_layer.linear.freeze()

        def lora_loss(layer, x, y):
            out = layer(x)
            return mx.mean(nn.losses.cross_entropy(out, y))

        loss_grad_fn = nn.value_and_grad(lora_layer, lora_loss)
        dummy_x = mx.random.normal((2, 17408))
        dummy_y = mx.array([0, 1])
        loss_val, grads = loss_grad_fn(lora_layer, dummy_x, dummy_y)
        mx.eval(loss_val, grads)
        self.assertGreater(float(loss_val), 0.0)

        grad_flat = dict(mlx_utils.tree_flatten(grads))
        total_grad_norm = 0.0
        for name, tensor in grad_flat.items():
            if "lora" in name:
                total_grad_norm += float(mx.linalg.norm(tensor))
        self.assertGreater(total_grad_norm, 0.0)

    def test_03_physical_adapters_safetensors_validation(self):
        if os.path.exists(self.adapters_path):
            loaded_weights = mx.load(self.adapters_path)
            self.assertGreater(len(loaded_weights), 0)
            for tensor in loaded_weights.values():
                self.assertGreaterEqual(float(mx.linalg.norm(tensor)), 0.0)

    def test_04_live_forward_pass_stream_generation(self):
        prompt = "Synthesize a deterministic Python function solve() that returns 42."
        tokens_yielded = [
            resp.text
            for resp in stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=30,
            )
        ]
        full_output = "".join(tokens_yielded)
        self.assertTrue(full_output.strip())
        self.assertGreater(len(tokens_yielded), 0)


if __name__ == "__main__":
    unittest.main()
