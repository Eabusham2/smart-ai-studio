"""Model-free protection against the two Bonsai 2 loader regressions."""
from dataclasses import dataclass
import sys
from types import ModuleType

import pytest

from core.mlx_repo_config_compat import compatible_qwen35_configs
from core.repository_runtime_imports import repository_runtime_imports


def test_publisher_packed_module_wins_over_unrelated_runtime(tmp_path, monkeypatch):
    (tmp_path / "runtime.py").write_text("class Packed: pass\n")
    (tmp_path / "codec.py").write_text("def transcode(value): return value\n")
    unrelated = ModuleType("runtime")
    unrelated.Packed = object()
    monkeypatch.setitem(sys.modules, "runtime", unrelated)
    with repository_runtime_imports(tmp_path):
        from runtime import Packed
        assert Packed is not unrelated.Packed
        assert Packed.__module__ == "runtime"
    assert sys.modules["runtime"] is unrelated


def test_mlx_vlm_050_nested_dicts_typed_before_model_initialization(monkeypatch):
    @dataclass
    class TextConfig:
        model_type: str
        hidden_size: int = 4

    @dataclass
    class VisionConfig:
        model_type: str = "qwen3_5"

    @dataclass
    class ModelConfig:
        text_config: TextConfig
        vision_config: VisionConfig
        model_type: str = "prism_hadamard_qwen35"

        def __post_init__(self):
            # MLX-VLM's model consumers dereference this during construction.
            if not self.text_config.model_type:
                raise ValueError("missing runtime type")

        @classmethod
        def from_dict(cls, params):
            # Mirrors MLX-VLM 0.5.0: no nested config deserialization.
            return cls(**{k: v for k, v in params.items() if k in cls.__dataclass_fields__})

    model_module = ModuleType("mlx_vlm.models.qwen3_5")
    model_module.TextConfig = TextConfig
    model_module.VisionConfig = VisionConfig
    model_module.ModelConfig = ModelConfig
    models = ModuleType("mlx_vlm.models")
    models.qwen3_5 = model_module
    parent = ModuleType("mlx_vlm")
    parent.models = models
    monkeypatch.setitem(sys.modules, "mlx_vlm", parent)
    monkeypatch.setitem(sys.modules, "mlx_vlm.models", models)
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.qwen3_5", model_module)
    params = {"text_config": {"model_type": "qwen3_5_text", "hidden_size": 4},
              "vision_config": {"model_type": "qwen3_5"}}
    with pytest.raises(AttributeError, match="model_type"):
        ModelConfig.from_dict(params)
    previous = ModelConfig.__dict__["from_dict"]
    with compatible_qwen35_configs():
        config = ModelConfig.from_dict(params)
        assert config.text_config.model_type == "qwen3_5_text"
        assert config.vision_config.model_type == "qwen3_5"
        assert config.model_type == "prism_hadamard_qwen35"
        assert isinstance(config.text_config, TextConfig)
        assert isinstance(config.vision_config, VisionConfig)
    assert ModelConfig.__dict__["from_dict"] is previous
    with pytest.raises(AttributeError, match="model_type"):
        ModelConfig.from_dict(params)


def test_typed_model_config_patch_restores_after_error(monkeypatch):
    @dataclass
    class TextConfig:
        model_type: str

    @dataclass
    class VisionConfig:
        model_type: str = "qwen3_5"

    class ModelConfig:
        @classmethod
        def from_dict(cls, params):
            raise RuntimeError("upstream failure")

    m = ModuleType("mlx_vlm.models.qwen3_5")
    m.TextConfig = TextConfig
    m.VisionConfig = VisionConfig
    m.ModelConfig = ModelConfig
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.qwen3_5", m)
    original = ModelConfig.__dict__["from_dict"]
    with pytest.raises(RuntimeError, match="upstream failure"):
        with compatible_qwen35_configs():
            ModelConfig.from_dict({"text_config": {"model_type": "qwen3_5_text"},
                                   "vision_config": {"model_type": "qwen3_5"}})
    assert ModelConfig.__dict__["from_dict"] is original


def test_qwen35_mtp_verification_uses_original_nonmetal_projections():
    mx = pytest.importorskip("mlx.core")
    if mx.metal.is_available():
        pytest.skip("This test exercises native CPU/CUDA verification")
    nn = pytest.importorskip("mlx.nn")
    language = pytest.importorskip("mlx_vlm.models.qwen3_5.language")
    from core.mlx_repo_config_compat import enable_portable_qwen35_verification

    enable_portable_qwen35_verification()
    x = mx.ones((2, 3, 512), dtype=mx.float16)
    layer = nn.Linear(512, 16, bias=False)
    layer.weight = layer.weight.astype(x.dtype)
    packed = nn.QuantizedLinear.from_linear(layer, group_size=32, bits=4)
    for projection in (layer, packed):
        expected = projection(x)
        actual = language._target_verify_linear(projection, x, True)
        assert mx.allclose(actual, expected, atol=.002, rtol=.002).item()
