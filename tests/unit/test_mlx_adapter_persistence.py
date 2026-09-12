import sys
import types

from core.mlx_adapter_persistence import _infer_rank, install_mlx_adapter_persistence


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape


class FakeModel:
    def __init__(self):
        self.updated = None

    def update(self, tree):
        self.updated = tree

    def parameters(self):
        return {"ok": 1}


class DummyMLX:
    def __init__(self, adapter_path):
        self.adapter_path = str(adapter_path)
        self.model = None
        self.tokenizer = None
        self.is_mlx_available = False
        self.original_saw_adapter_path = "unset"
        self.injected = None

    def load_model(self):
        self.original_saw_adapter_path = self.adapter_path
        self.model = FakeModel()
        self.tokenizer = object()
        self.is_mlx_available = True
        return True

    def inject_lora_adapters(self, r=8, scale=2.0):
        self.injected = (r, scale)
        return {}


install_mlx_adapter_persistence(DummyMLX)


def test_rank_inference_uses_small_lora_dimension():
    assert _infer_rank({"a": FakeTensor((4096, 8)), "b": FakeTensor((8, 4096))}) == 8


def test_raw_safetensors_checkpoint_is_restored_after_base_load(tmp_path, monkeypatch):
    checkpoint = tmp_path / "adapter.safetensors"
    checkpoint.write_bytes(b"placeholder")
    weights = {
        "layer.q_proj.lora_a": FakeTensor((4096, 8)),
        "layer.q_proj.lora_b": FakeTensor((8, 4096)),
    }

    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.__path__ = []
    mx = types.ModuleType("mlx.core")
    mx.load = lambda path: weights
    mx.eval = lambda *args, **kwargs: None
    utils = types.ModuleType("mlx.utils")
    utils.tree_unflatten = lambda items: list(items)
    mlx_pkg.core = mx
    mlx_pkg.utils = utils
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx.utils", utils)

    engine = DummyMLX(checkpoint)
    assert engine.load_model() is True
    assert engine.original_saw_adapter_path is None
    assert engine.adapter_path == str(checkpoint)
    assert engine.injected == (8, 2.0)
    assert engine.last_restored_adapter_path == str(checkpoint)
    assert engine.last_restored_adapter_rank == 8
    assert engine.model.updated is not None
    assert engine.adapters == weights
