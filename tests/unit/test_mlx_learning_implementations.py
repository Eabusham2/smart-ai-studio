"""Numerical parity with the historical MLX update, using only local model weights."""
import pytest

from core._mlx_engine_base import MLXReasoningBackend


@pytest.fixture
def tiny_mlx():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    pytest.importorskip("mlx_lm.tuner.lora")

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [ord(ch) % 9 for ch in text]

        def apply_chat_template(self, turns, tokenize=False, add_generation_prompt=False):
            return "".join(turn["content"] for turn in turns) + ("A" if add_generation_prompt else "")

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.down_proj = nn.Linear(4, 4, bias=False)
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(4, 4, bias=False)
            self.self_attn.v_proj = nn.Linear(4, 4, bias=False)
            self.linear_attn = nn.Module()
            self.linear_attn.out_proj = nn.Linear(4, 4, bias=False)

        def __call__(self, x):
            return self.mlp.down_proj(self.self_attn.q_proj(x) + self.self_attn.v_proj(x)) + self.linear_attn.out_proj(x)

    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self._forward_modes = []
            self.embed = nn.Embedding(9, 4)
            self.layers = [Block()]
            self.lm_head = nn.Linear(4, 9, bias=False)

        def __call__(self, ids):
            self._forward_modes.append(self.training)
            return self.lm_head(self.layers[0](self.embed(ids)))

    def make():
        mx.random.seed(78)
        engine = MLXReasoningBackend("fixture/whole-row")
        engine.model = LanguageModel()
        engine.model.eval()
        engine.tokenizer = Tokenizer()
        engine.is_mlx_available = True
        return engine

    return make


def test_original_full_row_adamw_matches_reference_steps_and_export(tiny_mlx, tmp_path):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.lora import LoRALinear

    rows = [{"prompt": "why?", "completion": "yes"}, {"prompt": "how?", "completion": "no"}]
    reference = tiny_mlx()
    model = reference.model
    model.freeze()
    layer = model.layers[0]
    # Original injection order and if/elif attention selection on a hybrid block.
    for parent, name in ((layer.mlp, "down_proj"), (layer.self_attn, "q_proj"), (layer.self_attn, "v_proj")):
        setattr(parent, name, LoRALinear.from_base(getattr(parent, name), r=8, scale=2.0))
        getattr(parent, name).unfreeze(keys=["lora_a", "lora_b"], recurse=False)

    before = dict(tree_flatten(model.trainable_parameters()))
    protected = "layers.0.self_attn.q_proj.lora_b"
    importance = mx.ones_like(before[protected])
    target = before[protected] + 0.05

    def loss(m, ids):
        ce = mx.mean(nn.losses.cross_entropy(m(ids)[:, :-1, :], ids[:, 1:]))
        weight = dict(tree_flatten(m.trainable_parameters()))[protected]
        return ce + mx.sum(importance * (weight - target) ** 2)

    model.train()
    optimizer = optim.AdamW(learning_rate=.002)
    gradient = nn.value_and_grad(model, loss)
    for _ in range(2):
        for row in rows:
            text = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>".format(
                row["prompt"], row["completion"])
            _, grads = gradient(model, mx.array([reference.tokenizer.encode(text)]))
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
    expected = dict(tree_flatten(model.trainable_parameters()))

    backend = tiny_mlx()
    file = tmp_path / "standalone.safetensors"
    weights, drift = backend.train_mini_batch(
        {}, rows, fisher_matrix={protected: importance, "unused": importance},
        reference_weights={protected: target}, lambda_ewc=2., learning_rate=.002, steps=2,
        save_path=str(file))
    assert drift > 0 and backend.last_learning_metrics["optimizer_steps"] == 4
    assert not backend.model.training and backend.model._forward_modes == [True] * 4
    assert backend.last_learning_metrics["learning_mode"] == "legacy"
    assert isinstance(backend.model.layers[0].linear_attn.out_proj, nn.Linear)
    exported = mx.load(str(file))
    assert set(weights) == set(expected) == set(exported)
    for key in weights:
        assert mx.allclose(weights[key], expected[key], rtol=1e-6, atol=1e-7).item(), key
        assert mx.array_equal(weights[key], exported[key]).item()
    assert not list(tmp_path.glob("*.next.safetensors"))


def test_original_fisher_weights_whole_anchors_equally(tiny_mlx):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    backend = tiny_mlx()
    backend.inject_lora_adapters()
    anchors = ["ab", "a longer anchor!"]

    def loss(m, ids):
        return mx.mean(nn.losses.cross_entropy(m(ids)[:, :-1, :], ids[:, 1:]))

    gradient = nn.value_and_grad(backend.model, loss)
    vectors = []
    for text in anchors:
        _, grads = gradient(backend.model, mx.array([backend.tokenizer.encode(text)]))
        vectors.append(dict(tree_flatten(grads)))
    expected = {key: (mx.square(vectors[0][key]) + mx.square(vectors[1][key])) / 2 for key in vectors[0]}
    fisher = backend.compute_mlx_fisher(["", *anchors])
    assert not backend.model.training
    assert fisher.keys() == expected.keys()
    assert any(mx.any(value > 0).item() for value in fisher.values())
    for key, value in fisher.items():
        assert mx.allclose(value, expected[key], rtol=1e-6, atol=1e-7).item()
    for value in backend.compute_mlx_fisher([""]).values():
        assert mx.all(value == 0).item()

    managed = backend.compute_mlx_fisher(anchors, learning_mode="managed", max_window_tokens=3)
    assert any(not mx.allclose(fisher[key], managed[key]).item() for key in fisher)


def test_mlx_mode_routes_plain_extended_and_managed_calls(tiny_mlx):
    backend = tiny_mlx()
    backend.learning_implementation = "legacy"
    rows = [{"prompt": "abcd", "completion": "efgh"}]
    backend.train_mini_batch({}, rows, steps=1, max_window_tokens=4)
    assert backend.last_learning_metrics["learning_mode"] == "managed"
    assert backend.last_learning_metrics["trained_windows"] > 1
    with pytest.raises(ValueError, match="require managed learning"):
        backend.train_mini_batch({}, rows, steps=1, learning_mode="legacy", objective="reward_policy")
    backend.train_mini_batch({}, rows, steps=1, learning_mode="legacy")
    assert backend.last_learning_metrics["trained_windows"] == 1
    assert backend.last_learning_metrics["learning_mode"] == "legacy"
    with pytest.raises(ValueError, match="learning implementation"):
        backend.train_mini_batch({}, rows, learning_mode="unknown")


def test_legacy_generation_keeps_immutable_parent_and_drops_managed_receipts(tiny_mlx, tmp_path, monkeypatch):
    import mlx.core as mx
    from pathlib import Path
    from core.learning_state import bind_learning_state, LearningTransaction

    backend = tiny_mlx()
    backend.learning_implementation = "legacy"
    store = bind_learning_state(backend, model_info={"resolved_revision": "a" * 40}, state_root=tmp_path)
    rows = [{"prompt": "xyz", "completion": "abc"}]

    backend.train_mini_batch({}, rows, learning_mode="managed", steps=1, learning_metadata={"checked": True})
    parent = Path(store.current_checkpoint())
    parent_bytes = parent.read_bytes()
    assert store.current_manifest()["metadata"]["checked"] is True

    weights, drift = backend.train_mini_batch({}, rows, steps=1, save_path=str(parent))
    committed = store.current_manifest()
    assert drift > 0 and parent.read_bytes() == parent_bytes
    assert committed["generation"] == backend._mlx_parent_generation
    assert store.current_checkpoint() == backend.adapter_path and Path(backend.adapter_path) != parent
    assert "checked" not in committed["metadata"]
    assert committed["metadata"]["metrics"]["learning_mode"] == "legacy"
    assert not committed["metadata"]["metrics"]["optimizer_resumed"]
    for key, value in mx.load(backend.adapter_path).items():
        assert mx.array_equal(value, weights[key]).item()

    def fail(*args, **kwargs):
        raise OSError("failed commit")

    export = tmp_path / "export.safetensors"
    export.write_bytes(b"original")
    monkeypatch.setattr(LearningTransaction, "commit", fail)
    with pytest.raises(OSError, match="failed commit"):
        backend.train_mini_batch({}, rows, steps=1, save_path=str(export))
    assert store.current_manifest() == committed
    assert export.read_bytes() == b"original"
    for key, value in weights.items():
        assert mx.array_equal(value, backend.adapters[key]).item()


@pytest.mark.parametrize("destination_mode", ["legacy", "managed"])
def test_mlx_export_cannot_overwrite_another_implementation_generation(tiny_mlx, tmp_path, destination_mode):
    from pathlib import Path
    from core.learning_state import bind_learning_state

    rows = [{"prompt": "xyz", "completion": "abc"}]

    def load(mode):
        backend = tiny_mlx()
        backend.learning_implementation = mode
        bind_learning_state(backend, model_info={"resolved_revision": "b" * 40}, state_root=tmp_path)
        return backend

    owner = load(destination_mode)
    owner.train_mini_batch({}, rows, steps=1)
    destination = Path(owner.adapter_path)
    original = destination.read_bytes()
    other = load("managed" if destination_mode == "legacy" else "legacy")
    assert other.learning_state.root != owner.learning_state.root
    with pytest.raises(ValueError, match="cannot overwrite managed learning state"):
        other.train_mini_batch({}, rows, steps=1, save_path=str(destination))
    assert destination.read_bytes() == original
    assert other.learning_state.current_manifest() is None
    assert not other.adapters


def test_packed_loader_with_legacy_selection_uses_its_native_head(tiny_mlx):
    backend = tiny_mlx()
    backend._vlm_runtime = "jang_vlm"
    backend.model.config = {"hidden_size": 4, "vocab_size": 9}
    assert set(backend.inject_lora_adapters(learning_mode="legacy")) == {"lm_head.lora_a", "lm_head.lora_b"}
    _, drift = backend.train_mini_batch({}, [{"prompt": "x", "completion": "y"}], learning_mode="legacy", steps=1)
    assert drift > 0
    assert backend.last_learning_metrics["learning_mode"] == "managed"
    assert backend.last_learning_metrics["adapter_target"] == "language_output_head"
