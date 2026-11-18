"""Real MLX, packed affine matmuls, Hadamard rotation, and hybrid Qwen3.5 decoder blocks."""
from types import SimpleNamespace

import pytest


class Tokens:
    def encode(self, text, **kwargs):
        return [int(s) for s in text.split()]


def packed_decoder(runtime):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    qwen = pytest.importorskip("mlx_lm.models.qwen3_5")
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import ArraysCache, KVCache

    def ternary_arrays(original):
        rows, width = original.weight.shape
        # Sixteen little-endian 2-bit codes per uint32. Codes 0/1/2
        # dequantize exactly to -scale, zero, +scale within each 128-wide group.
        dense = original.weight
        codes = mx.where(dense > .025, 2, mx.where(dense < -.025, 0, 1)).astype(mx.uint32)
        shifts = 2 * mx.arange(16, dtype=mx.uint32)
        packed = mx.sum(codes.reshape(rows, -1, 16) << shifts, axis=-1).astype(mx.uint32)
        scales = mx.full((rows, width//128), .04, dtype=mx.float16)
        return packed, scales, -scales

    def rotate(x, signs):
        return mx.hadamard_transform((x * signs).reshape(-1, 128), scale=128**-.5).reshape(x.shape)

    class PublisherPacked(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.weight, self.scales, self.biases = ternary_arrays(original)
            self.signs = mx.where(mx.arange(original.weight.shape[1]) % 3 == 0, -1., 1.)

        def __call__(self, x):
            return mx.quantized_matmul(rotate(x, self.signs), self.weight, self.scales, self.biases,
                                       group_size=128, bits=2)

    class JangAffine(nn.QuantizedLinear):
        def __init__(self, original):
            rows, width = original.weight.shape
            super().__init__(width, rows, bias=False, group_size=128, bits=2)
            self.weight, self.scales, self.biases = ternary_arrays(original)
            self.signs = mx.where(mx.arange(width) % 5 == 0, -1., 1.)

        def __call__(self, x):
            return super().__call__(rotate(x, self.signs))

    Packed = PublisherPacked if runtime == "mlx_repo_vlm" else JangAffine

    class Block(qwen.DecoderLayer):
        def __call__(self, x, mask=None, cache=None, position_ids=None):
            return super().__call__(x, mask=mask, cache=cache)

    class Language(nn.Module):
        def __init__(self):
            super().__init__()
            self.args = qwen.TextModelArgs(hidden_size=128, intermediate_size=128, vocab_size=128,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                    full_attention_interval=2, linear_num_key_heads=2, linear_num_value_heads=2,
                    # MLX-LM's Metal recurrent kernel assigns Dk / 32 entries per SIMD lane.
                    linear_key_head_dim=64, linear_value_head_dim=64)
            self.model = nn.Module()
            self.model.layers = [Block(self.args, i) for i in range(2)]
            self.model.norm = nn.RMSNorm(128)
            self.model.embed_tokens = nn.Embedding(128, 128)
            self.lm_head = Packed(nn.Linear(128, 128, bias=False))
            last = self.model.layers[-1]
            for parent, names in ((last.self_attn, ("q_proj", "v_proj", "k_proj", "o_proj")),
                                  (last.mlp, ("down_proj", "up_proj", "gate_proj"))):
                for key in names:
                    original = getattr(parent, key)
                    # K/V projection output width is half the input width.
                    setattr(parent, key, Packed(original))

        def make_cache(self):
            return [ArraysCache(size=2), KVCache()]

        def __call__(self, ids, cache=None, position_ids=None, capture_layer_ids=None):
            cache = cache if cache is not None else self.make_cache()
            x = self.model.embed_tokens(ids)
            states = []
            for i, layer in enumerate(self.model.layers):
                x = layer(x, None if layer.is_linear else create_attention_mask(x, cache[i]), cache[i])
                if i in (capture_layer_ids or ()):
                    states.append(x)
            return SimpleNamespace(logits=self.lm_head(self.model.norm(x)), hidden_states=states)

    from core._mlx_engine_base import MLXReasoningBackend
    engine = MLXReasoningBackend("fixture/" + runtime)
    engine.model, engine.is_mlx_available, engine.tokenizer = Language(), True, Tokens()
    engine._vlm_runtime = runtime
    engine.model.eval()
    return engine


@pytest.mark.parametrize("runtime", ["mlx_repo_vlm", "jang_vlm"])
def test_packed_terminal_replay_freezes_prefix_and_quantized_weights(runtime, tmp_path):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from mlx.utils import tree_flatten
    from core.query_adaptation import mlx_query_loss

    mx.random.seed(17)
    engine = packed_decoder(runtime)
    model = engine.model
    base = dict(tree_flatten(model.parameters()))
    ids = mx.array([[1, 5, 6, 4, 3, 6, 2, 7, 1, 9, 8, 7]])
    text = " ".join(map(str, ids.tolist()[0]))
    golden = model(ids).logits
    inputs = model.model.embed_tokens(ids)
    original_q_and_gate = model.model.layers[-1].self_attn.q_proj(inputs)
    weights = engine.inject_lora_adapters()
    assert any("q_proj" in k for k in weights), getattr(engine, "_packed_learning_fallback", "")
    assert all("layers.1." in k for k in weights)
    assert mx.array_equal(model(ids).logits, golden).item()
    loss, spans, _ = mlx_query_loss(model, Tokens(), text, 32, 3, 2, packed=True)
    for start, stop in spans:
        expected = nn.losses.cross_entropy(golden[:, start:stop], ids[:, start+1:stop+1], reduction="mean")
        assert loss(model, start, stop).item() == pytest.approx(expected.item(), abs=2e-4)
    previous = dict(tree_flatten(model.parameters()))
    path = tmp_path / "adapter.safetensors"
    _, drift = engine.train_mini_batch({}, [], steps=2, max_window_tokens=3, context_text=text,
            objective="query_only", save_path=str(path), validation_callback=lambda _: True)
    assert drift > 0
    packed_q = model.model.layers[-1].self_attn.q_proj(inputs)
    gates = lambda value: value.reshape(*value.shape[:-1], 4, 64)[..., 32:]
    assert mx.array_equal(gates(original_q_and_gate), gates(packed_q)).item()
    assert not mx.array_equal(original_q_and_gate, packed_q).item()
    after = dict(tree_flatten(model.parameters()))
    changed = [k for k, v in previous.items() if not mx.array_equal(v, after[k]).item()]
    assert changed and all("q_proj.lora_" in k for k in changed)
    assert engine.last_learning_metrics["query_replay"] == "terminal_attention_with_frozen_hybrid_prefix"
    for k, v in base.items():
        original = k if k in after else k.rsplit(".", 1)[0] + ".linear." + k.rsplit(".", 1)[1]
        assert v.dtype == after[original].dtype and mx.array_equal(v, after[original]).item()
    assert any(v.dtype == mx.uint32 for v in base.values())
    saved, info = mx.load(str(path), return_metadata=True)
    assert info["smartai_lora_layout"] == "packed_terminal"
    assert set(saved) == set(weights)
    # Exercise both internal training factors and an answer-token loss as well.
    engine.train_mini_batch({}, [{"prompt": "1 2", "completion": "4 3"}], steps=1,
                            validation_callback=lambda _: True)
    assert any(not mx.array_equal(after[k], v).item() for k, v in tree_flatten(model.parameters())
               if "down_proj.lora_" in k)


def test_packed_hot_scope_matches_cold_restore_and_rolls_back_bad_candidate(tmp_path):
    mx = pytest.importorskip("mlx.core")
    from core._mlx_engine_base import MLXReasoningBackend
    from core.memory_scopes import LearningScope
    from core.mlx_adapter_persistence import install_mlx_adapter_persistence, switch_learning_checkpoint

    source = tmp_path / "model"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"qwen3_5"}')
    (source / "model.safetensors").write_bytes(b"deterministic fixture")
    (source / "tokenizer.json").write_text("{}")

    class Engine(MLXReasoningBackend):
        _vlm_runtime = "mlx_repo_vlm"
        model_info = {}
        learning_state_root = tmp_path / "memory"

        def _resolve_mlx_source(self):
            return self.model_path

        def get_training_model(self):
            return self.model

        def invalidate_prompt_reuse(self, reason):
            self.cached_prefix = None
            self.last_invalidation = reason

        def load_model(self):
            mx.random.seed(81)
            fixture = packed_decoder(self._vlm_runtime)
            self.model, self.tokenizer, self.is_mlx_available = fixture.model, fixture.tokenizer, True
            return True

    install_mlx_adapter_persistence(Engine)
    engine = Engine(str(source))
    assert engine.load_model()
    resident = engine.model
    tokens = mx.array([[1, 7, 9]])
    initial = resident(tokens).logits
    rows = [{"prompt": "1 7", "completion": "9"}]
    exported = tmp_path / "exported.safetensors"
    engine.train_mini_batch({}, rows, steps=2, learning_rate=.001, save_path=str(exported), validation_callback=lambda _: True)
    global_scope, checkpoint = engine.learning_scope, engine.adapter_path
    raw, info = mx.load(str(exported), return_metadata=True)
    assert info["smartai_identity"] == engine.learning_state.identity_digest
    assert raw.keys() == engine.adapters.keys() and all(mx.array_equal(value, raw[key]).item() for key, value in engine.adapters.items())
    committed = resident(tokens).logits
    assert not mx.array_equal(initial, committed).item()
    empty = LearningScope("session", "other")
    engine.cached_prefix = object()
    assert switch_learning_checkpoint(engine, empty)
    assert engine.model is resident and engine.cached_prefix is None
    assert engine.adapter_path is None and mx.array_equal(initial, resident(tokens).logits).item()
    assert switch_learning_checkpoint(engine, global_scope)
    assert engine.adapter_path == checkpoint and mx.array_equal(committed, resident(tokens).logits).item()
    cold = Engine(str(source))
    assert cold.load_model() and mx.array_equal(committed, cold.model(tokens).logits).item()
    # Valid manifest and identity, but the restored tensor shape is inconsistent.
    store = engine.learning_state.for_scope(empty)
    bad, info = mx.load(checkpoint, return_metadata=True)
    key = next(k for k in bad if k.endswith("lora_a"))
    bad[key] = bad[key][:-1]
    with store.begin() as txn:
        mx.save_safetensors(str(txn.directory / "adapters.safetensors"), bad, metadata=info)
        txn.commit("adapters.safetensors")
    with pytest.raises(ValueError, match="shape"):
        switch_learning_checkpoint(engine, empty)
    assert engine.learning_scope == global_scope and engine.adapter_path == checkpoint
    assert mx.array_equal(committed, resident(tokens).logits).item()
    engine.train_mini_batch({}, rows, steps=1, validation_callback=lambda _: True)
    assert engine.last_learning_metrics["optimizer_resumed"]


def test_text_training_isolates_vlm_rope_state_on_success_and_rejection(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from core._mlx_engine_base import MLXReasoningBackend as Base
    from core.mlx_engine import MLXReasoningBackend

    ids, deltas = mx.array([[[71, 84]]]), mx.array([[69]])
    language = SimpleNamespace(_position_ids=ids, _rope_deltas=deltas)
    wrapper = SimpleNamespace(language_model=language)
    # The production method calls super() on the backend even though this test
    # only exercises the private VLM view, without allocating a second model.
    engine = object.__new__(MLXReasoningBackend)
    engine.model, engine._vlm_runtime = wrapper, "mlx_repo_vlm"
    monkeypatch.setattr(engine, "invalidate_prompt_reuse", lambda _: None)
    def inner(self, *args, **kwargs):
        assert self.model is language
        assert language._position_ids is None and language._rope_deltas is None
        language._position_ids, language._rope_deltas = mx.array([1]), mx.array([0])
        return kwargs["validation_callback"](self)
    monkeypatch.setattr(Base, "train_mini_batch", inner)
    def validate(backend):
        assert backend.model is wrapper
        return True
    assert engine._run_training_method("train_mini_batch", validation_callback=validate)
    assert engine.model is wrapper and language._position_ids is ids and language._rope_deltas is deltas
    def reject(backend):
        assert backend.model is wrapper
        raise InterruptedError()
    with pytest.raises(InterruptedError):
        engine._run_training_method("train_mini_batch", validation_callback=reject)
    assert engine.model is wrapper and language._position_ids is ids and language._rope_deltas is deltas
