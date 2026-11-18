"""Exercise the production MLX transaction with tiny, deterministic CPU protocol fixtures."""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import threading

import numpy as np
import pytest

from core._mlx_engine_base import MLXReasoningBackend
from core.learning_state import bind_learning_state
from core.learning_data import training_coverage


class Tokenizer:
    def encode(self, s, **kwargs):
        return [ord(c) % 5 for c in s]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "".join(("u" if m["role"] == "user" else "a") + m["content"] for m in messages) + (
            "a" if add_generation_prompt else "")


class Module:
    def named_modules(self):
        yield "", self
        for key, value in vars(self).items():
            if isinstance(value, Module):
                for path, module in value.named_modules():
                    yield key + ("." + path if path else ""), module

    def freeze(self):
        for _, mod in self.named_modules():
            mod._unfrozen = set()

    def unfreeze(self, keys, recurse=False):
        self._unfrozen = set(keys)

    def trainable_parameters(self):
        return {f"{p}.{k}": value for p, mod in self.named_modules()
                for k, value in vars(mod).items() if isinstance(value, np.ndarray)
                and (not hasattr(mod, "_unfrozen") or k in mod._unfrozen)}

    def update(self, tree):
        for key, value in tree.items():
            parent, _, name = key.rpartition(".")
            setattr(dict(self.named_modules())[parent], name, value)

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.train(False)


class PackedProjection(Module):
    def __init__(self):
        self.packed = np.array([0b01001000, 0b01100101], dtype=np.uint32)
        self.calls = 0

    def __call__(self, hidden):
        self.calls += 1
        shifts = 2 * np.arange(5, dtype="uint32")
        trits = ((self.packed[:, None] >> shifts) & 3).astype("float32") - 1
        return hidden @ trits


class Model(Module):
    def __init__(self):
        self.config = {"text_config": {"hidden_size": 2, "vocab_size": 5}}
        self.lm_head = PackedProjection()
        self.embedding = np.array([[1, 0], [1, 1], [-1, 0], [0, -1], [-1, 1]], dtype="float32")
        self.lengths = []
        self.training = False

    def __call__(self, ids):
        self.lengths.append(ids.shape[1])
        return SimpleNamespace(logits=self.lm_head(self.embedding[ids]))


@pytest.fixture
def cpu_protocol(monkeypatch):
    """NumPy reference derivatives and an immutable optimizer exercise backend operations on Linux."""
    mx = ModuleType("mlx.core")
    mx.float32 = np.float32
    for name in ("square", "all", "any", "stack", "isfinite", "zeros", "sum", "where", "array", "array_equal",
                 "sqrt", "exp", "clip", "minimum", "maximum"):
        setattr(mx, name, getattr(np, name))
    mx.random = SimpleNamespace(uniform=lambda lo, hi, shape: np.random.default_rng(7).uniform(lo, hi, shape).astype("float32"))
    mx.stop_gradient = lambda a: a
    mx.eval = lambda *values: None
    # This fixture deliberately uses npz; the conditional Apple test exercises
    # the actual safetensors writer and native autodiff.
    def save(path, weights, metadata=None):
        with open(path, "wb") as handle:
            np.savez(handle, **weights)
    mx.save_safetensors = save
    def load(path):
        with np.load(path) as saved:
            return dict(saved)
    mx.load = load

    nn = ModuleType("mlx.nn")
    nn.Module = Module
    def ce(z, y, reduction):
        z = z - z.max(axis=-1, keepdims=True)
        return np.log(np.exp(z).sum(axis=-1)) - np.take_along_axis(z, y[..., None], axis=-1)[..., 0]
    nn.losses = SimpleNamespace(cross_entropy=ce)
    def value_and_grad(model, loss_fn):
        def calculate(model, *args):
            loss = loss_fn(model, *args)
            grads = {}
            for key, value in model.trainable_parameters().items():
                grad = np.zeros_like(value)
                for index in np.ndindex(value.shape):
                    plus, minus = value.copy(), value.copy()
                    plus[index] += 1e-3
                    minus[index] -= 1e-3
                    model.update({key: plus})
                    a = loss_fn(model, *args)
                    model.update({key: minus})
                    b = loss_fn(model, *args)
                    grad[index] = (a - b) / 0.002
                model.update({key: value})
                grads[key] = grad
            return loss, grads
        return calculate
    nn.value_and_grad = value_and_grad

    class Optimizer:
        def __init__(self, learning_rate, weight_decay):
            self.learning_rate = learning_rate
            self.state = {"step": np.array(0)}

        def init(self, trainable):
            pass

        def update(self, model, grads):
            self.state["step"] = self.state["step"] + 1
            model.update({k: v - self.learning_rate * grads[k] for k, v in model.trainable_parameters().items()})

    utils = ModuleType("mlx.utils")
    utils.tree_flatten, utils.tree_unflatten = lambda t: list(t.items()), dict
    for name, module in (("mlx", ModuleType("mlx")), ("mlx.core", mx), ("mlx.nn", nn),
                         ("mlx.optimizers", SimpleNamespace(AdamW=Optimizer)), ("mlx.utils", utils)):
        monkeypatch.setitem(sys.modules, name, module)
    return mx


def test_mlx_partial_fisher_matches_zero_importance_on_remaining_parameters(cpu_protocol):
    def fresh():
        engine = MLXReasoningBackend("fixture/partial-fisher")
        engine.model, engine.tokenizer = Model(), Tokenizer()
        engine.is_mlx_available = True
        engine._vlm_runtime = "mlx_repo_vlm"
        engine.inject_lora_adapters()
        return engine

    partial, complete = fresh(), fresh()
    key = "lm_head.lora_b"
    weights = partial.adapters
    one = np.ones_like(weights[key])
    reference = {key: weights[key] + .1}
    rows = [{"prompt": "abc", "completion": "answer"}]
    options = dict(steps=2, max_window_tokens=6, learning_rate=.005, lambda_ewc=2.)
    subset, _ = partial.train_mini_batch({}, rows, fisher_matrix={key: one}, reference_weights=reference, **options)
    diagonal = {name: one if name == key else np.zeros_like(value) for name, value in weights.items()}
    full, _ = complete.train_mini_batch({}, rows, fisher_matrix=diagonal, reference_weights={**weights, **reference}, **options)
    assert partial.last_learning_metrics["ewc_used"] and partial.last_learning_metrics["fisher_tensors"] == 1
    for name in weights:
        np.testing.assert_allclose(subset[name], full[name])
    for fisher, refs in (({"unknown": one}, {"unknown": one}), ({key: one}, weights),
                         ({key: one[:-1]}, reference), ({key: -one}, reference),
                         ({key: one * np.nan}, reference), ({key: one}, {key: one * np.inf})):
        with pytest.raises((ValueError, FloatingPointError)):
            partial.train_mini_batch({}, rows, fisher_matrix=fisher, reference_weights=refs, **options)
    assert all(np.array_equal(subset[name], partial.adapters[name]) for name in subset)


def test_explicit_mlx_export_writes_and_rolls_back_alongside_managed_generation(tmp_path, cpu_protocol, monkeypatch):
    from core.learning_state import LearningTransaction, validate_checkpoint

    engine = MLXReasoningBackend("fixture/export")
    engine.model, engine.tokenizer = Model(), Tokenizer()
    engine.is_mlx_available = True
    engine._vlm_runtime = "mlx_repo_vlm"
    store = bind_learning_state(engine, model_info={"resolved_revision": "a" * 40}, state_root=tmp_path / "state")
    external = tmp_path / "public" / "my-adapter.safetensors"
    rows = [{"prompt": "abc", "completion": "answer"}]
    def train(path):
        return engine.train_mini_batch({}, rows, save_path=str(path), steps=1, max_window_tokens=5)

    train(external)
    first = Path(store.current_checkpoint())
    first_bytes = first.read_bytes()
    assert external.read_bytes() == first_bytes and validate_checkpoint(engine, external)
    # Existing evaluation callers pass the live path back to the trainer.
    train(first)
    second = Path(store.current_checkpoint())
    assert second != first and first.read_bytes() == first_bytes
    with pytest.raises(ValueError, match="overwrite managed"):
        train(first)
    assert store.current_checkpoint() == str(second) and first.read_bytes() == first_bytes
    sidecar = Path(str(external) + ".memory.json")
    previous_sidecar = sidecar.read_bytes()
    def refuse(*args, **kwargs):
        raise RuntimeError("another update won")
    monkeypatch.setattr(LearningTransaction, "commit", refuse)
    with pytest.raises(RuntimeError, match="another update won"):
        train(external)
    assert external.read_bytes() == first_bytes and sidecar.read_bytes() == previous_sidecar
    assert store.current_checkpoint() == str(second)
    assert not list(external.parent.glob("tmp*"))


@pytest.mark.parametrize("runtime", ["mlx_repo_vlm", "jang_vlm"])
def test_packed_training_transaction_and_recovery(tmp_path, runtime, cpu_protocol, monkeypatch):
    engine = MLXReasoningBackend("fixture/" + runtime)
    engine.model, engine.tokenizer = Model(), Tokenizer()
    engine.is_mlx_available = True
    engine._vlm_runtime = runtime
    original_head = engine.model.lm_head
    packed = original_head.packed.copy()
    assert engine.model.trainable_parameters()  # Real fresh MLX modules expose foundation trainables.
    store = bind_learning_state(engine, model_info={"resolved_revision": "a" * 40}, state_root=tmp_path)
    row = [{"prompt": "abcdefghi", "completion": "answer"}]
    kwargs = dict(steps=1, max_window_tokens=5, learning_rate=0.05)
    with pytest.raises(RuntimeError, match="rejected"):
        engine.train_mini_batch({}, row, validation_callback=lambda candidate: False, **kwargs)
    assert engine.model.lm_head is original_head and store.current_manifest() is None
    weights, drift = engine.train_mini_batch({}, row, anchor_texts=["abc"], **kwargs)
    assert drift > 0 and np.isfinite(drift)
    assert set(weights) == {"lm_head.lora_a", "lm_head.lora_b"}
    assert engine.model.lm_head.linear is original_head
    np.testing.assert_array_equal(original_head.packed, packed)
    assert max(engine.model.lengths) <= 4
    assert engine.last_learning_metrics["supervised_tokens"] == training_coverage(
        engine.tokenizer, row, max_tokens=5)["supervised_tokens"]
    assert engine.last_learning_metrics["ewc_used"]
    assert engine.last_learning_metrics["adapter_target"] == "language_output_head"
    assert Path(store.current_checkpoint()).exists()
    engine.train_mini_batch({}, row, validation_callback=lambda candidate: True, **kwargs)
    assert engine.last_learning_metrics["optimizer_resumed"] is True
    prior = store.current_manifest()
    prior_weights = {k: v.copy() for k, v in engine.adapters.items()}
    event = threading.Event()
    def stop_after_training(candidate):
        assert any(not np.array_equal(v, prior_weights[k]) for k, v in candidate.model.trainable_parameters().items())
        event.set()
        return True

    for validator, error in ((lambda candidate: False, RuntimeError), (stop_after_training, InterruptedError)):
        with pytest.raises(error):
            engine.train_mini_batch({}, row, validation_callback=validator, cancel_event=event, **kwargs)
        assert store.current_manifest() == prior
        for key, value in engine.adapters.items():
            np.testing.assert_array_equal(value, prior_weights[key])
    event.clear()
    monkeypatch.setattr(cpu_protocol, "save_safetensors", lambda *a, **kw: (_ for _ in ()).throw(OSError("full disk")))
    with pytest.raises(OSError, match="full disk"):
        engine.train_mini_batch({}, row, **kwargs)
    assert store.current_manifest() == prior
    for key, value in engine.adapters.items():
        np.testing.assert_array_equal(value, prior_weights[key])
    monkeypatch.setattr(original_head, "packed", np.array([1, 2], dtype="uint32"))
    engine.model.embedding[0] = np.nan
    with pytest.raises(FloatingPointError):
        engine.train_mini_batch({}, row, **kwargs)
    assert store.current_manifest() == prior


@pytest.mark.parametrize("runtime", ["mlx_repo_vlm", "jang_vlm"])
def test_native_mlx_packed_no_vjp_and_restart(tmp_path, runtime):
    """Runs real MLX autodiff, AdamW and safetensors when MLX is installed."""
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn
    from core.mlx_adapter_persistence import install_mlx_adapter_persistence

    @mx.custom_function
    def project(hidden, packed):
        shifts = 2 * mx.arange(5, dtype=mx.uint32)
        weights = ((packed[:, None] >> shifts) & 3).astype(mx.float32) - 1
        return hidden @ weights

    @project.vjp
    def no_backward(*args):
        raise AssertionError("An inference-only packed kernel received a VJP")

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.packed = mx.array([0b01001000, 0b01100101], dtype=mx.uint32)

        def __call__(self, hidden):
            return project(hidden, self.packed)

    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=2, vocab_size=5)
            self.embedding = mx.array([[1, 0], [1, 1], [-1, 0], [0, -1], [-1, 1]], dtype=mx.float32)
            self.lm_head = Head()

        def __call__(self, ids):
            return SimpleNamespace(logits=self.lm_head(self.embedding[ids]))

    class Backend(MLXReasoningBackend):
        def __init__(self):
            super().__init__("fixture/" + runtime)
            self.model_info = {"resolved_revision": "a" * 40}
            self.learning_state_root = tmp_path
            self._vlm_runtime = runtime

        def load_model(self):
            self.model, self.tokenizer = LanguageModel(), Tokenizer()
            self.is_mlx_available = True
            self.model.eval()
            return True

        def get_training_model(self):
            return self.model

    install_mlx_adapter_persistence(Backend)
    mx.random.seed(7)
    engine = Backend()
    assert engine.load_model()
    row = [{"prompt": "abcdefghi", "completion": "answer"}]
    baseline = engine.model(mx.array([[0, 1, 2]])).logits
    native = engine.model.lm_head
    engine.inject_lora_adapters(r=2)
    assert engine.model.lm_head.linear is native
    assert bool(mx.array_equal(baseline, engine.model(mx.array([[0, 1, 2]])).logits).item())
    learned, drift = engine.train_mini_batch({}, row, max_window_tokens=5, anchor_texts=["abc"], steps=1)
    assert drift > 0
    probe = engine.score_learning_probe("abc", "de")
    restored = Backend()
    assert restored.load_model()
    assert all(mx.array_equal(v, restored.adapters[k]).item() for k, v in learned.items())
    assert restored.score_learning_probe("abc", "de") == pytest.approx(probe)
    restored.train_mini_batch({}, row, max_window_tokens=5, steps=1)
    assert restored.last_learning_metrics["optimizer_resumed"]
    rollouts = [{"prompt": "ab", "completion": text, "rollout_prompt_ids": [1, 1],
                 "rollout_completion_ids": [idx], "reward": reward}
                for text, idx, reward in (("a", 2, 1.), ("b", 3, 0.))]
    before = restored.model.lm_head.linear.packed
    _, policy_drift = restored.train_mini_batch({}, rollouts, max_window_tokens=5, steps=1,
                                                objective="reward_policy", validation_callback=lambda _: True)
    assert policy_drift > 0 and restored.last_learning_metrics["objective"] == "reward_policy"
    assert bool(mx.array_equal(before, restored.model.lm_head.linear.packed).item())
    with pytest.raises(ValueError, match="query projection|output-head"):
        restored.train_mini_batch({}, row, objective="query_only", context_text="abc",
                                  max_window_tokens=5, steps=1, validation_callback=lambda _: True)



def test_legacy_projection_layout_reuses_and_restores_only_saved_targets(cpu_protocol, monkeypatch):
    """Exercise legacy q/v/down with narrow layers and prove alien packed layers fail without replacement."""
    from core.mlx_learning import ensure_adapters
    import mlx.nn as nn

    class Linear(Module):
        def __init__(self):
            self.weight = np.eye(2, dtype="float32")

        def __call__(self, hidden):
            return hidden @ self.weight

    class LoRALinear(Module):
        @classmethod
        def from_base(cls, base, r, scale):
            wrapped = cls()
            wrapped.linear = base
            wrapped.lora_a = np.ones((2, r), dtype="float32")
            wrapped.lora_b = np.zeros((r, 2), dtype="float32")
            wrapped.scale = scale
            return wrapped

    class Decoder(Module):
        def __init__(self):
            self.block = Module()
            self.block.self_attn = Module()
            self.block.self_attn.q_proj = Linear()
            self.block.self_attn.v_proj = Linear()
            self.block.mlp = Module()
            self.block.mlp.down_proj = Linear()

        @property
        def layers(self):
            return (self.block,)

    monkeypatch.setattr(nn, "Linear", Linear, raising=False)
    monkeypatch.setattr(nn, "QuantizedLinear", Linear, raising=False)
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner.lora", SimpleNamespace(LoRALinear=LoRALinear))
    backend = SimpleNamespace(model=Decoder())
    base = backend.model.block.self_attn.q_proj
    trainables = ensure_adapters(backend, rank=3)
    projections = {"block.self_attn.q_proj", "block.self_attn.v_proj", "block.mlp.down_proj"}
    assert set(trainables) == {p + "." + name for p in projections for name in ("lora_a", "lora_b")}
    head = backend.model.block.self_attn.q_proj
    head.lora_b += 1
    assert ensure_adapters(backend)["block.self_attn.q_proj.lora_b"] is head.lora_b
    assert backend.model.block.self_attn.q_proj is head and head.linear is base
    subset = SimpleNamespace(model=Decoder())
    ensure_adapters(subset, rank=3, layout="legacy", target_paths={"block.self_attn.q_proj"})
    assert set(subset.adapters) == {"block.self_attn.q_proj.lora_a", "block.self_attn.q_proj.lora_b"}
    assert isinstance(subset.model.block.self_attn.v_proj, Linear)
    incompatible = SimpleNamespace(model=Decoder(), _vlm_runtime="jang_vlm")
    incompatible.model.block.self_attn.v_proj = PackedProjection()
    prior_down = incompatible.model.block.mlp.down_proj
    with pytest.raises(RuntimeError, match="Unsupported loaded projection"):
        ensure_adapters(incompatible, layout="legacy", target_paths=projections)
    assert incompatible.model.block.mlp.down_proj is prior_down


def test_native_mlx_legacy_projection_update_and_restart(tmp_path):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm.tuner.lora")
    import mlx.nn as nn
    from core.mlx_adapter_persistence import install_mlx_adapter_persistence

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(4, 4, bias=False)
            self.self_attn.v_proj = nn.Linear(4, 4, bias=False)
            self.mlp = nn.Module()
            self.mlp.down_proj = nn.Linear(4, 4, bias=False)

        def __call__(self, x):
            return self.mlp.down_proj(self.self_attn.q_proj(x) + self.self_attn.v_proj(x))

    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(5, 4)
            self.layers = [Block()]
            self.lm_head = nn.Linear(4, 5, bias=False)

        def __call__(self, ids):
            return self.lm_head(self.layers[0](self.embedding(ids)))

    class Engine(MLXReasoningBackend):
        def load_model(self):
            mx.random.seed(11)
            self.model, self.tokenizer = LanguageModel(), Tokenizer()
            self.model.eval()
            self.is_mlx_available = True
            return True

        def get_training_model(self):
            return self.model

    install_mlx_adapter_persistence(Engine)

    def load():
        engine = Engine("fixture/legacy")
        engine.model_info = {"resolved_revision": "a" * 40}
        engine.learning_state_root = tmp_path
        assert engine.load_model()
        return engine

    engine = load()
    row = [{"prompt": "abc", "completion": "answer"}]
    base = engine.model.layers[0].self_attn.q_proj
    before = base.weight
    weights, delta = engine.train_mini_batch({}, row, steps=1)
    assert delta > 0
    assert len(weights) == 6 and set(weights) == set(engine.inject_lora_adapters())
    assert engine.model.layers[0].self_attn.q_proj.linear is base
    assert bool(mx.array_equal(base.weight, before).item())
    reference = engine.score_learning_probe("abc", "answer")
    restarted = load()
    assert restarted.score_learning_probe("abc", "answer") == pytest.approx(reference)
    assert restarted.last_restored_adapter_provenance == "model-bound"


def test_interrupt_after_commit_keeps_live_weights_generation_and_ewc_metrics(tmp_path, cpu_protocol, monkeypatch):
    from core.learning_state import LearningTransaction
    engine = MLXReasoningBackend("fixture/published")
    engine.model, engine.tokenizer = Model(), Tokenizer()
    engine.is_mlx_available = True
    engine._vlm_runtime = "mlx_repo_vlm"
    store = bind_learning_state(engine, model_info={"resolved_revision": "a" * 40}, state_root=tmp_path)
    commit = LearningTransaction.commit
    def signal_after_publish(txn, *args, **kwargs):
        commit(txn, *args, **kwargs)
        raise KeyboardInterrupt()
    monkeypatch.setattr(LearningTransaction, "commit", signal_after_publish)
    with pytest.raises(KeyboardInterrupt):
        engine.train_mini_batch({}, [{"prompt": "abc", "completion": "answer"}], steps=1,
                                anchor_texts=["abc"], max_window_tokens=5)
    checkpoint = store.current_checkpoint()
    manifest = store.current_manifest()
    assert checkpoint == engine.adapter_path == engine._last_managed_checkpoint
    assert engine._mlx_parent_generation == manifest["generation"]
    assert manifest["metadata"]["metrics"] == engine.last_learning_metrics
    assert engine.last_learning_metrics["ewc_used"] and engine.last_learning_metrics["fisher_tensors"] > 0
    for key, value in cpu_protocol.load(checkpoint).items():
        np.testing.assert_array_equal(value, engine.model.trainable_parameters()[key])


@pytest.mark.parametrize("runtime", ["mlx_repo_vlm", "jang_vlm"])
def test_packed_reward_updates_real_head_and_rejects_query(runtime, cpu_protocol, tmp_path):
    engine = MLXReasoningBackend("fixture/" + runtime)
    engine.model, engine.tokenizer = Model(), Tokenizer()
    engine.is_mlx_available = True
    engine._vlm_runtime = runtime
    store = bind_learning_state(engine, model_info={"resolved_revision": "c" * 40}, state_root=tmp_path)
    rows = [{"prompt": "c", "completion": answer, "rollout_prompt_ids": [1, 1], "rollout_completion_ids": [idx],
             "reward": reward} for idx, answer, reward in [(2, "a", 1.0), (3, "b", 0.0)]]
    packed = engine.model.lm_head.packed.copy()
    before = engine.model(np.array([[1, 1]])).logits[0, -1]
    weights, delta = engine.train_mini_batch({}, rows, objective="reward_policy", steps=1, learning_rate=.05,
                                              max_window_tokens=8, validation_callback=lambda _: True)
    after = engine.model(np.array([[1, 1]])).logits[0, -1]
    assert delta > 0 and after[2] > before[2] and after[3] < before[3]
    np.testing.assert_array_equal(packed, engine.model.lm_head.linear.packed)
    receipt = store.current_manifest()
    assert receipt["metadata"]["metrics"]["objective"] == "reward_policy"
    with pytest.raises(ValueError, match="query projection|output-head"):
        engine.train_mini_batch({}, rows, steps=1, objective="query_only", context_text="abcde",
                                validation_callback=lambda _: True)
    with pytest.raises(ValueError, match="Uniform"):
        engine.train_mini_batch({}, [dict(row, reward=0.) for row in rows], objective="reward_policy",
                                steps=1, validation_callback=lambda _: True)
    assert store.current_manifest() == receipt
    for key, value in weights.items():
        np.testing.assert_array_equal(engine.adapters[key], value)


@pytest.mark.parametrize("runtime", ["mlx_repo_vlm", "jang_vlm"])
def test_packed_entropic_policy_with_supplied_reference(tmp_path, runtime, cpu_protocol):
    engine = MLXReasoningBackend("fixture/" + runtime)
    engine.model, engine.tokenizer = Model(), Tokenizer()
    engine.is_mlx_available = True
    engine._vlm_runtime = runtime
    store = bind_learning_state(engine, model_info={"resolved_revision": "e" * 40}, state_root=tmp_path)
    logits = engine.model(np.array([[1, 1]])).logits[0, -1]
    initial = logits - np.log(np.exp(logits).sum())
    rows = [{"prompt": "x", "completion": str(i), "rollout_prompt_ids": [1, 1],
             "rollout_completion_ids": [i], "reference_logprob": float(initial[i]), "reward": reward}
            for i, reward in ((2, .99), (4, .25), (3, 0.))]
    _, delta = engine.train_mini_batch({}, rows, objective="reward_policy", reward_weighting="entropic",
            reference_kl_coef=.1, entropic_kl_target=.2, steps=1, max_window_tokens=8, learning_rate=.03,
            validation_callback=lambda _: True)
    metrics = store.current_manifest()["metadata"]["metrics"]
    assert delta > 0 and metrics["entropic_kl_achieved"] == pytest.approx(.2)
    assert metrics["sampled_initial_log_ratios"] == pytest.approx([0., 0., 0.], abs=2e-6)
    changed = engine.model(np.array([[1, 1]])).logits[0, -1]
    assert changed[2] - changed[3] > logits[2] - logits[3]
