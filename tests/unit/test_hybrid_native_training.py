"""Hybrid native training uses one optimizer and validates the actual serving candidate."""
import copy
from pathlib import Path
import threading

import pytest

from tests.unit.test_native_context_learning import tiny_controller


def load_hybrid(backend):
    backend.unload_model()
    backend.learning_implementation = "hybrid"
    assert backend.load_model()
    backend.tokenizer.chat_template = ("{% for message in messages %}{{ message.content }} {% endfor %}"
                                      "{% if not add_generation_prompt %}red{% endif %}")
    return backend


def test_hybrid_original_updates_match_and_validation_controls_publication(tiny_controller):
    import torch
    from core.legacy_torch_learning import LegacyControllerTrainer, update_legacy_lora

    backend = load_hybrid(tiny_controller)
    model = backend._ensure_transformers_lora()
    reference = copy.deepcopy(model)
    rows = [{"prompt": "vault color", "completion": "slate"}, {"prompt": "the vault", "completion": "red"}]
    rng = torch.random.get_rng_state()
    expected_metrics, expected_drift = update_legacy_lora(
        reference, backend.tokenizer, rows, trainer=LegacyControllerTrainer,
        learning_rate=.003, steps=2, max_length=16, output_directory=None)
    torch.random.set_rng_state(rng)
    calls = []

    def validate(candidate):
        calls.append(candidate.score_learning_probe("vault color", "slate"))
        assert all(torch.equal(candidate.model.get_parameter(k), p) for k, p in reference.named_parameters())
        return {"accepted": True, "score": calls[-1]}

    metrics, drift = backend.train_mini_batch({}, rows, learning_rate=.003, steps=2, max_window_tokens=16,
                                              learning_metadata={"witness_id": "one"}, validation_callback=validate)
    assert calls and len(calls) == 1
    assert metrics["learning_mode"] == "hybrid" and metrics["training_algorithm"] == "original_completion_adamw"
    assert metrics["optimizer_steps"] == expected_metrics["optimizer_steps"] == 4
    assert metrics["optimizer_resumed"] is False and drift == expected_drift
    manifest = backend.learning_state.current_manifest()
    assert manifest["metadata"]["validation"]["score"] == calls[0]
    assert manifest["metadata"]["metrics"] == metrics
    assert backend.adapter_path == backend.learning_state.current_checkpoint()
    for reason in ("reject", "cancel"):
        stop = threading.Event()
        def reject(candidate):
            assert any(not torch.equal(candidate.model.get_parameter(k), p) for k, p in reference.named_parameters())
            if reason == "cancel":
                stop.set()
                return True
            return False
        with pytest.raises(RuntimeError, match="cancelled|validation"):
            backend.train_mini_batch({}, rows, steps=1, cancel_event=stop, validation_callback=reject)
        assert backend.learning_state.current_manifest()["generation"] == manifest["generation"]
        assert all(torch.equal(backend.model.get_parameter(k), p) for k, p in reference.named_parameters())
    newer, _ = backend.train_mini_batch({}, rows, steps=1, validation_callback=lambda _: True)
    assert not newer["optimizer_resumed"] and newer["training_algorithm"] == "original_completion_adamw"
    assert backend.learning_state.current_manifest()["metadata"]["witness_id"] == "one"
    current = backend.adapter_path
    for mode in ("legacy", "managed"):
        with pytest.raises(ValueError, match="hybrid checkpoint namespace"):
            backend.train_mini_batch({}, rows, learning_mode=mode)
    assert backend.learning_state.current_checkpoint() == current


def test_hybrid_short_completion_honors_fisher_penalty_on_each_original_step(tiny_controller, monkeypatch):
    import torch
    from core import torch_context_learning as learning

    backend = load_hybrid(tiny_controller)
    model = backend._ensure_transformers_lora()
    named = learning.lora_parameters(model)
    fisher = {k: torch.ones_like(v) for k, v in named.items()}
    penalties = []
    original = learning.backward_ewc
    def observe(named, before, diagonal, strength):
        previous = {k: p.grad.clone() for k, p in named.items() if p.grad is not None}
        original(named, before, diagonal, strength)
        penalties.append(any(not torch.equal(named[k].grad, grad) for k, grad in previous.items()))
    monkeypatch.setattr(learning, "backward_ewc", observe)
    metrics, _ = backend.train_mini_batch({}, [{"prompt": "vault", "completion": "slate"}], steps=2,
                                          lambda_ewc=8., fisher_matrix=fisher, validation_callback=lambda _: True)
    assert metrics["training_algorithm"] == "original_completion_adamw" and metrics["optimizer_steps"] == 2
    assert metrics["ewc_used"] and metrics["fisher_tensors"] == len(named) and metrics["ewc_lambda"] == 8.
    assert penalties == [False, True]


@pytest.mark.parametrize("kind", ["long", "prompt", "weights", "row_weight", "prefix_eos"])
def test_hybrid_routes_specialized_causal_rows_to_bounded_training(tiny_controller, kind):
    from core.learning_data import training_coverage

    backend = load_hybrid(tiny_controller)
    rows = [{"prompt": "vault color", "completion": "slate"}]
    options = {"max_window_tokens": 8}
    if kind == "long":
        rows[0]["completion"] = "slate " * 30
    elif kind == "prompt":
        options["supervise_prompt"] = True
    elif kind == "weights":
        options["row_weights"] = [.5]
    elif kind == "row_weight":
        rows[0]["weight"] = .5
    else:
        from tokenizers.processors import TemplateProcessing
        backend.tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
            single="$A [EOS]", special_tokens=[("[EOS]", backend.tokenizer.eos_token_id)])
    expected = training_coverage(backend.tokenizer, rows, max_tokens=8, supervise_prompt=kind == "prompt")
    metrics, drift = backend.train_mini_batch({}, rows, steps=1, validation_callback=lambda _: True, **options)
    assert metrics["training_algorithm"] == "managed_causal_windows" and metrics["learning_mode"] == "hybrid"
    assert metrics["training_windows"] == expected["windows"] and metrics["supervised_tokens"] == expected["supervised_tokens"]
    assert drift > 0 and metrics["supervise_prompt"] == (kind == "prompt")


@pytest.mark.parametrize("tiny_controller", ["split_queries"], indirect=True)
@pytest.mark.parametrize("objective", ["query_only", "reward_policy"])
def test_hybrid_keeps_specialized_query_and_reward_optimizers(tiny_controller, objective):
    backend = load_hybrid(tiny_controller)
    options = dict(objective=objective, steps=1, max_window_tokens=8, validation_callback=lambda _: True)
    if objective == "query_only":
        rows = []
        options.update(context_text="What is the vault color ? the vault color is slate", max_context_tokens=16)
    else:
        rows = [{"prompt": "vault color", "completion": word,
                 "rollout_prompt_ids": backend.tokenizer.encode("vault color", add_special_tokens=False),
                 "rollout_completion_ids": backend.tokenizer.encode(word, add_special_tokens=False), "reward": reward}
                for word, reward in (("slate", 1.), ("red", 0.))]
    metrics, drift = backend.train_mini_batch({}, rows, **options)
    assert drift > 0 and metrics["training_algorithm"] == "managed_" + objective
    assert metrics["learning_mode"] == "hybrid" and metrics["objective"] == objective
    if objective == "query_only":
        assert metrics["kv_prefills"] == 1 and metrics["updated_projection"] == "q_proj"
    else:
        assert metrics["group_size"] == 2


@pytest.mark.parametrize("tiny_controller", ["split_queries"], indirect=True)
@pytest.mark.parametrize("kind", ["gguf", "bitnet"])
def test_standalone_hybrid_conversions_preserve_original_artifacts(kind, tiny_controller, tmp_path, monkeypatch):
    from core.bitnet_rebuild_trainer import BitNetRebuildTrainer
    from core.gguf_lora_trainer import GGUFLoRATrainer

    root = tmp_path / "artifacts"
    if kind == "gguf":
        trainer = GGUFLoRATrainer(model_path="selected.gguf", base_model_id=tiny_controller.model_path,
                                  adapter_root=str(root), rank=2, alpha=2)
        convert = "_convert_to_gguf"
    else:
        trainer = BitNetRebuildTrainer(base_model_id=tiny_controller.model_path, runtime_root=str(tmp_path),
                                      deploy_root=str(root), rank=2, alpha=2)
        monkeypatch.setattr(trainer, "_training_runtime_available", lambda: True)
        convert = "_convert_merged_checkpoint"
    def convert_weights(directory, destination):
        Path(destination).write_bytes(next(Path(directory).glob("*.safetensors")).read_bytes())
    monkeypatch.setattr(trainer, convert, convert_weights)
    rows = [{"prompt": "vault color", "completion": "slate"}]
    legacy, _, _, old = trainer.train(rows, steps=1, learning_mode="legacy")
    old = Path(old)
    snapshot = old.read_bytes()
    metrics, drift, touched, new = trainer.train(rows, learning_mode="hybrid", steps=1)
    assert metrics["training_algorithm"] == "original_completion_adamw" and metrics["learning_mode"] == "hybrid"
    assert legacy["learning_mode"] == "legacy" and drift > 0 and touched > 0
    assert Path(new).parent == root / "hybrid-implementation" and old.read_bytes() == snapshot
    assert Path(new).name == old.name and (Path(new).parent / "peft/adapter_config.json").exists()
    with pytest.raises(ValueError, match="own previous checkpoint"):
        trainer.train(rows, learning_mode="hybrid", previous_peft=root / "peft")


@pytest.mark.parametrize("tiny_controller", ["split_queries"], indirect=True)
@pytest.mark.parametrize("kind", ["gguf", "bitnet"])
def test_hybrid_native_serving_candidate_controls_commit_and_rollback(kind, tiny_controller, tmp_path, monkeypatch):
    """Train real CPU PEFT tensors, then exercise the public native backend with a converter/serving double."""
    import hashlib
    from core.bitnet_rebuild_trainer import BitNetRebuildTrainer
    from core.engines.bitnet_cpp_engine import BitNetCppReasoningBackend
    from core.engines.gguf_engine import GGUFReasoningBackend
    from core.gguf_lora_trainer import GGUFLoRATrainer
    from core.learning_state import bind_learning_state
    from tests.unit.gguf_fixtures import write_gguf

    root = tmp_path / "native"
    selected = tmp_path / "base.gguf"
    write_gguf(selected)
    source = tiny_controller.model_path
    if kind == "gguf":
        backend = GGUFReasoningBackend(str(selected), adapter_root=str(root), training_base_model_id=source)
        trainer, convert = GGUFLoRATrainer, "_convert_to_gguf"
    else:
        monkeypatch.setenv("SMARTAI_BITNET_TRAINING_ROOT", str(root))
        backend = BitNetCppReasoningBackend(str(selected), training_base_model_id=source)
        trainer, convert = BitNetRebuildTrainer, "_convert_merged_checkpoint"
        monkeypatch.setattr(trainer, "_training_runtime_available", staticmethod(lambda: True))
        monkeypatch.setattr(backend, "_check_deployment_format", lambda path: None)
    backend.learning_implementation = "hybrid"
    store = bind_learning_state(backend, model_path=selected, adapter_format=kind, state_root=root / "learned_memory")
    backend.model = b"base"
    def verify_parent(trainer):
        assert trainer.allow_parent_download is False
        trainer.training_source = source
        if kind == "bitnet":
            trainer.quantization = "i2_s"
        return "fixture-parent"
    def load():
        backend.adapter_path = backend._candidate_deployment or store.current_checkpoint() or ""
        backend.model = Path(backend.adapter_path).read_bytes() if backend.adapter_path else b"base"
        return True
    def conversion(trainer, directory, output):
        assert backend.model is None
        weights = next(Path(directory).glob("*.safetensors"))
        Path(output).write_bytes(hashlib.sha256(weights.read_bytes()).digest())
    monkeypatch.setattr(backend, "unload_model", lambda: setattr(backend, "model", None))
    monkeypatch.setattr(backend, "load_model", load)
    monkeypatch.setattr(backend, "training_ready", lambda: True)
    monkeypatch.setattr(backend, "_verify_training_parent", verify_parent)
    monkeypatch.setattr(trainer, convert, conversion)
    rows = [{"prompt": "the vault", "completion": "slate"}]
    observed = []
    def decide(candidate):
        assert candidate.model != b"base" and Path(candidate._candidate_deployment).read_bytes() == candidate.model
        observed.append(candidate.model)
        return {"accepted": len(observed) == 2, "deployed_sha256": candidate.model.hex()}
    with pytest.raises(RuntimeError, match="validation"):
        backend.train_mini_batch({}, rows, steps=1, validation_callback=decide)
    assert not store.current_checkpoint() and backend.model == b"base"
    metrics, drift = backend.train_mini_batch({}, rows, steps=1, validation_callback=decide)
    assert drift > 0 and metrics["training_algorithm"] == "original_completion_adamw"
    assert len(observed) == 2 and backend.model == observed[1] and store.current_checkpoint() == backend.adapter_path
    assert store.current_manifest()["metadata"]["validation"] == {
        "accepted": True, "deployed_sha256": observed[1].hex(),
    }


def test_hybrid_gguf_budget_can_use_verified_xpu_without_requiring_legacy_projection_loading(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import torch
    from core import gguf_lora_trainer

    (tmp_path / "model.safetensors").write_bytes(b"parent")
    trainer = gguf_lora_trainer.GGUFLoRATrainer(model_path="selected.gguf", base_model_id=str(tmp_path),
                                                adapter_root=str(tmp_path / "adapters"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "xpu", SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                                    mem_get_info=lambda _: (3 * 1024**3, 4 * 1024**3)))
    monkeypatch.setattr("core.transformers_placement.transformers_memory_limits", lambda *args: {"cpu": 2 * 1024**3})
    monkeypatch.setattr(trainer, "_training_backend_available", lambda: True)
    seen = []
    monkeypatch.setattr(gguf_lora_trainer, "_check_parent_nf4", lambda _, device: seen.append(device))
    assert trainer._training_placement(tmp_path, hybrid=True) == ("xpu:0", "nf4")
    assert not trainer._legacy_training and seen == ["xpu:0"]
    assert trainer._training_placement(tmp_path, hybrid=False) == ("cpu", "resident")
