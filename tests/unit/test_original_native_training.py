"""Verify the old completion-only optimizer through the public native training APIs."""
import copy
import math
from pathlib import Path

import pytest

from tests.unit.test_native_context_learning import tiny_controller


@pytest.mark.parametrize("managed_store", [True, False], ids=["scoped-store", "original-directory"])
def test_direct_controller_matches_original_adamw_step_for_step(tiny_controller, tmp_path, managed_store):
    import torch

    backend = tiny_controller
    backend.learning_implementation = "legacy"
    backend.unload_model()
    assert backend.load_model()
    backend.tokenizer.chat_template = "{{ messages[0]['content'] }} "
    if not managed_store:
        del backend.learning_state
        backend.adapter_path = str(tmp_path / "original-peft")
    model = backend._ensure_transformers_lora()
    reference = copy.deepcopy(model)
    rows = [{"prompt": "What color", "completion": "slate"}, {"prompt": "the vault", "completion": "red"}]
    for _ in range(2):
        # The pre-PR controller constructed a new AdamW, masked the tokenized
        # generation prefix, and took one step per row on each pass.
        rng = torch.random.get_rng_state()
        before = {k: p.detach().float().clone() for k, p in reference.named_parameters() if p.requires_grad}
        reference.train()
        optimizer = torch.optim.AdamW((p for p in reference.parameters() if p.requires_grad), lr=.01)
        for _ in range(2):
            for row in rows:
                prefix = row["prompt"] + " "
                encoded = backend.tokenizer(prefix + row["completion"], return_tensors="pt",
                                             truncation=True, max_length=512)
                labels = encoded["input_ids"].clone()
                labels[:, :len(backend.tokenizer.encode(prefix))] = -100
                optimizer.zero_grad(set_to_none=True)
                reference(**encoded, labels=labels).loss.backward()
                optimizer.step()
        torch.random.set_rng_state(rng)
        metrics, drift = backend.train_mini_batch({}, rows, learning_rate=.01, steps=2, lambda_ewc=5.0)
        assert metrics["learning_mode"] == "legacy"
        assert metrics["optimizer_steps"] == 4 and metrics["optimizer_resumed"] is False
        assert metrics["supervised_tokens"] == 4
        expected_drift = 0.
        for name, p in reference.named_parameters():
            actual = model.get_parameter(name)
            assert torch.equal(actual, p), name
            if name in before:
                expected_drift += (p.detach().float() - before[name]).square().sum().item()
        assert drift == pytest.approx(math.sqrt(expected_drift))
        assert Path(backend.adapter_path, "adapter_model.safetensors").is_file()
    if managed_store:
        assert backend.learning_state.current_checkpoint() == backend.adapter_path
        assert backend.learning_state.current_manifest()["metadata"]["validation"] == {"method": "parameter_drift"}


def test_checkpoint_namespace_switch_and_no_inherited_receipts(tiny_controller):
    from core.learning_state import bind_learning_state

    backend = tiny_controller
    managed = backend.learning_state
    rows = [{"prompt": "vault color", "completion": "slate"}]
    backend.train_mini_batch({}, rows, learning_mode="managed", validation_callback=lambda _: True,
                             learning_metadata={"retirement_receipt": "verified-managed"})
    managed_path = Path(backend.adapter_path)
    saved = (managed_path / "adapter_model.safetensors").read_bytes()
    # A direct legacy update against an explicitly bound store creates a new
    # immutable generation and cannot inherit managed behavioral evidence.
    backend.train_mini_batch({}, rows, learning_mode="legacy", steps=1)
    assert "retirement_receipt" not in managed.current_manifest()["metadata"]
    assert (managed_path / "adapter_model.safetensors").read_bytes() == saved
    original_managed_path = backend.adapter_path
    backend.unload_model()
    backend.learning_implementation = "legacy"
    assert backend.load_model() and backend.adapter_path is None
    legacy = backend.learning_state
    assert legacy.root == managed.root / "legacy-implementation"
    rebound = bind_learning_state(backend, model_path=backend.model_path, model_info=backend.model_info,
                                  state_root=legacy.root, adapter_format="peft")
    assert rebound is legacy
    backend.train_mini_batch({}, rows, steps=1)
    assert legacy.current_checkpoint() != managed.current_checkpoint()
    backend.unload_model()
    backend.learning_implementation = "managed"
    assert backend.load_model() and backend.adapter_path == original_managed_path


@pytest.mark.parametrize("tiny_controller", ["split_queries"], indirect=True)
@pytest.mark.parametrize("kind", ["gguf", "bitnet"])
def test_native_helpers_and_standalone_choose_the_old_update(kind, tiny_controller, tmp_path, monkeypatch):
    import torch
    from core.gguf_lora_trainer import GGUFLoRATrainer
    from core.bitnet_rebuild_trainer import BitNetRebuildTrainer

    root = tmp_path / "published"
    if kind == "gguf":
        trainer = GGUFLoRATrainer(model_path="selected.gguf", base_model_id=tiny_controller.model_path,
                                  adapter_root=str(root), rank=2, alpha=2)
        monkeypatch.setattr(trainer, "_training_placement", lambda _: ("cpu", "resident"))
        convert = "_convert_to_gguf"
    else:
        trainer = BitNetRebuildTrainer(base_model_id=tiny_controller.model_path, runtime_root=str(tmp_path),
                                       deploy_root=str(root), rank=2, alpha=2)
        monkeypatch.setattr(trainer, "_training_runtime_available", lambda: True)
        convert = "_convert_merged_checkpoint"
    # Keep native formatting distinct from the controller's prefix-plus-completion
    # construction. Native trainers used the complete assistant chat template.
    tok = tiny_controller.tokenizer
    tok.chat_template = "{% for m in messages %}{{ m.content }} {% endfor %}{% if not add_generation_prompt %}red{% endif %}"
    assert trainer._format_training_example(tok, "vault", "slate") == ("vault ", "vault slate red")
    probe = torch.nn.Linear(2, 2, bias=False)
    snap = trainer._snapshot_trainable(probe)
    with torch.no_grad():
        probe.weight.add_(1.)
    delta, count = trainer._drift_l2(snap, probe)
    assert count == 4 and delta == pytest.approx(2.)
    monkeypatch.setattr(trainer, convert, lambda _, destination: Path(destination).write_bytes(b"converter-fixture"))
    rows = [{"prompt": "vault color", "completion": "slate"}]
    first, drift, _, exported = trainer.train(rows, steps=1)
    second, _, _, _ = trainer.train(rows, steps=1)
    assert first["learning_mode"] == second["learning_mode"] == "legacy"
    assert not first["optimizer_resumed"] and not second["optimizer_resumed"]
    assert drift > 0 and Path(exported).parent == root
    managed, _, _, _ = trainer.train(rows, learning_mode="managed", steps=1)
    assert managed["optimizer_resumed"]


def test_original_qlora_can_dispatch_across_two_verified_accelerators(tmp_path, monkeypatch):
    from core import gguf_lora_trainer
    from core.gguf_lora_trainer import GGUFLoRATrainer

    source = tmp_path / "parent"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"data")
    trainer = GGUFLoRATrainer(model_path="selected.gguf", base_model_id=str(source), adapter_root=str(tmp_path / "state"))
    verified = []
    monkeypatch.setattr(trainer, "_training_backend_available", lambda: True)
    monkeypatch.setattr("core.transformers_placement.transformers_memory_limits",
                        lambda *args: {0: 384 * 1024**2, 1: 384 * 1024**2, "cpu": 128 * 1024**2})
    monkeypatch.setattr(gguf_lora_trainer, "_check_parent_nf4", lambda parent, device: verified.append((parent, device)))
    assert trainer._training_placement(source, legacy=True) == ("auto", "nf4")
    assert verified == [(source, 0), (source, 1)]


@pytest.mark.parametrize("kind", ["gguf", "bitnet"])
def test_legacy_hot_reload_does_not_require_serving_nll_and_keeps_original_files(tmp_path, monkeypatch, kind):
    from core.engines.gguf_engine import GGUFReasoningBackend
    from core.engines.bitnet_cpp_engine import BitNetCppReasoningBackend
    from core.gguf_lora_trainer import GGUFLoRATrainer
    from core.bitnet_rebuild_trainer import BitNetRebuildTrainer
    from core.learning_state import bind_learning_state
    from tests.unit.gguf_fixtures import write_gguf

    source = tmp_path / "base.gguf"
    write_gguf(source)
    root = tmp_path / "adapters"
    root.mkdir()
    (root / "peft").mkdir()
    (root / "peft/adapter_config.json").write_text("{}")
    if kind == "gguf":
        trainer_cls = GGUFLoRATrainer
        old = root / "adapter.gguf"
        backend = GGUFReasoningBackend(str(source), adapter_root=str(root), training_base_model_id="fixture/parent",
                                       legacy_adapter_path=str(old))
    else:
        trainer_cls = BitNetRebuildTrainer
        monkeypatch.setenv("SMARTAI_BITNET_TRAINING_ROOT", str(root))
        old = root / "learned-i2_s.gguf"
        backend = BitNetCppReasoningBackend(str(source), training_base_model_id="fixture/parent", legacy_adapter_path=str(old))
    old.write_bytes(b"old")
    backend.learning_implementation = "legacy"
    store = bind_learning_state(backend, model_path=source, adapter_format=kind, state_root=root / "learned_memory")
    backend._loaded_legacy_deployment = backend.adapter_path = str(old)
    backend.model = b"old"
    monkeypatch.setattr(backend, "training_ready", lambda: True)
    monkeypatch.setattr(backend, "_verify_training_parent", lambda trainer: "training-parent-digest")
    monkeypatch.setattr(backend, "score_learning_probe", lambda *_: pytest.fail("Original hot reload evaluated NLL"))
    if kind == "bitnet":
        monkeypatch.setattr(backend, "_check_deployment_format", lambda _: None)
    def unload():
        backend.model = None

    def load():
        backend.adapter_path = backend._candidate_deployment or store.current_checkpoint() or str(old)
        backend.model = Path(backend.adapter_path).read_bytes()
        return True

    def train(trainer, data, *, learning_mode, output_directory, previous_peft, **kwargs):
        assert learning_mode == "legacy" and trainer.allow_parent_download
        assert previous_peft == root / "peft" and backend.model is None
        deployment = output_directory / old.name
        deployment.write_bytes(b"candidate")
        return {"trainable_parameters_touched": 4, "learning_mode": "legacy"}, 0.25, 4, str(deployment)

    monkeypatch.setattr(trainer_cls, "train", train)
    monkeypatch.setattr(backend, "load_model", load)
    monkeypatch.setattr(backend, "unload_model", unload)
    result, drift = backend.train_mini_batch({}, [{"prompt": "Color?", "completion": "slate"}])
    assert drift == .25 and result["learning_mode"] == "legacy"
    assert backend.model == b"candidate" and old.read_bytes() == b"old"
    assert store.current_checkpoint() == backend.adapter_path != str(old)
    assert store.current_manifest()["metadata"]["validation"] == {"method": "deployment_reload"}


def test_original_qlora_selects_verified_intel_xpu_and_leaves_managed_placement_unchanged(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import torch
    from core import gguf_lora_trainer

    (tmp_path / "model.safetensors").write_bytes(b"parent")
    trainer = gguf_lora_trainer.GGUFLoRATrainer(model_path="selected.gguf", base_model_id=str(tmp_path),
                                                adapter_root=str(tmp_path / "adapters"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "xpu", SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                                    mem_get_info=lambda _: (3 * 1024**3, 4 * 1024**3)))
    monkeypatch.setattr(trainer, "_training_backend_available", lambda: True)
    monkeypatch.setattr("core.transformers_placement.transformers_memory_limits", lambda *args: {"cpu": 2 * 1024**3})
    verified = []
    monkeypatch.setattr(gguf_lora_trainer, "_check_parent_nf4", lambda _, device: verified.append(device))
    assert trainer._training_placement(tmp_path, legacy=True) == ("xpu:0", "nf4")
    assert verified == ["xpu:0"]
    assert trainer._training_placement(tmp_path, legacy=False) == ("cpu", "resident")
