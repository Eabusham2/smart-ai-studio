"""Chosen-checkpoint dispatch, live memory budgets and real CPU/disk inference."""
import json
import gc
import hashlib
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import controller_runtime as runtime
from core import transformers_placement as placement


GIB = placement.GIB


def torch_host(*, cuda_free=None, mps=False, probe_error=False):
    def memory(index):
        if probe_error:
            raise RuntimeError("free-memory query unavailable")
        return cuda_free[index], 8 * GIB
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda_free is not None,
            device_count=lambda: len(cuda_free or ()), mem_get_info=memory,
            empty_cache=lambda: None,
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


@pytest.fixture
def host_memory(monkeypatch):
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(total=32 * GIB, available=28 * GIB))
    monkeypatch.setattr("core.training_memory.process_memory_bytes", lambda: 2 * GIB)


def test_gpu_cpu_budgets_use_free_memory_and_reserve_context(host_memory):
    budgets = placement.transformers_memory_limits(torch_host(cuda_free=[8 * GIB, 4 * GIB]), kv_bytes=2 * GIB)
    assert budgets[0] == 5 * GIB
    assert budgets[1] == GIB
    assert budgets["cpu"] == 28 * GIB - int(32 * GIB * .05) - 2 * GIB


def test_unified_memory_is_never_counted_as_two_full_pools(host_memory):
    cpu = placement.transformers_memory_limits(torch_host())
    unified = placement.transformers_memory_limits(torch_host(mps=True))
    assert set(unified) == {"mps", "cpu"}
    assert sum(unified.values()) == cpu["cpu"]


def test_application_cap_subtracts_existing_process_and_kv(host_memory):
    budgets = placement.transformers_memory_limits(torch_host(), 8, kv_bytes=GIB)
    assert budgets["cpu"] == int(4.75 * GIB)
    with pytest.raises(RuntimeError, match="Too little available RAM"):
        placement.transformers_memory_limits(torch_host(), 2)


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_memory_caps_are_not_ignored(host_memory, value):
    with pytest.raises(ValueError, match="positive finite"):
        placement.transformers_memory_limits(torch_host(), value)


def test_unmeasured_gpu_is_not_given_its_total_capacity(host_memory):
    budgets = placement.transformers_memory_limits(torch_host(cuda_free=[8 * GIB], probe_error=True))
    assert set(budgets) == {"cpu"}


def test_kv_estimate_uses_real_gqa_metadata_and_requested_context(tmp_path):
    config = {"text_config": {"num_hidden_layers": 80, "num_attention_heads": 64,
              "num_key_value_heads": 8, "hidden_size": 8192, "torch_dtype": "bfloat16"}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert placement.estimate_transformers_kv_bytes(str(tmp_path), 32768) == 2 * 80 * 8 * 128 * 32768 * 2
    (tmp_path / "config.json").write_text('{"model_type": "unknown"}')
    assert placement.estimate_transformers_kv_bytes(str(tmp_path), 32768) is None


def test_actual_device_map_controls_inference_only_and_telemetry():
    model = SimpleNamespace(hf_device_map={"embed": 0, "layer1": "cpu", "layer2": "disk"})
    info = placement.describe_transformers_placement(model, {0: GIB, "cpu": 4 * GIB})
    assert info["devices"] == ["cpu", "cuda:0", "disk"]
    assert info["disk_offload"] and info["inference_only"]
    assert not info["measured_speedup"]
    resident = placement.describe_transformers_placement(SimpleNamespace(hf_device_map={"": "cuda:0"}))
    assert not resident["inference_only"] and not resident["disk_offload"]


def test_prompt_device_never_uses_disk_or_meta_placeholders():
    model = SimpleNamespace(device="meta", hf_device_map={"embed": "disk", "layer": 0})
    assert placement.transformers_input_device(model) == 0
    model._hf_hook = SimpleNamespace(execution_device="cpu")
    assert placement.transformers_input_device(model) == "cpu"
    assert placement.transformers_input_device(SimpleNamespace(device="meta", hf_device_map={"": "disk"})) == "cpu"


def fake_transformers(monkeypatch, tmp_path, *, fail=False):
    calls = []
    model = SimpleNamespace(hf_device_map={"embed": "cpu", "layers": "disk"},
                            config=SimpleNamespace(), device="meta", eval=lambda: None)
    tokenizer = SimpleNamespace(encode=lambda text: [1])
    def load(source, **kwargs):
        calls.append((source, kwargs))
        if fail:
            raise RuntimeError("unsupported checkpoint architecture")
        (Path(kwargs["offload_folder"]) / "weights.dat").write_bytes(b"owned offload fixture")
        return model
    module = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_host())
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setattr("config.paths.get_portable_data_dir", lambda: str(tmp_path / "state"))
    monkeypatch.setattr(placement, "transformers_memory_limits", lambda *args, **kwargs: {"cpu": 4 * GIB})
    monkeypatch.setattr(runtime, "release_training_memory", lambda: None)
    source = tmp_path / "chosen-model"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"selected checkpoint")
    return source, calls


def test_loader_preserves_weights_dtype_and_cleans_only_owned_offload(monkeypatch, tmp_path):
    source, calls = fake_transformers(monkeypatch, tmp_path)
    backend = runtime.UniversalControllerBackend(str(source), {"controller_runtime": "transformers_auto"})
    assert backend.load_model()
    requested_source, kwargs = calls[0]
    assert requested_source == str(source)
    assert kwargs["torch_dtype"] == "auto"
    assert kwargs["device_map"] == "auto"
    assert kwargs["offload_state_dict"] and kwargs["low_cpu_mem_usage"]
    assert kwargs["max_memory"] == {"cpu": 4 * GIB}
    directory = Path(kwargs["offload_folder"])
    assert directory.is_dir() and directory.parent == tmp_path / "state/runtime/transformers-offload"
    assert backend.memory_placement["disk_offload"]
    assert not backend.training_ready()
    backend.unload_model()
    assert not directory.exists()
    assert (source / "model.safetensors").read_bytes() == b"selected checkpoint"


def test_failed_architecture_load_removes_private_partial_offload(monkeypatch, tmp_path):
    source, calls = fake_transformers(monkeypatch, tmp_path, fail=True)
    backend = runtime.UniversalControllerBackend(str(source), {"controller_runtime": "transformers_auto"})
    with pytest.raises(RuntimeError, match="No compatible Transformers"):
        backend.load_model()
    assert calls and not Path(calls[0][1]["offload_folder"]).exists()
    assert (source / "model.safetensors").exists()


def saved_adapter(source, state_root):
    key = hashlib.sha256(str(source).encode()).hexdigest()[:16]
    directory = state_root / "controller_lora" / key
    directory.mkdir(parents=True)
    (directory / "adapter_config.json").write_text("{}")
    return directory


@pytest.mark.parametrize("offloaded", [False, True])
def test_saved_adapter_restore_failure_never_silently_omits_learning(monkeypatch, tmp_path, offloaded):
    source, calls = fake_transformers(monkeypatch, tmp_path)
    saved_adapter(source, tmp_path / "state")
    def restore(model, *args, **kwargs):
        raise RuntimeError("saved adapter is incompatible")
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=restore)))
    if not offloaded:
        monkeypatch.setattr(placement, "describe_transformers_placement", lambda *args: {"inference_only": False})
    backend = runtime.UniversalControllerBackend(str(source), {"controller_runtime": "transformers_auto"})
    with pytest.raises(RuntimeError, match="refusing to omit"):
        backend.load_model()
    assert backend.model is None and not backend.is_loaded
    assert backend.processor is None and backend.tokenizer is None
    assert not Path(calls[0][1]["offload_folder"]).exists()


def test_failed_adapter_drops_hook_cycles_before_deleting_offload_files(monkeypatch, tmp_path):
    source, calls = fake_transformers(monkeypatch, tmp_path)
    saved_adapter(source, tmp_path / "state")
    references = []
    class Model:
        hf_device_map = {"embed": "cpu", "layer": "disk"}
        config = None
        device = "meta"

    def create_model(source, **kwargs):
        calls.append((source, kwargs))
        model = Model()
        model._hf_hook = SimpleNamespace(model=model)
        references.append(weakref.ref(model))
        return model

    monkeypatch.setattr(sys.modules["transformers"].AutoModelForCausalLM, "from_pretrained", create_model)
    def restore(model, *args, **kwargs):
        raise RuntimeError("failure retains a model in this traceback")
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=restore)))
    monkeypatch.setattr(runtime, "release_training_memory", gc.collect)
    real_delete = runtime.shutil.rmtree
    def delete_after_release(directory):
        assert references and references[0]() is None, "offload deleted while model/hooks are still live"
        return real_delete(directory)
    monkeypatch.setattr(runtime.shutil, "rmtree", delete_after_release)
    backend = runtime.UniversalControllerBackend(str(source), {"controller_runtime": "transformers_auto"})
    with pytest.raises(RuntimeError, match="refusing to omit"):
        backend.load_model()
    assert backend._offload_directory is None


def test_adapter_redispatch_keeps_budget_directory_and_reports_result(monkeypatch, tmp_path):
    source, calls = fake_transformers(monkeypatch, tmp_path)
    saved_adapter(source, tmp_path / "state")
    seen = []
    def restore(model, *args, **kwargs):
        seen.append(kwargs)
        model.hf_device_map = {"embed": "cpu", "layer": "disk", "adapter": "cpu"}
        return model
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=restore)))
    backend = runtime.UniversalControllerBackend(str(source), {"controller_runtime": "transformers_auto"})
    assert backend.load_model()
    try:
        assert seen[0]["max_memory"] == calls[0][1]["max_memory"]
        assert seen[0]["offload_folder"] == calls[0][1]["offload_folder"]
        assert not seen[0]["is_trainable"]
        assert backend.memory_placement["device_map"]["adapter"] == "cpu"
        assert not backend.training_ready()
    finally:
        backend.unload_model()


def test_locked_offload_cleanup_remains_owned_and_blocks_replacement(monkeypatch, tmp_path):
    source, calls = fake_transformers(monkeypatch, tmp_path)
    backend = runtime.UniversalControllerBackend(str(source), {"controller_runtime": "transformers_auto"})
    assert backend.load_model()
    directory = backend._offload_directory
    real_delete = runtime.shutil.rmtree
    def locked(path):
        raise PermissionError("mapped file has not closed")
    monkeypatch.setattr(runtime.shutil, "rmtree", locked)
    backend.unload_model()
    assert backend.model is None and not backend.is_loaded
    assert backend._offload_directory == directory and Path(directory).exists()
    with pytest.raises(RuntimeError, match="retry unloading"):
        backend.load_model()
    assert len(calls) == 1 and backend._offload_directory == directory
    monkeypatch.setattr(runtime.shutil, "rmtree", real_delete)
    backend.unload_model()
    assert backend._offload_directory is None and not Path(directory).exists()


@pytest.mark.parametrize("with_saved_adapter", [False, True])
def test_real_small_checkpoint_runs_with_cpu_and_disk_offload(monkeypatch, tmp_path, with_saved_adapter):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    config = transformers.GPT2Config(vocab_size=64, n_positions=64, n_embd=32, n_layer=4, n_head=4)
    checkpoint = tmp_path / "tiny-checkpoint"
    original = transformers.GPT2LMHeadModel(config)
    original.eval()
    original.save_pretrained(checkpoint)
    prompt = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected = original(input_ids=prompt).logits.detach().clone()
    if with_saved_adapter:
        peft = pytest.importorskip("peft")
        trained = peft.get_peft_model(original, peft.LoraConfig(
            r=2, lora_alpha=2, target_modules=["c_attn"], task_type="CAUSAL_LM", fan_in_fan_out=True))
        with torch.no_grad():
            for name, parameter in trained.named_parameters():
                if "lora_" in name:
                    parameter.copy_(torch.sin(torch.arange(parameter.numel()).reshape(parameter.shape)) * 0.2)
        trained.eval()
        with torch.no_grad():
            learned = trained(input_ids=prompt).logits.detach().clone()
        assert not torch.allclose(expected, learned), "fixture must contain a meaningful saved adapter update"
        expected = learned
        trained.save_pretrained(saved_adapter(checkpoint, tmp_path / "state"))
        del trained
    del original
    tokenizer = SimpleNamespace(encode=lambda text: [1, 2])
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr("config.paths.get_portable_data_dir", lambda: str(tmp_path / "state"))
    monkeypatch.setattr(placement, "transformers_memory_limits", lambda *args, **kwargs: {"cpu": 80_000})
    backend = runtime.UniversalControllerBackend(str(checkpoint), {"controller_runtime": "transformers_auto", "default_context": 64})
    try:
        assert backend.load_model()
        assert backend.memory_placement["disk_offload"]
        assert not backend.training_ready()
        if with_saved_adapter:
            assert hasattr(backend.model, "peft_config")
        with torch.no_grad():
            output = backend.model(input_ids=prompt)
        assert output.logits.shape == (1, 3, 64)
        assert torch.isfinite(output.logits).all()
        assert torch.allclose(output.logits, expected, rtol=1e-5, atol=1e-5)
        assert next(backend.model.parameters()).dtype == torch.float32
    finally:
        backend.unload_model()
