"""Allocation planning and adapter behavior for an exact chosen GGUF."""
from __future__ import annotations

import io
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core import gguf_placement as placement
from core.engines.gguf_engine import GGUFReasoningBackend
from core.engines.prism_gguf_engine import PrismGGUFReasoningBackend
from core.engines import prism_gguf_worker
from tests.unit.gguf_fixtures import write_gguf


GIB = placement.GIB


def model_metadata(weight_gb=24, layers=24):
    layer = int(weight_gb * GIB / (layers + 2))
    return {"layer_count": layers, "layer_bytes": [layer] * layers, "embedding_length": 4096,
            "head_count": 32, "head_count_kv": 8, "key_length": 128, "value_length": 128,
            "vocab_size": 32768, "tensor_bytes": layer * (layers + 2),
            "non_layer_bytes": 2 * layer, "output_bytes": layer}


def hardware(*, ram=24, gpu=7, unified=False):
    return {"ram_available_bytes": ram * GIB, "ram_total_bytes": 32 * GIB, "process_rss_bytes": GIB,
            "gpu_free_bytes": gpu * GIB, "gpu_total_bytes": (32 if unified else 8) * GIB,
            "unified_memory": unified, "gpu_backend": "metal" if unified else "cuda"}


def test_oversized_gpu_model_gets_partial_offload_without_changing_context_or_precision():
    plan = placement.plan_gguf_placement(model_metadata(), context_tokens=8192, hardware=hardware())
    assert 0 < plan["n_gpu_layers"] < 24
    assert plan["context_tokens"] == 8192 and plan["kv_precision"] == "f16"
    assert plan["use_mmap"] and plan["use_mlock"] is False
    assert plan["gpu_weight_bytes"] + plan["gpu_kv_bytes"] <= plan["gpu_budget_bytes"]
    assert plan["minimum_host_bytes_estimate"] <= plan["host_budget_bytes"]
    assert not plan["measured"] and not plan["hard_rss_cap"]


def test_weights_larger_than_ram_use_honest_file_backed_paging_if_hot_set_fits():
    plan = placement.plan_gguf_placement(model_metadata(50), context_tokens=8192, hardware=hardware())
    assert plan["file_backed_paging"] is True
    assert plan["resident_host_bytes_estimate"] > plan["host_budget_bytes"]
    assert "may page from disk" in plan["message"]
    with pytest.raises(MemoryError):
        placement.plan_gguf_placement(model_metadata(50), context_tokens=8192, hardware=hardware(), mmap_allowed=False)


def test_unified_memory_is_one_pool_and_small_budget_fails_without_context_reduction():
    plan = placement.plan_gguf_placement(model_metadata(50), context_tokens=8192,
                                         hardware=hardware(ram=12, gpu=12, unified=True))
    assert plan["unified_memory"] and plan["file_backed_paging"]
    assert plan["minimum_host_bytes_estimate"] <= plan["host_budget_bytes"] < 12 * GIB
    assert plan["n_gpu_layers"] < placement.plan_gguf_placement(
        model_metadata(50), context_tokens=8192, hardware=hardware(ram=24, gpu=24, unified=True))["n_gpu_layers"]
    with pytest.raises(MemoryError, match="unchanged context"):
        placement.plan_gguf_placement(model_metadata(), context_tokens=262144, hardware=hardware(ram=4, gpu=1))


def test_runtime_without_gpu_support_and_unknown_vram_are_cpu_only():
    for hw, supported in ((hardware(), False), (hardware(gpu=0), True)):
        plan = placement.plan_gguf_placement(model_metadata(8), context_tokens=4096,
                                             hardware=hw, gpu_offload_supported=supported)
        assert plan["n_gpu_layers"] == 0


def test_apple_hardware_snapshot_counts_parent_and_worker_rss_once(monkeypatch):
    monkeypatch.setattr(placement.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(placement.platform, "machine", lambda: "arm64")
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=12 * GIB, total=16 * GIB),
        Process=lambda pid: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=GIB)),
    ))
    same_process = placement.placement_hardware(owner_pid=os.getpid())
    assert same_process["process_rss_bytes"] == GIB
    assert same_process["unified_memory"] is True
    assert same_process["gpu_free_bytes"] == same_process["ram_available_bytes"] == 12 * GIB
    other_process = placement.placement_hardware(owner_pid=os.getpid() + 1)
    assert other_process["process_rss_bytes"] == 2 * GIB


def test_native_gpu_probe_respects_the_visible_device_and_is_bounded(monkeypatch):
    monkeypatch.setattr(placement.platform, "system", lambda: "Linux")
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=12 * GIB, total=16 * GIB),
        Process=lambda pid: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=GIB)),
    ))
    monkeypatch.setattr(placement.shutil, "which", lambda name: "nvidia-smi")
    query = Mock(return_value=SimpleNamespace(stdout="4096,8192"))
    monkeypatch.setattr(placement.subprocess, "run", query)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a-real-device,1")
    result = placement.placement_hardware()
    assert result["gpu_free_bytes"] == 4 * GIB
    assert "--id=GPU-a-real-device" in query.call_args.args[0]
    assert query.call_args.kwargs["timeout"] == 3
    query.reset_mock()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert placement.placement_hardware()["gpu_free_bytes"] == 0
    query.assert_not_called()


def test_live_process_limit_and_context_overhead_participate_in_plan():
    unlimited = placement.plan_gguf_placement(model_metadata(24), context_tokens=4096, hardware=hardware())
    constrained = placement.plan_gguf_placement(model_metadata(24), context_tokens=4096,
                                                hardware=hardware(), memory_limit_gb=8)
    assert constrained["host_budget_bytes"] == 7 * GIB
    assert constrained["file_backed_paging"]
    assert constrained["context_tokens"] == unlimited["context_tokens"]
    longer = placement.plan_gguf_placement(model_metadata(24), context_tokens=65536, hardware=hardware())
    assert longer["n_gpu_layers"] < unlimited["n_gpu_layers"]


def test_retry_is_allocation_only_bounded_and_preserves_every_semantic_argument(tmp_path, monkeypatch):
    path = tmp_path / "exact.gguf"
    write_gguf(path)
    monkeypatch.setattr(placement, "placement_hardware", lambda **kwargs: hardware())
    calls = []
    model = object()

    def factory(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise MemoryError("CUDA out of memory")
        return model

    loaded, plan = placement.load_gguf_with_placement(factory, model_path=str(path), n_ctx=4096, type_k=1, type_v=1)
    assert loaded is model and plan["status"] == "applied"
    assert len(calls) == 2 and calls[1]["n_gpu_layers"] < calls[0]["n_gpu_layers"]
    assert {key: value for key, value in calls[0].items() if key != "n_gpu_layers"} == {
        key: value for key, value in calls[1].items() if key != "n_gpu_layers"}
    assert calls[1]["model_path"] == str(path) and calls[1]["n_ctx"] == 4096
    calls.clear()
    failure = Mock(side_effect=ValueError("unsupported tensor encoding"))
    with pytest.raises(RuntimeError, match="unsupported tensor encoding"):
        placement.load_gguf_with_placement(failure, model_path=str(path), n_ctx=4096)
    assert failure.call_count == 1
    failure = Mock(side_effect=MemoryError("failed to allocate"))
    with pytest.raises(RuntimeError):
        placement.load_gguf_with_placement(failure, model_path=str(path), n_ctx=4096)
    assert failure.call_count == 2


def test_malformed_header_never_reaches_native_factory_and_manual_placement_is_explicit(tmp_path):
    path = tmp_path / "bad.gguf"
    path.write_bytes(b"GGUF")
    factory = Mock()
    with pytest.raises(ValueError):
        placement.load_gguf_with_placement(factory, model_path=str(path), n_ctx=4096)
    factory.assert_not_called()
    write_gguf(path)
    model, plan = placement.load_gguf_with_placement(factory, model_path=str(path), n_ctx=1234,
                                                      n_gpu_layers=3, auto_memory_placement=False)
    assert plan["status"] == "applied" and plan["n_gpu_layers"] == 3
    assert factory.call_args.kwargs["n_ctx"] == 1234


def test_gguf_backend_load_reports_plan_and_invalid_format_error(tmp_path, monkeypatch):
    path = tmp_path / "selected.gguf"
    write_gguf(path)
    model = SimpleNamespace()
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=factory, llama_supports_gpu_offload=lambda: True))
    monkeypatch.setattr(placement, "placement_hardware", lambda **kwargs: hardware())
    backend = GGUFReasoningBackend(str(path), n_ctx=1234, memory_limit_gb=16, adapter_root=str(tmp_path / "adapter"))
    assert backend.load_model()
    assert backend.model is model and backend.memory_placement["status"] == "applied"
    assert backend.memory_placement["context_tokens"] == 1234
    assert factory.call_args.kwargs["use_mmap"] and not factory.call_args.kwargs["use_mlock"]
    factory.reset_mock()
    path.write_bytes(b"invalid")
    assert not backend.load_model()
    assert backend.last_error
    factory.assert_not_called()


def test_prism_rejects_invalid_source_before_starting_a_worker(tmp_path, monkeypatch):
    path = tmp_path / "invalid.gguf"
    path.write_bytes(b"invalid")
    backend = PrismGGUFReasoningBackend(str(path), adapter_root=str(tmp_path / "adapter"))
    launch = Mock(side_effect=AssertionError("Invalid model started a native worker"))
    monkeypatch.setattr("core.engines.prism_gguf_engine.subprocess.Popen", launch)
    assert not backend.load_model()
    assert backend.last_error
    launch.assert_not_called()


def test_prism_surfaces_current_native_encoding_error_without_stale_log(tmp_path, monkeypatch):
    path = tmp_path / "selected.gguf"
    write_gguf(path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    error_log = adapter / "prism-worker.stderr.log"
    error_log.write_text("unrelated previous launch error\n")
    backend = PrismGGUFReasoningBackend(str(path), adapter_root=str(adapter))
    monkeypatch.setattr(backend, "_worker_env", lambda: {})

    def launch(*args, **kwargs):
        kwargs["stderr"].write("native loader: unsupported tensor encoding 143\n")
        kwargs["stderr"].flush()
        return SimpleNamespace(poll=lambda: 1)

    monkeypatch.setattr("core.engines.prism_gguf_engine.subprocess.Popen", launch)
    with pytest.raises(RuntimeError, match="unsupported tensor encoding 143"):
        backend.load_model()
    assert "unrelated previous launch" not in backend.last_error
    assert backend.model is None


def test_legacy_all_logits_reads_last_filled_row_not_allocated_tail():
    class LegacyModel:
        _logits_all = True
        n_tokens = 2
        # Older llama-cpp-python stores n_ctx rows but _scores is a filled view.
        scores = [[9.0, 1.0], [1.0, 9.0], [1000.0, -1000.0]]

        @property
        def _scores(self):
            return self.scores[:self.n_tokens]

    assert placement.last_token_logits(LegacyModel()) == [1.0, 9.0]


def test_prism_worker_preserves_last_token_entropy_without_context_wide_logits(tmp_path, monkeypatch):
    path = tmp_path / "selected.gguf"
    write_gguf(path)
    expected_logits = [0.0, 0.0, 0.0, 0.0]
    get_logits = Mock(return_value=expected_logits)
    model = SimpleNamespace(
        tokenize=lambda raw: [1, 2], eval=Mock(), n_vocab=lambda: 4, n_ctx=lambda: 4096,
        _ctx=SimpleNamespace(get_logits_ith=get_logits), _logits_all=False,
    )
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(
        Llama=factory, llama_get_logits_ith=lambda *a: None, llama_supports_gpu_offload=lambda: False))
    monkeypatch.setattr(placement, "placement_hardware", lambda **kwargs: hardware())
    config = tmp_path / "worker.json"
    config.write_text(json.dumps({"model_path": str(path), "n_ctx": 4096, "memory_limit_gb": 8}))
    output = io.StringIO()
    monkeypatch.setattr(sys, "argv", ["worker", str(config)])
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"op":"ping"}\n{"op":"entropy","prompt":"question"}\n{"op":"shutdown"}\n'))
    monkeypatch.setattr(sys, "stdout", output)
    assert prism_gguf_worker.main() == 0
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert factory.call_args.kwargs["logits_all"] is False
    assert records[0]["memory_placement"]["context_tokens"] == 4096
    assert records[1]["entropy"] == 1.0
    get_logits.assert_called_once_with(-1)
