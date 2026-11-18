"""The original CUDA text loader and the managed Transformers controller coexist."""
import pytest

from config.settings import get_settings
from core.entropy_router import EntropyRouter
from core.generation_metrics import RequestMetrics, observe_generations
from core.memory_scopes import LearningScope
from core.temperature_policy import pro_ladder
from core.pro_engine import ProReasoningEngine
from tests.unit.test_native_context_learning import tiny_controller


@pytest.mark.parametrize("tiny_controller", [96], indirect=True)
def test_migrated_direct_solver_uses_logits_and_preserves_nucleus_sampling(tiny_controller, monkeypatch, tmp_path):
    import torch

    source = tiny_controller.model_path
    tiny_controller.unload_model()
    engine = ProReasoningEngine(settings=get_settings(use_mock=False, auto_download=False, learning_implementation="managed",
                                database_path=str(tmp_path / "chat.db"), max_new_tokens=2, search_top_p=.73))
    try:
        result = engine.load_model("Local causal model", model_path=source, backend="torch")
        assert result["status"] == "loaded"
        backend = engine.controller_backend
        assert engine.model is backend.model and engine.tokenizer is backend.tokenizer
        prompt = "What is the vault color?"
        tokens = backend.tokenizer(prompt, return_tensors="pt").to(backend.model.device)
        with torch.no_grad():
            expected = EntropyRouter.compute_entropy_from_logits(backend.model(**tokens).logits[:, -1, :])
        actual = engine.calculate_token_entropy(prompt)
        assert actual == pytest.approx(expected) and expected != pytest.approx(.35)
        assert engine.router.route(actual)[1] == engine.settings.pro_branch_count_default

        with engine.use_learning_scope(LearningScope("project", "torch-compat"), seed_from_current=False):
            assert engine.controller_backend.direct_torch_generation
            assert engine.model is engine.controller_backend.model and engine.tokenizer is engine.controller_backend.tokenizer
            assert engine.calculate_token_entropy(prompt) == pytest.approx(expected)
        backend = engine.controller_backend
        assert backend.direct_torch_generation
        calls = []
        generate = backend.model.generate

        def observe(**kwargs):
            sampled = generate(**kwargs)
            prefix = kwargs["input_ids"].shape[-1]
            calls.append((kwargs, backend.tokenizer.decode(sampled[0, prefix:], skip_special_tokens=True)))
            return sampled

        monkeypatch.setattr(backend.model, "generate", observe)
        metrics = RequestMetrics()
        with observe_generations(metrics.update):
            branches = engine.generate_parallel_branches(prompt, branch_count=2)
        assert metrics.snapshot()["invocations"] == 3 and metrics.snapshot()["tokens"] >= 3
        assert branches == [answer for _, answer in calls]
        assert [kwargs["temperature"] for kwargs, _ in calls] == pro_ladder(2)
        assert all(kwargs["top_p"] == .73 and kwargs["num_return_sequences"] == 1 and
                   kwargs["pad_token_id"] == backend.tokenizer.eos_token_id for kwargs, _ in calls)
        assert prompt not in "".join(branches)
        calls.clear()
        text = "".join(backend.stream_generate_tokens(prompt, max_tokens=2, temperature=.65, top_p=.61))
        kwargs, _ = calls[0]
        assert kwargs["top_p"] == .61 and kwargs["temperature"] == .65 and prompt not in text
        base = engine.model
        backend.train_mini_batch({}, [{"prompt": prompt, "completion": "slate"}], steps=1, validation_callback=lambda _: True)
        assert engine.model is backend.model and engine.model is not base and engine.model.get_base_model() is base
    finally:
        engine.unload_model()
    assert engine.model is None and engine.tokenizer is None


@pytest.mark.parametrize("backend,device,cuda_available,model_info,uses_bfloat16", [
    ("cuda", "cuda", True, {}, True),
    ("torch", "cuda", True, {}, True),
    ("cuda", "cuda", False, {}, False),
    ("torch", "cpu", True, {}, False),
    ("cpu", "cuda", True, {}, False),
    ("cuda", "cuda", True, {"controller_runtime": "transformers_auto"}, False),
])
def test_loader_preserves_only_the_original_cuda_precision(
        tmp_path, monkeypatch, backend, device, cuda_available, model_info, uses_bfloat16):
    import sys
    from core.pro_runtime_hardening import _active_backend_name
    from tests.unit.test_transformers_placement import fake_transformers

    engine = ProReasoningEngine(device=device, settings=get_settings(use_mock=False, auto_download=False, learning_implementation="managed",
                                database_path=str(tmp_path / "chat.db")))
    source, calls = fake_transformers(monkeypatch, tmp_path)
    torch_host = sys.modules["torch"]
    torch_host.bfloat16 = object()
    torch_host.cuda.is_available = lambda: cuda_available
    torch_host.cuda.ipc_collect = lambda: None
    try:
        result = engine.load_model("Local precision test", model_path=str(source), backend=backend, model_info=model_info)
        assert result["status"] == "loaded", result
        _, options = calls[0]
        assert options["torch_dtype"] == (torch_host.bfloat16 if uses_bfloat16 else "auto")
        assert options["device_map"] == "auto"
        if uses_bfloat16:
            assert result["backend"] == "cuda" and result["runtime"] == "transformers_auto"
            assert _active_backend_name(engine) == backend
            assert engine.model is engine.controller_backend.model
            assert engine.tokenizer.encode("test") == [1]
    finally:
        engine.unload_model()


@pytest.mark.parametrize("requested,active", [("torch", "torch"), ("cuda", "cuda"), ("auto", "cuda"), (None, "cuda")])
def test_legacy_cuda_dispatch_preserves_original_loader_and_handles(tmp_path, monkeypatch, requested, active):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock
    from core import pro_engine
    from tests.unit.test_transformers_placement import torch_host

    source = str(tmp_path / "selected")
    model = Mock()
    tokenizer = Mock()
    model_loader, token_loader = Mock(return_value=model), Mock(return_value=tokenizer)
    torch_module = torch_host()
    torch_module.bfloat16 = object()
    torch_module.cuda.is_available = lambda: True
    torch_module.cuda.ipc_collect = lambda: None
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_loader),
        AutoTokenizer=SimpleNamespace(from_pretrained=token_loader),
    ))
    monkeypatch.setattr(pro_engine, "resolve_optimal_backend", lambda _: ("cuda", "cuda"))
    engine = ProReasoningEngine(device="cuda", backend="auto", settings=get_settings(
        use_mock=False, auto_download=False, learning_implementation="legacy", database_path=str(tmp_path / "chat.db")))
    try:
        result = engine.load_model("Local causal model", model_path=source, backend=requested)
        assert (result["status"], result["model"], result["backend"], result["path"]) == (
            "loaded", "Local causal model", "cuda", source)
        assert "runtime" not in result
        token_loader.assert_called_once_with(source, trust_remote_code=True)
        model_loader.assert_called_once_with(source, torch_dtype=torch_module.bfloat16, device_map="auto",
                                             trust_remote_code=True)
        model.eval.assert_called_once_with()
        assert engine.active_backend == active and engine.controller_backend is None
        assert engine.model is model and engine.tokenizer is tokenizer and engine.is_model_loaded
    finally:
        engine.unload_model()
    assert engine.active_backend is None and engine.model is None and engine.tokenizer is None


@pytest.mark.parametrize("tiny_controller", [96], indirect=True)
def test_original_direct_handles_drive_real_logits_and_generation_on_cpu(tiny_controller, tmp_path, monkeypatch):
    """Exercise the raw CUDA route with real CPU tensors in place of the unavailable GPU."""
    import torch
    import transformers

    model, tokenizer = tiny_controller.model, tiny_controller.tokenizer
    source = tiny_controller.model_path
    tiny_controller.unload_model()
    transfers = []

    class Inputs(dict):
        def to(self, device):
            transfers.append(device)
            return self

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return Inputs(tokenizer(*args, **kwargs))

        def __getattr__(self, name):
            return getattr(tokenizer, name)

    engine = ProReasoningEngine(device="cuda", settings=get_settings(
        use_mock=False, auto_download=False, learning_implementation="legacy", database_path=str(tmp_path / "chat.db"),
        max_new_tokens=2, search_top_p=.73))
    try:
        with monkeypatch.context() as loader:
            loader.setattr(torch.cuda, "is_available", lambda: True)
            loader.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
            loader.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model)
            assert engine.load_model("Selected", model_path=source, backend="torch")["status"] == "loaded"
        prompt = "What is the vault color?"
        with torch.no_grad():
            expected = EntropyRouter.compute_entropy_from_logits(model(**tokenizer(prompt, return_tensors="pt")).logits[:, -1])
        assert engine.calculate_token_entropy(prompt) == pytest.approx(expected)
        calls = []
        generate = model.generate

        def record(**kwargs):
            sampled = generate(**kwargs)
            answer = sampled[0][kwargs["input_ids"].shape[1]:]
            calls.append((kwargs, tokenizer.decode(answer, skip_special_tokens=True)))
            return sampled

        monkeypatch.setattr(model, "generate", record)
        branches = engine.generate_parallel_branches(prompt, branch_count=2)
        assert len(calls) == len(pro_ladder(2))
        assert branches == [answer for _, answer in calls]
        assert [options["temperature"] for options, _ in calls] == pro_ladder(2)
        assert all(options["top_p"] == .73 and options["num_return_sequences"] == 1 for options, _ in calls)
        assert transfers and set(transfers) == {"cuda"}
    finally:
        engine.unload_model()


def test_direct_cuda_switch_restores_exact_model_and_managed_checkpoint(tiny_controller, tmp_path, monkeypatch):
    import copy
    import torch
    import transformers
    from core.controller_runtime import controller_learning_root
    from peft import peft_model

    source = tiny_controller.model_path
    reference, tokenizer = tiny_controller.model, tiny_controller.tokenizer
    tiny_controller.unload_model()
    engine = ProReasoningEngine(device="cuda", settings=get_settings(
        use_mock=False, auto_download=False, learning_implementation="legacy", database_path=str(tmp_path / "chat.db")))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: None)
    monkeypatch.setattr(peft_model, "infer_device", lambda: "cpu")
    monkeypatch.setattr("core.transformers_placement.transformers_memory_limits", lambda *a, **k: {"cpu": 2**30})
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: copy.deepcopy(reference))
    try:
        assert engine.load_model("Original CUDA", model_path=source, backend="torch")["status"] == "loaded"
        assert engine.controller_backend is None and engine.active_backend == "torch"
        assert engine.set_learning_implementation("managed")["learning_implementation"] == "managed"
        backend = engine.controller_backend
        assert engine.model is backend.model and backend.direct_backend_name == "torch"
        assert backend.learning_state.root == controller_learning_root()
        # Run the optimizer against the tiny real CPU model under the managed route.
        with monkeypatch.context() as cpu:
            cpu.setattr(torch.cuda, "is_available", lambda: False)
            backend.train_mini_batch({}, [{"prompt": "vault color", "completion": "slate"}], steps=1,
                                     validation_callback=lambda _: True)
        learned = backend.learning_state.current_checkpoint()
        identity = backend.learning_state.identity
        assert engine.set_learning_implementation("legacy")["learning_implementation"] == "legacy"
        assert engine.controller_backend is None and engine.active_backend == "torch"
        assert not getattr(engine.model, "peft_config", None)
        assert engine.set_learning_implementation("managed")["learning_implementation"] == "managed"
        assert engine.controller_backend.adapter_path == learned
        assert engine.controller_backend.learning_state.identity == identity
        assert engine.model is engine.controller_backend.model
    finally:
        engine.unload_model()
