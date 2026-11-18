"""The direct Torch loader retains its original generation contract through the PEFT controller."""
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
    engine = ProReasoningEngine(settings=get_settings(use_mock=False, auto_download=False,
                                database_path=str(tmp_path / "chat.db"), max_new_tokens=2, search_top_p=.73))
    try:
        result = engine.load_model("Local causal model", model_path=source, backend="torch")
        assert result["status"] == "loaded"
        backend = engine.controller_backend
        prompt = "What is the vault color?"
        tokens = backend.tokenizer(prompt, return_tensors="pt").to(backend.model.device)
        with torch.no_grad():
            expected = EntropyRouter.compute_entropy_from_logits(backend.model(**tokens).logits[:, -1, :])
        actual = engine.calculate_token_entropy(prompt)
        assert actual == pytest.approx(expected) and expected != pytest.approx(.35)
        assert engine.router.route(actual)[1] == engine.settings.pro_branch_count_default

        with engine.use_learning_scope(LearningScope("project", "torch-compat"), seed_from_current=False):
            assert engine.controller_backend.direct_torch_generation
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
    finally:
        engine.unload_model()
