"""Online task augmentation, trajectory synthesis, search budgets and frozen-policy scoring."""
import json
import threading
from copy import deepcopy

import pytest

from core import arc_augmentation as arc
from core.online_consolidator import ManagedAwakeOnlineConsolidator
from core.policy_reference import policy_reference
from core.test_time_learning import OnlineLearning
from tests.unit.context_backend_fixture import TransactionalBackend
from tests.unit.test_native_context_learning import tiny_controller


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setattr("core.online_consolidator.release_training_memory", lambda backend: {})
    return OnlineLearning(ManagedAwakeOnlineConsolidator(TransactionalBackend(tmp_path / "state"), lambda_ewc=0))


def test_arc_grid_variants_are_invertible_on_rectangles():
    original = [[1, 2, 3], [0, 8, 9]]
    transformed = [arc.transform(original, variant) for variant in arc.VARIANTS]
    assert len({arc.grid_text(a) for a in transformed}) == len(arc.VARIANTS) == 16
    for result, variant in zip(transformed, arc.VARIANTS):
        assert arc.transform(result, variant, inverse=True) == original
    for malformed in [[[True]], [[10]], [[-1]], [[1, 2], [3]], [[]], "1", "{"]:
        assert arc.grid(malformed) is None


def test_arc_votes_canonically_and_reports_invalid_samples(runner):
    tests = [{"input": [[1, 2]], "output": [[1, 2]]}, {"input": [[4], [0]], "output": [[4], [0]]}]
    query = [[9, 3, 7], [2, 1, 0]]
    prompts = []
    def identity(prompt, **kwargs):
        prompts.append(prompt)
        return "invalid" if len(prompts) == 2 else prompt.rsplit("Input: ", 1)[-1]
    runner.generate = identity
    result = runner.run_request(dict(method="few_shot", demonstrations=tests, query=query, votes=16))
    assert result["votes"] == {arc.grid_text(query): 15}
    assert result["invalid_votes"] == 1 and result["arc_views"] == 16
    assert result["training_rows"] > len(tests)
    assert len(set(prompts)) == 16
    assert all(arc.grid_text(query) not in str(row) for row in runner.backend.received_batches[0])
    runner.generate = lambda *a, **kw: "[]"
    result = runner.few_shot(tests, query, votes=1)
    assert result["answer"] is None and result["invalid_votes"] == 1
    with pytest.raises(ValueError, match="rectangular"):
        runner.few_shot(tests, "plain text", arc_augment=True)


def test_larger_text_demonstrations_and_generation_and_budget(runner):
    demos = [{"prompt": str(i) + "?" * 550, "completion": "a" * 600} for i in range(13)]
    runner.generate = lambda prompt, *, max_tokens, temperature: str(max_tokens) * 12000
    result = runner.few_shot(demos, "q" * 2100, votes=9, steps=9, max_tokens=300)
    assert len(result["answer"]) == 36000 and result["votes"][result["answer"]] == 9
    assert result["arc_views"] == 0 and result["training_rows"] == 26


def test_agent_messages_are_jointly_conditioned_and_synthesized_only_after_commit(runner):
    traces = [{"agent": "one", "messages": [{"role": "tool", "content": "found route P"},
                                            {"role": "assistant", "content": "use P\nFinal answer: 36"}]},
              {"agent": "two", "messages": [{"role": "tool", "content": "found route Q"},
                                            {"role": "assistant", "content": "use Q\nFinal answer: 36"}]},
              {"agent": "three", "completion": "Final answer: 20"}]
    original = deepcopy(traces)
    def synthesize(prompt, **kwargs):
        assert runner.backend.learning_state.current_manifest() and kwargs["temperature"] == 0
        assert prompt == "Square six" and runner.backend.train_calls == 1
        return "Six squared is thirty-six."
    runner.generate = synthesize
    result = runner.run_request({"method": "consolidate_candidates", "prompt": "Square six", "trajectories": traces})
    assert result["answer"] == "Six squared is thirty-six." and result["synthesized"]
    assert result["best_sample"] != result["answer"] and traces == original
    pairs = [r for r in runner.backend.received_batches[-1] if "reasoning records" in r["prompt"]]
    assert len(pairs) == 2
    assert all("found route P" in r["prompt"] and "found route Q" in r["prompt"] for r in pairs)
    assert all(r["completion"] not in r["prompt"] and "20" not in r["completion"] for r in pairs)
    with pytest.raises(ValueError, match="ending"):
        runner.consolidate_candidates("task", trajectories=["text", {"messages": [{"role": "tool", "content": "note"}]}])


def test_discovery_large_schedule_uses_fresh_groups_and_does_not_fake_reference(runner, monkeypatch):
    monkeypatch.setattr(runner, "_require", lambda name: None)
    count = 0
    def sample(prompt, *, max_tokens, temperature):
        nonlocal count
        count += 1
        assert max_tokens == 500 and temperature == .75
        return f"candidate {count}"
    runner.generate = sample
    monkeypatch.setattr(runner.backend, "score_learning_probe",
                        lambda prompt, text: 9 - runner.backend.weight * (2 if text[-1] == "0" else 1))
    result = runner.discover("x"*2500, rollouts=20, steps=9, max_tokens=500, temperature=.75,
                             archive_size=200, children_per_expansion=10, entropic_kl_target=.3,
                             verifier=lambda text: float(text.endswith("0")), schedule=[{}, {"learning_rate": .0003}])
    assert count == 40 and len(result["rounds"]) == 2
    assert result["updates"] == 2 and len(result["rounds"][0]["inserted"]) == 10
    assert result["rounds"][1]["policy_generation"] == result["rounds"][0]["update"]["generation"]
    assert result["reference_policy"] == "disabled" and result["reference_kl_coefficient"] == 0
    with pytest.raises(ValueError, match="frozen policy"):
        runner.discover("q", verifier=lambda a: .5, reference_kl_coef=.01)
    with pytest.raises(ValueError, match="steps"):
        runner.discover("q", verifier=lambda a: .5, schedule=[{}, {"steps": 0}])
    assert count == 40


@pytest.mark.parametrize("tiny_controller", [96], indirect=True)
@pytest.mark.parametrize("starting_adapter", [False, True])
def test_torch_reference_tracks_first_policy_exact_tokens_and_restores(tiny_controller, starting_adapter):
    import torch
    from core.learning_recall import EmptyGeneration
    from core.torch_context_learning import lora_parameters

    backend = tiny_controller
    if starting_adapter:
        backend.train_mini_batch({}, [{"prompt": "vault color", "completion": "red"}],
                                 steps=1, learning_rate=.01, validation_callback=lambda _: True)
    reference = policy_reference(backend)
    original = deepcopy(backend.model).eval()
    backend.train_mini_batch({}, [{"prompt": "vault color", "completion": "slate"}],
                             steps=2, learning_rate=.02, validation_callback=lambda _: True)
    before = {name: value.detach().clone() for name, value in lora_parameters(backend.model).items()}
    with torch.random.fork_rng():
        torch.manual_seed(91)
        for _ in range(16):
            try:
                sample = reference.sample("vault color", 3, 1., None)
                break
            except EmptyGeneration:
                continue
        else:
            pytest.fail("No usable sampled actions")
    sampled_ids = sample["rollout_prompt_ids"] + sample["rollout_completion_ids"]
    prefix = len(sample["rollout_prompt_ids"])

    def logp(model):
        with torch.no_grad():
            ids = torch.tensor([sampled_ids], device=model.get_input_embeddings().weight.device)
            logits = model(ids).logits[:, prefix - 1:-1].float().log_softmax(-1)
            targets = ids[:, prefix:, None].to(logits.device)
            return logits.gather(-1, targets).sum().item()

    assert sample["reference_logprob"] == pytest.approx(logp(original), abs=1e-5)
    assert abs(logp(backend.model) - sample["reference_logprob"]) > 1e-4
    assert all(torch.equal(value, before[name]) for name, value in lora_parameters(backend.model).items())
    def interrupt_score(module, args, kwargs):
        if kwargs.get("use_cache") is False:
            raise InterruptedError("failed frozen forward")
    handle = reference.base.register_forward_pre_hook(interrupt_score, with_kwargs=True)
    try:
        with torch.random.fork_rng(), pytest.raises(InterruptedError, match="failed frozen forward"):
            torch.manual_seed(91)
            for _ in range(16):
                try:
                    reference.sample("vault color", 3, 1., None)
                except EmptyGeneration:
                    continue
    finally:
        handle.remove()
    assert logp(backend.model) != pytest.approx(sample["reference_logprob"], abs=1e-4)
    assert all(torch.equal(value, before[name]) for name, value in lora_parameters(backend.model).items())
    stopped = threading.Event()
    stopped.set()
    with pytest.raises(InterruptedError):
        reference.sample("vault color", 3, 1., stopped)


def test_mlx_native_discover_measures_reference_across_real_updates(tmp_path, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    llama = pytest.importorskip("mlx_lm.models.llama")
    from core._mlx_engine_base import MLXReasoningBackend
    from core.learning_state import bind_learning_state

    class CharacterTokens:
        def encode(self, text, **kwargs):
            return [ord(c) % 26 for c in text]
        def decode(self, ids, **kwargs):
            return "".join(chr(104 + i) for i in ids)  # ord(c) % 26 == i

    mx.random.seed(79)
    args = llama.ModelArgs(model_type="llama", hidden_size=16, num_hidden_layers=1, intermediate_size=32,
                           num_attention_heads=2, num_key_value_heads=1, vocab_size=26, rms_norm_eps=1e-5)
    backend = MLXReasoningBackend("fixture/discover")
    backend.is_mlx_available = True
    backend.model, backend.tokenizer = llama.Model(args), CharacterTokens()
    backend.get_training_model = lambda: backend.model
    backend.context_budget_tokens = 512
    store = bind_learning_state(backend, model_info={"resolved_revision": "f"*40}, state_root=tmp_path)
    workflow = OnlineLearning(ManagedAwakeOnlineConsolidator(backend, lambda_ewc=0))
    result = workflow.discover("q", verifier=lambda text: (ord(text[0]) % 26) / 25,
                               rounds=2, rollouts=8, max_tokens=2, learning_rate=.001, entropic_kl_target=.15)
    assert result["reference_policy"] == "frozen_at_start" and result["reference_generation"] is None
    assert result["updates"] == 2, result["rounds"]
    manifest = store.current_manifest()
    metrics = manifest["metadata"]["metrics"]
    assert metrics["sampled_token_ids"] and metrics["reference_kl_coef"] == pytest.approx(.1)
    assert max(map(abs, metrics["sampled_initial_log_ratios"])) > 1e-6
    assert metrics["reference_logprob_source"] == "captured_initial_trajectory_logprobs"
    assert metrics["reference_logprob_distribution"] == "raw_causal_logits"
    assert result["rounds"][-1]["update"]["reward_validation_sources"] == ["sampled_token_ids"]
    sidecar = "sampled-rollouts.json"
    assert sidecar in manifest["artifacts"] and "_sampled_rollout_group" not in manifest["metadata"]
    artifact = store.directory / "generations" / manifest["generation"] / sidecar
    group = json.loads(artifact.read_text())
    assert len(group["samples"]) == 8 and len(group["rollout_prompt_ids"]) > 1
    assert group["reference_generation"] is None and all(len(s["rollout_completion_ids"]) == 2 for s in group["samples"])
    assert not manifest["metadata"]["memory"].get("replay_head")

    # Failure during a frozen forward must restore the live, later-generation
    # adapter even on a backend whose functional update swaps parameter leaves.
    reference = policy_reference(backend)
    from core.mlx_learning import flat_trainables
    from mlx.utils import tree_unflatten
    changed = {k: v + .25 for k, v in flat_trainables(backend.model).items()}
    backend.model.update(tree_unflatten(list(changed.items())))
    def broken_score(*args):
        assert any(not mx.array_equal(changed[k], p).item() for k, p in flat_trainables(backend.model).items())
        raise InterruptedError("interrupted during scoring")
    monkeypatch.setattr("core.policy_reference._mlx_logprob", broken_score)
    with pytest.raises(InterruptedError, match="during scoring"):
        reference.sample("q", 2, 1., None)
    assert all(mx.array_equal(changed[k], p).item() for k, p in flat_trainables(backend.model).items())

    artifact.write_text("{}")
    with pytest.raises(RuntimeError, match="integrity|verification|digest|corrupt|hash",):
        store.current_manifest()


def test_large_python_rewards_respect_declared_budgets():
    pytest.importorskip("resource")
    from core.learning_code_verifier import PythonTaskVerifier
    source = {"function": "identity", "cases": [{"args": ["ab" * 1000], "expected": "ab" * 1000}] * 50,
              "baseline_ms": 100000, "timeout_seconds": 6, "memory_mb": 256}
    verify = PythonTaskVerifier(source)
    assert verify("def identity(value):\n    return value") > 0, verify.results
    with pytest.raises(ValueError, match="max_case_chars"):
        PythonTaskVerifier(dict(source, max_case_chars=100))
    assert PythonTaskVerifier(dict(source, max_ast_nodes=3))("def identity(value):\n    return value") == 0


@pytest.mark.parametrize("tiny_controller", [256], indirect=True)
def test_real_cpu_joint_consolidation_runs_synthesis_on_the_new_generation(tiny_controller, monkeypatch):
    import torch
    runner = OnlineLearning(ManagedAwakeOnlineConsolidator(tiny_controller, lambda_ewc=0))
    generations = []
    original = tiny_controller.stream_generate_tokens
    def observe(*args, **kwargs):
        generations.append(tiny_controller.learning_state.current_manifest()["generation"])
        yield from original(*args, **kwargs)
    monkeypatch.setattr(tiny_controller, "stream_generate_tokens", observe)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        result = runner.consolidate_candidates("vault color", trajectories=[
            {"agent": "red", "completion": "red"},
            {"agent": "slate", "messages": [{"role": "user", "content": "color"}, {"role": "assistant", "content": "slate"}]}],
            mode="verifier", verifier={"kind": "exact", "expected": "slate"},
            steps=1, learning_rate=.003, max_tokens=4)
    update = result["rounds"][0]["update"]
    assert update["parameter_drift_l2"] > 0 and generations == [update["generation"]]
    assert result["best_sample"] == "slate"
    assert result["synthesized"] and result["answer_reward"] == float(result["answer"].strip() == "slate")


def test_source_uses_declared_overlap_and_evidence_budget(runner, tmp_path):
    path = tmp_path / "lesson.txt"
    path.write_text("abcde" + "fghij")
    def select(prompt, **kwargs):
        assert kwargs["max_tokens"] == 320
        return json.dumps({"spans": ["defgh"]})
    runner.generate = select
    result = runner.ingest(path, strategy="evidence", question="q" * 900, chunk_chars=5,
                           overlap_chars=2, span_chars=5000, steps=9, selection_tokens=320)
    assert result["selected_chars"] == 5 and result["trained_chunks"] == 1


def test_rollout_audit_cannot_publish_an_empty_checkpoint(tmp_path):
    backend = TransactionalBackend(tmp_path / "state")
    store = backend.learning_state
    with store.begin(metadata={"_sampled_rollout_group": {"samples": []}}) as txn:
        with pytest.raises(RuntimeError, match="no authenticated artifact"):
            txn.commit(".")
    assert store.current_manifest() is None


def test_empty_post_update_generation_preserves_accepted_candidate_receipt(runner):
    runner.generate = lambda *args, **kw: " "
    result = runner.consolidate_candidates("q", trajectories=["yes", "no"],
        verifier={"kind": "exact", "expected": "yes"}, mode="verifier")
    assert result["updates"] == 1 and result["best_sample"] == "yes"
    assert result["answer"] is None and not result["synthesized"] and result["synthesis_error"]
    assert runner.backend.learning_state.current_manifest()["generation"] == result["rounds"][0]["update"]["generation"]


@pytest.mark.parametrize("tiny_controller", [96], indirect=True)
def test_exact_sampler_preserves_model_forced_and_suppressed_token_rules(tiny_controller):
    config = tiny_controller.model.generation_config
    # The first token can only be "slate", and the model's forced last token is "color".
    config.suppress_tokens = [i for i in range(16) if i not in (3, 4)]
    config.begin_suppress_tokens = [3]
    config.forced_eos_token_id = 3
    original_options = deepcopy(config.to_dict())
    sample = policy_reference(tiny_controller).sample("vault color", 2, .6, None)
    assert sample["rollout_completion_ids"] == [4, 3]
    assert sample["reference_logprob"] < -1  # Raw model law, before forced/suppressed-token processing.
    assert config.to_dict() == original_options


def test_mlx_text_scoring_preserves_prior_multimodal_positions():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from types import SimpleNamespace
    from core.policy_reference import sampled_action_nll

    class PositionAware(nn.Module):
        def __call__(self, ids):
            assert self._position_ids is None and self._rope_deltas is None
            self._position_ids = mx.arange(ids.shape[1])
            self._rope_deltas = mx.zeros(1)
            return mx.zeros((1, ids.shape[1], 3))

    model = PositionAware()
    positions, deltas = mx.array([0, 12, 12, 12]), mx.array([10])
    model._position_ids, model._rope_deltas = positions, deltas
    backend = SimpleNamespace(model=model, is_mlx_available=True)
    loss = sampled_action_nll(backend, {"rollout_prompt_ids": [1, 2], "rollout_completion_ids": [2, 1]})
    assert loss == pytest.approx(1.09861229)
    assert model._position_ids is positions and model._rope_deltas is deltas
