import pytest

from core.reward_adaptation import reward_group


class Tokens:
    def encode(self, text, **kwargs):
        return [int(s) for s in text.split()]


def sampled_group(rewards=(1., 0.)):
    return [{"prompt": "1 2 3", "completion": str(action), "rollout_prompt_ids": [1, 2, 3],
             "rollout_completion_ids": [action], "reward": reward} for action, reward in zip([7, 8], rewards)]


@pytest.fixture
def llama(request):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    threads = torch.get_num_threads()
    with torch.random.fork_rng():
        torch.set_num_threads(1)
        torch.manual_seed(111)
        try:
            family = getattr(request, "param", "Llama")
            if not hasattr(transformers, family + "Config"):
                pytest.skip(f"The installed Transformers version does not include {family}")
            config_class = getattr(transformers, family + "Config")
            model_class = getattr(transformers, family + "ForCausalLM")
            config = config_class(vocab_size=32, hidden_size=32, intermediate_size=48,
                                              num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                              max_position_embeddings=128)
            model = model_class(config)
            yield peft.get_peft_model(model, peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")).eval()
        finally:
            torch.set_num_threads(threads)


def test_policy_group_masks_conditioning_and_rejects_truncation():
    group = reward_group(Tokens(), sampled_group(), 16)
    assert group.advantages == (1., -1.)
    assert group.windows[0].labels == (-100, -100, -100, 7)
    with pytest.raises(ValueError, match="exceeds"):
        reward_group(Tokens(), sampled_group(), 3)
    mixed = sampled_group()
    mixed[1]["rollout_prompt_ids"][0] = 4
    with pytest.raises(ValueError, match="different conditioning"):
        reward_group(Tokens(), mixed, 16)


def test_policy_moves_two_logits_in_opposite_directions(llama):
    import torch
    from core.torch_context_learning import update_lora
    def scores():
        with torch.no_grad():
            return llama(input_ids=torch.tensor([[1, 2, 3]])).logits[0, -1].log_softmax(-1)[[7, 8]]
    before = scores()
    metrics, drift = update_lora(llama, Tokens(), sampled_group(), learning_rate=.005,
                                 steps=1, max_tokens=16, objective="reward_policy")
    after = scores()
    assert after[0] > before[0] and after[1] < before[1]
    assert drift > 0 and metrics["group_size"] == 2 and metrics["supervised_tokens"] == 2
    old = {k: p.detach().clone() for k, p in llama.named_parameters()}
    for flat in ((0., 0.), (1., 1.)):
        with pytest.raises(ValueError, match="Uniform"):
            update_lora(llama, Tokens(), sampled_group(flat), learning_rate=.005, steps=1,
                        max_tokens=16, objective="reward_policy")
        assert all(torch.equal(old[k], p) for k, p in llama.named_parameters())


@pytest.mark.parametrize("llama", ["Llama", "Mistral", "Qwen2", "Qwen3"], indirect=True)
def test_frozen_query_spans_match_original_logits_and_change_only_queries(llama):
    import torch
    from torch.nn.functional import cross_entropy
    from core.query_adaptation import torch_query_loss
    from core.torch_context_learning import update_lora

    context = "1 2 3 4 5 7 3 4 8 2 6 2 4 1 5 3 4 8 6 7"
    tokens = torch.tensor([Tokens().encode(context)])
    with torch.no_grad():
        full = llama(input_ids=tokens).logits
        span_loss, spans, _ = torch_query_loss(llama, Tokens(), context, 32, 4, 3)
        for start, stop in spans:
            assert span_loss(start, stop).item() == pytest.approx(
                cross_entropy(full[0, start:stop], tokens[0, start + 1:stop + 1]).item(), abs=1e-6)
    saved = {k: p.detach().clone() for k, p in llama.named_parameters()}
    seen = []
    hook = llama.get_base_model().base_model.register_forward_pre_hook(lambda m, a, kw: seen.append(kw["input_ids"].shape[-1]), with_kwargs=True)
    try:
        metrics, drift = update_lora(llama, Tokens(), [], learning_rate=.005, steps=3, max_tokens=4,
                                     context_text=context, max_context_tokens=32, objective="query_only")
    finally:
        hook.remove()
    assert seen == [20, 4, 4, 4]
    assert drift > 0 and metrics["kv_prefills"] == 1
    changed = [k for k, p in llama.named_parameters() if not torch.equal(p, saved[k])]
    assert changed and all(".q_proj.lora_" in k for k in changed)
    assert all(("lora_" in k) == p.requires_grad for k, p in llama.named_parameters())
    assert not any(p.grad is not None for p in llama.parameters())


def test_native_mlx_fixed_kv_updates_only_query():
    mx = pytest.importorskip("mlx.core", reason="Requires the native MLX runtime")
    llama = pytest.importorskip("mlx_lm.models.llama")
    from core._mlx_engine_base import MLXReasoningBackend
    from core.query_adaptation import mlx_query_loss
    from mlx.utils import tree_flatten
    import mlx.nn as nn

    mx.random.seed(9)
    args = llama.ModelArgs(model_type="llama", hidden_size=16, num_hidden_layers=2,
                           intermediate_size=32, num_attention_heads=4, num_key_value_heads=2,
                           vocab_size=32, rms_norm_eps=1e-5)
    backend = MLXReasoningBackend("fixture/llama")
    backend.model, backend.tokenizer, backend.is_mlx_available = llama.Model(args), Tokens(), True
    backend.model.eval()
    context = "1 2 3 4 5 6 7 4 3 2 1 2 3 4 5 6"
    ids = mx.array([Tokens().encode(context)])
    original = backend.model(ids)
    scorer, spans, _ = mlx_query_loss(backend.model, backend.tokenizer, context, 16, 4, 2)
    for start, stop in spans:
        expected = nn.losses.cross_entropy(original[:, start:stop], ids[:, start + 1:stop + 1], reduction="mean")
        assert scorer(backend.model, start, stop).item() == pytest.approx(expected.item(), abs=1e-5)
    backend.inject_lora_adapters()
    before = dict(tree_flatten(backend.model.parameters()))
    updated, drift = backend.train_mini_batch({}, [], objective="query_only", context_text=context, steps=2,
                                              max_window_tokens=4, max_context_tokens=16,
                                              validation_callback=lambda candidate: True)
    assert drift > 0 and backend.last_learning_metrics["kv_prefills"] == 1
    changed = [k for k, v in tree_flatten(backend.model.parameters()) if not mx.array_equal(v, before[k]).item()]
    assert changed and all("q_proj.lora_" in k for k in changed)


def test_entropic_calibration_and_leave_one_out_are_scale_invariant():
    from math import exp, log
    from core.reward_adaptation import reward_advantages
    rewards = (.05, .4, .55, .99)
    original = {}
    _, advantages = reward_advantages([{"reward": r} for r in rewards], reward_weighting="entropic",
                                       diagnostics=original)
    beta = original["entropic_beta"]
    masses = [exp(beta * (r - max(rewards))) for r in rewards]
    prob = [w / sum(masses) for w in masses]
    assert sum(p * log(4 * p) for p in prob) == pytest.approx(log(2), abs=1e-10)
    eps = original["entropic_loo_epsilon"]
    for i, a in enumerate(advantages):
        peers = sum(w for j, w in enumerate(masses) if j != i) / 3
        assert a == pytest.approx(masses[i] / (eps + peers) - 1)
    for scale, offset in ((.3, .44), (1e-10, 0.), (1e-200, 0.)):
        metrics = {}
        _, transformed = reward_advantages([{"reward": scale * r + offset} for r in rewards],
                                            reward_weighting="entropic", diagnostics=metrics)
        assert transformed == pytest.approx(advantages, rel=1e-12)
        assert metrics["entropic_beta"] * scale == pytest.approx(beta)
    assert not original["entropic_kl_saturated"]


def test_entropic_ties_and_unreachable_budget_remain_finite():
    from math import isfinite, log
    from core.reward_adaptation import reward_advantages
    report = {}
    _, a = reward_advantages([{"reward": r} for r in (0, 1, 1)], reward_weighting="entropic",
                             entropic_kl_target=2.0, diagnostics=report)
    assert report["entropic_kl_saturated"] and report["entropic_kl_achieved"] == pytest.approx(log(1.5))
    assert all(isfinite(x) for x in a) and a[0] < 0 < a[1] == a[2]
    tiny = {}
    _, b = reward_advantages([{"reward": r} for r in (0., 5e-324)], reward_weighting="entropic", diagnostics=tiny)
    assert tiny["entropic_beta_clamped"] and all(isfinite(x) for x in b)
    for gamma in (float("nan"), float("inf"), 0, -1):
        with pytest.raises(ValueError, match="KL target"):
            reward_advantages(sampled_group(), reward_weighting="entropic", entropic_kl_target=gamma)
    with pytest.raises(ValueError, match="Unknown reward weighting"):
        reward_advantages(sampled_group(), reward_weighting="fake")


def test_entropic_step_uses_measured_trajectory_logprobs_and_restores_on_failure(llama):
    import torch
    from core.torch_context_learning import update_lora
    rows = sampled_group()
    # Use two-token completions to distinguish trajectory sums from token means.
    for row in rows:
        row["rollout_completion_ids"] *= 2
    def scores():
        with torch.no_grad():
            vals = []
            for row in rows:
                prefix, actions = row["rollout_prompt_ids"], row["rollout_completion_ids"]
                logits = llama(input_ids=torch.tensor([prefix + actions])).logits[0].log_softmax(-1)
                vals.append(sum(logits[len(prefix) - 1 + i, token].item() for i, token in enumerate(actions)))
            return vals

    # Actual initial policy scores for these exact sampled actions.
    initial = scores()
    options = dict(learning_rate=.003, steps=1, max_tokens=16, objective="reward_policy",
                   reward_weighting="entropic", entropic_kl_target=.15)
    plain, _ = update_lora(llama, Tokens(), rows, **options)
    assert plain["reward_weighting"] == "entropic" and plain["reference_logprob_source"] == "disabled"
    current = scores()
    assert current[0] - current[1] > initial[0] - initial[1]
    snapshot = {k: p.detach().clone() for k, p in llama.named_parameters()}
    for invalid_rows, coef, reason in ((rows, .1, "reference_logprob"),
             ([dict(r, reward=0.) for r in rows], 0., "Uniform"), (rows, float("nan"), "coefficient")):
        with pytest.raises(ValueError, match=reason):
            update_lora(llama, Tokens(), invalid_rows, reference_kl_coef=coef, **options)
        assert all(torch.equal(p, snapshot[k]) for k, p in llama.named_parameters())
    for row, score in zip(rows, initial):
        row["reference_logprob"] = score
    receipt, drift = update_lora(llama, Tokens(), rows, reference_kl_coef=.5, **options)
    ratios = [a - b for a, b in zip(current, initial)]
    assert receipt["sampled_initial_log_ratios"] == pytest.approx(ratios, abs=2e-6)
    assert receipt["shaped_advantages"] == pytest.approx([a - .5 * r for a, r in zip(plain["advantages"], ratios)], abs=2e-6)
    assert drift > 0 and receipt["reference_logprob_source"] == "supplied_initial_trajectory_logprobs"


def test_entropic_validation_requires_a_relative_improvement():
    from core.torch_context_learning import default_validation

    class Probe:
        def __init__(self, scores):
            self.scores = scores

        def score_learning_probe(self, prompt, answer):
            return self.scores.get(answer, 1.)

    rows = sampled_group()
    check = default_validation(Probe({"7": 2., "8": 4.}), rows, objective="reward_policy", reward_weighting="entropic")
    assert not check(Probe({"7": 1.5, "8": 3.5}))["accepted"]
    assert not check(Probe({"7": 1.5, "8": 3.}))["accepted"]
    assert check(Probe({"7": 1., "8": 3.5}))["accepted"]
    decimal = default_validation(Probe({"7": 1.2, "8": 1.}), rows,
                                 objective="reward_policy", reward_weighting="entropic")
    assert not decimal(Probe({"7": 1.1, "8": .9}))["accepted"]
