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
