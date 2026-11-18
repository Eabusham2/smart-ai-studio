"""Sample actual actions and score them with the policy at the start of a discovery run.

Only adapter leaves need a snapshot: learning freezes the base model. All model
operations, including restoring an MLX view after an error, share the serving lock.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import deepcopy
import math
import inspect

from core.learning_recall import EmptyGeneration, check_cancel, generation_prefix, generation_room
from core.native_speed import runtime_lock


def _decode(tokenizer, ids):
    try:
        result = tokenizer.decode(ids, skip_special_tokens=True)
    except TypeError:
        result = tokenizer.decode(ids)
    if not isinstance(result, str) or not result.strip():
        raise EmptyGeneration("The sampled trajectory contains no decodable answer")
    return result


def _trajectory(text, prefix, actions, logprob=None):
    row = {"completion": text, "rollout_prompt_ids": prefix, "rollout_completion_ids": actions}
    if logprob is None:
        return row
    if not math.isfinite(logprob) or logprob > 0:
        raise RuntimeError("The frozen policy did not produce a finite trajectory log probability")
    return {**row, "reference_logprob": logprob, "reference_logprob_source": "captured_initial_trajectory_logprobs"}


def _base(model):
    getter = getattr(model, "get_base_model", None)
    return getter() if callable(getter) else model


def _torch_logprob(model, sequence, prefix_length, factors=None):
    import torch
    kwargs = {"input_ids": sequence[:, :-1], "use_cache": False}
    count = sequence.shape[1] - prefix_length
    forward = inspect.signature(_base(model).forward).parameters
    if "logits_to_keep" in forward:
        kwargs["logits_to_keep"] = count
    elif "num_logits_to_keep" in forward:
        kwargs["num_logits_to_keep"] = count
    output = torch.func.functional_call(model, factors, (), kwargs) if factors else model(**kwargs)
    logits = output.logits[0, -count:].float().log_softmax(-1)
    targets = sequence[0, prefix_length:].to(logits.device)
    return logits.gather(1, targets[:, None]).sum().item()


def _mlx_logprob(model, prefix, actions):
    import mlx.core as mx
    import mlx.nn as nn
    ids = mx.array([prefix + actions])
    output = model(ids[:, :-1])
    logits = getattr(output, "logits", output)[:, len(prefix) - 1:].astype(mx.float32)
    return -float(nn.losses.cross_entropy(logits, ids[:, len(prefix):], reduction="sum").item())


@contextmanager
def private_text_positions(model):
    """Keep a text-only forward independent of a VLM's previous image positions."""
    saved = {key: getattr(model, key) for key in ("_position_ids", "_rope_deltas") if hasattr(model, key)}
    try:
        for key in saved:
            setattr(model, key, None)
        yield
    finally:
        for key, value in saved.items():
            setattr(model, key, value)


def _mlx_call(backend, operation, *args):
    def locked():
        with runtime_lock(backend), private_text_positions(_mlx_model(backend)):
            return operation(*args)
    owner = getattr(backend, "run_on_mlx_owner", None)
    return owner(locked) if callable(owner) else locked()


def _mlx_model(backend):
    getter = getattr(backend, "get_training_model", None)
    return getter() if callable(getter) else backend.model


def sampled_action_nll(backend, row):
    """Mean NLL on exactly the sampled actions, including stop tokens, on supported serving models."""
    from core.reward_adaptation import _token_ids
    prefix = list(_token_ids(row["rollout_prompt_ids"]))
    actions = list(_token_ids(row["rollout_completion_ids"]))
    if getattr(backend, "runtime", "") in {"transformers", "hf_transformers", "transformers_auto"}:
        import torch
        from core.transformers_placement import transformers_input_device
        with runtime_lock(backend), torch.no_grad():
            model = backend.model
            training = model.training
            try:
                model.eval()
                ids = torch.tensor([prefix + actions], device=transformers_input_device(model))
                return -_torch_logprob(model, ids, len(prefix)) / len(actions)
            finally:
                model.train(training)
    if getattr(backend, "is_mlx_available", False):
        def score():
            model = _mlx_model(backend)
            training = model.training
            try:
                model.eval()
                return -_mlx_logprob(model, prefix, actions) / len(actions)
            finally:
                model.train(training)
        return _mlx_call(backend, score)
    return None


class TorchReference:
    sampling_rule = "model_generation_config"

    def __init__(self, backend, frozen):
        from core.torch_context_learning import lora_parameters

        self.backend = backend
        self.frozen = frozen
        with runtime_lock(backend):
            model = backend.model
            self.base = _base(model)
            self.factors = ({k: p.detach().clone() for k, p in lora_parameters(model).items()}
                            if frozen and getattr(model, "peft_config", None) else {})

    def sample(self, prompt, max_tokens, temperature, cancel_event):
        import torch
        from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
        from core.transformers_placement import transformers_input_device

        class Cancel(StoppingCriteria):
            def __call__(self, *args, **kwargs):
                check_cancel(cancel_event)
                return False

        backend = self.backend
        with runtime_lock(backend), torch.no_grad():
            check_cancel(cancel_event)
            model = backend.model
            if _base(model) is not self.base:
                raise RuntimeError("The serving model changed during discovery")
            training = model.training
            model.eval()
            try:
                prefix_text = generation_prefix(backend, prompt)
                room = generation_room(backend, prefix_text, max_tokens)
                prefix = backend.tokenizer.encode(prefix_text)
                if not prefix:
                    raise ValueError("The rollout prompt tokenized to an empty sequence")
                ids = torch.tensor([prefix], device=transformers_input_device(model), dtype=torch.long)
                config = getattr(model, "generation_config", None)
                sampling = deepcopy(config) if config is not None else GenerationConfig.from_model_config(model.config)
                sampling.max_new_tokens = room
                sampling.do_sample = bool(temperature)
                sampling.temperature = temperature or 1.
                # Every sample represents one completion. Keep model-specific
                # EOS, forced/suppressed tokens, stop strings and logits rules.
                sampling.num_return_sequences = 1
                sampling.return_dict_in_generate = False
                options = {"tokenizer": backend.tokenizer} if getattr(sampling, "stop_strings", None) else {}
                sequence = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **options,
                    generation_config=sampling, stopping_criteria=StoppingCriteriaList([Cancel()]))
                check_cancel(cancel_event)
                actions = sequence[0, ids.shape[1]:].tolist()
                answer = _decode(backend.tokenizer, actions)
                if not self.frozen:
                    return _trajectory(answer, list(prefix), actions)
                # Score all sampled actions, including EOS, under the raw frozen
                # causal policy. No decode/re-encode step can change these IDs.
                inactive = (model.disable_adapter() if not self.factors and getattr(model, "peft_config", None)
                            else nullcontext())
                with inactive:
                    logprob = _torch_logprob(model, sequence, len(prefix), self.factors)
                return _trajectory(answer, list(prefix), actions, logprob)
            finally:
                model.train(training)


class MLXReference:
    sampling_rule = "temperature_categorical_or_greedy"

    def __init__(self, backend, frozen):
        self.backend = backend
        self.frozen = frozen
        _mlx_call(backend, self._capture)

    @staticmethod
    def _factors(model):
        from mlx.utils import tree_flatten
        from core.mlx_learning import is_adapter_key
        return {key: leaf for key, leaf in tree_flatten(model.parameters()) if is_adapter_key(key)}

    def _capture(self):
        import mlx.core as mx
        self.base = _mlx_model(self.backend)
        self.factors = self._factors(self.base) if self.frozen else {}
        if self.factors:
            mx.eval(*self.factors.values())

    def sample(self, prompt, max_tokens, temperature, cancel_event):
        return _mlx_call(self.backend, self._sample, prompt, max_tokens, temperature, cancel_event)

    def _sample(self, prompt, max_tokens, temperature, cancel_event):
        import mlx.core as mx
        from mlx.utils import tree_unflatten

        check_cancel(cancel_event)
        model = _mlx_model(self.backend)
        if model is not self.base:
            raise RuntimeError("The serving model changed during discovery")
        backend = self.backend
        prefix_text = generation_prefix(backend, prompt)
        room = generation_room(backend, prefix_text, max_tokens)
        tokenizer = backend.tokenizer
        prefix = list(tokenizer.encode(prefix_text))
        if not prefix:
            raise ValueError("The rollout prompt tokenized to an empty sequence")
        eos = getattr(tokenizer, "eos_token_ids", None) or getattr(tokenizer, "eos_token_id", None)
        stops = set(eos) if isinstance(eos, (list, tuple, set)) else {eos}
        training = model.training
        model.eval()
        try:
            actions = []
            # Standard MLX-LM decoders can reuse a private KV cache. Other
            # trainable text heads expose the same full forward used by their trainer.
            cache = None
            if callable(getattr(model, "make_cache", None)):
                cache = model.make_cache()
            elif type(model).__module__.startswith("mlx_lm.models."):
                from mlx_lm.models.cache import make_prompt_cache
                cache = make_prompt_cache(model)
            for _ in range(room):
                check_cancel(cancel_event)
                inputs = actions[-1:] if cache is not None and actions else prefix + actions
                ids = mx.array([inputs])
                output = model(ids, cache=cache) if cache is not None else model(ids)
                logits = getattr(output, "logits", output)[0, -1].astype(mx.float32)
                token = int((mx.random.categorical(logits / temperature) if temperature else mx.argmax(logits)).item())
                actions.append(token)
                if token in stops:
                    break
            check_cancel(cancel_event)
            answer = _decode(tokenizer, [a for a in actions if a not in stops])
            if not self.frozen:
                return _trajectory(answer, prefix, actions)
            current = self._factors(model)
            if self.factors and self.factors.keys() != current.keys():
                raise RuntimeError("The MLX adapter layout changed during discovery")
            # A fresh LoRA residual was absent at the initial checkpoint. Zero B
            # reconstructs that policy after the first update without changing A.
            initial = self.factors or {k: mx.zeros_like(v) if k.endswith("lora_b") else v for k, v in current.items()}
            try:
                if initial:
                    model.update(tree_unflatten(list(initial.items())))
                return _trajectory(answer, prefix, actions, _mlx_logprob(model, prefix, actions))
            finally:
                if current:
                    model.update(tree_unflatten(list(current.items())))
        finally:
            model.train(training)


def policy_reference(backend, *, frozen=True):
    tokenizer = getattr(backend, "tokenizer", None)
    if not callable(getattr(tokenizer, "encode", None)) or not callable(getattr(tokenizer, "decode", None)):
        return None
    if getattr(backend, "runtime", "") in {"transformers", "hf_transformers", "transformers_auto"}:
        return TorchReference(backend, frozen)
    if getattr(backend, "is_mlx_available", False) and getattr(backend, "model", None) is not None:
        return MLXReference(backend, frozen)
    return None
