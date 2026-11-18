"""Report test-time objectives exposed by the currently loaded serving backend."""
from __future__ import annotations

from core.query_adaptation import MLX_QUERY_MODELS, TORCH_QUERY_MODELS, is_query_adapter


def adaptation_capabilities(backend):
    result = {"causal_lm": False, "reward_policy": False, "query_only": False, "reasons": {}}
    reasons = result["reasons"]
    model = getattr(backend, "model", None)
    runtime = str(getattr(backend, "runtime", ""))
    scope = getattr(getattr(backend, "learning_state", None), "scope", None)
    if getattr(scope, "pin", ""):
        reason = "The selected document generation is read-only."
    elif model is None:
        reason = "Load a model before requesting parameter adaptation."
    elif callable(getattr(backend, "get_training_model", None)) and getattr(backend, "is_mlx_available", False):
        def inspect_language_model():
            from core.mlx_learning import _projection_targets
            model = backend.get_training_model()
            packed = getattr(backend, "_vlm_runtime", None) in {"mlx_repo_vlm", "jang_vlm"}
            existing = [name for name, m in model.named_modules() if hasattr(m, "lora_a") and hasattr(m, "lora_b")]
            if packed:
                reason = "Packed MLX learning exposes output-head adapters only."
                return bool(existing or callable(getattr(model, "lm_head", None))), False, reason
            targets = list(_projection_targets(model, None))
            import mlx.nn as nn
            trainable = bool(existing) or bool(targets) and all(
                isinstance(getattr(parent, key), (nn.Linear, nn.QuantizedLinear)) for parent, key in targets)
            queries = any(is_query_adapter(name) for name in existing) if existing else any(k == "q_proj" for _, k in targets)
            query_ready = bool(trainable and queries and type(model).__module__.split(".")[-1] in MLX_QUERY_MODELS)
            if query_ready:
                try:
                    from mlx_lm.models.cache import KVCache, make_prompt_cache
                    layers = make_prompt_cache(model)
                    query_ready = bool(layers) and all(type(layer) is KVCache for layer in layers)
                except (ImportError, AttributeError, ValueError):
                    query_ready = False
            return trainable, query_ready, "This MLX language decoder has no supported full-attention query-cache adapter."

        owner = getattr(backend, "run_on_mlx_owner", None)
        try:
            ready, query_ready, reason = (owner(inspect_language_model)
                if callable(owner) and getattr(backend, "_mlx_owner", None) is not None else inspect_language_model())
            result["causal_lm"] = result["reward_policy"] = ready
            result["query_only"] = query_ready
        except (ImportError, ValueError, AttributeError, RuntimeError) as error:
            reason = f"Cannot verify the selected MLX decoder: {error}"
    elif runtime in {"transformers", "hf_transformers", "transformers_auto"}:
        ready = bool(backend.training_ready())
        result["causal_lm"] = result["reward_policy"] = ready
        base_cfg = getattr(model, "config", None)
        if getattr(model, "peft_config", None):
            targets = [name for name, _ in model.named_modules() if is_query_adapter(name) and ".lora_A." in name]
        else:
            targets = [name for name in backend._transformers_lora_targets(model) if is_query_adapter(name)] if ready else []
        result["query_only"] = bool(ready and targets and getattr(base_cfg, "model_type", "") in TORCH_QUERY_MODELS
                                    and getattr(base_cfg, "_attn_implementation", None) in {None, "eager", "sdpa"})
        reason = ("This controller is inference-only or has no trainable language projection." if not ready else
                  "Query-only adaptation requires a supported split query projection and full KV cache.")
    elif callable(getattr(backend, "_verify_training_parent", None)):
        ready = bool(backend.training_ready())
        result["causal_lm"] = result["reward_policy"] = ready
        reason = ("The native runtime cannot differentiate queries against its serving KV cache." if ready else
                  "Native adaptation needs a matching local training parent, memory and converter.")
    else:
        reason = getattr(backend, "learning_unavailable_reason", None) or (
            "The selected serving backend exposes no validated parameter update path.")
    for key in ("causal_lm", "reward_policy", "query_only"):
        if not result[key]:
            reasons[key] = reason
    return result
