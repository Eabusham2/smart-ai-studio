"""Normalize nested qwen3_5 config records across mlx-vlm releases.

Some mlx-vlm versions return a ModelConfig whose nested text/vision config
is still a dict. LanguageModel expects typed objects with .model_type.
Scope this compatibility hook strictly to the selected repository load.
"""
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import threading

_LOCK = threading.RLock()


def _typed(value, cls):
    if not isinstance(value, dict):
        return value
    if not is_dataclass(cls):
        raise TypeError("Unsupported mlx-vlm config class")
    accepted = {f.name for f in fields(cls)}
    # Preserve the publisher's fields that this runtime understands; never
    # invent architecture values or change the checkpoint/weight format.
    return cls(**{k: v for k, v in value.items() if k in accepted})


@contextmanager
def compatible_qwen35_configs():
    with _LOCK:
        from mlx_vlm.models.qwen3_5 import ModelConfig, TextConfig, VisionConfig
        descriptor = ModelConfig.__dict__.get("from_dict")
        if descriptor is None:
            raise RuntimeError("mlx-vlm qwen3_5 ModelConfig.from_dict is unavailable")
        original = ModelConfig.from_dict

        def normalized(cls, params):
            # MLX-VLM 0.5.0 directly instantiates ModelConfig from the raw
            # JSON dict. Pre-convert nested configs BEFORE __post_init__ runs:
            # some releases dereference text_config.model_type there.
            typed = dict(params)
            typed["text_config"] = _typed(typed.get("text_config"), TextConfig)
            typed["vision_config"] = _typed(typed.get("vision_config"), VisionConfig)
            cfg = original(typed)
            if isinstance(cfg, dict):
                cfg = cls(**cfg)
            cfg.text_config = _typed(cfg.text_config, TextConfig)
            cfg.vision_config = _typed(cfg.vision_config, VisionConfig)
            return cfg

        try:
            ModelConfig.from_dict = classmethod(normalized)
            yield
        finally:
            ModelConfig.from_dict = descriptor


def _no_metal_kernel(*args, **kwargs):
    # Returning None selects mlx-vlm's existing per-token/per-pad-group path.
    return None


def enable_portable_qwen35_verification():
    """Use mlx-vlm 0.6.3's native fallback for Metal-only MTP fast paths.

    GatedDeltaNet already dispatches to differentiable MLX ops on CPU/CUDA.
    These optional projection and ragged-attention helpers omit that check in
    0.6.3. Their callers implement the identical operation when they return None.
    """
    import mlx.core as mx
    if mx.metal.is_available():
        return
    from importlib.metadata import version
    if version("mlx-vlm") != "0.6.3":
        raise RuntimeError("Bonsai 2.5 requires Studio's pinned mlx-vlm 0.6.3 runtime.")
    from mlx_vlm.models.qwen3_5 import language
    with _LOCK:
        for name in ("_target_verify_weight", "_target_verify_quantized_linear",
                     "_target_verify_quantized_argmax", "_qwen3_5_ragged_decode_attention"):
            if not callable(getattr(language, name, None)):
                raise RuntimeError(f"Missing mlx-vlm Qwen3.5 verification helper: {name}")
            setattr(language, name, _no_metal_kernel)
