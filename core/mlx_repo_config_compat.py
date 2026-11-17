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
            cfg = original(params)
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
