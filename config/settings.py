"""Full cross-platform settings plus continual-learning controls."""
from __future__ import annotations
from dataclasses import dataclass, is_dataclass
from config._settings_base import *
from config import _settings_base as _base
_BaseSettings=_base.Settings

if is_dataclass(_BaseSettings):
    @dataclass
    class Settings(_BaseSettings):
        enable_awake_ogp_daemon: bool=True
        polling_interval_seconds: float=300.0
        daemon_queue_check_seconds: float=2.0
        min_surprise_threshold: float=0.85
        min_batch_queue_size: int=5
        ogp_ortho_tolerance: float=1e-5
        ogp_anchor_samples: int=8
        base_learning_rate: float=1e-4
        lora_rank: int=4
        lora_alpha: float=8.0
        lora_chunk_size: int=6
        h2o_sink_tokens: int=4
        h2o_heavy_tokens: int=1024
        h2o_max_budget: int=2048
        moe_adapter_bank_dir: str="adapter_banks"
        enable_neuromorphic_lif: bool=True
else:
    from pydantic import Field
    class Settings(_BaseSettings):
        enable_awake_ogp_daemon: bool=Field(default=True)
        polling_interval_seconds: float=Field(default=300.0)
        daemon_queue_check_seconds: float=Field(default=2.0)
        min_surprise_threshold: float=Field(default=0.85)
        min_batch_queue_size: int=Field(default=5)
        ogp_ortho_tolerance: float=Field(default=1e-5)
        ogp_anchor_samples: int=Field(default=8)
        base_learning_rate: float=Field(default=1e-4)
        lora_rank: int=Field(default=4)
        lora_alpha: float=Field(default=8.0)
        lora_chunk_size: int=Field(default=6)
        h2o_sink_tokens: int=Field(default=4)
        h2o_heavy_tokens: int=Field(default=1024)
        h2o_max_budget: int=Field(default=2048)
        moe_adapter_bank_dir: str=Field(default="adapter_banks")
        enable_neuromorphic_lif: bool=Field(default=True)

def get_settings(**overrides):
    return Settings(**overrides)
