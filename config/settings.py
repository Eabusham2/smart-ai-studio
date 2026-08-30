import os
import psutil
from dataclasses import dataclass, field

def compute_auto_kv_budget(total_ram_gb: float) -> int:
    """Dynamically sizes the KV token arena based on physical unified memory budget."""
    if total_ram_gb <= 8.0:
        return 1024
    elif total_ram_gb <= 16.0:
        return 2048
    elif total_ram_gb <= 32.0:
        return 4096
    return 8192

@dataclass
class EngineSettings:
    total_ram_gb: float = field(default_factory=lambda: psutil.virtual_memory().total / (1024 ** 3))
    mlx_model_path: str = "prism-ml/Ternary-Bonsai-27B-mlx-2bit"
    max_kv_tokens: int = field(init=False)
    h2o_sink_tokens: int = 4
    h2o_heavy_tokens: int = 64
    h2o_max_budget: int = 128
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_chunk_size: int = 6
    total_layers: int = 60
    base_learning_rate: float = 1e-4
    ogp_ortho_tolerance: float = 1e-5
    polling_interval_seconds: float = 300.0
    min_surprise_threshold: float = 0.85
    min_batch_queue_size: int = 5
    sandbox_timeout_seconds: float = 4.0
    sandbox_max_memory_mb: int = 512
    enable_awake_ogp_daemon: bool = True
    db_path: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "memory.db")

    def __post_init__(self):
        self.max_kv_tokens = compute_auto_kv_budget(self.total_ram_gb)

# Backward-compatibility alias
Settings = EngineSettings

def get_settings() -> EngineSettings:
    return EngineSettings()
