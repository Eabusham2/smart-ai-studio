"""
Configuration management for the Autonomous Reasoning Engine.
Provides strongly-typed settings with environment variable support,
cross-platform detection (macOS / Apple Silicon MLX, Windows, Linux/Unix),
speculative acceleration configuration, and fallback defaults.
"""

import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


def detect_system_platform() -> str:
    """Detects current operating system family."""
    system = platform.system().lower()
    if "darwin" in system:
        return "macos"
    elif "windows" in system:
        return "windows"
    else:
        return "linux"


def detect_optimal_device() -> str:
    """Detects best available compute hardware across platforms."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return "cuda"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mps"

    return "cpu"


def detect_optimal_backend() -> str:
    """Detects optimal inference/training engine: mlx, torch, or mock."""
    env_backend = os.getenv("BACKEND", "").lower()
    if env_backend in ("mlx", "torch", "mock"):
        return env_backend

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx.core as mx
            import mlx_lm
            return "mlx"
        except ImportError:
            pass

    return "torch"


try:
    from pydantic import Field
    from pydantic_settings import BaseSettings

    class Settings(BaseSettings):
        # Platform & Backend
        os_platform: str = Field(default_factory=detect_system_platform, description="Detected OS platform")
        backend: str = Field(default_factory=detect_optimal_backend, description="Inference engine backend: mlx, torch")
        device: str = Field(default_factory=detect_optimal_device, description="Compute device: cuda, mps, cpu")
        live_mode: bool = Field(default=True, description="Enable live neural model execution")
        small_model: bool = Field(default=False, description="Use lightweight local model fallback")
        kv_bits: int = Field(default=4, description="KV-cache quantization bitwidth (4-bit for 16GB M1 Mac)")

        # Hardware & Model Configuration (Dedicated 27B Ternary Architecture)
        base_model_path: str = Field(default="prism-ml/Ternary-Bonsai-27B-mlx-2bit", description="Path/ID for 27B Ternary base model")
        mlx_model_path: str = Field(default="prism-ml/Ternary-Bonsai-27B-mlx-2bit", description="Path/ID for Apple MLX 27B 2-bit/1.58-bit model")
        lora_adapter_path: Optional[str] = Field(default=None, description="Path to consolidated Slow-LoRA adapter")
        small_model_path: str = Field(default="prism-ml/Ternary-Bonsai-mlx-2bit", description="Model checkpoint ID")
        flash_model_path: str = Field(default="Qwen/Qwen-3.8B-Flash-Next-1.58bit", description="Qwen 3.8 Flash Next 1.58-bit model checkpoint")

        # Speculative Acceleration (Lossless Decoding Speedups)
        speculative_mode: str = Field(default="pld", description="Speculative decoding mode: pld, lookahead, dflash, eagle, medusa, none")
        speculative_tokens: int = Field(default=4, description="Number of draft tokens K to speculate ahead")
        draft_model_path: Optional[str] = Field(default=None, description="Path to secondary draft model or DFlash drafter")
        draft_device: Optional[str] = Field(default=None, description="Device for secondary draft head")

        # Reasoning & Test-Time Search
        entropy_low_threshold: float = Field(default=0.25, description="Low entropy cutoff for Instant (N=1)")
        entropy_high_threshold: float = Field(default=0.70, description="High entropy cutoff for 16 branches")
        pro_branch_count_default: int = Field(default=16, description="Parallel candidate sequence count")
        instant_branch_count: int = Field(default=1, description="Single-pass branch count")
        search_temperature: float = Field(default=0.75, description="Sampling temperature for Pro rollouts")
        search_top_p: float = Field(default=0.92, description="Top-p nucleus sampling cutoff")
        max_new_tokens: int = Field(default=1536, description="Max generated reasoning tokens")
        prm_pruning_threshold: float = Field(default=0.40, description="Process reward model step cutoff")

        # Ground-Truth Sandbox Verification
        sandbox_timeout_seconds: float = Field(default=4.0, description="Subprocess timeout cap in seconds")
        use_docker_sandbox: bool = Field(default=False, description="Whether to route sandbox to Docker")
        docker_image: str = Field(default="python:3.10-slim", description="Docker base image for sandbox")

        # Memory & Elastic Weight Consolidation (EWC)
        database_path: str = Field(default_factory=lambda: os.getenv("DATABASE_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "memory.db")), description="SQLite episodic memory database path")
        ewc_lambda: float = Field(default=400.0, description="Synaptic quadratic penalty multiplier")
        consolidation_lr: float = Field(default=2e-5, description="Learning rate for slow-LoRA sleep consolidation")
        consolidation_weight_decay: float = Field(default=0.01, description="Weight decay for AdamW in sleep cycle")
        episodic_replay_ratio: float = Field(default=0.25, description="Ratio of user memories vs anchor replay")
        idle_sleep_threshold_minutes: int = Field(default=30, description="Inactivity time before sleep cycle triggers")

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            extra = "ignore"

except ImportError:
    @dataclass
    class Settings:
        os_platform: str = field(default_factory=detect_system_platform)
        backend: str = field(default_factory=detect_optimal_backend)
        device: str = field(default_factory=detect_optimal_device)
        live_mode: bool = os.getenv("LIVE_MODE", "true").lower() in ("1", "true", "yes")
        small_model: bool = os.getenv("SMALL_MODEL", "false").lower() in ("1", "true", "yes")
        kv_bits: int = int(os.getenv("KV_BITS", "4"))

        base_model_path: str = os.getenv("BASE_MODEL_PATH", "prism-ml/Ternary-Bonsai-27B-mlx-2bit")
        mlx_model_path: str = os.getenv("MLX_MODEL_PATH", "prism-ml/Ternary-Bonsai-27B-mlx-2bit")
        lora_adapter_path: Optional[str] = os.getenv("LORA_ADAPTER_PATH", None)
        small_model_path: str = os.getenv("SMALL_MODEL_PATH", "prism-ml/Ternary-Bonsai-27B-mlx-2bit")
        flash_model_path: str = os.getenv("FLASH_MODEL_PATH", "Qwen/Qwen-3.8B-Flash-Next-1.58bit")

        speculative_mode: str = os.getenv("SPECULATIVE_MODE", "pld")
        speculative_tokens: int = int(os.getenv("SPECULATIVE_TOKENS", "4"))
        draft_model_path: Optional[str] = os.getenv("DRAFT_MODEL_PATH", None)
        draft_device: Optional[str] = os.getenv("DRAFT_DEVICE", None)

        entropy_low_threshold: float = float(os.getenv("ENTROPY_LOW_THRESHOLD", "0.25"))
        entropy_high_threshold: float = float(os.getenv("ENTROPY_HIGH_THRESHOLD", "0.70"))
        pro_branch_count_default: int = int(os.getenv("PRO_BRANCH_COUNT_DEFAULT", "16"))
        instant_branch_count: int = int(os.getenv("INSTANT_BRANCH_COUNT", "1"))
        search_temperature: float = float(os.getenv("SEARCH_TEMPERATURE", "0.75"))
        search_top_p: float = float(os.getenv("SEARCH_TOP_P", "0.92"))
        max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "1536"))
        prm_pruning_threshold: float = float(os.getenv("PRM_PRUNING_THRESHOLD", "0.40"))

        sandbox_timeout_seconds: float = float(os.getenv("SANDBOX_TIMEOUT_SECONDS", "4.0"))
        use_docker_sandbox: bool = os.getenv("USE_DOCKER_SANDBOX", "false").lower() in ("1", "true", "yes")
        docker_image: str = os.getenv("DOCKER_IMAGE", "python:3.10-slim")

        database_path: str = os.getenv("DATABASE_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "memory.db"))
        ewc_lambda: float = float(os.getenv("EWC_LAMBDA", "400.0"))
        consolidation_lr: float = float(os.getenv("CONSOLIDATION_LR", "0.00002"))
        consolidation_weight_decay: float = float(os.getenv("CONSOLIDATION_WEIGHT_DECAY", "0.01"))
        episodic_replay_ratio: float = float(os.getenv("EPISODIC_REPLAY_RATIO", "0.25"))
        idle_sleep_threshold_minutes: int = int(os.getenv("IDLE_SLEEP_THRESHOLD_MINUTES", "30"))


_global_settings: Optional[Settings] = None


def get_settings(**kwargs) -> Settings:
    """Retrieve global settings instance or initialize with overrides."""
    global _global_settings
    if _global_settings is None or kwargs:
        _global_settings = Settings(**kwargs) if kwargs else Settings()
    return _global_settings
