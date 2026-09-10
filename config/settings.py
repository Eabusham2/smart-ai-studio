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
from typing import Any, Dict, List, Literal, Optional, Tuple


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
    """Detects optimal inference/training engine: mlx, gguf, bitnet, or torch."""
    env_backend = os.getenv("BACKEND", "").lower()
    if env_backend in ("mlx", "gguf", "bitnet", "torch"):
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
        use_mock: bool = Field(default=False, description="Enable mock mode for testing without GPU/neural weights")
        small_model: bool = Field(default=False, description="Use lightweight local model fallback")
        kv_bits: int = Field(default=4, description="KV-cache quantization bitwidth (4-bit for 16GB M1 Mac)")

        # Hardware & Model Configuration (Dedicated 27B Qwen3.8 Architectures)
        base_model_path: str = Field(default="orcarouter/Qwen3.8-27B-Uncensored-MLX", description="Path/ID for 27B Uncensored MLX model")
        mlx_model_path: str = Field(default="orcarouter/Qwen3.8-27B-Uncensored-MLX", description="Path/ID for Apple MLX 27B 2-bit uncensored model")
        lora_adapter_path: Optional[str] = Field(default=None, description="Path to consolidated Slow-LoRA adapter")
        small_model_path: str = Field(default="h34v7/Ternary-Qwen3.5-3.8B-mlx", description="Model checkpoint ID")
        flash_model_path: str = Field(default="Qwen/Qwen-3.8B-Flash-Next-1.58bit", description="Qwen 3.8 Flash Next 1.58-bit model checkpoint")
        ternary_qwen_3_8b_path: str = Field(default="h34v7/Ternary-Qwen3.5-3.8B-mlx", description="Ternary Qwen 3.8B Fast checkpoint")
        ternary_qwen_27b_path: str = Field(default="jayPark777/Qwen3.8-27B-Axon-MLQT", description="Ternary Qwen 27B Axon MLQT {-1,0,+1} checkpoint")
        gguf_model_path: str = Field(default="jayPark777/Qwen3.8-27B-Axon-MLQT", description="GGUF / Ternary Qwen 27B Axon MLQT checkpoint")
        flash_qwen_7b_path: str = Field(default="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", description="Flash Next Qwen 7B Coder checkpoint")
        vision_model_path: str = Field(default="mlx-community/nanoLLaVA-1.5-mlx", description="Uncensored Multimodal Vision checkpoint")
        vision_mmproj_path: Optional[str] = Field(default=None, description="GGUF vision clip projector path")
        auto_download: bool = Field(default=True, description="Automatically stream missing weights from Hugging Face")

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
        use_mock: bool = os.getenv("USE_MOCK", "false").lower() in ("1", "true", "yes")
        small_model: bool = os.getenv("SMALL_MODEL", "false").lower() in ("1", "true", "yes")
        kv_bits: int = int(os.getenv("KV_BITS", "4"))

        base_model_path: str = os.getenv("BASE_MODEL_PATH", "orcarouter/Qwen3.8-27B-Uncensored-MLX")
        mlx_model_path: str = os.getenv("MLX_MODEL_PATH", "orcarouter/Qwen3.8-27B-Uncensored-MLX")
        lora_adapter_path: Optional[str] = os.getenv("LORA_ADAPTER_PATH", None)
        small_model_path: str = os.getenv("SMALL_MODEL_PATH", "h34v7/Ternary-Qwen3.5-3.8B-mlx")
        flash_model_path: str = os.getenv("FLASH_MODEL_PATH", "Qwen/Qwen-3.8B-Flash-Next-1.58bit")
        ternary_qwen_3_8b_path: str = os.getenv("TERNARY_QWEN_3_8B_PATH", "h34v7/Ternary-Qwen3.5-3.8B-mlx")
        ternary_qwen_27b_path: str = os.getenv("TERNARY_QWEN_27B_PATH", "jayPark777/Qwen3.8-27B-Axon-MLQT")
        gguf_model_path: str = os.getenv("GGUF_MODEL_PATH", "jayPark777/Qwen3.8-27B-Axon-MLQT")
        flash_qwen_7b_path: str = os.getenv("FLASH_QWEN_7B_PATH", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
        vision_model_path: str = os.getenv("VISION_MODEL_PATH", "mlx-community/nanoLLaVA-1.5-mlx")
        vision_mmproj_path: Optional[str] = os.getenv("VISION_MMPROJ_PATH", None)
        auto_download: bool = os.getenv("AUTO_DOWNLOAD", "true").lower() in ("1", "true", "yes")

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


# Comprehensive Multi-Backend Model Registry Presets
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "model_1": {
        "id": "model_1",
        "key": "qwen_27b_uncensored_mlx",
        "name": "Qwen3.8-27B Uncensored (MLX 2-Bit)",
        "short_name": "Qwen 27B Uncensored (MLX)",
        "tag": "🔥 Qwen 27B Uncensored",
        "type": "mlx_2bit",
        "base_params": "27B",
        "raw_params": 27_000_000_000,
        "precision": "2-Bit Uncensored MLX",
        "max_context": 131_072,
        "vram": "~14.5 GB / 16 GB",
        "accent": "#38bdf8",
        "artifacts": {
            "mlx": "orcarouter/Qwen3.8-27B-Uncensored-MLX",
            "gguf": "orcarouter/Qwen3.8-27B-Uncensored-GGUF",
            "bitnet": "orcarouter/Qwen3.8-27B-Uncensored",
            "torch": "orcarouter/Qwen3.8-27B-Uncensored"
        },
        "default_repo_id": "orcarouter/Qwen3.8-27B-Uncensored-MLX"
    },
    "model_2": {
        "id": "model_2",
        "key": "qwen_27b_abliterated_gguf",
        "name": "Qwen3.8-27B Abliterated (Lowest Quant GGUF)",
        "short_name": "Qwen 27B Abliterated (GGUF)",
        "tag": "🔓 Qwen 27B Abliterated",
        "type": "gguf_quant",
        "base_params": "27B",
        "raw_params": 27_000_000_000,
        "precision": "Q2_K / Q3_K_M Lowest Quant",
        "max_context": 131_072,
        "vram": "~14.2 GB / 16 GB",
        "accent": "#f43f5e",
        "artifacts": {
            "gguf": "douyamv/Qwen3.8-27B-abliterated-GGUF",
            "mlx": "douyamv/Qwen3.8-27B-abliterated-MLX",
            "torch": "douyamv/Qwen3.8-27B-abliterated"
        },
        "default_repo_id": "douyamv/Qwen3.8-27B-abliterated-GGUF"
    },
    "model_3": {
        "id": "model_3",
        "key": "realvisxl_v5_image_gen",
        "name": "RealVisXL V5.0 (High-Res Photoreal Uncensored)",
        "short_name": "RealVisXL V5.0 (SDXL)",
        "tag": "📸 RealVisXL V5.0 (SDXL)",
        "type": "image_diffusion",
        "base_params": "6.6B SDXL",
        "raw_params": 6_600_000_000,
        "precision": "FP16 / SafeTensors (16GB RAM Optimized)",
        "max_context": 4_096,
        "vram": "~6.2 GB / 16 GB",
        "accent": "#ec4899",
        "artifacts": {
            "torch": "SG161222/RealVisXL_V5.0",
            "diffusers": "SG161222/RealVisXL_V5.0"
        },
        "default_repo_id": "SG161222/RealVisXL_V5.0"
    },
    "model_4": {
        "id": "model_4",
        "key": "z_image_turbo_nsfw_v2",
        "name": "Z-Image Turbo NSFW v2 (Q8 GGUF High-Res)",
        "short_name": "Z-Image Turbo NSFW v2",
        "tag": "⚡ Z-Image Turbo Q8",
        "type": "image_diffusion_gguf",
        "base_params": "4.0B Diffusion",
        "raw_params": 4_000_000_000,
        "precision": "Q8_0 GGUF Quantized",
        "max_context": 4_096,
        "vram": "~4.5 GB / 16 GB",
        "accent": "#a855f7",
        "artifacts": {
            "gguf": "lesliemore/z-image-turbo-nsfw-v2-GGUF",
            "torch": "lesliemore/z-image-turbo-nsfw-v2"
        },
        "default_repo_id": "lesliemore/z-image-turbo-nsfw-v2-GGUF"
    },
    "model_5": {
        "id": "model_5",
        "key": "qwen_image_edit_rapid",
        "name": "Qwen Image Edit Rapid AIO (Text & Image in GGUF)",
        "short_name": "Qwen Image Edit AIO",
        "tag": "🎨 Qwen Image Edit Rapid",
        "type": "image_edit_rapid",
        "base_params": "7.0B Multimodal",
        "raw_params": 7_000_000_000,
        "precision": "Rapid AIO GGUF (Text & Image In)",
        "max_context": 32_768,
        "vram": "~5.2 GB / 16 GB",
        "accent": "#06b6d4",
        "artifacts": {
            "gguf": "Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF",
            "torch": "Phil2Sat/Qwen-Image-Edit-Rapid-AIO"
        },
        "default_repo_id": "Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF"
    },
    "model_6": {
        "id": "model_6",
        "key": "ideogram_instant_sdxl",
        "name": "Ideogram Instant Uncensored (16GB RAM Turbo)",
        "short_name": "Ideogram Instant (16GB)",
        "tag": "✨ Ideogram Instant NSFW",
        "type": "image_diffusion_turbo",
        "base_params": "3.5B Turbo",
        "raw_params": 3_500_000_000,
        "precision": "4-Step Instant Diffusion",
        "max_context": 2_048,
        "vram": "~3.8 GB / 16 GB",
        "accent": "#eab308",
        "artifacts": {
            "torch": "stabilityai/sdxl-turbo",
            "diffusers": "SG161222/RealVisXL_V5.0"
        },
        "default_repo_id": "SG161222/RealVisXL_V5.0"
    },
    "model_7": {
        "id": "model_7",
        "key": "ltx_video_mlx",
        "name": "LTX-Video 2.5 (MLX Q4 High-Res Video & Audio)",
        "short_name": "LTX-Video 2.5 (MLX)",
        "tag": "🎬 LTX-Video 2.5 Q4",
        "type": "video_mlx",
        "base_params": "5.0B Video",
        "raw_params": 5_000_000_000,
        "precision": "Q4 Apple Silicon MLX Native",
        "max_context": 8_192,
        "vram": "~5.8 GB / 16 GB",
        "accent": "#10b981",
        "artifacts": {
            "mlx": "dgrauet/ltx-2.5-mlx-q4",
            "torch": "Lightricks/LTX-Video"
        },
        "default_repo_id": "dgrauet/ltx-2.5-mlx-q4"
    },
    "model_8": {
        "id": "model_8",
        "key": "wan_remix_video",
        "name": "Wan 2.2 Remix (GGUF Q4 Motion Engine)",
        "short_name": "Wan 2.2 Remix (GGUF)",
        "tag": "🎥 Wan 2.2 Remix Q4",
        "type": "video_gguf",
        "base_params": "5.0B Video",
        "raw_params": 5_000_000_000,
        "precision": "Q4 GGUF Quantized",
        "max_context": 8_192,
        "vram": "~5.9 GB / 16 GB",
        "accent": "#6366f1",
        "artifacts": {
            "gguf": "freeguyfroverrrr/Wan-2.2-Remix-GGUF",
            "torch": "Wan-AI/Wan2.1-T2V-1.3B"
        },
        "default_repo_id": "freeguyfroverrrr/Wan-2.2-Remix-GGUF"
    },
    "model_9": {
        "id": "model_9",
        "key": "minimax_h3_aftermidnight",
        "name": "MiniMax-H3 MLX 4-bit (AfterMidnight NSFW LoRA)",
        "short_name": "MiniMax-H3 AfterMidnight",
        "tag": "🌙 MiniMax-H3 AfterMidnight",
        "type": "video_audio_lora",
        "base_params": "4.0B Video/Audio",
        "raw_params": 4_000_000_000,
        "precision": "4-bit MLX + Rank 32 LoRA",
        "max_context": 8_192,
        "vram": "~5.7 GB / 16 GB",
        "accent": "#8b5cf6",
        "artifacts": {
            "mlx": "pipenetwork/MiniMax-H3-MLX-4bit",
            "lora": "SexGod1979/AfterMidnight-MiniMax-H3-NSFW"
        },
        "default_repo_id": "pipenetwork/MiniMax-H3-MLX-4bit"
    }
}


_global_settings: Optional[Settings] = None


def get_settings(**kwargs) -> Settings:
    """Retrieve global settings instance or initialize with overrides."""
    global _global_settings
    if _global_settings is None or kwargs:
        _global_settings = Settings(**kwargs) if kwargs else Settings()
    return _global_settings
