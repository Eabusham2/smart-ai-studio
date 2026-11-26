"""Pinned Bonsai 2.5 presets shared by the desktop and programmatic entry points.

Repository revisions and GGUF/sidecar SHA-256 values were read from the publishers'
Hugging Face API on 2026-11-25. The CRACK entries are standalone AtomicChat-derived
ternary weights, with the original Prism MTP head retained; neither is a LoRA.
"""
from __future__ import annotations

BONSAI_MLX_REPO = "prism-ml/Ternary-Bonsai-2.5-27B-mlx-2bit"
BONSAI_MLX_REVISION = "aa432fd0e17612f6684e728d4d3c42b6656bcabb"
BONSAI_GGUF_REPO = "prism-ml/Ternary-Bonsai-2.5-27B-GGUF"
BONSAI_GGUF_REVISION = "28a7093cc25bc151cc2b284c5570f12ce343fdcf"
CRACK_MLX_REPO = "barozp/Ternary-Bonsai-2.5-27B-Uncensored-MLX-2bit"
CRACK_MLX_REVISION = "0a50dabd13f96ad36394921a03b073dec5b1ab30"
CRACK_GGUF_REPO = "kelsey1337/Ternary-Bonsai-2.5-27B-Uncensored-MTP-GGUF"
CRACK_GGUF_REVISION = "ad67f3e0770aefcdac514848ee3595d7edcdf7ad"
BONSAI_GGUF_FILE = "Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf"
CRACK_GGUF_FILE = "Ternary-Bonsai-2.5-27B-Uncensored-PTQ1_0-mtp.gguf"
MMPROJ_FILE = "Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf"
MMPROJ_SHA256 = "6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903"
MLX_MTP_FILE = "mtp/mtp-v2-head.safetensors"
MLX_MTP_SHA256 = "56bbaf0da659f7db9b5b8878d02088775ec8324fcf92dba450cd929fb3431885"

# The selected packing has the same ternary values as PQ2_0 and Q2_0. Only its
# storage differs. Each accepted file includes the official v2 draft head.
GGUF_MTP_HASHES = {
    "Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf": "e0fc5f3d47b9c926d52677d303b19df356a2376614b3e1cb9973865fc04ca431",
    "Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf": "ededc9fc0a8a6a8dd30914d107b2d2ce7674d789f1d05a1a481f240609aa6c23",
    "Ternary-Bonsai-2.5-27B-Q2_0-mtp.gguf": "b0f83de8481c3200c529aef5fb495dbfedf2a4a3f017f7a3ea0a69c67be1d306",
    "Ternary-Bonsai-2.5-27B-Uncensored-PTQ1_0-mtp.gguf": "29331c5477242fad9f667cc428bcf18ab22f304ba6b02684bf9a37ae935e8e9c",
    "Ternary-Bonsai-2.5-27B-Uncensored-PQ2_0-mtp.gguf": "d2761cd8929758d25793c27d5221e4f10e11952a0683db535272013f66e539e1",
    "Ternary-Bonsai-2.5-27B-Uncensored-Q2_0-mtp.gguf": "6b321b5604ad32bcffe45c220f5926d51a9f5367eeb512b15f19ce63eb8ca386",
}
MLX_REQUIRED_FILES = [
    "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
    "chat_template.jinja", "generation_config.json", "hadamard.json",
    "preprocessor_config.json", "video_preprocessor_config.json",
    "runtime/artifact.py", "runtime/codec.py", "runtime/runtime.py",
    "runtime/vision_artifact.py", "runtime/mtp_artifact.py",
    "mtp/head-manifest.json", MLX_MTP_FILE,
]


def bonsai_model_info(*, crack: bool = False, is_apple_silicon: bool = True) -> dict:
    """Return a fresh desktop-shaped preset; importing this module downloads nothing."""
    mlx_repo = CRACK_MLX_REPO if crack else BONSAI_MLX_REPO
    gguf_repo = CRACK_GGUF_REPO if crack else BONSAI_GGUF_REPO
    gguf_file = CRACK_GGUF_FILE if crack else BONSAI_GGUF_FILE
    short_name = "Bonsai 2.5 27B CRACK" if crack else "Bonsai 2.5 27B Ternary"
    info = {
        "name": short_name + (" (Abliterated/Uncensored)" if crack else " Multimodal"),
        "short_name": short_name,
        "repo_id": mlx_repo if is_apple_silicon else gguf_repo,
        "repo_revision": (CRACK_MLX_REVISION if crack else BONSAI_MLX_REVISION)
        if is_apple_silicon else (CRACK_GGUF_REVISION if crack else BONSAI_GGUF_REVISION),
        "model_path": None,
        "gguf_file": None if is_apple_silicon else gguf_file,
        "gguf_sha256": None if is_apple_silicon else GGUF_MTP_HASHES[gguf_file],
        "gguf_preference": None if is_apple_silicon else "PTQ1_0-mtp",
        "mmproj_file": None if is_apple_silicon else MMPROJ_FILE,
        "mmproj_repo_id": None if is_apple_silicon else BONSAI_GGUF_REPO,
        "mmproj_repo_revision": None if is_apple_silicon else BONSAI_GGUF_REVISION,
        "gguf_training_base_model_id": None if is_apple_silicon else "Qwen/Qwen3.8-27B",
        "prism_llama_fork": not is_apple_silicon,
        "bonsai25_mtp": True,
        "mtp_file": MLX_MTP_FILE if is_apple_silicon else None,
        "mtp_sha256": MLX_MTP_SHA256 if is_apple_silicon else None,
        "model_type": "text", "input_modalities": ["text", "image", "video"],
        "ternary": True, "runtime_family": "bonsai2_hadamard",
        "precision": "True ternary g128 + MTP v2" + ("; AtomicChat uncensored weights" if crack else ""),
        "raw_params": 27_360_000_000, "base_params": "27.36B",
        "est_speed": "Native ternary + target-verified MTP",
        "max_context": 262_144, "default_context": 32768,
        "vram": "~8.6 GB MLX + 0.27 GB MTP / 6.23 GB GGUF + optional vision; KV extra",
        "tag": ("🔓 " if crack else "👁️ ") + short_name,
        "accent": "#f43f5e" if crack else "#38bdf8",
    }
    if is_apple_silicon:
        info["download_required_files"] = list(MLX_REQUIRED_FILES)
        info["download_allow_patterns"] = list(MLX_REQUIRED_FILES)
    return info


def bonsai_registry_preset(*, crack: bool = False) -> dict:
    info = bonsai_model_info(crack=crack)
    info.update({
        "id": "model_2" if crack else "model_1",
        "key": "bonsai25_27b_ternary_crack" if crack else "bonsai25_27b_ternary_multimodal",
        "type": "multimodal_ternary",
        "artifacts": {"mlx": CRACK_MLX_REPO if crack else BONSAI_MLX_REPO,
                      "gguf": CRACK_GGUF_REPO if crack else BONSAI_GGUF_REPO},
        "default_repo_id": CRACK_MLX_REPO if crack else BONSAI_MLX_REPO,
    })
    return info


def cross_platform_mlx_presets() -> dict:
    """Explicit original MLX choices alongside the existing GGUF desktop tabs."""
    presets = {}
    for crack, key in ((False, "bonsai25_mlx"), (True, "bonsai25_crack_mlx")):
        info = bonsai_model_info(crack=crack, is_apple_silicon=True)
        info["name"] += " (MLX)"
        info["short_name"] += " (MLX)"
        info["tag"] += " MLX"
        info["controller_runtime"] = "mlx_repo_vlm"
        info["mlx_runtime"] = "native"
        presets[key] = info
    return presets
