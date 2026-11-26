"""Exact downloads for the media pipelines implemented by Studio.

These inventories were checked against each publisher's immutable Hugging Face
revision. A repository can also contain single-file exports, other precisions,
older versions, and unrelated examples; none is an implicit download choice.
An unsupported component layout is kept visible with an explicit explanation.
"""
from __future__ import annotations

from copy import deepcopy


REALVIS_REPO = "SG161222/RealVisXL_V5.0"

_REALVIS_FILES = (
    "model_index.json", "scheduler/scheduler_config.json",
    "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
    "text_encoder_2/config.json", "text_encoder_2/model.fp16.safetensors",
    "tokenizer/merges.txt", "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json", "tokenizer/vocab.json",
    "tokenizer_2/merges.txt", "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json", "tokenizer_2/vocab.json",
    "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
)
_FLUX_DIFFUSERS_FILES = (
    "model_index.json", "scheduler/scheduler_config.json",
    "text_encoder/config.json", "text_encoder/generation_config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/added_tokens.json", "tokenizer/chat_template.jinja",
    "tokenizer/merges.txt", "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "tokenizer/vocab.json",
    "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
)
_FLUX_MLX_FILES = (
    "text_encoder/0.safetensors", "text_encoder/1.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/chat_template.jinja", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
    "transformer/0.safetensors", "transformer/1.safetensors",
    "transformer/model.safetensors.index.json",
    "vae/0.safetensors", "vae/model.safetensors.index.json",
)
_WAN_FILES = (
    "model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
    "text_encoder/model-00001-of-00005.safetensors",
    "text_encoder/model-00002-of-00005.safetensors",
    "text_encoder/model-00003-of-00005.safetensors",
    "text_encoder/model-00004-of-00005.safetensors",
    "text_encoder/model-00005-of-00005.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json", "tokenizer/spiece.model",
    "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
)
_COGVIDEO_FILES = (
    "model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/added_tokens.json", "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model", "tokenizer/tokenizer_config.json",
    "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
)
_AUDIO_FILES = (
    "model.safetensors", "model_config.json",
    "t5gemma-b-b-ul2/config.json", "t5gemma-b-b-ul2/generation_config.json",
    "t5gemma-b-b-ul2/model.safetensors", "t5gemma-b-b-ul2/special_tokens_map.json",
    "t5gemma-b-b-ul2/tokenizer.json", "t5gemma-b-b-ul2/tokenizer.model",
    "t5gemma-b-b-ul2/tokenizer_config.json",
)
_AUDIO_MLX_FILES = ("config.json", "model.safetensors.index.json", *_AUDIO_FILES)
_BONSAI_IMAGE_MLX_FILES = (
    "manifest.json", "model_index.json", "scheduler/scheduler_config.json",
    "text_encoder-mlx-4bit/added_tokens.json", "text_encoder-mlx-4bit/config.json",
    "text_encoder-mlx-4bit/merges.txt", "text_encoder-mlx-4bit/model.safetensors",
    "text_encoder-mlx-4bit/model.safetensors.index.json",
    "text_encoder-mlx-4bit/special_tokens_map.json", "text_encoder-mlx-4bit/tokenizer.json",
    "text_encoder-mlx-4bit/tokenizer_config.json", "text_encoder-mlx-4bit/vocab.json",
    "tokenizer/added_tokens.json", "tokenizer/chat_template.jinja", "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json", "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json", "tokenizer/vocab.json",
    "transformer-packed-mflux/config.json",
    "transformer-packed-mflux/diffusion_pytorch_model.safetensors",
    "transformer-packed-mflux/quantization_config.json",
    "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
)


def _manifest(revision, files, byte_count, **options):
    return {
        "builtin_media_preset": True,
        "repo_revision": revision,
        "download_required_files": list(files),
        "download_allow_patterns": list(files),
        "download_bytes": byte_count,
        "est_speed": "Speed depends on your hardware",
        "vram": "Memory depends on the runtime and generation settings",
        **options,
    }


_DOWNLOADS = {
    REALVIS_REPO: _manifest(
        "ac93e0dda1f6d448cae19bbfab8c5e720a5e48bc", _REALVIS_FILES, 6_941_384_878,
        model_type="image", image_backend="diffusers",
        diffusers_variant="fp16", diffusers_dtype="float16", inference_steps=30,
    ),
    "mlx-community/flux2-klein-4b-4bit": _manifest(
        "860e87183ceb29e39627c0612ebd66d8ea66e68c", _FLUX_MLX_FILES, 4_619_700_117,
        model_type="image", image_backend="mflux", inference_steps=4,
    ),
    "black-forest-labs/FLUX.2-klein-4B": _manifest(
        "e7b7dc27f91deacad38e78976d1f2b499d76a294", _FLUX_DIFFUSERS_FILES, 15_980_131_745,
        model_type="image", image_backend="diffusers", diffusers_dtype="bfloat16",
        inference_steps=4, guidance_scale=1.0,
    ),
    "prism-ml/bonsai-image-ternary-4B-unpacked": _manifest(
        "30bbc032dd26939bff95c3d523477ad5065c3e1a", _FLUX_DIFFUSERS_FILES, 15_980_131_835,
        model_type="image", image_backend="diffusers", diffusers_dtype="float16",
        inference_steps=4, guidance_scale=1.0,
    ),
    "prism-ml/bonsai-image-ternary-4B-mlx-2bit": _manifest(
        "2c24c81b934a658ba5590cf39088ba929985b4a8", _BONSAI_IMAGE_MLX_FILES, 3_888_256_018,
        model_type="image", image_backend="bonsai_mlx", inference_steps=4, guidance_scale=1.0,
        runtime_setup_note="First use installs a private publisher image runtime; the app's chat runtime is kept intact.",
    ),
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": _manifest(
        "0fad780a534b6463e45facd96134c9f345acfa5b", _WAN_FILES, 28_928_887_859,
        model_type="video", diffusers_dtype="bfloat16", diffusers_vae_dtype="float32",
        inference_steps=30, guidance_scale=5.0, frames=49, fps=15,
    ),
    "zai-org/CogVideoX-2b": _manifest(
        "1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01", _COGVIDEO_FILES, 13_775_527_819,
        model_type="video", diffusers_dtype="float16",
        inference_steps=50, guidance_scale=6.0, frames=49, fps=8,
    ),
    "mlx-community/stable-audio-3-small-music": _manifest(
        "48bf34ca57d6f5a2e37282602066210dc6ff3eac", _AUDIO_MLX_FILES, 3_492_138_310,
        model_type="audio", audio_backend="mlx_audio", audio_variant="small-music",
    ),
    "mlx-community/stable-audio-3-small-sfx": _manifest(
        "ca6c38718d27cd2b446a4661c8851f1f3d0a3870", _AUDIO_MLX_FILES, 3_492_138_536,
        model_type="audio", audio_backend="mlx_audio", audio_variant="small-sfx",
    ),
    "stabilityai/stable-audio-3-small-music": _manifest(
        "8bdfb53c6b0fbc62113cd571fd37373efe15fa14", _AUDIO_FILES, 3_492_071_426,
        model_type="audio", audio_backend="stable_audio_3", audio_variant="small-music",
        requires_hf_auth=True,
    ),
    "stabilityai/stable-audio-3-small-sfx": _manifest(
        "05ca3857881c94a2b84026b1111c537d78f5f9f0", _AUDIO_FILES, 3_492_071_539,
        model_type="audio", audio_backend="stable_audio_3", audio_variant="small-sfx",
        requires_hf_auth=True,
    ),
}

_UNAVAILABLE = {
    "lesliemore/z-image-turbo-nsfw-v2-GGUF": (
        "image",
        "This repository contains a diffusion transformer GGUF, not a complete image pipeline. "
        "Studio does not yet implement its GGUF media loader with the separate Z-Image text encoder "
        "and VAE. No model files were substituted or downloaded."
    ),
    "Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF": (
        "image",
        "The publisher states this GGUF release is not an all-in-one pipeline: it needs a separately "
        "selected text encoder and VAE. Studio does not yet implement this GGUF image-edit runtime. "
        "Downloading every version and quantization in the repository would not make it loadable."
    ),
    "dgrauet/ltx-2.5-mlx-q4": (
        "video",
        "This split LTX MLX release requires its matching MLX video runtime and a selected component "
        "set. That runtime is not integrated in Studio's Diffusers video engine. The repository also "
        "requires Hugging Face access approval. Its weights have not been replaced by another model."
    ),
    "freeguyfroverrrr/Wan-2.2-Remix-GGUF": (
        "video",
        "This repository contains separate T2V/I2V, high/low-noise and quantization variants. "
        "Studio does not yet implement the Wan GGUF media pipeline and its required companion "
        "selection. Downloading all variants is not a supported model installation."
    ),
    "pipenetwork/MiniMax-H3-MLX-4bit": (
        "video",
        "This repository contains only the MiniMax-H3 MLX diffusion transformer. It requires the "
        "publisher's MLX pipeline plus upstream text encoder and video/audio VAEs; the advertised "
        "AfterMidnight LoRA is a separate artifact. This runtime is not integrated in Studio."
    ),
}

_DISPLAY = {
    REALVIS_REPO: {
        "name": "RealVisXL V5.0 (SDXL FP16)", "short_name": "RealVisXL V5.0 (FP16)",
        "tag": "📸 RealVisXL V5.0 (FP16)", "precision": "FP16 SafeTensors",
        "raw_params": None, "base_params": "SDXL",
    },
    "lesliemore/z-image-turbo-nsfw-v2-GGUF": {
        "name": "Z-Image Turbo NSFW v2 (GGUF transformer)", "short_name": "Z-Image Turbo GGUF",
        "precision": "Q8_0 transformer; separate text encoder and VAE required",
        "tag": "Z-Image Turbo · runtime required",
    },
    "Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF": {
        "name": "Qwen Image Edit Rapid (GGUF components)", "short_name": "Qwen Image Edit GGUF",
        "precision": "Separate diffusion, text encoder and VAE components",
        "tag": "Qwen Image Edit · runtime required",
    },
    "dgrauet/ltx-2.5-mlx-q4": {
        "name": "LTX-2.5 (MLX components)", "short_name": "LTX-2.5 MLX",
        "precision": "Publisher's split MLX weights",
        "tag": "LTX-2.5 · runtime required",
    },
    "freeguyfroverrrr/Wan-2.2-Remix-GGUF": {
        "name": "Wan 2.2 Remix (GGUF components)", "short_name": "Wan 2.2 Remix GGUF",
        "precision": "Separate high/low-noise and quantization variants",
        "tag": "Wan 2.2 Remix · runtime required",
    },
    "pipenetwork/MiniMax-H3-MLX-4bit": {
        "name": "MiniMax-H3 (4-bit MLX transformer)", "short_name": "MiniMax-H3 MLX",
        "raw_params": 33_000_000_000, "base_params": "33B transformer",
        "precision": "4-bit transformer; text encoder, VAEs and LoRA are separate",
        "tag": "MiniMax-H3 · runtime required",
    },
    "prism-ml/bonsai-image-ternary-4B-unpacked": {
        "precision": "Ternary weights in unpacked FP16 SafeTensors",
        "base_params": "4B transformer",
    },
    "prism-ml/bonsai-image-ternary-4B-mlx-2bit": {
        "precision": "2-bit packed ternary transformer; 4-bit text encoder",
        "base_params": "4B transformer",
    },
}


def media_download_metadata(repo_id: str) -> dict:
    """Return a fresh exact selection or a known unsupported-runtime reason."""
    if repo_id in _DOWNLOADS:
        result = deepcopy(_DOWNLOADS[repo_id])
    elif repo_id in _UNAVAILABLE:
        kind, reason = _UNAVAILABLE[repo_id]
        result = {
            "builtin_media_preset": True, "model_type": kind, "download_unavailable_reason": reason,
            "est_speed": "Publisher runtime setup required", "vram": "Memory requirements not verified",
            "raw_params": None, "base_params": "Parameter count not verified",
        }
    else:
        return {}
    result.update(_DISPLAY.get(repo_id, {}))
    result["model_card_url"] = "https://huggingface.co/" + repo_id
    return result


def apply_builtin_media_catalog(models: dict) -> None:
    """Attach download requirements before user-created selections are loaded.

    Existing model IDs and repository choices stay intact, including the old
    model_6 slot. Its former Ideogram label was not the selected repository.
    """
    for info in models.values():
        repo = info.get("repo_id") or info.get("default_repo_id")
        info.update(media_download_metadata(repo))
