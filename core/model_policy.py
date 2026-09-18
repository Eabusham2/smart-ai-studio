"""Strict model-admission policy for Smart AI Studio.

Only text/chat models must prove native ternary / 1.58-bit style weights.
Image, video, and audio generation models are admitted by their media task.
"2-bit" by itself is NOT accepted as evidence of ternary weights.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, Optional, Tuple

TERNARY_MARKERS = (
    "ternary",
    "1.58-bit",
    "1.58bit",
    "1.58 bit",
    "trit",
    "tritplane",
    "bitnet",
    "{-1, 0, +1}",
    "{−1, 0, +1}",
    "-1,0,+1",
)

MEDIA_PIPELINE_MARKERS = (
    "text-to-image",
    "image-to-image",
    "text-to-video",
    "image-to-video",
    "text-to-audio",
    "audio-generation",
    "text-to-speech",
)

TEXT_PIPELINE_MARKERS = (
    "text-generation",
    "text2text-generation",
    "image-text-to-text",
    "image-to-text",
    "video-text-to-text",
    "video-to-text",
    "audio-text-to-text",
    "audio-to-text",
    "automatic-speech-recognition",
    "conversational",
)


def _blob(values: Iterable[Any]) -> str:
    return " ".join(str(v or "") for v in values).lower()


def _has_ternary_proof(values: Iterable[Any]) -> bool:
    text = _blob(values)
    return any(marker.lower() in text for marker in TERNARY_MARKERS)


def infer_input_modalities(*values: Any) -> list[str]:
    """Conservative input-capability inference for imported text controllers."""
    blob = _blob(values)
    modalities = {"text"}

    image_markers = (
        "image-text-to-text", "image-to-text", "visual-question-answering",
        "vision-language", "vision_language", "vlm", "qwen-vl", "qwen2-vl",
        "qwen2.5-vl", "qwen3-vl", "llava", "idefics", "pixtral",
        "image_input", "pixel_values",
    )
    audio_markers = (
        "audio-text-to-text", "audio-to-text", "audio-language",
        "audio_language", "speech-language", "speech_language",
        "input_audio", "audio_values", "qwen-audio",
    )
    video_markers = (
        "video-text-to-text", "video-language", "video_language",
        "video_input", "video_values", "video-llava",
    )

    if any(marker in blob for marker in image_markers):
        modalities.add("image")
    if any(marker in blob for marker in audio_markers):
        modalities.add("audio")
    if any(marker in blob for marker in video_markers):
        modalities.add("video")
    return sorted(modalities)



def infer_controller_runtime(*values: Any) -> str:
    """Infer a controller runtime family from model metadata without model-name special cases."""
    blob = _blob(values)
    modalities = set(infer_input_modalities(*values))

    # BitNet checkpoints are commonly distributed as GGUF too; architecture/runtime
    # metadata wins over the container extension.
    if any(marker in blob for marker in ("bitnet", "i2_s", "bitlinear", "1bitllm")):
        return "bitnet"
    if "gguf" in blob:
        return "gguf"
    if "jang" in blob and "mlx" in blob:
        return "jang_vlm"
    if "prism_hadamard" in blob and "mlx" in blob:
        return "mlx_repo_vlm"
    if "mlx" in blob:
        return "mlx_vlm" if modalities - {"text"} else "mlx_lm"
    if any(marker in blob for marker in ("transformers", "safetensors", "pytorch_model")):
        return "transformers_auto"
    return "auto"


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(value)


def _extract_base_model_id(values: Iterable[Any]) -> str:
    """Best-effort training lineage from structured card/tags/README metadata."""
    text = " ".join(str(v or "") for v in values)
    patterns = (
        r"base_model:(?:finetune:|quantized:)?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
        r"base model(?: id)?\s*[:=]\s*[\`\"']?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
        r"base_model(?:_name_or_path)?[\"']?\s*[:=]\s*[\"']([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _extract_projector_repo(values: Iterable[Any]) -> str:
    """Find a Hub repo explicitly associated with mmproj/projector text."""
    text = "\n".join(str(v or "") for v in values)
    for line in text.splitlines():
        low = line.lower()
        if not any(marker in low for marker in ("mmproj", "projector", "vision projector")):
            continue
        match = re.search(
            r"(?:https?://huggingface\.co/)?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
            line,
        )
        if match:
            return match.group(1)
    return ""


def _pick_gguf_artifacts(file_names: Iterable[str], ternary: bool) -> Dict[str, Any]:
    """Pick a language GGUF and projector from metadata without guessing at runtime."""
    names = [str(x or "") for x in file_names if str(x or "").lower().endswith(".gguf")]
    projectors = [
        name for name in names
        if any(marker in name.lower() for marker in ("mmproj", "projector", "vision"))
    ]
    language = [name for name in names if name not in projectors]

    def model_score(name: str) -> tuple:
        low = name.lower()
        score = 0
        # True ternary/BitNet-native representations first.
        for marker, points in (
            ("ptq1_0", 120),
            ("i2_s", 115),
            ("tl1", 112),
            ("tl2", 110),
            ("1.58", 108),
            ("1bit", 106),
            ("ternary", 104),
            ("trit", 102),
            ("pq2_0", 96),
        ):
            if marker in low:
                score = max(score, points)
        if ternary and not score:
            # A model admitted as ternary may still use a generic filename; keep it
            # below explicit ternary encodings but above unrelated projector files.
            score = 30
        # Avoid silently selecting ordinary quant variants when an explicit ternary
        # artifact exists in the same repository.
        for marker in ("q8_", "q6_", "q5_", "q4_", "q3_"):
            if marker in low:
                score -= 25
        return (score, -len(name), name)

    model_file = max(language, key=model_score) if language else ""
    mmproj_file = ""
    if projectors:
        mmproj_file = max(
            projectors,
            key=lambda name: (
                20 if "q8" in name.lower() else 10 if "f16" in name.lower() else 0,
                name,
            ),
        )

    preference = ""
    if model_file:
        low = model_file.lower()
        for marker in ("PTQ1_0", "I2_S", "TL1", "TL2", "PQ2_0"):
            if marker.lower() in low:
                preference = marker
                break

    return {
        "gguf_file": model_file or None,
        "gguf_preference": preference or None,
        "mmproj_file": mmproj_file or None,
    }


def derive_runtime_metadata(
    *values: Any,
    file_names: Optional[Iterable[str]] = None,
    repo_id: str = "",
    ternary: bool = False,
) -> Dict[str, Any]:
    """Derive portable backend/runtime metadata from actual model metadata/files."""
    names = [str(x or "") for x in (file_names or [])]
    all_values = [*values, *names]
    blob = _blob(all_values)
    runtime = infer_controller_runtime(*all_values)
    artifacts = _pick_gguf_artifacts(names, ternary=bool(ternary))

    is_bitnet = runtime == "bitnet" or any(
        marker in blob for marker in ("bitnet", "i2_s", "bitlinear", "1bitllm")
    )
    is_gguf = bool(artifacts.get("gguf_file")) or runtime == "gguf"
    is_prism_gguf = bool(
        is_gguf
        and any(
            marker in blob
            for marker in (
                "prism_hadamard",
                "prismml",
                "prism-ml",
                "ptq1_0",
                "pq2_0",
                "hadamard qwen",
            )
        )
    )

    if is_bitnet:
        backend_family = "bitnet"
        runtime = "bitnet"
    elif is_prism_gguf:
        backend_family = "prism_gguf"
        runtime = "gguf"
    elif is_gguf:
        backend_family = "gguf"
        runtime = "gguf"
    elif runtime in ("mlx_lm", "mlx_vlm", "mlx_repo_vlm", "jang_vlm"):
        backend_family = runtime
    elif runtime == "transformers_auto":
        backend_family = "transformers"
    else:
        backend_family = "auto"

    base_model_id = _extract_base_model_id(all_values)
    projector_repo = _extract_projector_repo(all_values)
    if artifacts.get("mmproj_file") and not projector_repo:
        projector_repo = str(repo_id or "")

    result: Dict[str, Any] = {
        "controller_runtime": runtime,
        "backend_family": backend_family,
        "runtime_family": (
            "bonsai2_hadamard"
            if is_prism_gguf and "bonsai" in blob
            else ""
        ),
        "prism_llama_fork": bool(is_prism_gguf),
        **artifacts,
    }
    if projector_repo:
        result["mmproj_repo_id"] = projector_repo
    if base_model_id:
        if is_bitnet:
            result["bitnet_training_base_model_id"] = base_model_id
        elif is_gguf:
            result["gguf_training_base_model_id"] = base_model_id
    elif is_bitnet and repo_id:
        rid = str(repo_id).strip().strip("/")
        low = rid.lower()
        # Microsoft's published family separates deployment GGUF/packed weights
        # from the BF16 training master with a stable sibling naming convention.
        if low.startswith("microsoft/bitnet-"):
            if low.endswith("-gguf"):
                result["bitnet_training_base_model_id"] = rid[:-5] + "-bf16"
            elif not low.endswith("-bf16"):
                result["bitnet_training_base_model_id"] = rid + "-bf16"
    return result


def _classify_pipeline(pipeline_tag: str, tags: Iterable[str]) -> str:
    blob = _blob((pipeline_tag, *list(tags)))
    if any(marker in blob for marker in MEDIA_PIPELINE_MARKERS):
        return "media"
    if any(marker in blob for marker in TEXT_PIPELINE_MARKERS):
        return "text"
    # Fail closed for model families that are overwhelmingly chat/text models.
    if any(marker in blob for marker in ("qwen", "llama", "mistral", "gemma", "bonsai", "causal-lm", "causallm")):
        return "text"
    return "unknown"


def inspect_hf_model(repo_id: str) -> Dict[str, Any]:
    """Inspect Hub card/config/files and return admission + runtime metadata."""
    repo_id = str(repo_id or "").strip().strip("/")
    if not repo_id:
        return {"ok": False, "reason": "No Hugging Face repository ID provided."}
    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        info = api.model_info(repo_id)
        tags = list(getattr(info, "tags", None) or [])
        pipeline_tag = str(getattr(info, "pipeline_tag", "") or "")
        library_name = str(getattr(info, "library_name", "") or "")
        model_id = str(getattr(info, "id", repo_id) or repo_id)
        config = getattr(info, "config", None) or {}
        card_data = getattr(info, "card_data", None)
        if card_data is None:
            card_data = getattr(info, "cardData", None)
        siblings = [
            str(getattr(item, "rfilename", "") or "")
            for item in (getattr(info, "siblings", None) or [])
            if str(getattr(item, "rfilename", "") or "")
        ]

        # Card/config text often contains runtime/projector/base-lineage details that
        # are not promoted to Hub tags. Fetch only these tiny metadata files.
        metadata_text = []
        for filename in ("README.md", "config.json", "model_config.json"):
            if filename not in siblings:
                continue
            try:
                path = hf_hub_download(repo_id=model_id, filename=filename)
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    metadata_text.append(fh.read(256_000))
            except Exception:
                pass

        evidence = [
            model_id,
            pipeline_tag,
            library_name,
            *tags,
            _json_text(config),
            _json_text(card_data),
            *siblings,
            *metadata_text,
        ]
        task = _classify_pipeline(pipeline_tag, evidence)
        ternary = _has_ternary_proof(evidence)
        input_modalities = infer_input_modalities(*evidence)
        runtime_meta = derive_runtime_metadata(
            *evidence,
            file_names=siblings,
            repo_id=model_id,
            ternary=bool(ternary),
        )
        return {
            "ok": True,
            "repo_id": model_id,
            "pipeline_tag": pipeline_tag,
            "library_name": library_name,
            "tags": tags,
            "task": task,
            "ternary": bool(ternary),
            "input_modalities": input_modalities,
            **runtime_meta,
        }
    except Exception as exc:
        return {
            "ok": False,
            "repo_id": repo_id,
            "reason": f"Could not verify Hugging Face metadata: {exc}",
        }


def enforce_hf_text_ternary(repo_id: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Reject text/chat Hub repos unless the Hub metadata explicitly proves ternary."""
    meta = inspect_hf_model(repo_id)
    if not meta.get("ok"):
        return False, str(meta.get("reason") or "Unable to verify model metadata."), meta
    if meta.get("task") != "text":
        return True, "Media/non-text model; text-only ternary restriction does not apply.", meta
    if not meta.get("ternary"):
        return (
            False,
            "Text models must explicitly advertise native ternary / 1.58-bit / trit weights. "
            "A plain 2-bit, 4-bit, GGUF, MLX, AWQ, or FP8 quantization is not accepted as ternary proof.",
            meta,
        )
    return True, "Verified ternary text model.", meta


def inspect_local_model(path: str) -> Dict[str, Any]:
    """Conservative local-folder inspection used by the custom model importer."""
    expanded = os.path.abspath(os.path.expanduser(str(path or "")))
    if not os.path.exists(expanded):
        return {"ok": False, "reason": "Local model path does not exist.", "path": expanded}

    snippets = [expanded]
    candidates = []
    if os.path.isdir(expanded):
        for name in (
            "config.json",
            "model_config.json",
            "tokenizer_config.json",
            "README.md",
            "README.txt",
        ):
            p = os.path.join(expanded, name)
            if os.path.isfile(p):
                candidates.append(p)
    elif os.path.isfile(expanded):
        candidates.append(expanded)

    for p in candidates:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                snippets.append(fh.read(256_000))
        except Exception:
            pass

    blob = _blob(snippets)
    media = any(marker in blob for marker in MEDIA_PIPELINE_MARKERS) or any(
        marker in blob for marker in ("diffusion", "text-to-image", "text-to-video", "text-to-audio", "audio-generation")
    )
    text = any(
        marker in blob
        for marker in (
            "causallm",
            "causal-lm",
            "text-generation",
            "qwen",
            "llama",
            "mistral",
            "gemma",
            "bonsai",
            "chat_template",
        )
    )
    task = "media" if media and not text else "text" if text else "unknown"
    file_names = []
    try:
        if os.path.isdir(expanded):
            for root, _dirs, files in os.walk(expanded):
                for name in files:
                    file_names.append(os.path.relpath(os.path.join(root, name), expanded))
                    if len(file_names) >= 4096:
                        break
                if len(file_names) >= 4096:
                    break
        else:
            file_names = [os.path.basename(expanded)]
    except Exception:
        file_names = []

    ternary = _has_ternary_proof([*snippets, *file_names])
    runtime_meta = derive_runtime_metadata(
        *snippets,
        file_names=file_names,
        ternary=bool(ternary),
    )
    return {
        "ok": True,
        "path": expanded,
        "task": task,
        "ternary": bool(ternary),
        "input_modalities": infer_input_modalities(*snippets, *file_names),
        **runtime_meta,
    }


def enforce_local_text_ternary(path: str) -> Tuple[bool, str, Dict[str, Any]]:
    meta = inspect_local_model(path)
    if not meta.get("ok"):
        return False, str(meta.get("reason") or "Unable to inspect local model."), meta
    if meta.get("task") == "media":
        return True, "Media model; text-only ternary restriction does not apply.", meta
    # Unknown local model folders fail closed because the user explicitly requires
    # text additions to be ternary and an unclassified folder might be a text model.
    if not meta.get("ternary"):
        return (
            False,
            "Local text/unknown model does not contain verifiable ternary / 1.58-bit metadata. "
            "Add explicit ternary metadata/config or use a verified Hugging Face ternary repo.",
            meta,
        )
    return True, "Verified ternary local text model.", meta
