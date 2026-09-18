"""Strict model-admission policy for Smart AI Studio.

Only text/chat models must prove native ternary / 1.58-bit style weights.
Image, video, and audio generation models are admitted by their media task.
"2-bit" by itself is NOT accepted as evidence of ternary weights.
"""
from __future__ import annotations

import json
import os
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
    "automatic-speech-recognition",
)

TEXT_PIPELINE_MARKERS = (
    "text-generation",
    "text2text-generation",
    "image-text-to-text",
    "conversational",
)


def _blob(values: Iterable[Any]) -> str:
    return " ".join(str(v or "") for v in values).lower()


def _has_ternary_proof(values: Iterable[Any]) -> bool:
    text = _blob(values)
    return any(marker.lower() in text for marker in TERNARY_MARKERS)


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
    """Inspect Hub metadata and return a task + ternary proof summary."""
    repo_id = str(repo_id or "").strip().strip("/")
    if not repo_id:
        return {"ok": False, "reason": "No Hugging Face repository ID provided."}
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id)
        tags = list(getattr(info, "tags", None) or [])
        pipeline_tag = str(getattr(info, "pipeline_tag", "") or "")
        library_name = str(getattr(info, "library_name", "") or "")
        model_id = str(getattr(info, "id", repo_id) or repo_id)
        task = _classify_pipeline(pipeline_tag, tags)
        ternary = _has_ternary_proof((model_id, pipeline_tag, library_name, *tags))
        return {
            "ok": True,
            "repo_id": model_id,
            "pipeline_tag": pipeline_tag,
            "library_name": library_name,
            "tags": tags,
            "task": task,
            "ternary": bool(ternary),
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
    ternary = _has_ternary_proof(snippets)
    return {"ok": True, "path": expanded, "task": task, "ternary": bool(ternary)}


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
