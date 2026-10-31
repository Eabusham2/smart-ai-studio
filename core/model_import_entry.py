"""Turn validated model admission into a saved selector entry without losing identity."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


_RUNTIME_PREFERENCES = {"auto", "native", "strata"}
_STRATA_UNSUPPORTED_TERNARY = {34: "TQ1_0", 35: "TQ2_0", 143: "PTQ1_0"}
# The pinned Qwen native kernels support these formats. This only chooses an
# inspection route: the official preflight still checks their tensor roles,
# model layout, tokenizer and this computer. DeepSeek uses a separate engine.
_STRATA_QWEN_TYPES = {
    0, 1, 2, 3, 6, 7, 8, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 29, 30,
    42, 144, 145, 146,
}


def _observed_types(checked: dict) -> set[int] | None:
    types = checked.get("quantization_types")
    if (not isinstance(types, (list, tuple)) or not types
            or any(type(value) is not int or value < 0 for value in types)):
        return None
    return set(types)


def _automatic_strata_candidate(info: dict, checked: dict, observed: set[int] | None) -> bool:
    if (info.get("requires_local_validation") or not info.get("strata_candidate")
            or set(info.get("input_modalities") or ["text"]) != {"text"}
            or checked.get("gguf_version") != 3
            or not observed or observed.intersection(_STRATA_UNSUPPORTED_TERNARY)):
        return False
    architecture = checked.get("architecture")
    if architecture == "qwen4exp":
        return observed <= _STRATA_QWEN_TYPES
    # DeepSeek's own preflight knows its separate encoding combinations. Do not
    # reject those using the Qwen kernel table (for example, its MXFP4 path).
    return architecture == "deepseek41"


def _native_selection(info: dict, source: str) -> dict:
    """Restore the admitted native route when changing a saved Strata choice."""
    from core.model_policy import admit_local_model
    allowed, reason, native = admit_local_model(source)
    if not allowed:
        raise ValueError(reason)
    preserved = {key: value for key, value in info.items()
                 if key in {"short_name", "name", "source_origin", "source_repo_id", "source_revision",
                            "repo_id", "repo_revision", "download_required_files", "download_allow_patterns"}}
    return {**preserved, **native}


def make_imported_model_entry(
    metadata: dict[str, Any], *, display_name: str = "", local_path: str = "",
    repo_id: str | None = None, fetched: bool = False,
    runtime_preference: str | None = None,
) -> dict[str, Any]:
    """Keep a saved runtime choice separate from the selected artifact's identity.

    Pending Hub sources retain the choice until their exact downloaded files
    pass local inspection. Automatic placement never converts model weights.
    """
    info = dict(metadata)
    preference = runtime_preference if runtime_preference is not None else info.get("runtime_preference", "auto")
    if not isinstance(preference, str) or preference not in _RUNTIME_PREFERENCES:
        raise ValueError("Choose Automatic, Native, or Strata as the model runtime.")
    checked = info.get("gguf_metadata") if isinstance(info.get("gguf_metadata"), dict) else {}
    observed = _observed_types(checked)
    chosen_path = str(checked.get("model_path") or info.get("strata_model_path") or info.get("model_path") or local_path)
    local = bool(local_path or info.get("source_kind") == "strata_local" or (
        info.get("source_kind") in {"local", "local_import", "hf_fetch"}
        and info.get("model_path") and not info.get("requires_local_validation")))
    if local_path and Path(local_path).is_file() and Path(local_path).suffix.lower() == ".gguf" and checked:
        specified = os.path.abspath(os.path.expanduser(local_path))
        if specified not in {os.path.abspath(str(path)) for path in checked.get("files") or []}:
            raise ValueError("The selected GGUF path differs from the inspected model. Inspect the chosen file again.")
    if preference == "strata":
        unsupported = (observed or set()).intersection(_STRATA_UNSUPPORTED_TERNARY)
        if unsupported:
            formats = ", ".join(_STRATA_UNSUPPORTED_TERNARY[kind] for kind in sorted(unsupported))
            raise ValueError(f"The pinned Strata engine cannot execute the selected {formats} tensor encoding. "
                             "Choose Native to retain these exact weights; no conversion or replacement is performed.")
        pending_gguf = (not local and info.get("requires_local_validation")
                        and str(info.get("gguf_file") or "").lower().endswith(".gguf"))
        if not pending_gguf and (not local or not checked or not observed or info.get("requires_local_validation")):
            raise ValueError("Strata requires an inspected local GGUF model. Download and inspect the selected GGUF first, or choose Native.")
    use_strata = local and (preference == "strata" or
                           (preference == "auto" and _automatic_strata_candidate(info, checked, observed)))
    if info.get("controller_runtime") == "strata" and not use_strata:
        info = _native_selection(info, chosen_path)
        checked = info.get("gguf_metadata") or {}
        chosen_path = str(checked.get("model_path") or info.get("model_path") or chosen_path)
    if use_strata:
        from config.strata_models import make_local_strata_selection
        info = {**info, **make_local_strata_selection(
            chosen_path,
            gguf_files=checked.get("files"), metadata=info, display_name=display_name or None,
        )}
    source = str(local_path or repo_id or info.get("path") or "Custom model")
    name = str(display_name or info.get("short_name") or info.get("name") or Path(source).name or "Custom model").strip()
    pipeline = str(info.get("pipeline_tag") or "").lower()
    model_type = str(info.get("model_type") or "")
    if not model_type:
        if info.get("task") == "text":
            model_type = "text"
        elif "audio" in pipeline or "speech" in pipeline:
            model_type = "audio"
        elif "video" in pipeline:
            model_type = "video"
        elif "image" in pipeline:
            model_type = "image"
        else:
            model_type = "media"
    is_strata = info.get("controller_runtime") == "strata"
    fetched = bool(fetched or info.get("source_origin") == "hf_fetch")
    repository = repo_id or info.get("source_repo_id") or info.get("repo_id")
    maximum = int(info.get("max_context") or 0) if model_type == "text" else 0
    info.update({
        "name": f"{name} ({'HF Custom' if fetched else 'Custom'})",
        "short_name": name,
        "model_type": model_type,
        "real_media_engine": model_type in {"image", "video"},
        "model_path": info.get("strata_model_path") if is_strata else chosen_path if local else None,
        "repo_id": None if is_strata else repository,
        "source_kind": "strata_local" if is_strata else "hf_fetch" if fetched else "local_import" if local else "hf_import",
        "source_origin": "hf_fetch" if fetched else "local" if local else "hf_import",
        "source_repo_id": repository,
        "source_revision": info.get("source_revision") or info.get("repo_revision") or info.get("revision"),
        "runtime_preference": preference,
        "input_modalities": list(info.get("input_modalities") or ["text"]) if model_type == "text" else [],
        "precision": str(info.get("precision") or "Checkpoint precision; unchanged"),
        "ternary": bool(info.get("ternary", False)),
        "raw_params": int(info.get("raw_params") or 0),
        "base_params": str(info.get("base_params") or "Custom"),
        "max_context": maximum,
        "vram": "Automatic placement" if model_type == "text" else "Auto",
        "tag": f"{'🌐' if fetched else '🧩'} {name}",
    })
    if model_type == "text":
        info.setdefault("default_context", min(maximum, 32768) if maximum else 32768)
        info.setdefault("auto_memory_placement", True)
    if is_strata:
        info.setdefault("storage_note", "Using your selected local GGUF files. Preparation may require additional disk space; no replacement weights are downloaded.")
        info.setdefault("memory_note", "Strata checks the selected model and configures GPU, RAM and SSD placement where supported.")
    return info
