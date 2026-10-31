"""Download an admitted, immutable model selection and inspect its actual files."""
from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

from core.hf_downloader import download_model_from_hf
from core.model_policy import admit_local_model


def download_imported_model(
    repo_id: str, metadata: dict[str, Any], *, progress_callback=None, cancel_event=None,
) -> dict[str, Any]:
    """Keep the selected revision and all shards; never register uninspected weights."""
    revision = str(metadata.get("repo_revision") or metadata.get("source_revision") or "")
    required = list(metadata.get("download_required_files") or [])
    allowed = list(metadata.get("download_allow_patterns") or [])
    if metadata.get("repo_id") != repo_id or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return {"status": "error", "error": "The model selection must have a verified repository and immutable revision."}
    for name in required + allowed:
        if (not isinstance(name, str) or not name or "\\" in name
                or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                or ":" in name or any(char in name for char in "*?[]")):
            return {"status": "error", "error": "The model selection contains an invalid artifact path."}
    if not required or not set(required).issubset(allowed):
        return {"status": "error", "error": "The model selection has no complete, exact artifact list."}
    result = download_model_from_hf(
        repo_id, revision=revision, allow_patterns=allowed, required_files=required,
        progress_callback=progress_callback, cancel_event=cancel_event,
    )
    if result.get("status") != "success":
        return result
    if cancel_event is not None and cancel_event.is_set():
        return {**result, "status": "cancelled", "error": "Import cancelled; completed cache files are kept."}
    local_dir = str(result.get("local_dir") or "")
    ok, reason, inspected = admit_local_model(local_dir, gguf_file=metadata.get("gguf_file") or None)
    if not ok:
        return {**result, "status": "error", "error": f"Downloaded model inspection failed: {reason}"}
    if metadata.get("task") == "media":
        # Some audio/media checkpoints declare their task only in immutable Hub
        # metadata. Local weight inspection must not turn that into a text model.
        for key in ("task", "model_type", "pipeline_tag", "controller_runtime", "backend_family", "input_modalities"):
            if key in metadata:
                inspected[key] = metadata[key]
    inspected.update({
        "repo_id": repo_id, "repo_revision": revision, "source_revision": revision,
        "source_repo_id": repo_id, "source_origin": "hf_fetch",
        "requires_local_validation": False,
        "download_required_files": required, "download_allow_patterns": allowed,
        "runtime_preference": str(metadata.get("runtime_preference") or "auto"),
    })
    return {**result, "metadata": inspected}
