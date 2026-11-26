"""
Streaming Model Auto-Downloader & Local Weight Cache Manager.
Manages downloading and verifying MLX, GGUF, BitNet, and SafeTensors model weights from Hugging Face
with real-time progress callbacks, resume support, and offline cache resolution.
"""

import os
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from config.paths import get_portable_data_dir
from core.hf_downloader import _snapshot_looks_installed


def get_models_cache_dir() -> str:
    """Returns directory for storing downloaded model weights."""
    portable_dir = get_portable_data_dir()
    models_dir = os.path.join(portable_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def is_model_available_locally(artifact_id: str) -> Tuple[bool, Optional[str]]:
    """Checks whether model artifact is available in local directory or HuggingFace cache."""
    if not artifact_id:
        return False, None

    # 1. Direct path check
    if _snapshot_looks_installed(artifact_id):
        return True, artifact_id

    # 2. Check portable models directory
    models_dir = get_models_cache_dir()
    sanitized = artifact_id.replace("/", "--")
    candidate_path = os.path.join(models_dir, sanitized)
    if _snapshot_looks_installed(candidate_path):
        return True, candidate_path

    # Check for direct file in models dir
    file_candidate = os.path.join(models_dir, os.path.basename(artifact_id))
    if _snapshot_looks_installed(file_candidate):
        return True, file_candidate

    # 3. Check HuggingFace Hub standard cache
    try:
        from core.controller_runtime import resolve_local_snapshot
        snapshot = resolve_local_snapshot(artifact_id)
        if snapshot and _snapshot_looks_installed(snapshot):
            return True, snapshot
    except Exception:
        pass

    return False, None


def _selected_local_path(path, filename=None, required_files=None):
    if not path:
        return None
    directory = path if os.path.isdir(path) else os.path.dirname(path)
    try:
        if any(not os.path.isfile(os.path.join(directory, name)) or
               os.path.getsize(os.path.join(directory, name)) == 0 for name in required_files or []):
            return None
    except OSError:
        return None
    if filename:
        if os.path.isdir(path):
            path = os.path.join(path, filename)
        elif os.path.basename(path) != os.path.basename(filename):
            return None
    return path if _snapshot_looks_installed(path) else None


def ensure_model_available(
    repo_or_file_id: str,
    backend: str = "mlx",
    filename: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    auto_download: bool = True,
    revision: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    required_files: Optional[List[str]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """
    Verifies that model weights exist locally.
    If missing and auto_download is enabled, streams download from Hugging Face with progress callbacks.
    """
    if cancel_event is not None and cancel_event.is_set():
        return {"status": "cancelled", "path": None, "cached": False, "error": "Download cancelled."}

    local_path = None
    selection_patterns = allow_patterns
    if selection_patterns is None and filename:
        selection_patterns = list(dict.fromkeys([filename, *(required_files or [])]))
    if revision:
        # An arbitrary cached revision or an unversioned portable directory is
        # not proof that the user's pinned selection is installed.
        try:
            from core.hf_transfer import snapshot_download
            local_path = snapshot_download(repo_id=repo_or_file_id, revision=revision,
                                           allow_patterns=selection_patterns, local_files_only=True,
                                           cancel_event=cancel_event)
        except Exception:
            pass
    else:
        try:
            _, local_path = is_model_available_locally(repo_or_file_id)
        except OSError as error:
            from core.hf_transfer import format_download_error
            return {"status": "error", "path": None, "cached": False,
                    "error": format_download_error(error)}
    if cancel_event is not None and cancel_event.is_set():
        return {"status": "cancelled", "path": None, "cached": False, "error": "Download cancelled."}
    local_path = _selected_local_path(local_path, filename, required_files)
    if local_path:
        if progress_callback:
            progress_callback(1.0, f"✓ Model weights verified locally at `{local_path}`")
        return {
            "status": "ready",
            "path": local_path,
            "cached": True
        }

    if not auto_download:
        return {
            "status": "not_downloaded",
            "path": None,
            "message": f"Weights for `{repo_or_file_id}` are not downloaded yet."
        }

    # Execute Download
    if progress_callback:
        progress_callback(0.05, f"⬇️ Connecting to Hugging Face for `{repo_or_file_id}`...")

    try:
        from core.hf_transfer import snapshot_download, hf_hub_download, DownloadCancelled

        target_options = {}
        if not revision:
            models_dir = get_models_cache_dir()
            sanitized = repo_or_file_id.replace("/", "--")
            target_dir = os.path.join(models_dir, sanitized)
            os.makedirs(target_dir, exist_ok=True)
            target_options["local_dir"] = target_dir

        def transfer_progress(message, percent):
            if progress_callback:
                progress_callback(percent / 100.0, message)

        if filename and not required_files:
            # Download specific file (e.g. GGUF single binary)
            if progress_callback:
                progress_callback(0.25, f"⬇️ Downloading `{filename}` from `{repo_or_file_id}`...")
            dl_path = hf_hub_download(
                repo_id=repo_or_file_id,
                filename=filename,
                **target_options,
                revision=revision,
                cancel_event=cancel_event,
                progress_callback=transfer_progress,
            )
            if cancel_event is not None and cancel_event.is_set():
                raise DownloadCancelled("Download cancelled; completed model files were kept.")
            if not _snapshot_looks_installed(dl_path):
                raise ValueError("The selected model file is missing or empty after download.")
            if progress_callback:
                progress_callback(1.0, f"✓ Successfully downloaded `{filename}`")
            return {
                "status": "ready",
                "path": dl_path,
                "cached": False
            }
        else:
            # Download full snapshot (e.g. MLX or SafeTensors folder)
            if progress_callback:
                progress_callback(0.20, f"⬇️ Downloading snapshot for `{repo_or_file_id}`...")
            dl_path = snapshot_download(
                repo_id=repo_or_file_id,
                **target_options,
                revision=revision,
                allow_patterns=selection_patterns,
                cancel_event=cancel_event,
                progress_callback=transfer_progress,
            )
            if cancel_event is not None and cancel_event.is_set():
                raise DownloadCancelled("Download cancelled; completed model files were kept.")
            if not _snapshot_looks_installed(dl_path):
                raise ValueError("The download did not produce a complete model snapshot.")
            missing = [name for name in required_files or []
                       if not os.path.isfile(os.path.join(dl_path, name))
                       or os.path.getsize(os.path.join(dl_path, name)) == 0]
            if missing:
                raise ValueError("Selected model files are missing: " + ", ".join(missing))
            selected_path = _selected_local_path(dl_path, filename, required_files)
            if not selected_path:
                raise ValueError("The selected model file is missing or empty after download.")
            if progress_callback:
                progress_callback(1.0, f"✓ Successfully downloaded `{repo_or_file_id}`")
            return {
                "status": "ready",
                "path": selected_path,
                "cached": False
            }

    except Exception as e:
        from core.hf_transfer import DownloadCancelled, format_download_error
        cancelled = isinstance(e, DownloadCancelled) or (cancel_event is not None and cancel_event.is_set())
        message = format_download_error(e)
        if progress_callback:
            progress_callback(0.0, "Download cancelled." if cancelled else message)
        return {
            "status": "cancelled" if cancelled else "error",
            "path": None,
            "error": message,
            "cached": False
        }
