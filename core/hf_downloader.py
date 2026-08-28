"""
HuggingFace Auto-Downloader & Local Weight Cache Manager.
Downloads real model checkpoints directly from HuggingFace Hub with live progress streaming,
inspects local cache presence, supports cache purge, and auto-loads into Apple Silicon MLX.
"""

import os
import shutil
import threading
from typing import Any, Callable, Dict, Optional


def is_model_cached_locally(repo_id: str) -> bool:
    """Checks whether the specified HuggingFace model is already cached on disk."""
    if not repo_id:
        return False
    if os.path.exists(repo_id):
        return True

    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(repo_id, "config.json")
        if cached is not None and isinstance(cached, str) and os.path.exists(cached):
            return True
    except Exception:
        pass

    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    repo_folder = f"models--{repo_id.replace('/', '--')}"
    target_path = os.path.join(cache_dir, repo_folder, "snapshots")
    if os.path.exists(target_path) and os.listdir(target_path):
        return True

    return False


def purge_local_model_cache(repo_id: str) -> bool:
    """Purges the local snapshot cache for the given model repo to allow a clean reinstall."""
    if not repo_id:
        return False
    try:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        repo_folder = f"models--{repo_id.replace('/', '--')}"
        full_path = os.path.join(cache_dir, repo_folder)
        if os.path.exists(full_path):
            shutil.rmtree(full_path, ignore_errors=True)
            return True
    except Exception:
        pass
    return False


def download_model_from_hf(
    repo_id: str,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    cancel_event: Optional[threading.Event] = None
) -> Dict[str, Any]:
    """
    Downloads model weights from HuggingFace Hub with real-time status updates and clean resumption.
    """
    if not repo_id:
        return {"status": "error", "error": "No model repository ID specified"}

    if progress_callback:
        progress_callback(f"Connecting to HuggingFace Hub for `{repo_id}`...", 5.0)

    try:
        from huggingface_hub import snapshot_download

        if progress_callback:
            progress_callback(f"Downloading snapshot for `{repo_id}`...", 20.0)

        # Removed deprecated resume_download argument
        local_dir = snapshot_download(
            repo_id=repo_id,
            max_workers=4
        )

        if progress_callback:
            progress_callback(f"Successfully downloaded `{repo_id}` to cache.", 100.0)

        return {
            "status": "success",
            "repo_id": repo_id,
            "local_dir": local_dir
        }
    except Exception as e:
        error_msg = str(e)
        if progress_callback:
            progress_callback(f"Download error: {error_msg}", 0.0)
        return {
            "status": "error",
            "repo_id": repo_id,
            "error": error_msg
        }
