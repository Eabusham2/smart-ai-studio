"""
HuggingFace Auto-Downloader & Local Weight Cache Manager.
Downloads real model checkpoints directly from HuggingFace Hub with live progress streaming,
inspects local cache presence, supports cache purge, and tracks models proven loaded in memory.
"""

import os
import shutil
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional


_WEIGHT_SUFFIXES = (
    ".safetensors",
    ".gguf",
    ".bin",
    ".npz",
    ".pt",
    ".pth",
)
_METADATA_NAMES = {
    "config.json",
    "tokenizer_config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
}
_LOADED_MODEL_KEYS = set()
_LOADED_MODEL_LOCK = threading.Lock()


def _model_key(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    if os.path.exists(os.path.expanduser(value)):
        return os.path.abspath(os.path.expanduser(value))
    return value


def register_loaded_model(identifier: str) -> None:
    """Record a model identifier/path that has successfully loaded into memory."""
    key = _model_key(identifier)
    if not key:
        return
    with _LOADED_MODEL_LOCK:
        _LOADED_MODEL_KEYS.add(key)


def unregister_loaded_model(identifier: Optional[str] = None) -> None:
    """Remove one loaded model identifier, or clear all when the active model unloads."""
    with _LOADED_MODEL_LOCK:
        if identifier is None:
            _LOADED_MODEL_KEYS.clear()
        else:
            _LOADED_MODEL_KEYS.discard(_model_key(identifier))


def is_model_registered_loaded(identifier: str) -> bool:
    key = _model_key(identifier)
    if not key:
        return False
    with _LOADED_MODEL_LOCK:
        return key in _LOADED_MODEL_KEYS


def _candidate_hf_cache_roots() -> Iterable[str]:
    """Returns every Hugging Face hub cache root that may be active on this machine."""
    seen = set()

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        if HF_HUB_CACHE:
            path = os.path.abspath(os.path.expanduser(str(HF_HUB_CACHE)))
            if path not in seen:
                seen.add(path)
                yield path
    except Exception:
        pass

    env_hub = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")
    if env_hub:
        path = os.path.abspath(os.path.expanduser(env_hub))
        if path not in seen:
            seen.add(path)
            yield path

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        path = os.path.abspath(os.path.expanduser(os.path.join(hf_home, "hub")))
        if path not in seen:
            seen.add(path)
            yield path

    default = os.path.abspath(os.path.expanduser("~/.cache/huggingface/hub"))
    if default not in seen:
        yield default


def _snapshot_looks_installed(path: str) -> bool:
    """Require real model assets, not merely an empty/partial HF snapshot directory."""
    if not path or not os.path.exists(path):
        return False
    if os.path.isfile(path):
        return path.lower().endswith(_WEIGHT_SUFFIXES)

    has_weights = False
    has_metadata = False
    try:
        for root, _, files in os.walk(path):
            for filename in files:
                lower = filename.lower()
                if lower.endswith(_WEIGHT_SUFFIXES):
                    has_weights = True
                if filename in _METADATA_NAMES or lower.endswith("config.json"):
                    has_metadata = True
                if has_weights and has_metadata:
                    return True
    except Exception:
        return False

    return has_weights


def is_model_cached_locally(repo_id: str) -> bool:
    """
    Reliably determines whether a model is installed locally.

    A model that is *already loaded successfully* is authoritative and therefore
    always reports ready, even if Hugging Face cache metadata is temporarily stale
    or lives under a nonstandard root. Otherwise a real weight file must be found.
    """
    if not repo_id:
        return False

    if is_model_registered_loaded(repo_id):
        return True

    try:
        from config.paths import get_bundled_model_path
        if get_bundled_model_path(repo_id):
            return True
    except Exception:
        pass

    expanded = os.path.abspath(os.path.expanduser(repo_id))
    if os.path.exists(expanded):
        return _snapshot_looks_installed(expanded)

    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if getattr(repo, "repo_id", None) != repo_id:
                continue
            for revision in getattr(repo, "revisions", ()):
                snapshot_path = str(getattr(revision, "snapshot_path", "") or "")
                if _snapshot_looks_installed(snapshot_path):
                    return True
    except Exception:
        pass

    try:
        from huggingface_hub import try_to_load_from_cache

        for marker in (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer_config.json",
        ):
            cached = try_to_load_from_cache(repo_id, marker)
            if isinstance(cached, str) and os.path.exists(cached):
                if _snapshot_looks_installed(os.path.dirname(cached)):
                    return True
    except Exception:
        pass

    repo_folder = f"models--{repo_id.replace('/', '--')}"
    for cache_root in _candidate_hf_cache_roots():
        snapshots = os.path.join(cache_root, repo_folder, "snapshots")
        if not os.path.isdir(snapshots):
            continue
        try:
            for revision in os.listdir(snapshots):
                if _snapshot_looks_installed(os.path.join(snapshots, revision)):
                    return True
        except Exception:
            continue

    return False


def purge_local_model_cache(repo_id: str) -> bool:
    """Purges all known local Hugging Face cache copies for a model repo."""
    if not repo_id:
        return False

    unregister_loaded_model(repo_id)
    removed = False
    repo_folder = f"models--{repo_id.replace('/', '--')}"
    for cache_root in _candidate_hf_cache_roots():
        full_path = os.path.join(cache_root, repo_folder)
        try:
            if os.path.exists(full_path):
                shutil.rmtree(full_path, ignore_errors=True)
                removed = True
        except Exception:
            pass
    return removed


def download_model_from_hf(
    repo_id: str,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    allow_patterns: Optional[List[str]] = None,
    required_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Downloads model weights from HuggingFace Hub with real-time status updates."""
    if not repo_id:
        return {"status": "error", "error": "No model repository ID specified"}

    if progress_callback:
        progress_callback(f"Connecting to HuggingFace Hub for `{repo_id}`...", 5.0)

    try:
        from huggingface_hub import snapshot_download

        if progress_callback:
            progress_callback(f"Downloading snapshot for `{repo_id}`...", 20.0)

        local_dir = snapshot_download(
            repo_id=repo_id,
            max_workers=4,
            allow_patterns=list(allow_patterns) if allow_patterns else None,
        )

        if required_files:
            missing = [
                name for name in required_files
                if not os.path.isfile(os.path.join(local_dir, name))
            ]
            if missing:
                return {
                    "status": "error",
                    "repo_id": repo_id,
                    "local_dir": local_dir,
                    "error": "Pinned Hugging Face artifact(s) missing: " + ", ".join(missing),
                }

        if not _snapshot_looks_installed(local_dir):
            return {
                "status": "error",
                "repo_id": repo_id,
                "error": "Hugging Face download returned without a usable model-weight snapshot",
            }

        if progress_callback:
            progress_callback(f"Successfully downloaded `{repo_id}` to cache.", 100.0)

        return {
            "status": "success",
            "repo_id": repo_id,
            "local_dir": local_dir,
        }
    except Exception as e:
        error_msg = str(e)
        if progress_callback:
            progress_callback(f"Download error: {error_msg}", 0.0)
        return {
            "status": "error",
            "repo_id": repo_id,
            "error": error_msg,
        }
