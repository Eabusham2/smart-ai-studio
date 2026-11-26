"""
HuggingFace Auto-Downloader & Local Weight Cache Manager.
Downloads real model checkpoints directly from HuggingFace Hub with live progress streaming,
inspects local cache presence, supports cache purge, and tracks models proven loaded in memory.
"""

import os
import json
import re
import shutil
import threading
from pathlib import PurePosixPath
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
    "model_index.json",
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
        return path.lower().endswith(_WEIGHT_SUFFIXES) and os.path.getsize(path) > 0

    has_weights = False
    has_metadata = False
    try:
        for root, _, files in os.walk(path):
            for filename in files:
                lower = filename.lower()
                full_path = os.path.join(root, filename)
                if lower.endswith(".index.json"):
                    with open(full_path, encoding="utf-8") as handle:
                        index = json.load(handle)
                    shards = set((index.get("weight_map") or {}).values())
                    if shards and not all(os.path.isfile(os.path.join(root, shard)) and
                                          os.path.getsize(os.path.join(root, shard)) > 0
                                          for shard in shards):
                        return False
                if lower.endswith(_WEIGHT_SUFFIXES) and os.path.isfile(full_path) and os.path.getsize(full_path) > 0:
                    has_weights = True
                if filename in _METADATA_NAMES or lower.endswith("config.json"):
                    has_metadata = True
    except Exception:
        return False

    return has_weights


def _local_selection_present(path: str, required_files=None) -> bool:
    """Validate the selected files, including nested companions and shards."""
    try:
        if not path or not os.path.exists(path):
            return False
        if required_files is not None:
            if not required_files or isinstance(required_files, (str, bytes)):
                return False
            names = []
            for name in required_files:
                if not isinstance(name, str) or not name or "\\" in name or ":" in name:
                    return False
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts:
                    return False
                names.append(relative)
            directory = path
            if os.path.isfile(path):
                # An explicit selected file may live below the snapshot root.
                # Recover that root only from an exact selected relative suffix.
                absolute = os.path.abspath(path).replace(os.sep, "/")
                matching = [name for name in names if absolute.endswith("/" + name.as_posix())]
                if not matching:
                    return False
                selected = max(matching, key=lambda name: len(name.parts))
                directory = path
                for _ in selected.parts:
                    directory = os.path.dirname(directory)
            if not all(os.path.isfile(os.path.join(directory, *name.parts)) and
                       os.path.getsize(os.path.join(directory, *name.parts)) > 0 for name in names):
                return False
            for name in names:
                if name.as_posix().endswith(".index.json"):
                    with open(os.path.join(directory, *name.parts), encoding="utf-8") as handle:
                        index = json.load(handle)
                    for shard in set((index.get("weight_map") or {}).values()):
                        relative = PurePosixPath(shard)
                        if relative.is_absolute() or ".." in relative.parts or "\\" in shard:
                            return False
                        shard_path = os.path.join(directory, *name.parent.parts, *relative.parts)
                        if not os.path.isfile(shard_path) or os.path.getsize(shard_path) == 0:
                            return False
            # Unselected alternate-format indexes in this same commit must not
            # make a complete selected checkpoint look incomplete.
            return any(name.as_posix().lower().endswith(_WEIGHT_SUFFIXES) for name in names)
        return _snapshot_looks_installed(path)
    except (OSError, TypeError, ValueError, AttributeError):
        return False


def resolve_cached_snapshot(repo_id: str, revision=None, *, allow_patterns=None, required_files=None) -> Optional[str]:
    """Resolve only the selected local Hub snapshot; never fetch or change refs.

    Recent Hub versions verify cached repository inventories during offline
    lookup. A filtered download must be looked up with the same selection, or
    an intentionally omitted README/alternate quant makes it look incomplete.
    """
    if not repo_id:
        return None
    if allow_patterns is None and required_files is not None:
        allow_patterns = required_files
    try:
        from core.hf_transfer import snapshot_download
        options = {"repo_id": repo_id, "revision": revision, "local_files_only": True}
        if allow_patterns is not None:
            options["allow_patterns"] = allow_patterns
        # The default API cache is the same one downloads use. Additional known
        # roots cover a changed HF_HOME/cache setting without trying other pins.
        cache_roots = list(_candidate_hf_cache_roots())
        for index, cache_root in enumerate(cache_roots or [None]):
            try:
                snapshot = snapshot_download(**options, **({"cache_dir": cache_root} if index else {}))
                if _local_selection_present(snapshot, required_files):
                    return os.path.abspath(snapshot)
            except Exception:
                continue
    except (ImportError, OSError, TypeError, ValueError):
        pass
    return None


def resolve_cached_model_path(model_info: Dict[str, Any], explicit_path=None) -> Optional[str]:
    """Return the complete selected local model for UI, Eval, and loading."""
    info = model_info or {}
    revision = info.get("repo_revision") or info.get("source_revision")
    pinned = isinstance(revision, str) and len(revision) == 40 and all(c in "0123456789abcdefABCDEF" for c in revision)
    required = info.get("download_required_files")
    if required is None and info.get("gguf_file"):
        required = list(info.get("gguf_files") or [info["gguf_file"]])
        if info.get("mmproj_file") and info["mmproj_file"] not in required and (not info.get("mmproj_repo_id") or info.get("mmproj_repo_id") == info.get("repo_id")):
            required.append(info["mmproj_file"])
    for candidate in (explicit_path, info.get("path"), info.get("model_path")):
        if not candidate:
            continue
        expanded = os.path.abspath(os.path.expanduser(str(candidate)))
        if os.path.exists(expanded):
            parts = expanded.replace(os.sep, "/").split("/")
            # A stale recorded Hub cache path cannot override a newly selected
            # commit. Explicit custom folders retain their normal semantics.
            if pinned and any(part == "snapshots" and index > 0 and parts[index - 1].startswith("models--")
                                and index + 1 < len(parts) and parts[index + 1].lower() != revision.lower()
                                for index, part in enumerate(parts)):
                continue
            return expanded if _local_selection_present(expanded, required) else None
    repo_id = info.get("repo_id") or explicit_path or info.get("model_path")
    return resolve_cached_snapshot(repo_id, revision,
                                   allow_patterns=info.get("download_allow_patterns"),
                                   required_files=required)


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
    revision: Optional[str] = None,
) -> Dict[str, Any]:
    """Downloads model weights from HuggingFace Hub with real-time status updates."""
    if not repo_id:
        return {"status": "error", "error": "No model repository ID specified"}
    if cancel_event is not None and cancel_event.is_set():
        return {"status": "cancelled", "repo_id": repo_id, "error": "Download cancelled."}

    if progress_callback:
        progress_callback(f"Connecting to HuggingFace Hub for `{repo_id}`...", 5.0)

    try:
        from core.hf_transfer import snapshot_download

        if progress_callback:
            progress_callback(f"Downloading snapshot for `{repo_id}`...", 20.0)

        local_dir = snapshot_download(
            repo_id=repo_id,
            max_workers=4,
            allow_patterns=list(allow_patterns) if allow_patterns is not None else None,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
            **({"revision": revision} if revision else {}),
        )
        if cancel_event is not None and cancel_event.is_set():
            return {"status": "cancelled", "repo_id": repo_id, "local_dir": local_dir,
                    "error": "Download cancelled; completed cache files are kept."}

        if required_files:
            missing = [
                name for name in required_files
                if not os.path.isfile(os.path.join(local_dir, name))
                or os.path.getsize(os.path.join(local_dir, name)) == 0
            ]
            if missing:
                # Hub can consider a completed zero-byte blob cached and return
                # it on every retry. Repair only the failed selected artifacts;
                # other components and quantizations retain their bytes/mtimes.
                from core.hf_transfer import hf_hub_download
                from pathlib import Path, PurePosixPath
                snapshot_path = Path(local_dir)
                cache_options = {}
                repair_revision = revision
                if snapshot_path.parent.name == "snapshots" and snapshot_path.parent.parent.name.startswith("models--"):
                    cache_options["cache_dir"] = str(snapshot_path.parent.parent.parent)
                    if re.fullmatch(r"[0-9a-fA-F]{40}", snapshot_path.name):
                        if revision and re.fullmatch(r"[0-9a-fA-F]{40}", str(revision)):
                            if str(revision).lower() != snapshot_path.name.lower():
                                raise ValueError("The downloaded snapshot does not match the selected repository revision.")
                        else:
                            # A branch can move between snapshot transfer and
                            # repair. Keep every companion on the resolved commit.
                            repair_revision = snapshot_path.name
                for name in missing:
                    relative = PurePosixPath(name)
                    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                        raise ValueError("The selected model has an invalid artifact filename.")
                    hf_hub_download(
                        repo_id=repo_id, filename=name, revision=repair_revision,
                        force_download=True, cancel_event=cancel_event,
                        progress_callback=progress_callback, **cache_options,
                    )
                if cancel_event is not None and cancel_event.is_set():
                    return {"status": "cancelled", "repo_id": repo_id, "local_dir": local_dir,
                            "error": "Download cancelled; completed cache files are kept."}
                missing = [name for name in required_files
                           if not os.path.isfile(os.path.join(local_dir, name))
                           or os.path.getsize(os.path.join(local_dir, name)) == 0]
            if missing:
                return {
                    "status": "error",
                    "repo_id": repo_id,
                    "local_dir": local_dir,
                    "error": "Pinned Hugging Face artifact(s) missing: " + ", ".join(missing),
                }

        if not _local_selection_present(local_dir, required_files):
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
        from core.hf_transfer import DownloadCancelled, format_download_error
        if isinstance(e, DownloadCancelled) or (cancel_event is not None and cancel_event.is_set()):
            return {"status": "cancelled", "repo_id": repo_id,
                    "error": "Download cancelled; completed cache files are kept."}
        error_msg = format_download_error(e)
        if progress_callback:
            progress_callback(f"Download error: {error_msg}", 0.0)
        return {
            "status": "error",
            "repo_id": repo_id,
            "error": error_msg,
        }
