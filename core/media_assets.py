"""Install and resolve the complete, pinned component set of a media preset.

A diffusion transformer alone is not an installed pipeline.  The same resolver
is used by the Install button, restart checks, and native generation workers.
All companions use the shared Hub transfer recovery and retain completed files.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import platform


def validate_media_platform(info):
    if info.get("platform_requirement") == "Apple Silicon Mac" and not (
            platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}):
        raise RuntimeError(f"{info.get('short_name') or info.get('repo_id')} requires an Apple Silicon Mac for its MLX runtime.")


def _selections(info):
    primary = {
        "role": "model", "repo_id": info.get("repo_id"),
        "revision": info.get("repo_revision"),
        "required_files": info.get("download_required_files"),
        "size_bytes": info.get("download_primary_bytes", info.get("download_bytes", 0)),
    }
    selections = [primary, *list(info.get("download_components") or [])]
    roles = set()
    for entry in selections:
        role, repo = entry.get("role"), entry.get("repo_id")
        revision, files = entry.get("revision"), entry.get("required_files")
        if not isinstance(role, str) or not role or role in roles:
            raise ValueError("The media component list has a missing or repeated role.")
        roles.add(role)
        if not repo or not re.fullmatch(r"[0-9a-fA-F]{40}", str(revision or "")):
            raise ValueError(f"The {role} component needs an exact repository revision.")
        if not isinstance(files, (list, tuple)) or not files or len(files) != len(set(files)):
            raise ValueError(f"The {role} component has no complete, exact artifact list.")
        for filename in files:
            if (not isinstance(filename, str) or not filename
                    or PurePosixPath(filename).is_absolute()
                    or ".." in PurePosixPath(filename).parts
                    or any(char in filename for char in "*?[]\\")):
                raise ValueError(f"The {role} component has an invalid artifact filename.")
    return selections


def _component_info(info, selection):
    if selection["role"] == "model":
        return info
    return {
        "repo_id": selection["repo_id"], "repo_revision": selection["revision"],
        "download_required_files": selection["required_files"],
        "download_allow_patterns": selection["required_files"],
        "path": (info.get("component_paths") or {}).get(selection["role"]),
    }


def _resolved(info, selections, paths):
    result = dict(info)
    result["path"] = result["model_path"] = paths["model"]
    result["component_paths"] = {role: path for role, path in paths.items() if role != "model"}
    result["component_files"] = {
        selection["role"]: [str(Path(paths[selection["role"]]) / name)
                            for name in selection["required_files"]]
        for selection in selections if selection["role"] != "model"
    }
    return result


def resolve_media_model_assets(info):
    """Return an offline-ready copy, or None if any selected component is absent."""
    from core.hf_downloader import resolve_cached_model_path
    selections = _selections(info)
    paths = {}
    for selection in selections:
        path = resolve_cached_model_path(_component_info(info, selection))
        if path is None:
            return None
        paths[selection["role"]] = path
    return _resolved(info, selections, paths)


def download_media_model_assets(info, *, progress_callback=None, cancel_event=None):
    """Install only required files, with aggregate progress across companions.

    The callback has the existing downloader signature (message, percent).
    Report success only after the complete selection resolves offline.
    """
    from core.hf_downloader import download_model_from_hf, resolve_cached_model_path
    validate_media_platform(info)
    selections = _selections(info)
    sizes = [max(1, int(entry.get("size_bytes") or 1)) for entry in selections]
    total, completed = sum(sizes), 0
    paths = {}
    for selection, size in zip(selections, sizes):
        if cancel_event is not None and cancel_event.is_set():
            return {"status": "cancelled"}
        role = selection["role"]
        selected_info = _component_info(info, selection)
        path = resolve_cached_model_path(selected_info)
        if path is None:
            def progress(message, percent):
                if progress_callback:
                    progress_callback(f"{role}: {message}",
                                      100.0 * (completed + size * min(100.0, max(0.0, percent)) / 100.0) / total)
            result = download_model_from_hf(
                selection["repo_id"], revision=selection["revision"],
                required_files=list(selection["required_files"]),
                allow_patterns=list(selection["required_files"]),
                progress_callback=progress, cancel_event=cancel_event,
            )
            if result.get("status") != "success":
                return {**result, "component": role}
            path = resolve_cached_model_path({**selected_info, "path": result.get("local_dir")})
            if path is None:
                return {"status": "error", "component": role,
                        "error": f"The {role} download finished, but its selected files are incomplete. Completed files were kept."}
        paths[role] = path
        completed += size
        if progress_callback:
            progress_callback(f"{role}: selected files ready", 100.0 * completed / total)
    if cancel_event is not None and cancel_event.is_set():
        return {"status": "cancelled"}
    resolved = _resolved(info, selections, paths)
    return {"status": "success", "local_dir": paths["model"], "model_info": resolved,
            "component_paths": resolved["component_paths"], "component_files": resolved["component_files"]}


def ensure_media_model_assets(info, *, progress_callback=None, cancel_event=None):
    """Resolve or install a preset for a generation request using fraction progress."""
    resolved = resolve_media_model_assets(info)
    if resolved is not None:
        return resolved
    result = download_media_model_assets(
        info, cancel_event=cancel_event,
        progress_callback=(lambda message, percent: progress_callback(0.02 + 0.06 * percent / 100.0, message))
        if progress_callback else None,
    )
    if result.get("status") != "success":
        raise RuntimeError(result.get("error") or "Media model installation cancelled.")
    return result["model_info"]
