"""Canonical path validation for workspace-scoped tools."""

import os
from pathlib import Path
from typing import Optional, Union


PathInput = Union[str, os.PathLike]


class WorkspacePathError(ValueError):
    """Raised when a tool path is unavailable or leaves the active workspace."""


def canonical_workspace(workspace_dir: Optional[PathInput]) -> Optional[str]:
    """Return an existing canonical workspace directory, or ``None`` when cleared."""
    if workspace_dir is None:
        return None
    try:
        raw_workspace = os.fspath(workspace_dir)
        if not isinstance(raw_workspace, str):
            raise WorkspacePathError("Workspace directory is unavailable.")
        if not raw_workspace.strip():
            return None
        if "\x00" in raw_workspace:
            raise WorkspacePathError("Workspace directory is unavailable.")
        root = Path(raw_workspace).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise WorkspacePathError("Workspace directory is unavailable.") from exc
    if not root.is_dir():
        raise WorkspacePathError("Workspace path must be an existing directory.")
    return str(root)


def resolve_workspace_path(
    workspace_dir: Optional[PathInput],
    requested_path: PathInput,
    *,
    for_creation: bool = False,
    basename_only: bool = False,
) -> str:
    """Resolve a relative tool path and require canonical workspace containment."""
    root_value = canonical_workspace(workspace_dir)
    if root_value is None:
        raise WorkspacePathError("No workspace is selected.")
    if requested_path is None:
        raise WorkspacePathError("Workspace path is required.")

    root = Path(root_value)
    try:
        selected_root = Path(os.fspath(workspace_dir)).expanduser()
    except (TypeError, ValueError) as exc:
        raise WorkspacePathError("Workspace directory is unavailable.") from exc
    if selected_root != root:
        raise WorkspacePathError("Selected workspace identity has changed.")
    try:
        raw_path = os.fspath(requested_path)
        if not isinstance(raw_path, str) or not raw_path.strip() or "\x00" in raw_path:
            raise WorkspacePathError("Workspace path is required.")
        requested = Path(raw_path)
    except (TypeError, ValueError) as exc:
        raise WorkspacePathError("Workspace path is invalid.") from exc
    if requested.is_absolute() or requested.drive or requested.root:
        raise WorkspacePathError("Workspace paths must be relative.")
    if any(part == os.pardir for part in requested.parts):
        raise WorkspacePathError("Workspace paths cannot contain parent traversal.")
    if basename_only and (len(requested.parts) != 1 or requested.name in ("", ".", "..")):
        raise WorkspacePathError("Artifact names must be plain filenames.")

    candidate = root / requested
    try:
        target = candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise WorkspacePathError("Workspace path is unavailable.") from exc

    try:
        target.relative_to(root)
    except ValueError as exc:
        raise WorkspacePathError("Workspace path escapes the selected workspace.") from exc

    if for_creation:
        # Resolve the nearest existing parent as a separate creation-time check.
        parent = candidate.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        try:
            resolved_parent = parent.resolve(strict=True)
            resolved_parent.relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise WorkspacePathError("Artifact parent escapes the selected workspace.") from exc

        # Existing symlink components are unsafe for writes because their target
        # can change between validation and creation.
        current = root
        for part in requested.parts:
            current = current / part
            if current.is_symlink():
                raise WorkspacePathError("Artifact paths cannot use symlinks.")

    return str(target)
