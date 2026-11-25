"""Immutable source and artifact checks for auto-built native runtimes."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


_FULL_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
_MAX_RECEIPT_BYTES = 64 * 1024


def require_full_commit(value: str, setting: str) -> str:
    """Return a normalized immutable commit or reject mutable Git references."""
    commit = str(value or "").strip()
    if not _FULL_COMMIT.fullmatch(commit):
        raise RuntimeError(
            f"{setting} must name a reviewed full 40-character Git commit; "
            "branches, tags, abbreviated hashes, and empty values are refused."
        )
    return commit.lower()


def _run_git(git: str, source: Optional[Path], *args: str) -> subprocess.CompletedProcess:
    command = [git, "-c", "credential.helper=", "-c", f"core.hooksPath={os.devnull}"]
    if source is not None:
        command += ["-C", str(source)]
    command += [str(arg) for arg in args]
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    return subprocess.run(command, capture_output=True, text=True, env=env)


def _checked_git(git: str, source: Optional[Path], *args: str) -> str:
    proc = _run_git(git, source, *args)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "git command failed").strip()
        raise RuntimeError(detail)
    # Leading status markers are significant for `git submodule status`.
    return (proc.stdout or "").rstrip()


def verify_pinned_checkout(
    git: str, source: Path, commit: str, *, recursive: bool = False,
) -> str:
    """Verify that executable source is the exact clean approved commit."""
    approved = require_full_commit(commit, "approved runtime commit")
    if not (source / ".git").exists():
        raise RuntimeError(f"Native runtime source is not a Git checkout: {source}")

    head = _checked_git(git, source, "rev-parse", "--verify", "HEAD^{commit}").lower()
    if head != approved:
        raise RuntimeError(
            f"Native runtime checkout HEAD {head or 'unknown'} does not match approved commit {approved}."
        )

    changed = _checked_git(
        git,
        source,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--ignore-submodules=none",
    )
    if changed:
        raise RuntimeError("Native runtime checkout has modified tracked source; refusing to execute it.")

    if recursive:
        submodules = _checked_git(git, source, "submodule", "status", "--recursive")
        if any(line and line[0] != " " for line in submodules.splitlines()):
            raise RuntimeError("Native runtime submodules do not match the commits pinned by the approved source.")
    return approved


def ensure_pinned_checkout(
    git: str,
    source: Path,
    repository: str,
    commit: str,
    *,
    recursive: bool = False,
) -> str:
    """Create or verify a detached checkout fetched by full commit identity."""
    approved = require_full_commit(commit, "approved runtime commit")
    source = Path(source)

    if (source / ".git").exists():
        verify_pinned_checkout(git, source, approved)
        if recursive:
            _checked_git(git, source, "submodule", "update", "--init", "--recursive", "--depth", "1")
        return verify_pinned_checkout(git, source, approved, recursive=recursive)

    if source.exists() and (not source.is_dir() or any(source.iterdir())):
        raise RuntimeError(f"Refusing to replace non-Git native runtime source: {source}")

    source.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{source.name}-", dir=str(source.parent)))
    try:
        _checked_git(git, None, "init", str(stage))
        _checked_git(git, stage, "remote", "add", "origin", repository)
        _checked_git(git, stage, "fetch", "--depth", "1", "--no-tags", "origin", approved)
        _checked_git(git, stage, "checkout", "--detach", approved)
        if recursive:
            _checked_git(git, stage, "submodule", "update", "--init", "--recursive", "--depth", "1")
        verify_pinned_checkout(git, stage, approved, recursive=recursive)
        if source.exists():
            source.rmdir()
        stage.replace(source)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return approved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_records(root: Path, artifacts: Mapping[str, Path]) -> Dict[str, Dict[str, str]]:
    resolved_root = Path(root).resolve(strict=True)
    records: Dict[str, Dict[str, str]] = {}
    for name, artifact in artifacts.items():
        resolved = Path(artifact).resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"Native runtime artifact is not a file: {artifact}")
        relative = resolved.relative_to(resolved_root)
        records[str(name)] = {"path": relative.as_posix(), "sha256": sha256_file(resolved)}
    if not records:
        raise ValueError("A native runtime receipt requires at least one artifact")
    return records


def write_runtime_receipt(
    receipt: Path,
    root: Path,
    repository: str,
    commit: str,
    artifacts: Mapping[str, Path],
    build_inputs: Mapping[str, Any],
) -> None:
    """Atomically bind built artifact bytes and build inputs to approved source."""
    payload = {
        "schema_version": 1,
        "source": {
            "repository": str(repository),
            "commit": require_full_commit(commit, "approved runtime commit"),
        },
        "build_inputs": dict(build_inputs),
        "artifacts": _artifact_records(root, artifacts),
    }
    receipt = Path(receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=str(receipt.parent), prefix=f".{receipt.name}-", delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, receipt)
        temporary = None
    finally:
        if temporary:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass


def validate_runtime_receipt(
    receipt: Path,
    root: Path,
    repository: str,
    commit: str,
    artifacts: Mapping[str, Path],
    build_inputs: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Return true only when source identity, inputs, paths, and bytes still match."""
    try:
        receipt = Path(receipt)
        if not receipt.is_file() or receipt.stat().st_size > _MAX_RECEIPT_BYTES:
            return False
        with receipt.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            return False
        expected_source = {
            "repository": str(repository),
            "commit": require_full_commit(commit, "approved runtime commit"),
        }
        if payload.get("source") != expected_source:
            return False
        if build_inputs is not None and payload.get("build_inputs") != dict(build_inputs):
            return False
        return payload.get("artifacts") == _artifact_records(root, artifacts)
    except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError):
        return False
