"""Embed the exact CI-built original Bonsai 2 runtime in release packages."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_smartai_bundled_native", ROOT / "core/bundled_native.py")
_native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_native)


def embed_native_runtime(code, target, *, source=None):
    """Copy a validated static artifact; fail a supported build if it is absent."""
    if target not in _native.TARGET_BACKENDS:
        if target.startswith("Windows-"):
            return None
        raise RuntimeError("No verified native artifact configured for " + target)
    source = source if source is not None else os.environ.get("SMARTAI_NATIVE_RUNTIME_DIR")
    if not source:
        raise RuntimeError("SMARTAI_NATIVE_RUNTIME_DIR must identify the tested native artifact")
    source, code = Path(source), Path(code)
    _native.validate_native_bundle(source, target, code_root=code)
    # The build uses BUILD_SHARED_LIBS=OFF. Reject additional files rather than
    # silently omitting a future shared library that needs its own receipt.
    allowed = {*_native.BUNDLE_FILES, "README.md"}
    for path in source.iterdir():
        if path.name not in allowed or path.is_symlink() or not path.is_file():
            raise ValueError("Unexpected file in the static native artifact: " + path.name)
    destination = code / "runtime/native"
    if destination.exists():
        raise RuntimeError("The native runtime destination must be new")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    (destination / "llama-server").chmod(0o755)
    _native.validate_native_bundle(destination, target, code_root=code)
    return destination
