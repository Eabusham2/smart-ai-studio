"""Discover the included original Bonsai 2 verifier without loading a model."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import platform


BUNDLE_FILES = ("llama-server", "llama-server.smartai.json", "Prism-llama-LICENSE")
TARGET_BACKENDS = {"Linux-x86_64": "cpu", "macOS-arm64": "metal"}
INCLUDED_RUNTIME = "Included Bonsai 2 runtime"


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_native_bundle(directory, target, *, code_root=None):
    """Check the static CI artifact against this app's source and platform."""
    if target not in TARGET_BACKENDS:
        raise ValueError("No included original Bonsai 2 runtime for this platform")
    root = Path(directory)
    code = Path(code_root) if code_root is not None else Path(__file__).resolve().parents[1]
    for name in BUNDLE_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file() or not path.stat().st_size:
            raise ValueError("Missing or invalid included native runtime file: " + name)
    receipt_path = root / "llama-server.smartai.json"
    if receipt_path.stat().st_size > 64 * 1024:
        raise ValueError("Included native runtime receipt exceeds the allowed size")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    patcher = code / "core/native_exactness.py"
    constants = {node.targets[0].id: ast.literal_eval(node.value)
                 for node in ast.parse(patcher.read_text(encoding="utf-8")).body
                 if isinstance(node, ast.Assign) and len(node.targets) == 1
                 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in {"CONTRACT", "SOURCE_ARCHIVE_SHA256"}}
    expected = {"schema_version": 1, "backend": TARGET_BACKENDS[target],
                "source_archive_sha256": constants["SOURCE_ARCHIVE_SHA256"],
                "exactness_contract": constants["CONTRACT"],
                "patcher_sha256": _digest(patcher),
                "binary_sha256": _digest(root / "llama-server")}
    if not isinstance(receipt, dict) or any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Included native runtime does not match its receipt, platform or current source")
    with (root / "llama-server").open("rb") as handle:
        header = handle.read(20)
    if target == "Linux-x86_64":
        matching_binary = (header[:6] == b"\x7fELF\x02\x01"
                           and int.from_bytes(header[18:20], "little") == 62)
    else:
        matching_binary = (header[:4] == b"\xcf\xfa\xed\xfe"
                           and int.from_bytes(header[4:8], "little") == 0x0100000C)
    if not matching_binary:
        raise ValueError("Included native executable has the wrong operating system or architecture")
    return receipt


def bundled_runtime_options(code_root=None, *, system=None, machine=None):
    """Return verified selectable paths; never change a model or preference."""
    system = platform.system() if system is None else system
    machine = (platform.machine() if machine is None else machine).lower()
    target = {("Linux", "x86_64"): "Linux-x86_64", ("Linux", "amd64"): "Linux-x86_64",
              ("Darwin", "arm64"): "macOS-arm64", ("Darwin", "aarch64"): "macOS-arm64"}.get((system, machine))
    if target is None:
        return None
    code = Path(code_root) if code_root is not None else Path(__file__).resolve().parents[1]
    root = code / "runtime/native"
    if not root.exists():
        return None
    validate_native_bundle(root, target, code_root=code)
    return {"speculative_server_path": str((root / "llama-server").resolve()),
            "speculative_runtime_receipt_path": str((root / "llama-server.smartai.json").resolve())}


def resolve_runtime_selection(server_path, receipt_path=None, *, code_root=None):
    """Resolve only the explicit included-runtime choice after any app move."""
    if server_path != INCLUDED_RUNTIME:
        return server_path, receipt_path
    if receipt_path:
        raise ValueError("The included runtime uses its own receipt. Clear the custom receipt field.")
    paths = bundled_runtime_options(code_root)
    if paths is None:
        raise ValueError("This application does not include the selected Bonsai 2 runtime for this platform.")
    return paths["speculative_server_path"], paths["speculative_runtime_receipt_path"]
