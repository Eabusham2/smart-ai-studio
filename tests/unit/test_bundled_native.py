"""Verify native packaging and discovery using inert executable-header fixtures."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from tools.bundled_native import embed_native_runtime


ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("bundled_native_test", ROOT / "core/bundled_native.py")
native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(native)


def fixture_bundle(tmp_path, target="Linux-x86_64"):
    code, source = tmp_path / "app", tmp_path / "native-artifact"
    (code / "core").mkdir(parents=True)
    source.mkdir()
    patcher = code / "core/native_exactness.py"
    shutil.copyfile(ROOT / "core/native_exactness.py", patcher)
    header = bytearray(20)
    if target.startswith("Linux"):
        header[:6] = b"\x7fELF\x02\x01"
        header[18:20] = (62).to_bytes(2, "little")
        backend = "cpu"
    else:
        header[:4] = b"\xcf\xfa\xed\xfe"
        header[4:8] = (0x0100000C).to_bytes(4, "little")
        backend = "metal"
    (source / "llama-server").write_bytes(header + b"inert fixture; never executed")
    (source / "Prism-llama-LICENSE").write_text("synthetic license fixture")
    data = {"schema_version": 1, "backend": backend,
            "source_archive_sha256": "8c0f589673b25574f27f013bb3278824384eb35eb984034a2445af3c437b9d05",
            "exactness_contract": "smartai-serial-target-v1",
            "patcher_sha256": hashlib.sha256(patcher.read_bytes()).hexdigest(),
            "binary_sha256": hashlib.sha256((source / "llama-server").read_bytes()).hexdigest()}
    (source / "llama-server.smartai.json").write_text(json.dumps(data))
    return code, source, data


@pytest.mark.parametrize("target,system,machine", [
    ("Linux-x86_64", "Linux", "x86_64"), ("macOS-arm64", "Darwin", "arm64")])
def test_embed_exact_runtime_and_discover_after_app_relocation(tmp_path, monkeypatch, target, system, machine):
    code, source, data = fixture_bundle(tmp_path, target)
    monkeypatch.setenv("SMARTAI_NATIVE_RUNTIME_DIR", str(source))
    destination = embed_native_runtime(code, target)
    assert destination == code / "runtime/native"
    assert (destination / "llama-server").stat().st_mode & 0o111
    for name in native.BUNDLE_FILES:
        assert (source / name).read_bytes() == (destination / name).read_bytes()
    moved = tmp_path / "moved app"
    code.rename(moved)
    paths = native.bundled_runtime_options(moved, system=system, machine=machine)
    assert Path(paths["speculative_server_path"]) == moved / "runtime/native/llama-server"
    assert native.validate_native_bundle(moved / "runtime/native", target, code_root=moved) == data
    monkeypatch.setattr(native.platform, "system", lambda: system)
    monkeypatch.setattr(native.platform, "machine", lambda: machine)
    # Saving the explicit choice does not save the original absolute app path.
    saved = json.loads(json.dumps({"server": native.INCLUDED_RUNTIME, "receipt": None}))
    selected = native.resolve_runtime_selection(saved["server"], saved["receipt"], code_root=moved)
    assert selected == (paths["speculative_server_path"], paths["speculative_runtime_receipt_path"])


@pytest.mark.parametrize("field", ["binary_sha256", "patcher_sha256", "source_archive_sha256", "exactness_contract", "backend"])
def test_mismatched_runtime_never_enters_package(tmp_path, field):
    code, source, data = fixture_bundle(tmp_path)
    data[field] = "mismatch"
    (source / "llama-server.smartai.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="does not match"):
        embed_native_runtime(code, "Linux-x86_64", source=source)
    assert not (code / "runtime/native").exists()


def test_wrong_binary_architecture_rejected_even_with_matching_hash(tmp_path):
    code, source, data = fixture_bundle(tmp_path)
    binary = source / "llama-server"
    header = bytearray(binary.read_bytes())
    header[18:20] = (183).to_bytes(2, "little")  # ELF AArch64 cannot run in x64 package.
    binary.write_bytes(header)
    data["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    (source / "llama-server.smartai.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="architecture"):
        embed_native_runtime(code, "Linux-x86_64", source=source)


def test_supported_build_requires_artifact_and_license(tmp_path, monkeypatch):
    monkeypatch.delenv("SMARTAI_NATIVE_RUNTIME_DIR", raising=False)
    assert embed_native_runtime(tmp_path, "Windows-x86_64") is None
    with pytest.raises(RuntimeError, match="SMARTAI_NATIVE_RUNTIME_DIR"):
        embed_native_runtime(tmp_path, "macOS-arm64")
    code, source, _ = fixture_bundle(tmp_path)
    (source / "Prism-llama-LICENSE").unlink()
    with pytest.raises(ValueError, match="LICENSE"):
        embed_native_runtime(code, "Linux-x86_64", source=source)


def test_discovery_leaves_unsupported_or_unbundled_app_alone(tmp_path):
    assert native.bundled_runtime_options(tmp_path, system="Windows", machine="AMD64") is None
    assert native.bundled_runtime_options(tmp_path, system="Linux", machine="x86_64") is None


def test_discovery_rejects_modified_installed_binary(tmp_path):
    code, source, _ = fixture_bundle(tmp_path)
    destination = embed_native_runtime(code, "Linux-x86_64", source=source)
    (destination / "llama-server").write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="does not match"):
        native.bundled_runtime_options(code, system="Linux", machine="x86_64")


def test_runtime_selection_never_reinterprets_custom_paths_or_receipts(monkeypatch):
    monkeypatch.setattr(native, "bundled_runtime_options", lambda *args: pytest.fail("Custom paths must not resolve an included runtime"))
    assert native.resolve_runtime_selection("/custom/llama-server", "/custom/receipt") == ("/custom/llama-server", "/custom/receipt")
    assert native.resolve_runtime_selection(None) == (None, None)
    with pytest.raises(ValueError, match="custom receipt"):
        native.resolve_runtime_selection(native.INCLUDED_RUNTIME, "/custom/receipt")
