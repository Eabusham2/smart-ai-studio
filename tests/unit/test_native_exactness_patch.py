"""Compile the row policy and exercise source/archive guards without a model."""
from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

from core import native_exactness as exact


def build_tool():
    path = Path(__file__).resolve().parents[2] / "tools/prepare_speculative_runtime.py"
    spec = importlib.util.spec_from_file_location("_test_native_build_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_row_policy_compiles_and_keeps_prefill_and_drafter_batching(tmp_path):
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Standalone C++ compiler unavailable; native-build CI compiles the full runtime")
    (tmp_path / "smartai-target-microbatch.h").write_text(exact.MICROBATCH_HEADER)
    source = tmp_path / "contract.cpp"
    source.write_text('''#include "smartai-target-microbatch.h"
// These are execution contract examples, including the separate noncausal
// drafter, single-token decode, mixed prefill and a multi-token verifier.
static_assert(smartai_target_microbatch(false, 4, 4, 512) == 512);
static_assert(smartai_target_microbatch(true, 400, 1, 512) == 512);
static_assert(smartai_target_microbatch(true, 512, 0, 512) == 512);
static_assert(smartai_target_microbatch(true, 4, 4, 512) == 1);
static_assert(smartai_target_microbatch(true, 1, 1, 512) == 1);
static_assert(smartai_target_microbatch(true, 0, 0, 512) == 512);
static_assert(smartai_target_microbatch(true, 400, 1, 128) == 128);
int main() { return 0; }
''')
    output = tmp_path / "contract-test"
    result = subprocess.run([compiler, "-std=c++11", str(source), "-o", str(output)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    # Executes only the tiny integer contract above; no inference library.
    subprocess.run([str(output)], check=True, timeout=10)


def test_wrong_source_preimage_cannot_partially_patch(tmp_path, monkeypatch):
    first = tmp_path / "first.cpp"
    first.write_text("reviewed first source\n")
    second = tmp_path / "second.cpp"
    second.write_text("unexpected replacement\n")
    original = first.read_bytes()
    monkeypatch.setattr(exact, "SOURCE_SHA256", {
        "first.cpp": hashlib.sha256(original).hexdigest(),
        "second.cpp": "0" * 64,
    })
    with pytest.raises(ValueError, match="Native source mismatch"):
        exact.apply_native_patch(tmp_path)
    assert first.read_bytes() == original
    assert not (tmp_path / "src/smartai-target-microbatch.h").exists()


def test_nonunique_patch_anchor_is_rejected():
    with pytest.raises(ValueError, match="exactly once"):
        exact._replace_once("anchor anchor", "anchor", "change")
    with pytest.raises(ValueError, match="exactly once"):
        exact._replace_once("missing", "anchor", "change")


def archive_fixture(tmp_path, *, name="llama/source.cpp", kind=None):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        item = tarfile.TarInfo(name)
        if kind is not None:
            item.type = kind
            item.linkname = "../../escape"
            handle.addfile(item)
        else:
            data = b"reviewed source fixture"
            item.size = len(data)
            handle.addfile(item, io.BytesIO(data))
    return archive


def test_unpinned_archive_is_refused_before_extraction(tmp_path):
    tool = build_tool()
    archive = archive_fixture(tmp_path)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="SHA256"):
        tool.extract_pinned_source(archive, output)
    assert not output.exists()


@pytest.mark.parametrize("name,kind", [
    ("../escape", None), ("/absolute", None), ("llama/../../escape", None),
    ("llama\\escape", None), ("C:/escape", None),
    ("llama/link", tarfile.SYMTYPE), ("llama/link", tarfile.LNKTYPE),
])
def test_archive_guards_validate_all_members_before_any_write(tmp_path, monkeypatch, name, kind):
    tool = build_tool()
    archive = archive_fixture(tmp_path, name=name, kind=kind)
    monkeypatch.setattr(tool, "SOURCE_ARCHIVE_SHA256", exact.sha256_file(archive))
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="Unsafe path|link or special"):
        tool.extract_pinned_source(archive, output)
    assert not output.exists()


def test_source_tool_never_overwrites_an_existing_build(tmp_path, monkeypatch):
    tool = build_tool()
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "user-owned"
    marker.write_bytes(b"keep")
    monkeypatch.setattr(tool, "extract_pinned_source", lambda *args: pytest.fail("Must reject before extraction"))
    with pytest.raises(SystemExit):
        tool.main(["--source-archive", str(tmp_path / "missing"), "--output-dir", str(output)])
    assert marker.read_bytes() == b"keep"


def test_native_runtime_is_included_before_app_packaging():
    # Tested native artifacts are embedded in both package flavors. The publisher
    # must never repack them separately or run a builder that clears dist.
    source = (Path(__file__).resolve().parents[2] / ".github/workflows/ci-build-release.yml").read_text(encoding="utf-8")
    package = source.split("  package:", 1)[1].split("  build-and-release:", 1)[0]
    assert package.index("Fetch matching verified native runtime for the app") < package.index("Build bundled runtime and test relocated first launch")
    publisher = source.split("  build-and-release:", 1)[1]
    assert "python build_app.py" not in publisher
    assert "needs: [preflight, test, mlx-learning, native-build, reuse-native, package]" in source
    assert "needs.native-build.result == 'success'" in source
