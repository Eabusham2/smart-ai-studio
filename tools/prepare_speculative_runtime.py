"""Build the pinned exact-verification runtime without downloading/running a model.

Supply ProCreations' pinned source archive locally. This program downloads
nothing, executes only the selected compiler/build tools, and never launches
the resulting model server. Receipts are produced only after a successful build.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep this source/build utility independent of core's neural-runtime imports.
_spec = importlib.util.spec_from_file_location("_smartai_native_exactness", ROOT / "core" / "native_exactness.py")
_native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_native)
CONTRACT = _native.CONTRACT
SOURCE_ARCHIVE_SHA256 = _native.SOURCE_ARCHIVE_SHA256
apply_native_patch = _native.apply_native_patch
sha256_file = _native.sha256_file


def extract_pinned_source(archive: Path, destination: Path) -> Path:
    if sha256_file(archive) != SOURCE_ARCHIVE_SHA256:
        raise ValueError("Source archive SHA256 does not match the reviewed Bonsai runtime")
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        roots = set()
        total = 0
        for item in members:
            relative = PurePosixPath(item.name)
            if (relative.is_absolute() or ".." in relative.parts or "\\" in item.name
                    or not relative.parts or ":" in item.name):
                raise ValueError("Unsafe path in native source archive")
            if not (item.isdir() or item.isfile()):
                raise ValueError("Native source archive contains a link or special file")
            roots.add(relative.parts[0])
            total += item.size
        if len(roots) != 1 or total > 1024 ** 3 or len(members) > 50_000:
            raise ValueError("Native source archive layout/size is unexpected")
        for item in members:
            target = destination.joinpath(*PurePosixPath(item.name).parts)
            if item.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = bundle.extractfile(item)
                if stream is None:
                    raise ValueError("Unreadable native source archive member")
                with stream, target.open("xb") as output:
                    shutil.copyfileobj(stream, output)
                target.chmod(0o755 if item.mode & 0o111 else 0o644)
    return destination / roots.pop()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda", "hip"), default="cpu")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--patch-only", action="store_true", help="Validate and patch source; do not build or create a runtime receipt")
    args = parser.parse_args(argv)
    if not 1 <= args.jobs <= 64:
        parser.error("--jobs must be between 1 and 64")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        parser.error("--output-dir must be a new directory; existing builds are never overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage alongside the destination; publish only a complete patch/build tree.
    with tempfile.TemporaryDirectory(prefix=".smartai-native-", dir=output.parent) as temporary:
        stage = Path(temporary)
        source = extract_pinned_source(args.source_archive, stage)
        patched = apply_native_patch(source)
        build = source / "build-smartai"
        if not args.patch_only:
            cmake = shutil.which("cmake")
            if not cmake:
                raise RuntimeError("CMake is required to build the native runtime")
            command = [cmake, "-S", str(source), "-B", str(build),
                       "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF",
                       "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_OPENSSL=OFF",
                       "-DGGML_OPENMP=OFF",
                       "-DGGML_NATIVE=OFF", "-DLLAMA_BUILD_SERVER=ON",
                       "-DGGML_METAL=" + ("ON" if args.backend == "metal" else "OFF"),
                       "-DGGML_CUDA=" + ("ON" if args.backend == "cuda" else "OFF"),
                       "-DGGML_HIP=" + ("ON" if args.backend == "hip" else "OFF")]
            subprocess.run(command, check=True)
            subprocess.run([cmake, "--build", str(build), "--config", "Release",
                            "--target", "llama-server", "--parallel", str(args.jobs)], check=True)
            filename = "llama-server.exe" if os.name == "nt" else "llama-server"
            candidates = [p for p in build.rglob(filename) if p.is_file()]
            if len(candidates) != 1:
                raise RuntimeError("Build did not produce exactly one native server")
            binary = candidates[0]
            receipt = {"schema_version": 1, "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                       "exactness_contract": CONTRACT, "binary_sha256": sha256_file(binary),
                       "patcher_sha256": sha256_file(ROOT / "core" / "native_exactness.py"),
                       "backend": args.backend, "patched_source_sha256": patched,
                       "validation": "Compiled only; no model inference or throughput benchmark"}
            Path(str(binary) + ".smartai.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Relocation is safe for the static server and its sidecar receipt.
        shutil.move(str(source), str(output))
    print(json.dumps({"status": "patched" if args.patch_only else "built", "path": str(output),
                      "exactness_contract": CONTRACT, "model_execution": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
