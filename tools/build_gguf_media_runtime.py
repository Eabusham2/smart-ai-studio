"""Build the pinned, model-free sd-cli bundle inside each native package job."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gguf_image_runtime import (GGML_COMMIT, RECEIPT_NAME, SD_CPP_COMMIT, SOURCE_ARCHIVES,
                                    _machine, _sha256, probe_sd_cpp_runtime,
                                    validate_runtime_bundle)


def _download_source(url, destination, expected):
    if destination.is_file():
        if _sha256(destination) != expected:
            raise RuntimeError(f"Cached publisher source failed SHA-256 verification: {destination}")
        return
    partial = destination.with_suffix(".download")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "SmartAIStudio/native-media-builder"})
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 512 * 1024 * 1024:
                    raise RuntimeError("The publisher source archive exceeds its expected size.")
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError("Publisher source archive failed SHA-256 verification.")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def _extract_source(archive, destination):
    """Strip the codeload root safely; example images are not build inputs."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        roots = {PurePosixPath(entry.filename).parts[0] for entry in source.infolist() if entry.filename}
        if len(roots) != 1:
            raise RuntimeError("Unexpected publisher source archive layout.")
        for entry in source.infolist():
            name = PurePosixPath(entry.filename)
            if (name.is_absolute() or ".." in name.parts or "\\" in entry.filename or
                    ":" in entry.filename or stat.S_ISLNK(entry.external_attr >> 16)):
                raise RuntimeError("Unsafe path in publisher source archive.")
            relative = name.parts[1:]
            if not relative or relative[0] in {"assets", ".github"}:
                continue
            target = destination.joinpath(*relative)
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(entry) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)


def _vulkan_sdk_available():
    # This GGML revision uses these Vulkan extension structures without header
    # guards. In particular Ubuntu 22.04's 1.3.204 headers cannot compile it.
    # A shader compiler alone is therefore not evidence of a usable SDK.
    required_types = ("VkPhysicalDevicePipelineRobustnessFeaturesEXT", "VkPhysicalDeviceFaultFeaturesEXT")

    def complete(prefix, shader):
        if not shader:
            return False
        include = next((prefix / name for name in ("Include", "include")
                        if (prefix / name / "vulkan/vulkan_core.h").is_file()), None)
        if include is None or not (include / "vulkan/vulkan.hpp").is_file():
            return False
        header = (include / "vulkan/vulkan_core.h").read_text(encoding="utf-8", errors="replace")
        if not all(name in header for name in required_types):
            return False
        return any((prefix / part / "SPIRV-HeadersConfig.cmake").is_file() for part in (
            "Lib/cmake/SPIRV-Headers", "lib/cmake/SPIRV-Headers",
            "lib64/cmake/SPIRV-Headers", "share/cmake/SPIRV-Headers"))

    sdk = os.getenv("VULKAN_SDK") or os.getenv("VulkanSDK")
    if sdk:
        root = Path(sdk)
        shader = any((root / part).is_file() for part in ("Bin/glslc.exe", "bin/glslc", "bin/glslc.exe"))
        if complete(root, shader):
            return True
    return any(complete(Path(prefix), shutil.which("glslc")) for prefix in ("/usr", "/usr/local"))


def select_backend(requested="auto"):
    if requested == "auto":
        selected = "metal" if platform.system() == "Darwin" else "vulkan" if _vulkan_sdk_available() else "cpu"
        if selected == "cpu":
            print("GGUF media: compatible Vulkan SDK/compiler not available; building the portable CPU backend.", flush=True)
        return selected
    if requested == "metal" and platform.system() != "Darwin":
        raise RuntimeError("The Metal native media backend must be built on macOS.")
    if requested == "vulkan" and not _vulkan_sdk_available():
        raise RuntimeError("A Vulkan SDK with headers and glslc is required for this requested build.")
    if requested not in {"cpu", "metal", "vulkan"}:
        raise ValueError("Choose auto, cpu, metal, or vulkan for the native media backend.")
    return requested


def _configure_command(cmake, source, build, backend):
    shared_backends = platform.system() != "Darwin"
    args = [cmake, "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
            "-DSD_BUILD_EXAMPLES=ON", "-DSD_SERVER_BUILD_FRONTEND=OFF",
            "-DSD_WEBP=OFF", "-DSD_WEBM=OFF", "-DSD_BUILD_SHARED_LIBS=ON",
            "-DGGML_NATIVE=OFF", "-DGGML_OPENMP=OFF",
            "-DSD_BUILD_SHARED_GGML_LIB=" + ("ON" if shared_backends else "OFF"),
            "-DGGML_BACKEND_DL=" + ("ON" if shared_backends else "OFF"),
            "-DGGML_CPU_ALL_VARIANTS=" + ("ON" if shared_backends else "OFF"),
            "-DSD_METAL=" + ("ON" if backend == "metal" else "OFF"),
            "-DSD_VULKAN=" + ("ON" if backend == "vulkan" else "OFF"),
            # Source archives have no .git directory. Supply the exact source
            # identity without accidentally reading a parent checkout's HEAD.
            "-DGIT_EXE=OFF", f"-DSDCPP_BUILD_COMMIT={SD_CPP_COMMIT}",
            "-DSDCPP_BUILD_VERSION=smartai-pinned",
            "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"]
    if platform.system() == "Darwin":
        args.extend(["-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0", "-DCMAKE_OSX_ARCHITECTURES=" + _machine(),
                     "-DGGML_METAL_EMBED_LIBRARY=ON"])
    elif platform.system() == "Windows":
        # A user should not need a separate Visual C++ Redistributable install.
        args.extend(["-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded", "-DSD_MSVC_MP_COUNT=2"])
    else:
        args.extend(["-DCMAKE_INSTALL_RPATH=$ORIGIN", "-DCMAKE_BUILD_RPATH=$ORIGIN",
                     "-DCMAKE_EXE_LINKER_FLAGS=-static-libstdc++ -static-libgcc",
                     "-DCMAKE_SHARED_LINKER_FLAGS=-static-libstdc++ -static-libgcc",
                     "-DCMAKE_MODULE_LINKER_FLAGS=-static-libstdc++ -static-libgcc"])
    return args


def _bundle_build_outputs(source, build, destination, backend):
    executable_name = "sd-cli.exe" if os.name == "nt" else "sd-cli"
    binaries = build / "bin"
    if (binaries / "Release" / executable_name).is_file():
        binaries /= "Release"
    if not (binaries / executable_name).is_file():
        raise RuntimeError("The native build did not produce sd-cli.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gguf-media-stage-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "runtime"
        stage.mkdir()
        for path in binaries.iterdir():
            if path.is_file() and (path.name == executable_name or ".so" in path.name or
                                   path.suffix.lower() in {".dll", ".dylib", ".metal", ".metallib"}):
                # Dereference CMake's relative library symlinks so all three
                # archive extractors receive the same self-contained files.
                shutil.copy2(path, stage / path.name, follow_symlinks=True)
        licenses = {
            "stable-diffusion.cpp-LICENSE.txt": source / "LICENSE",
            "ggml-LICENSE.txt": source / "ggml" / "LICENSE",
            "oniguruma-LICENSE.txt": source / "thirdparty" / "oniguruma" / "COPYING",
            "utf8proc-LICENSE.txt": source / "thirdparty" / "utf8proc" / "LICENSE.md",
        }
        for name, path in licenses.items():
            shutil.copy2(path, stage / name)
        devices = probe_sd_cpp_runtime(stage / executable_name)
        receipt = {
            "schema": 1, "source_commit": SD_CPP_COMMIT, "ggml_commit": GGML_COMMIT,
            "source_archives": [list(item) for item in SOURCE_ARCHIVES],
            "platform": platform.system(), "machine": _machine(), "backend": backend,
            "native_probe": "passed", "devices": devices,
            "files": {path.name: _sha256(path) for path in sorted(stage.iterdir()) if path.is_file()},
        }
        (stage / RECEIPT_NAME).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        validate_runtime_bundle(stage)
        if destination.exists():
            raise RuntimeError(f"Native media output already exists; refusing to overwrite it: {destination}")
        stage.replace(destination)
    print("GGUF_MEDIA_RUNTIME_BUILT", backend, SD_CPP_COMMIT, destination, flush=True)
    return destination / executable_name


def build_runtime(destination, *, backend="auto", work_dir=None, jobs=2, cmake=None):
    destination = Path(destination).absolute()
    selected = select_backend(backend)
    if destination.exists():
        executable, receipt = validate_runtime_bundle(destination)
        if receipt.get("backend") != selected:
            raise RuntimeError("The existing native media bundle uses a different build backend.")
        probe_sd_cpp_runtime(executable)
        return executable
    cmake = cmake or shutil.which("cmake")
    if not cmake:
        raise RuntimeError("Native packaging requires CMake and a C/C++ compiler; users receive the finished runtime.")
    if jobs < 1:
        raise ValueError("Build parallelism must be positive.")

    def compile_in(work):
        work.mkdir(parents=True, exist_ok=True)
        source = work / "source"
        source_marker = source / ".smartai-source.json"
        contract = [list(item) for item in SOURCE_ARCHIVES]
        if source.exists():
            if not source_marker.is_file() or json.loads(source_marker.read_text()) != contract:
                raise RuntimeError("The build source directory contains an unrecognized checkout.")
        else:
            for name, url, digest in SOURCE_ARCHIVES:
                archive = work / (name + "-" + (SD_CPP_COMMIT if name == "stable-diffusion" else GGML_COMMIT) + ".zip")
                _download_source(url, archive, digest)
                _extract_source(archive, source if name == "stable-diffusion" else source / "ggml")
            source_marker.write_text(json.dumps(contract), encoding="utf-8")
        build = work / ("build-" + selected)
        command = _configure_command(str(cmake), source, build, selected)
        print("Configuring pinned native media runtime:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        subprocess.run([str(cmake), "--build", str(build), "--config", "Release", "--target", "sd-cli",
                        "--parallel", str(jobs)], check=True)
        return _bundle_build_outputs(source, build, destination, selected)

    if work_dir is not None:
        return compile_in(Path(work_dir).absolute())
    with tempfile.TemporaryDirectory(prefix="smartai-gguf-build-", dir=os.getenv("RUNNER_TEMP") or None) as temporary:
        return compile_in(Path(temporary))


build_gguf_media_runtime = build_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--backend", choices=("auto", "cpu", "metal", "vulkan"), default="auto")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--cmake")
    args = parser.parse_args()
    build_runtime(args.destination, backend=args.backend, work_dir=args.work_dir,
                  jobs=args.jobs, cmake=args.cmake)


if __name__ == "__main__":
    main()
