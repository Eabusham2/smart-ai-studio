"""Install a pinned Prism llama-server on first use, without compiling or fetching models."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import threading
from urllib.request import Request, urlopen

from config.paths import get_portable_data_dir
from core.speculative_runtime import sha256_file

RELEASE = "prism-b12220-41c86f7"
RELEASE_URL = f"https://github.com/PrismML-Eng/llama.cpp/releases/download/{RELEASE}"
ARTIFACTS = {
    "bin-linux-cuda-12.4-x64.tar.gz": {
        "file": "llama-prism-b12220-41c86f7-bin-linux-cuda-12.4-x64.tar.gz",
        "size": 361385770,
        "sha256": "cc59206cea97c807684cda94f8001af0ad697e35f0e4e1485e754c2b00516069"
    },
    "bin-macos-arm64.tar.gz": {
        "file": "llama-prism-b12220-41c86f7-bin-macos-arm64.tar.gz",
        "size": 13216851,
        "sha256": "0281adb71a1d4000181aa3e9a0acbfec4189b1dc67dfb8091847b1c42aa0b10d"
    },
    "bin-macos-x64.tar.gz": {
        "file": "llama-prism-b12220-41c86f7-bin-macos-x64.tar.gz",
        "size": 12636048,
        "sha256": "97f2f1b4237d02c02602102e01a2acfed8d5347c4df34e255ec0857380d11275"
    },
    "bin-ubuntu-arm64.tar.gz": {
        "file": "llama-prism-b12220-41c86f7-bin-ubuntu-arm64.tar.gz",
        "size": 15093542,
        "sha256": "97d1b5f06b87f8c4ffaf9a0b69acf9bb46c49b4f5912df644629999d0eb33716"
    },
    "bin-ubuntu-rocm-7.2-x64.tar.gz": {
        "file": "llama-prism-b12220-41c86f7-bin-ubuntu-rocm-7.2-x64.tar.gz",
        "size": 206404615,
        "sha256": "f857f0aa96e9651fc97c2c2b68847ffbeb864b393345f170be25298d702123f9"
    },
    "bin-ubuntu-vulkan-arm64.tar.gz": {
        "file": "llama-prism-b12220-41c86f7-bin-ubuntu-vulkan-arm64.tar.gz",
        "size": 26238469,
        "sha256": "006d0e79974162e1f52c4b62078ab37afbcb3f692e79cace0a9df08eb2e37f20"
    },
    "bin-ubuntu-vulkan-x64.tar.gz": {
        "file": "llama-prism-b12220-41c86f7-bin-ubuntu-vulkan-x64.tar.gz",
        "size": 34014604,
        "sha256": "63c4f59cd1c6055f49e58488b1d89b5ac0a6f4d9a3cbfa17597d9e6c0a13714b"
    },
    "bin-ubuntu-x64.tar.gz": {
        "file": "llama-prism-b12220-41c86f7-bin-ubuntu-x64.tar.gz",
        "size": 19834862,
        "sha256": "daa1b6e0ddc169fed0dcccdde7499ef5e10f0533b76554d6e26e3bb2db5a7a9b"
    },
    "bin-win-cpu-arm64.zip": {
        "file": "llama-prism-b12220-41c86f7-bin-win-cpu-arm64.zip",
        "size": 13517085,
        "sha256": "803dc50c193c66b0da514219f0d08e4ff3eb35b9ecab0f2685bc7d657bd73c90"
    },
    "bin-win-cpu-x64.zip": {
        "file": "llama-prism-b12220-41c86f7-bin-win-cpu-x64.zip",
        "size": 21709815,
        "sha256": "babbc9ea87e788373d293704582d8360e5f9167e16324cb0bc85c946c58eeb4d"
    },
    "bin-win-cuda-12.4-x64.zip": {
        "file": "llama-prism-b12220-41c86f7-bin-win-cuda-12.4-x64.zip",
        "size": 345013520,
        "sha256": "96d851b80992df6edf72e41e59b117529360eba4365c5665625587b68cb4de56"
    },
    "bin-win-hip-radeon-x64.zip": {
        "file": "llama-prism-b12220-41c86f7-bin-win-hip-radeon-x64.zip",
        "size": 387894326,
        "sha256": "8ae03cf6a02b0966300e3475ade521a5fa06a74a87bfa2156469e01e64c2aa19"
    },
    "bin-win-vulkan-x64.zip": {
        "file": "llama-prism-b12220-41c86f7-bin-win-vulkan-x64.zip",
        "size": 28904089,
        "sha256": "fe9bb194abcd7ac9cbaed7e2c35154fa481af5fedd022ecfbf23fc6bb1291009"
    },
    "cudart-win-cuda": {
        "file": "cudart-llama-bin-win-cuda-12.4-x64.zip",
        "size": 391443627,
        "sha256": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"
    }
}
_INSTALL_LOCK = threading.Lock()


def runtime_variant():
    """Use a matching published binary; an explicit variant never silently falls back."""
    machine = platform.machine().lower()
    arch = "arm64" if machine in {"aarch64", "arm64"} else "x64" if machine in {"x86_64", "amd64"} else None
    if not arch:
        raise RuntimeError("No pinned Bonsai runtime is available for this processor")
    system = platform.system()
    variant = os.getenv("SMARTAI_BONSAI_NATIVE_VARIANT", "auto").strip().lower()
    if variant not in {"auto", "cpu", "cuda", "vulkan", "rocm"}:
        raise ValueError("SMARTAI_BONSAI_NATIVE_VARIANT must be auto/cpu/cuda/vulkan/rocm")
    if system == "Darwin":
        return f"bin-macos-{arch}.tar.gz"
    if variant == "auto":
        variant = "cuda" if arch == "x64" and shutil.which("nvidia-smi") else "cpu"
    if system == "Windows":
        kind = "hip-radeon" if variant == "rocm" else "cuda-12.4" if variant == "cuda" else variant
        selected = f"bin-win-{kind}-{arch}.zip"
    elif system == "Linux":
        kind = "rocm-7.2-" if variant == "rocm" else "vulkan-" if variant == "vulkan" else ""
        selected = f"bin-linux-cuda-12.4-{arch}.tar.gz" if variant == "cuda" else f"bin-ubuntu-{kind}{arch}.tar.gz"
    else:
        raise RuntimeError("No pinned Bonsai runtime is available for this operating system")
    if selected not in ARTIFACTS:
        raise RuntimeError(f"The pinned Bonsai runtime has no {variant} build for {system} {arch}; select CPU or a compatible executable")
    return selected


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise RuntimeError("Bonsai runtime installation cancelled")


def _download(artifact, path, cancelled):
    digest = hashlib.sha256()
    size = 0
    request = Request(RELEASE_URL + "/" + artifact["file"], headers={"User-Agent": "Smart-AI-Studio"})
    with urlopen(request, timeout=30) as source, path.open("xb") as output:
        while chunk := source.read(1024 * 1024):
            _check_cancelled(cancelled)
            size += len(chunk)
            if size > artifact["size"]:
                raise ValueError("Bonsai runtime download exceeds its pinned size")
            digest.update(chunk)
            output.write(chunk)
    if size != artifact["size"] or digest.hexdigest() != artifact["sha256"]:
        raise ValueError("Bonsai runtime download checksum mismatch")


def ensure_bonsai_server(cancelled=None):
    """Return a verified cached executable, installing a publisher release if missing."""
    explicit = os.getenv("SMARTAI_BONSAI_SERVER", "").strip()
    if explicit:
        server = Path(explicit).expanduser().resolve(strict=True)
        if not server.is_file():
            raise ValueError("SMARTAI_BONSAI_SERVER must name a llama-server executable")
        # The backend checks this explicitly selected binary's supported flags.
        return str(server)
    key = runtime_variant()
    artifact = ARTIFACTS[key]
    base = Path(get_portable_data_dir()) / "bonsai_native" / RELEASE
    destination = base / key.replace(".tar.gz", "").replace(".zip", "")
    receipt_path = destination / "studio-runtime.json"
    with _INSTALL_LOCK:
        _check_cancelled(cancelled)
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            server = destination / receipt.get("executable", "")
            if (receipt.get("archive_sha256") == artifact["sha256"] and server.is_file()
                    and server.resolve().is_relative_to(destination.resolve())
                    and sha256_file(str(server), cancelled) == receipt.get("binary_sha256")):
                return str(server)
            raise RuntimeError("Cached Bonsai runtime verification failed; remove its runtime cache and reload")
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".install-", dir=base) as temporary:
            stage = Path(temporary)
            archive = stage / artifact["file"]
            _download(artifact, archive, cancelled)
            content = stage / "runtime"
            # Reuse Studio's already-tested bounded, path-checked native archive
            # extractor (regular files first; internal links checked afterwards).
            from core.strata_runtime import extract_terminal
            extract_terminal(archive, content, lambda: _check_cancelled(cancelled))
            filename = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
            candidates = list(content.rglob(filename))
            if len(candidates) != 1:
                raise RuntimeError("Pinned Bonsai archive did not contain one llama-server")
            server = candidates[0]
            if key == "bin-win-cuda-12.4-x64.zip":
                runtime = ARTIFACTS["cudart-win-cuda"]
                cuda_archive = stage / runtime["file"]
                _download(runtime, cuda_archive, cancelled)
                cuda_root = stage / "cuda"
                extract_terminal(cuda_archive, cuda_root, lambda: _check_cancelled(cancelled))
                for library in cuda_root.rglob("*.dll"):
                    target = server.parent / library.name
                    if not target.exists():
                        shutil.copy2(library, target)
            if os.name != "nt":
                server.chmod(server.stat().st_mode | 0o700)
            receipt = {"release": RELEASE, "archive_sha256": artifact["sha256"],
                       "executable": str(server.relative_to(content)),
                       "binary_sha256": sha256_file(str(server), cancelled)}
            (content / "studio-runtime.json").write_text(json.dumps(receipt), encoding="utf-8")
            _check_cancelled(cancelled)
            if destination.exists():
                raise RuntimeError("Bonsai runtime destination already exists without a verified receipt")
            os.replace(content, destination)
        return str(destination / receipt["executable"])
