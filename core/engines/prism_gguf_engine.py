"""PrismML-only GGUF runtime.

This backend is intentionally separate from the universal GGUF backend.  It launches
llama-cpp-python in an isolated worker process with LLAMA_CPP_LIB_PATH / MTMD_CPP_LIB
pointing at a build of PrismML-Eng/llama.cpp, so Prism-private PTQ1_0 kernels never
replace or contaminate the normal process-wide llama.cpp runtime.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from config.paths import get_portable_data_dir
from core.engines.gguf_engine import GGUFReasoningBackend


PRISM_REPO = "https://github.com/PrismML-Eng/llama.cpp.git"
PRISM_REF = os.getenv("SMARTAI_PRISM_LLAMA_REF", "prism").strip() or "prism"


def _lib_names(base: str) -> Tuple[str, ...]:
    if os.name == "nt":
        return (f"{base}.dll", f"lib{base}.dll")
    if platform.system() == "Darwin":
        return (f"lib{base}.dylib", f"lib{base}.so")
    return (f"lib{base}.so",)


def _find_library_dir(root: Path, base: str) -> Optional[Path]:
    names = set(_lib_names(base))
    for path in root.rglob("*"):
        if path.is_file() and path.name in names:
            return path.parent
    return None


def ensure_prism_native_runtime() -> Tuple[str, str]:
    """Return directories containing Prism libllama and libmtmd, building once if needed."""
    explicit_llama = os.getenv("SMARTAI_PRISM_LLAMA_LIB_PATH", "").strip()
    explicit_mtmd = os.getenv("SMARTAI_PRISM_MTMD_LIB_PATH", "").strip()
    if explicit_llama and explicit_mtmd:
        return explicit_llama, explicit_mtmd

    base = Path(get_portable_data_dir()) / "prism_llama_runtime"
    source = Path(os.getenv("PRISM_LLAMA_CPP_DIR", "").strip() or (base / "llama.cpp"))
    build = base / "build"
    base.mkdir(parents=True, exist_ok=True)

    llama_dir = _find_library_dir(build, "llama") if build.exists() else None
    mtmd_dir = _find_library_dir(build, "mtmd") if build.exists() else None
    if llama_dir and mtmd_dir:
        return str(llama_dir), str(mtmd_dir)

    git = shutil.which("git")
    cmake = shutil.which("cmake")
    if not git or not cmake:
        raise RuntimeError(
            "Prism Bonsai GGUF requires the PrismML llama.cpp native runtime. "
            "Install git + cmake, or set SMARTAI_PRISM_LLAMA_LIB_PATH and "
            "SMARTAI_PRISM_MTMD_LIB_PATH to an existing Prism build."
        )

    if not (source / ".git").exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [git, "clone", "--depth", "1", "-b", PRISM_REF, PRISM_REPO, str(source)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Could not clone PrismML llama.cpp: "
                + (proc.stderr or proc.stdout or "git clone failed").strip()
            )

    configure = [
        cmake, "-S", str(source), "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=ON",
        "-DLLAMA_BUILD_TOOLS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_MTMD=ON",
    ]
    if shutil.which("nvcc"):
        configure.append("-DGGML_CUDA=ON")
    elif shutil.which("hipcc") and platform.system() != "Darwin":
        configure.append("-DGGML_HIP=ON")
    elif platform.system() == "Darwin":
        configure.append("-DGGML_METAL=ON")

    proc = subprocess.run(configure, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Could not configure PrismML llama.cpp: "
            + (proc.stderr or proc.stdout or "cmake configure failed").strip()
        )

    command = [cmake, "--build", str(build), "--target", "llama", "mtmd", "--parallel"]
    if os.name == "nt":
        command += ["--config", "Release"]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Could not build PrismML libllama/libmtmd: "
            + (proc.stderr or proc.stdout or "cmake build failed").strip()
        )

    llama_dir = _find_library_dir(build, "llama")
    mtmd_dir = _find_library_dir(build, "mtmd")
    if not llama_dir or not mtmd_dir:
        raise RuntimeError("PrismML build completed but libllama/libmtmd could not be located")
    return str(llama_dir), str(mtmd_dir)


class _PrismModelProxy:
    def __init__(self, backend: "PrismGGUFReasoningBackend"):
        self.backend = backend

    def tokenize(self, raw: bytes):
        text = raw.decode("utf-8", errors="replace")
        result = self.backend._call({"op": "tokenize", "text": text})
        return list(result.get("tokens") or [])

    def n_ctx(self) -> int:
        result = self.backend._call({"op": "n_ctx"})
        return int(result.get("n_ctx") or self.backend.n_ctx)


class PrismGGUFReasoningBackend(GGUFReasoningBackend):
    """Prism PTQ1_0 backend isolated from the universal llama.cpp runtime."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._worker: Optional[subprocess.Popen] = None
        self._io_lock = threading.RLock()
        self._worker_error_log = os.path.join(self.adapter_root, "prism-worker.stderr.log")

    def _worker_env(self) -> Dict[str, str]:
        llama_dir, mtmd_dir = ensure_prism_native_runtime()
        env = dict(os.environ)
        env["LLAMA_CPP_LIB_PATH"] = llama_dir
        env["MTMD_CPP_LIB"] = mtmd_dir
        lib_dirs = [llama_dir, mtmd_dir]
        if os.name == "nt":
            env["PATH"] = os.pathsep.join(lib_dirs + [env.get("PATH", "")])
        elif platform.system() == "Darwin":
            env["DYLD_LIBRARY_PATH"] = os.pathsep.join(lib_dirs + [env.get("DYLD_LIBRARY_PATH", "")])
        else:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_dirs + [env.get("LD_LIBRARY_PATH", "")])
        return env

    def _call(self, payload: Dict[str, Any], *, progress=None) -> Dict[str, Any]:
        with self._io_lock:
            if self._worker is None or self._worker.poll() is not None:
                raise RuntimeError("Prism GGUF worker is not running")
            assert self._worker.stdin is not None and self._worker.stdout is not None
            self._worker.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._worker.stdin.flush()
            while True:
                line = self._worker.stdout.readline()
                if not line:
                    raise RuntimeError("Prism GGUF worker exited without a response")
                result = json.loads(line)
                if result.get("event") == "speed_progress":
                    if progress is not None:
                        try:
                            progress(str(result.get("message") or ""))
                        except Exception:
                            pass
                    continue
                if result.get("status") == "error":
                    raise RuntimeError(str(result.get("error") or "Prism worker error"))
                return result

    def configure_auto_speed(self, enabled=True, force=False, progress=None, cancelled=None) -> Dict[str, Any]:
        """Keep tuning and private-kernel measurements inside the Prism worker."""
        if self.model is None:
            return {"status": "unsupported", "settings": {},
                    "message": "Load a Prism model before tuning its speed."}
        # The worker cannot read another stdin command while a native request is
        # active. A private marker conveys cancellation at token boundaries.
        with tempfile.TemporaryDirectory(prefix="smartai-prism-tune-") as directory:
            marker = Path(directory) / "cancel"
            stopped = threading.Event()

            def cancellation_requested():
                try:
                    return cancelled is not None and cancelled()
                except Exception:
                    return True

            # A cached profile can apply before the OS schedules the watcher.
            # Carry an existing cancellation into the worker's first check.
            if cancellation_requested():
                marker.touch()

            def watch_cancel():
                while not stopped.is_set():
                    if cancellation_requested():
                        marker.touch()
                        return
                    stopped.wait(0.05)

            watcher = threading.Thread(target=watch_cancel, name="prism-speed-cancel", daemon=True)
            watcher.start()
            try:
                response = self._call({
                    "op": "tune_speed", "enabled": bool(enabled), "force": bool(force),
                    "cancel_path": str(marker),
                    "max_process_gb": getattr(self, "speed_memory_limit_gb", None),
                    "memory_owner_pid": os.getpid(),
                }, progress=progress)
                self.auto_speed_result = dict(response.get("result") or {})
            except Exception as exc:
                self.auto_speed_result = {"status": "error", "settings": {},
                                          "message": f"Prism speed tuning failed: {str(exc)[:250]}"}
            finally:
                stopped.set()
                watcher.join(timeout=1.0)
        return dict(self.auto_speed_result)

    def load_model(self) -> bool:
        if not os.path.isfile(self.model_path):
            self.last_error = "The selected Prism GGUF file does not exist."
            return False
        self.unload_model()
        self.last_error = ""
        try:
            from core.gguf_metadata import inspect_gguf
            metadata = inspect_gguf(self.model_path)
            self.model_path = metadata["model_path"]
        except (OSError, ValueError) as exc:
            self.last_error = str(exc)
            return False
        os.makedirs(self.adapter_root, exist_ok=True)
        config = {
            "model_path": self.model_path,
            "mmproj_path": self.mmproj_path,
            "adapter_path": self.adapter_path if os.path.isfile(self.adapter_path) else None,
            "n_gpu_layers": self.n_gpu_layers,
            "n_ctx": self.n_ctx,
            "verbose": self.verbose,
            "n_threads": self.n_threads,
            "n_threads_batch": self.n_threads_batch,
            "auto_memory_placement": self.auto_memory_placement,
            "memory_limit_gb": self.memory_limit_gb,
            "memory_owner_pid": os.getpid(),
        }
        config_path = os.path.join(self.adapter_root, "prism-worker-config.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle)

        err = open(self._worker_error_log, "a", encoding="utf-8")
        error_start = err.tell()
        try:
            self._worker = subprocess.Popen(
                [sys.executable, "-m", "core.engines.prism_gguf_worker", config_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=err,
                text=True,
                bufsize=1,
                env=self._worker_env(),
            )
        finally:
            err.close()

        deadline = time.time() + 90.0
        last_error = None
        while time.time() < deadline:
            if self._worker.poll() is not None:
                break
            try:
                ready = self._call({"op": "ping"})
                self.memory_placement = dict(ready.get("memory_placement") or {})
                if "n_gpu_layers" in self.memory_placement:
                    self.n_gpu_layers = int(self.memory_placement["n_gpu_layers"])
                self.model = _PrismModelProxy(self)
                self.chat_handler = object() if self.mmproj_path else None
                self.is_gguf_available = True
                return True
            except Exception as exc:
                last_error = exc
                time.sleep(0.25)

        self.unload_model()
        detail = str(last_error or "worker exited")
        try:
            # Keep the native encoding/allocation error visible without reading
            # an unbounded log or reporting an unrelated previous launch.
            with open(self._worker_error_log, "rb") as log:
                log.seek(0, os.SEEK_END)
                log.seek(max(error_start, log.tell() - 4096))
                native_error = log.read(4096).decode("utf-8", errors="replace").strip()
            if native_error:
                detail += ": " + native_error
        except OSError:
            pass
        self.last_error = f"Prism GGUF worker failed to start: {detail}"
        raise RuntimeError(self.last_error)

    def generate_branches(
        self,
        prompt: str,
        branch_count: int = 1,
        max_tokens: int = 1536,
        temperature: Any = 0.75,
        top_p: float = 0.92,
        image_bytes: Optional[bytes] = None,
    ) -> List[str]:
        del image_bytes
        out = []
        for idx in range(max(1, int(branch_count))):
            temp = float(temperature[idx % len(temperature)]) if isinstance(temperature, (list, tuple)) else float(temperature)
            result = self._call({
                "op": "generate",
                "prompt": prompt,
                "max_tokens": int(max_tokens),
                "temperature": temp,
                "top_p": float(top_p),
            })
            out.append(str(result.get("text") or ""))
        return out

    def stream_generate_tokens(
        self,
        prompt: str,
        max_tokens: int = 1536,
        temperature: float = 0.75,
        top_p: float = 0.92,
        image_bytes: Optional[bytes] = None,
    ) -> Generator[str, None, None]:
        del image_bytes
        with self._io_lock:
            if self._worker is None or self._worker.poll() is not None:
                return
            assert self._worker.stdin is not None and self._worker.stdout is not None
            self._worker.stdin.write(json.dumps({
                "op": "stream",
                "prompt": prompt,
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
            }) + "\n")
            self._worker.stdin.flush()
            while True:
                line = self._worker.stdout.readline()
                if not line:
                    break
                event = json.loads(line)
                if event.get("event") == "token":
                    token = str(event.get("text") or "")
                    if token:
                        yield token
                elif event.get("event") == "done":
                    break
                elif event.get("status") == "error":
                    raise RuntimeError(str(event.get("error") or "Prism stream failed"))

    def calculate_token_entropy(self, prompt: str) -> float:
        try:
            result = self._call({"op": "entropy", "prompt": prompt})
            return float(result.get("entropy"))
        except Exception:
            return 0.35

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        text = "\n".join(str(m.get("content", "")) for m in messages or [])
        try:
            result = self._call({"op": "tokenize", "text": text})
            return len(result.get("tokens") or [])
        except Exception:
            return max(1, len(text) // 4)

    def supports_media_input(self, kind: str) -> bool:
        return str(kind or "").lower() in {"image", "video"} and bool(self.mmproj_path) and self.model is not None

    def review_media_input(self, path: str, kind: str, prompt: str = "") -> Dict[str, Any]:
        return self._call({
            "op": "review_media",
            "path": os.path.abspath(path),
            "kind": str(kind or "").lower(),
            "prompt": str(prompt or ""),
        })

    def unload_model(self):
        worker = getattr(self, "_worker", None)
        if worker is not None:
            try:
                if worker.poll() is None and worker.stdin is not None:
                    worker.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                    worker.stdin.flush()
                    worker.wait(timeout=3.0)
            except Exception:
                try:
                    worker.terminate()
                    worker.wait(timeout=2.0)
                except Exception:
                    try:
                        worker.kill()
                    except Exception:
                        pass
        self._worker = None
        self._speed_tuner = None
        self.model = None
        self.chat_handler = None
