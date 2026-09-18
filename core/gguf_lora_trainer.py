"""Exact-GGUF LoRA training bridge for Smart AI Studio.

The base GGUF stays frozen. Learning is performed by llama.cpp's real
llama-finetune-lora tool directly against the exact currently selected GGUF, so
custom quantization/rotation semantics remain owned by the matching llama.cpp
runtime. The resulting GGUF LoRA adapter is then hot-loaded by llama-cpp-python.
"""
from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _exe_name(name: str) -> str:
    return name + (".exe" if os.name == "nt" else "")


class GGUFLoRATrainer:
    def __init__(
        self,
        *,
        model_path: str,
        adapter_root: str,
        rank: int = 8,
        alpha: int = 16,
    ):
        self.model_path = os.path.abspath(str(model_path))
        self.adapter_root = os.path.abspath(str(adapter_root))
        self.adapter_path = os.path.join(self.adapter_root, "adapter.gguf")
        self.rank = int(rank)
        self.alpha = int(alpha)
        self._tool_root: Optional[Path] = None

    def _candidate_binaries(self) -> List[Path]:
        name = _exe_name("llama-finetune-lora")
        out: List[Path] = []

        for env_name in ("SMARTAI_GGUF_FINETUNE_BIN", "LLAMA_FINETUNE_LORA"):
            value = os.getenv(env_name, "").strip()
            if value:
                out.append(Path(value))

        found = shutil.which("llama-finetune-lora")
        if found:
            out.append(Path(found))

        roots = [
            Path.cwd(),
            Path.cwd().parent,
            Path(self.adapter_root).parent,
            Path(__file__).resolve().parents[2],
        ]
        explicit = os.getenv("PRISM_LLAMA_CPP_DIR", "").strip()
        if explicit:
            roots.insert(0, Path(explicit))

        relative = [
            Path("llama.cpp/build/bin") / name,
            Path("llama.cpp/build/bin/Release") / name,
            Path("build/bin") / name,
            Path("build/bin/Release") / name,
            Path("bin/cuda") / name,
            Path("bin/vulkan") / name,
            Path("bin/rocm") / name,
            Path("bin/hip") / name,
            Path("bin/cpu") / name,
            Path("bin/mac") / name,
            Path("Bonsai-demo/bin/cuda") / name,
            Path("Bonsai-demo/bin/vulkan") / name,
            Path("Bonsai-demo/bin/rocm") / name,
            Path("Bonsai-demo/bin/hip") / name,
            Path("Bonsai-demo/bin/cpu") / name,
            Path("Bonsai-demo/bin/mac") / name,
        ]
        for root in roots:
            for rel in relative:
                out.append(root / rel)

        dedup: List[Path] = []
        seen = set()
        for candidate in out:
            try:
                key = str(candidate.resolve())
            except Exception:
                key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(candidate)
        return dedup

    def _build_prism_trainer(self) -> Path:
        git = shutil.which("git")
        cmake = shutil.which("cmake")
        if not git or not cmake:
            raise RuntimeError(
                "GGUF parameter learning needs llama-finetune-lora. "
                "Install the Prism Bonsai llama.cpp tools, or install git + cmake "
                "so Smart AI Studio can build the training target."
            )

        tool_root = Path(self.adapter_root).parent / "gguf_train_tools"
        source = tool_root / "llama.cpp"
        build = source / "build-smartai-train"
        tool_root.mkdir(parents=True, exist_ok=True)

        if not (source / ".git").exists():
            proc = subprocess.run(
                [
                    git, "clone", "--depth", "1", "-b", "prism",
                    "https://github.com/PrismML-Eng/llama.cpp.git",
                    str(source),
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "Could not clone the Prism llama.cpp training runtime: "
                    + (proc.stderr or proc.stdout or "git clone failed").strip()
                )

        configure = [cmake, "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"]
        # Prefer a GPU training backend when its toolchain is already installed.
        if shutil.which("nvcc"):
            configure.append("-DGGML_CUDA=ON")
        elif shutil.which("hipcc") and platform.system() != "Darwin":
            configure.append("-DGGML_HIP=ON")
        elif platform.system() == "Darwin":
            configure.append("-DGGML_METAL=ON")

        proc = subprocess.run(configure, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Could not configure the Prism llama.cpp trainer: "
                + (proc.stderr or proc.stdout or "cmake configure failed").strip()
            )

        command = [cmake, "--build", str(build), "--target", "llama-finetune-lora", "--parallel"]
        if os.name == "nt":
            command += ["--config", "Release"]
        proc = subprocess.run(command, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Could not build llama-finetune-lora: "
                + (proc.stderr or proc.stdout or "cmake build failed").strip()
            )

        candidates = [
            build / "bin" / _exe_name("llama-finetune-lora"),
            build / "bin" / "Release" / _exe_name("llama-finetune-lora"),
        ]
        for candidate in candidates:
            if candidate.is_file():
                self._tool_root = source
                return candidate
        raise RuntimeError("llama-finetune-lora built successfully but its binary was not found")

    def find_existing_binary(self) -> Optional[Path]:
        for candidate in self._candidate_binaries():
            if candidate.is_file():
                for parent in [candidate.parent, *candidate.parents]:
                    if (parent / "gguf-py").is_dir():
                        self._tool_root = parent
                        break
                return candidate
        return None

    def can_prepare(self) -> bool:
        """Whether exact-GGUF training can be attempted without fabricating support."""
        if self.find_existing_binary() is not None:
            return True
        return bool(shutil.which("git") and shutil.which("cmake"))

    def _resolve_binary(self) -> Path:
        existing = self.find_existing_binary()
        if existing is not None:
            return existing
        return self._build_prism_trainer()

    @staticmethod
    def _require_supported_cli(binary: Path) -> None:
        proc = subprocess.run([str(binary), "--help"], capture_output=True, text=True)
        help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        required = (
            "--output-adapter",
            "--lora-rank",
            "--lora-alpha",
            "--lora-modules",
            "--assistant-loss-only",
            "--learning-rate",
        )
        missing = [flag for flag in required if flag not in help_text]
        if missing:
            raise RuntimeError(
                "Installed llama-finetune-lora is too old/incompatible for Smart AI Studio "
                "realtime GGUF learning; missing: " + ", ".join(missing)
            )

    def _write_dataset(self, data: List[Dict[str, str]], steps: int) -> str:
        rows = []
        for item in data or []:
            prompt = str(item.get("prompt") or "").strip()
            completion = str(item.get("completion") or "").strip()
            if not prompt or not completion:
                continue
            rows.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ]
                }
            )
        if not rows:
            raise RuntimeError("GGUF LoRA trainer received no prompt/completion pairs")

        # Repeat the tiny online-learning batch so the native trainer always has enough
        # tokens to form several training windows; this is the GGUF equivalent of the
        # existing MLX trainer's small multi-step update.
        repeated = rows * max(1, int(steps))
        while sum(len(json.dumps(row, ensure_ascii=False)) for row in repeated) < 8192:
            repeated += rows

        fd, path = tempfile.mkstemp(prefix="smartai-gguf-learn-", suffix=".jsonl")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as handle:
            for row in repeated:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def _reader_root(self, binary: Path) -> Optional[Path]:
        if self._tool_root and (self._tool_root / "gguf-py").is_dir():
            return self._tool_root
        for parent in [binary.parent, *binary.parents]:
            if (parent / "gguf-py").is_dir():
                return parent
        return None

    @staticmethod
    def _tensor_map(path: str, reader_root: Optional[Path]) -> Dict[str, Any]:
        if not path or not os.path.isfile(path) or reader_root is None:
            return {}
        gguf_py = str(reader_root / "gguf-py")
        inserted = False
        if gguf_py not in sys.path:
            sys.path.insert(0, gguf_py)
            inserted = True
        try:
            import numpy as np
            from gguf.gguf_reader import GGUFReader
            reader = GGUFReader(path)
            result = {}
            for tensor in reader.tensors:
                arr = tensor.data
                if getattr(arr.dtype, "kind", "") not in ("f", "i", "u"):
                    continue
                # LoRA adapters are normally float tensors. Keep only manageable data.
                result[tensor.name] = np.asarray(arr, dtype=np.float32).copy()
            return result
        finally:
            if inserted:
                try:
                    sys.path.remove(gguf_py)
                except ValueError:
                    pass

    @staticmethod
    def _drift_l2(before: Dict[str, Any], after: Dict[str, Any]) -> Tuple[float, int]:
        import numpy as np
        total = 0.0
        touched = 0
        for name, new in after.items():
            old = before.get(name)
            if old is not None and getattr(old, "shape", None) == getattr(new, "shape", None):
                delta = new.astype(np.float64) - old.astype(np.float64)
            else:
                delta = new.astype(np.float64)
            total += float(np.sum(delta * delta))
            touched += int(new.size)
        return math.sqrt(max(total, 0.0)), touched

    def train(
        self,
        data: List[Dict[str, str]],
        *,
        learning_rate: float = 1e-4,
        steps: int = 3,
    ) -> Tuple[Dict[str, Any], float, int, str]:
        binary = self._resolve_binary()
        self._require_supported_cli(binary)
        reader_root = self._reader_root(binary)
        dataset = self._write_dataset(data, steps)
        os.makedirs(self.adapter_root, exist_ok=True)

        before = self._tensor_map(self.adapter_path, reader_root)
        tmp_adapter = self.adapter_path + ".tmp.gguf"
        if os.path.exists(tmp_adapter):
            os.remove(tmp_adapter)

        n_gpu_layers = os.getenv("SMARTAI_GGUF_TRAIN_NGL", "999")
        command = [
            str(binary),
            "-m", self.model_path,
            "-f", dataset,
            "--output-adapter", tmp_adapter,
            "--assistant-loss-only",
            "-ngl", str(n_gpu_layers),
            "-c", "256",
            "-b", "32",
            "-ub", "32",
            "-fa", "off",
            "--lora-rank", str(self.rank),
            "--lora-alpha", str(self.alpha),
            "--lora-modules", "attn_q,attn_k,attn_v,attn_o",
            "--learning-rate", str(float(learning_rate)),
            "--lora-seed", "1",
        ]
        if os.path.isfile(self.adapter_path):
            command += ["--lora", self.adapter_path]

        try:
            proc = subprocess.run(command, capture_output=True, text=True)
        finally:
            try:
                os.remove(dataset)
            except OSError:
                pass

        if proc.returncode != 0 or not os.path.isfile(tmp_adapter):
            try:
                os.remove(tmp_adapter)
            except OSError:
                pass
            raise RuntimeError(
                "Exact-GGUF LoRA training failed. The current Prism llama.cpp backend "
                "may not expose backward kernels for this Bonsai packing on this device. "
                + (proc.stderr or proc.stdout or "trainer returned no adapter").strip()
            )

        after = self._tensor_map(tmp_adapter, reader_root)
        drift, touched = self._drift_l2(before, after)
        if drift <= 0.0 or touched <= 0:
            try:
                os.remove(tmp_adapter)
            except OSError:
                pass
            raise RuntimeError("GGUF LoRA trainer completed but no real adapter parameter change was measured")

        os.replace(tmp_adapter, self.adapter_path)
        return {"trainable_parameters_touched": touched}, float(drift), touched, self.adapter_path
