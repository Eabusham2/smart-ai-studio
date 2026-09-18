"""Real BitNet/bitnet.cpp inference backend.

Uses Microsoft's official BitNet runtime for supported 1.58-bit models.  This module
never fabricates model output.  Online parameter updates remain fail-closed until a
real BitNet adapter-training backend is available.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Generator, List, Optional

from config.paths import get_portable_data_dir


BITNET_REPO = "https://github.com/microsoft/BitNet.git"


class BitNetReasoningBackend:
    def __init__(
        self,
        model_path: str,
        vocab_size: int = 32000,
        hidden_dim: int = 2048,
        num_layers: int = 16,
        device: str = "cpu",
        n_ctx: int = 32768,
    ):
        del vocab_size, hidden_dim, num_layers
        self.model_path = str(model_path)
        self.device = str(device or "cpu")
        self.n_ctx = int(n_ctx)
        self.model: Optional[str] = None
        self.tokenizer: Optional[Any] = None
        self.is_loaded = False
        self.runtime_root: Optional[Path] = None
        self.cli_path: Optional[Path] = None

    @staticmethod
    def _find_model_file(value: str) -> Optional[Path]:
        path = Path(os.path.abspath(os.path.expanduser(value)))
        if path.is_file() and path.suffix.lower() == ".gguf":
            return path
        if path.is_dir():
            preferred = [
                path / "ggml-model-i2_s.gguf",
                path / "ggml-model-tl1.gguf",
                path / "ggml-model-tl2.gguf",
            ]
            for candidate in preferred:
                if candidate.is_file():
                    return candidate
            ggufs = [p for p in path.rglob("*.gguf") if p.is_file()]
            if ggufs:
                bitnet = [p for p in ggufs if any(x in p.name.lower() for x in ("i2_s", "tl1", "tl2", "bitnet"))]
                return max(bitnet or ggufs, key=lambda p: p.stat().st_size)
        return None

    @staticmethod
    def _find_cli(root: Path) -> Optional[Path]:
        names = ["llama-cli.exe", "llama-cli"] if os.name == "nt" else ["llama-cli"]
        candidates = [
            root / "build" / "bin",
            root / "build" / "bin" / "Release",
        ]
        for directory in candidates:
            for name in names:
                path = directory / name
                if path.is_file():
                    return path
        return None

    def _runtime_root(self) -> Path:
        explicit = os.getenv("BITNET_CPP_DIR", "").strip()
        if explicit:
            return Path(os.path.abspath(os.path.expanduser(explicit)))
        return Path(get_portable_data_dir()) / "bitnet_cpp" / "BitNet"

    def _ensure_runtime(self, model_file: Path) -> Path:
        root = self._runtime_root()
        cli = self._find_cli(root) if root.exists() else None
        if cli is not None:
            self.runtime_root = root
            return cli

        git = shutil.which("git")
        if not git:
            raise RuntimeError(
                "Real BitNet inference requires microsoft/BitNet. Install git or set "
                "BITNET_CPP_DIR to an existing bitnet.cpp checkout."
            )

        if not (root / ".git").exists():
            root.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                [git, "clone", "--recursive", "--depth", "1", BITNET_REPO, str(root)],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "Could not clone microsoft/BitNet: "
                    + (proc.stderr or proc.stdout or "git clone failed").strip()
                )

        model_dir = model_file.parent
        quant_type = "tl1" if "tl1" in model_file.name.lower() else "i2_s"
        setup = root / "setup_env.py"
        proc = subprocess.run(
            [
                sys.executable,
                str(setup),
                "--model-dir", str(model_dir),
                "--quant-type", quant_type,
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "bitnet.cpp could not prepare this model/runtime: "
                + (proc.stderr or proc.stdout or "setup_env.py failed").strip()
            )

        cli = self._find_cli(root)
        if cli is None:
            raise RuntimeError("bitnet.cpp setup completed but llama-cli was not found")
        self.runtime_root = root
        return cli

    def load_model(self) -> bool:
        model_file = self._find_model_file(self.model_path)
        if model_file is None:
            return False
        try:
            self.cli_path = self._ensure_runtime(model_file)
            self.model_path = str(model_file)
            self.model = self.model_path
            self.is_loaded = True
            return True
        except Exception:
            self.model = None
            self.is_loaded = False
            raise

    @staticmethod
    def _clean_cli_output(text: str, prompt: str) -> str:
        value = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", str(text or ""))
        # llama-cli may echo the prompt. Strip only an exact leading echo.
        stripped = value.strip()
        if stripped.startswith(prompt):
            stripped = stripped[len(prompt):].lstrip()
        return stripped

    def _generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        if not self.is_loaded or self.cli_path is None or self.model is None:
            raise RuntimeError("BitNet model is not loaded")
        threads = int(os.getenv("SMARTAI_BITNET_THREADS", str(max(1, os.cpu_count() or 2))))
        ngl = os.getenv("SMARTAI_BITNET_NGL", "0")
        command = [
            str(self.cli_path),
            "-m", str(self.model),
            "-n", str(max(1, int(max_tokens))),
            "-t", str(max(1, threads)),
            "-p", str(prompt),
            "-ngl", str(ngl),
            "-c", str(max(2048, int(self.n_ctx))),
            "--temp", str(float(temperature)),
        ]
        proc = subprocess.run(
            command,
            cwd=str(self.runtime_root) if self.runtime_root else None,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "bitnet.cpp inference failed: "
                + (proc.stderr or proc.stdout or "llama-cli failed").strip()
            )
        return self._clean_cli_output(proc.stdout, prompt)

    def generate_branches(
        self,
        prompt: str,
        branch_count: int = 1,
        max_tokens: int = 1536,
        temperature: Any = 0.75,
        top_p: float = 0.92,
    ) -> List[str]:
        del top_p
        outputs = []
        for idx in range(max(1, int(branch_count))):
            temp = float(temperature[idx % len(temperature)]) if isinstance(temperature, (list, tuple)) else float(temperature)
            outputs.append(self._generate(prompt, max_tokens, temp))
        return outputs

    def stream_generate_tokens(
        self,
        prompt: str,
        max_tokens: int = 1536,
        temperature: float = 0.75,
        top_p: float = 0.92,
    ) -> Generator[str, None, None]:
        del top_p
        text = self._generate(prompt, max_tokens, temperature)
        for match in re.finditer(r"\S+\s*", text):
            yield match.group(0)

    def calculate_token_entropy(self, prompt: str) -> float:
        del prompt
        # Official bitnet.cpp CLI does not expose stable next-token logits through
        # its public Python wrapper, so routing keeps the neutral fallback rather
        # than running a fake model.
        return 0.35

    def training_ready(self) -> bool:
        return False

    def train_mini_batch(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "BitNet inference is real, but this runtime exposes no verified online "
            "adapter-training API; refusing to fabricate a parameter update."
        )

    def unload_model(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
