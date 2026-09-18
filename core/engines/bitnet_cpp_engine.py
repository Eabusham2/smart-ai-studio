"""Real Microsoft bitnet.cpp inference backend.

This replaces the old synthetic BitNet placeholder in ProReasoningEngine routing.
It talks to bitnet.cpp's llama-server and never fabricates output. Parameter training
is intentionally fail-closed until bitnet.cpp exposes a compatible persistent adapter
training path.
"""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from config.paths import get_portable_data_dir


_SUPPORTED_REPOS = {
    "microsoft/BitNet-b1.58-2B-4T",
    "1bitLLM/bitnet_b1_58-large",
    "1bitLLM/bitnet_b1_58-3B",
    "HF1BitLLM/Llama3-8B-1.58-100B-tokens",
    "tiiuae/Falcon3-1B-Instruct-1.58bit",
    "tiiuae/Falcon3-3B-Instruct-1.58bit",
    "tiiuae/Falcon3-7B-Instruct-1.58bit",
    "tiiuae/Falcon3-10B-Instruct-1.58bit",
}


class _BitNetServerProxy:
    def __init__(self, backend: "BitNetCppReasoningBackend"):
        self.backend = backend

    def tokenize(self, raw: bytes) -> List[int]:
        result = self.backend._request("/tokenize", {"content": raw.decode("utf-8", errors="ignore")})
        return [int(x) for x in result.get("tokens", [])]

    def encode(self, text: str, *args, **kwargs) -> List[int]:
        del args, kwargs
        return self.tokenize(str(text).encode("utf-8"))

    def n_ctx(self) -> int:
        return int(self.backend.n_ctx)


class BitNetCppReasoningBackend:
    def __init__(
        self,
        model_path: str,
        hf_repo_id: Optional[str] = None,
        n_ctx: int = 32768,
        threads: Optional[int] = None,
        training_base_model_id: Optional[str] = None,
    ):
        self.original_model_path = str(model_path)
        self.model_path = str(model_path)
        self.hf_repo_id = str(hf_repo_id or "").strip()
        self.training_base_model_id = str(training_base_model_id or "").strip()
        self.n_ctx = int(n_ctx)
        self.threads = int(threads or max(1, (os.cpu_count() or 4) // 2))
        key_src = self.hf_repo_id or self.original_model_path
        key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:16]
        self.training_root = Path(get_portable_data_dir()) / "bitnet_learning" / key
        self.learned_model_path = self.training_root / "learned-i2_s.gguf"
        self.adapter_path = str(self.learned_model_path)
        self.adapters: Dict[str, Any] = {}
        self.model: Optional[_BitNetServerProxy] = None
        self.tokenizer: Optional[_BitNetServerProxy] = None
        self.is_loaded = False
        self._process: Optional[subprocess.Popen] = None
        self._base_url: Optional[str] = None

    def _root(self) -> Path:
        return Path(get_portable_data_dir()) / "bitnet_cpp_runtime"

    def _candidate_servers(self) -> List[Path]:
        out: List[Path] = []
        for env_name in ("SMARTAI_BITNET_SERVER", "BITNET_SERVER"):
            value = os.getenv(env_name, "").strip()
            if value:
                out.append(Path(value))
        root = self._root()
        source = root / "BitNet"
        out += [
            source / "build/bin/llama-server",
            source / "build/bin/Release/llama-server.exe",
            Path.cwd() / "BitNet/build/bin/llama-server",
            Path.cwd() / "BitNet/build/bin/Release/llama-server.exe",
        ]
        return out

    def find_server(self) -> Optional[Path]:
        for candidate in self._candidate_servers():
            if candidate.is_file():
                return candidate
        return None

    def _resolve_model_file(self) -> Optional[str]:
        # Learned deployment state is persistent and takes precedence on reload.
        if self.learned_model_path.is_file():
            return str(self.learned_model_path)
        value = os.path.abspath(os.path.expanduser(self.original_model_path))
        if os.path.isfile(value):
            return value
        if os.path.isdir(value):
            files = list(Path(value).rglob("*.gguf"))
        else:
            files = []
            try:
                from huggingface_hub import snapshot_download
                root = snapshot_download(repo_id=self.model_path, local_files_only=True)
                files = list(Path(root).rglob("*.gguf"))
            except Exception:
                pass
        if not files:
            return None
        preferred = [
            p for p in files
            if any(marker in p.name.lower() for marker in ("i2_s", "tl1", "tl2"))
        ]
        pool = preferred or files
        return str(max(pool, key=lambda p: p.stat().st_size))

    def _prepare_official_runtime(self) -> Optional[Path]:
        existing = self.find_server()
        if existing is not None:
            return existing

        git = shutil.which("git")
        if not git:
            return None
        root = self._root()
        source = root / "BitNet"
        root.mkdir(parents=True, exist_ok=True)
        if not (source / ".git").exists():
            proc = subprocess.run(
                [git, "clone", "--depth", "1", "https://github.com/microsoft/BitNet.git", str(source)],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                return None

        # setup_env.py owns BitNet's model-specific code generation and build.
        repo_id = self.hf_repo_id or (self.model_path if self.model_path in _SUPPORTED_REPOS else "")
        if repo_id not in _SUPPORTED_REPOS:
            return None
        models_dir = root / "models"
        command = [
            sys.executable,
            "setup_env.py",
            "--hf-repo", repo_id,
            "--model-dir", str(models_dir),
            "--quant-type", "i2_s",
        ]
        proc = subprocess.run(command, cwd=str(source), capture_output=True, text=True)
        if proc.returncode != 0:
            return None
        return self.find_server()

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _request(self, path: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 120) -> Dict[str, Any]:
        if not self._base_url:
            raise RuntimeError("bitnet.cpp server is not running")
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {"value": parsed}

    def load_model(self) -> bool:
        model_file = self._resolve_model_file()
        server = self._prepare_official_runtime()
        if not model_file or server is None:
            return False
        self.model_path = model_file
        self.unload_model()
        port = self._free_port()
        command = [
            str(server),
            "-m", self.model_path,
            "-c", str(self.n_ctx),
            "-t", str(self.threads),
            "-ngl", "0",
            "--host", "127.0.0.1",
            "--port", str(port),
            "-cb",
        ]
        try:
            self._process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._base_url = f"http://127.0.0.1:{port}"
            deadline = time.time() + 180
            while time.time() < deadline:
                if self._process.poll() is not None:
                    break
                try:
                    self._request("/health", timeout=2)
                    proxy = _BitNetServerProxy(self)
                    self.model = proxy
                    self.tokenizer = proxy
                    self.is_loaded = True
                    return True
                except Exception:
                    time.sleep(0.25)
        except Exception:
            pass
        self.unload_model()
        return False

    def unload_model(self):
        proc = self._process
        self._process = None
        self._base_url = None
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        text = "\n".join(str(m.get("content", "")) for m in messages or [])
        if self.tokenizer is None:
            return max(1, len(text) // 4)
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            return max(1, len(text) // 4)

    def generate_branches(
        self,
        prompt: str,
        branch_count: int = 1,
        max_tokens: int = 1536,
        temperature: Any = 0.65,
        top_p: float = 0.92,
    ) -> List[str]:
        out = []
        for idx in range(max(1, int(branch_count))):
            temp = float(temperature[idx % len(temperature)]) if isinstance(temperature, (list, tuple)) else float(temperature)
            result = self._request(
                "/completion",
                {
                    "prompt": prompt,
                    "n_predict": int(max_tokens),
                    "temperature": temp,
                    "top_p": float(top_p),
                    "stream": False,
                },
                timeout=1800,
            )
            out.append(str(result.get("content") or result.get("text") or "").strip())
        return out

    def stream_generate_tokens(
        self,
        prompt: str,
        max_tokens: int = 1536,
        temperature: float = 0.65,
        top_p: float = 0.92,
    ) -> Generator[str, None, None]:
        if not self._base_url:
            return
        payload = {
            "prompt": prompt,
            "n_predict": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": True,
        }
        req = urllib.request.Request(
            self._base_url + "/completion",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        response = urllib.request.urlopen(req, timeout=1800)
        try:
            for raw in response:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    item = json.loads(body)
                except Exception:
                    continue
                chunk = item.get("content") or item.get("text") or ""
                if chunk:
                    yield str(chunk)
        finally:
            response.close()

    def calculate_token_entropy(self, prompt: str) -> float:
        # bitnet.cpp server does not expose stable full-vocabulary logits through
        # its public server contract. Keep routing functional without an extra fake
        # model pass.
        del prompt
        return 0.32

    def training_ready(self) -> bool:
        """True only when the declared BF16 lineage can be really rebuilt for bitnet.cpp."""
        if not self.training_base_model_id:
            return False
        try:
            from core.bitnet_rebuild_trainer import BitNetRebuildTrainer
            trainer = BitNetRebuildTrainer(
                base_model_id=self.training_base_model_id,
                runtime_root=str(self._root()),
                deploy_root=str(self.training_root),
            )
            return bool(trainer.can_prepare())
        except Exception:
            return False

    def train_mini_batch(
        self,
        adapters: Any,
        data: List[Dict[str, str]],
        fisher_matrix: Any = None,
        lambda_ewc: float = 0.0,
        learning_rate: float = 1e-4,
        steps: int = 3,
        save_path: Optional[str] = None,
        **_kwargs,
    ):
        """Fine-tune the BF16 sibling, rebuild I2_S weights, then hot-reload bitnet.cpp."""
        del adapters, fisher_matrix, lambda_ewc, save_path
        if not self.training_base_model_id:
            raise RuntimeError("BitNet learning requires metadata with a BF16 training-base model id")

        from core.bitnet_rebuild_trainer import BitNetRebuildTrainer
        trainer = BitNetRebuildTrainer(
            base_model_id=self.training_base_model_id,
            runtime_root=str(self._root()),
            deploy_root=str(self.training_root),
        )
        if not trainer.can_prepare():
            raise RuntimeError(
                "BitNet inference is available, but this model/device does not expose a "
                "verified BF16→PEFT→BitNet rebuild training path."
            )

        self.training_root.mkdir(parents=True, exist_ok=True)
        backup = str(self.learned_model_path) + ".previous"
        had_learned = self.learned_model_path.is_file()
        if os.path.isfile(backup):
            os.remove(backup)
        if had_learned:
            shutil.copy2(str(self.learned_model_path), backup)

        self.unload_model()
        try:
            meta, drift, _touched, learned_path = trainer.train(
                data,
                learning_rate=float(learning_rate),
                steps=max(1, int(steps)),
            )
            self.model_path = learned_path
            self.adapters = dict(meta or {})
            if not self.load_model():
                raise RuntimeError("BitNet training succeeded but bitnet.cpp failed to reload learned weights")
            try:
                os.remove(backup)
            except OSError:
                pass
            return dict(self.adapters), float(drift)
        except Exception:
            if os.path.isfile(backup):
                os.replace(backup, str(self.learned_model_path))
            elif not had_learned:
                try:
                    os.remove(str(self.learned_model_path))
                except OSError:
                    pass
            self.model_path = str(self.learned_model_path if self.learned_model_path.is_file() else self.original_model_path)
            self.load_model()
            raise

    def supports_media_input(self, kind: str) -> bool:
        del kind
        return False
