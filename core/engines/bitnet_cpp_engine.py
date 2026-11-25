"""Real Microsoft bitnet.cpp inference and persistent-learning backend.

This replaces the old synthetic BitNet placeholder in ProReasoningEngine routing.
It talks to bitnet.cpp's llama-server and never fabricates output. Learning is
fail-closed unless metadata provides a compatible BF16 training lineage; when it
does, the backend performs a real PEFT update, rebuilds the selected ternary deployment weights,
hot-reloads them, and rolls back atomically on failure or cancellation.
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
from core.native_runtime_pinning import (
    ensure_pinned_checkout,
    require_full_commit,
    validate_runtime_receipt,
    verify_pinned_checkout,
    write_runtime_receipt,
)


BITNET_REPO = "https://github.com/microsoft/BitNet.git"
# Deliberately no network default: a trusted packager/operator selects the reviewed commit.
BITNET_COMMIT_ENV = "SMARTAI_BITNET_COMMIT"
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
        legacy_adapter_path: Optional[str] = None,
    ):
        self.original_model_path = str(model_path)
        self.model_path = str(model_path)
        self.hf_repo_id = str(hf_repo_id or "").strip()
        self.training_base_model_id = str(training_base_model_id or "").strip()
        self.n_ctx = int(n_ctx)
        self.threads = int(threads or max(1, (os.cpu_count() or 4) // 2))
        key_src = self.hf_repo_id or self.original_model_path
        key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:16]
        eval_training_root = str(os.getenv("SMARTAI_BITNET_TRAINING_ROOT", "") or "").strip()
        self.training_root = (
            Path(eval_training_root)
            if eval_training_root
            else Path(get_portable_data_dir()) / "bitnet_learning" / key
        )
        self.learned_model_path = self.training_root / "learned-i2_s.gguf"
        self.legacy_adapter_path = legacy_adapter_path
        self.adapter_path = ""
        self._candidate_deployment = None
        self._selected_model_file = None
        self._selected_tensor_types = None
        self.adapters: Dict[str, Any] = {}
        self.model: Optional[_BitNetServerProxy] = None
        self.tokenizer: Optional[_BitNetServerProxy] = None
        self.is_loaded = False
        self._process: Optional[subprocess.Popen] = None
        self._base_url: Optional[str] = None
        self.last_error = ""

    def _root(self) -> Path:
        return Path(get_portable_data_dir()) / "bitnet_cpp_runtime"

    @staticmethod
    def _explicit_server() -> Optional[Path]:
        for env_name in ("SMARTAI_BITNET_SERVER", "BITNET_SERVER"):
            value = os.getenv(env_name, "").strip()
            if value:
                candidate = Path(value)
                if candidate.is_file():
                    return candidate
        return None

    @staticmethod
    def _managed_servers(source: Path) -> List[Path]:
        return [
            source / "build/bin/llama-server",
            source / "build/bin/Release/llama-server.exe",
        ]

    def find_server(self) -> Optional[Path]:
        explicit = self._explicit_server()
        if explicit is not None:
            return explicit

        git = shutil.which("git")
        root = self._root()
        source = root / "BitNet"
        command = self._setup_command(root)
        if not git or command is None:
            return None
        try:
            commit = require_full_commit(os.getenv(BITNET_COMMIT_ENV, ""), BITNET_COMMIT_ENV)
            verify_pinned_checkout(git, source, commit, recursive=True)
        except RuntimeError:
            return None
        server = next((path for path in self._managed_servers(source) if path.is_file()), None)
        if server is None:
            return None
        receipt = source / "build" / "smartai-runtime.json"
        build_inputs = {"python": sys.executable, "setup_command": command}
        if validate_runtime_receipt(
            receipt,
            source / "build",
            BITNET_REPO,
            commit,
            {"llama-server": server},
            build_inputs,
        ):
            return server
        return None

    def _resolve_model_file(self) -> Optional[str]:
        value = os.path.abspath(os.path.expanduser(self.original_model_path))
        if os.path.isfile(value):
            return value
        if os.path.isdir(value):
            files = list(Path(value).rglob("*.gguf"))
        else:
            files = []
            try:
                from huggingface_hub import snapshot_download
                root = snapshot_download(repo_id=self.original_model_path, local_files_only=True)
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

    def _runtime_repo_id(self) -> str:
        """Map deployment/BF16 siblings to the canonical model family used by bitnet.cpp setup."""
        repo = str(self.hf_repo_id or "").strip()
        low = repo.lower()
        if low == "microsoft/bitnet-b1.58-2b-4t-gguf" or low == "microsoft/bitnet-b1.58-2b-4t-bf16":
            return "microsoft/BitNet-b1.58-2B-4T"
        if low == "microsoft/bitnet-b1.58-2b-4t":
            return "microsoft/BitNet-b1.58-2B-4T"
        return repo

    def _setup_command(self, root: Path) -> Optional[List[str]]:
        repo_id = self._runtime_repo_id()
        if not repo_id and self.original_model_path in _SUPPORTED_REPOS:
            repo_id = self.original_model_path
        if repo_id not in _SUPPORTED_REPOS:
            return None
        return [
            sys.executable,
            "setup_env.py",
            "--hf-repo", repo_id,
            "--model-dir", str(root / "models"),
            "--quant-type", "i2_s",
        ]

    def _prepare_official_runtime(self) -> Optional[Path]:
        explicit = self._explicit_server()
        if explicit is not None:
            return explicit

        root = self._root()
        command = self._setup_command(root)
        if command is None:
            return None

        git = shutil.which("git")
        if not git:
            self.last_error = "BitNet runtime preparation requires Git or an explicit SMARTAI_BITNET_SERVER."
            return None
        source = root / "BitNet"
        root.mkdir(parents=True, exist_ok=True)
        try:
            approved = require_full_commit(os.getenv(BITNET_COMMIT_ENV, ""), BITNET_COMMIT_ENV)
            commit = ensure_pinned_checkout(
                git,
                source,
                BITNET_REPO,
                approved,
                recursive=True,
            )
        except RuntimeError as exc:
            self.last_error = str(exc)
            return None

        build_inputs = {"python": sys.executable, "setup_command": command}
        receipt = source / "build" / "smartai-runtime.json"
        existing = next((path for path in self._managed_servers(source) if path.is_file()), None)
        if existing is not None and validate_runtime_receipt(
            receipt,
            source / "build",
            BITNET_REPO,
            commit,
            {"llama-server": existing},
            build_inputs,
        ):
            return existing

        # Never incrementally reuse an executable produced without a matching receipt.
        if (source / "build").exists():
            shutil.rmtree(source / "build")

        # setup_env.py owns BitNet's model-specific code generation and build.
        proc = subprocess.run(command, cwd=str(source), capture_output=True, text=True)
        if proc.returncode != 0:
            self.last_error = (proc.stderr or proc.stdout or "BitNet setup failed").strip()
            return None
        try:
            verify_pinned_checkout(git, source, commit, recursive=True)
            server = next((path for path in self._managed_servers(source) if path.is_file()), None)
            if server is None:
                self.last_error = "BitNet setup completed but llama-server was not found."
                return None
            write_runtime_receipt(
                receipt,
                source / "build",
                BITNET_REPO,
                commit,
                {"llama-server": server},
                build_inputs,
            )
            return server
        except (OSError, RuntimeError, ValueError) as exc:
            self.last_error = str(exc)
            return None

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
        from core.native_context_learning import select_deployment
        selected = self._selected_model_file or self._resolve_model_file()
        server = self._prepare_official_runtime()
        if not selected or server is None:
            return False
        self._selected_model_file = selected
        self.adapter_path = select_deployment(self, selected, self.training_root, "bitnet", self.learned_model_path) or ""
        self.model_path = self.adapter_path or selected
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
        progress_callback=None,
        stop_at_tool_boundary: bool = False, cancelled=None,
    ) -> List[str]:
        from core.tool_boundary_stop import ToolBoundaryStopper
        out = []
        count = max(1, int(branch_count))
        for idx in range(count):
            if callable(cancelled) and cancelled():
                return out
            temp = float(temperature[idx % len(temperature)]) if isinstance(temperature, (list, tuple)) else float(temperature)
            started = time.perf_counter()
            tokens = None
            if stop_at_tool_boundary or cancelled is not None:
                detector = ToolBoundaryStopper() if stop_at_tool_boundary else None
                pieces = []
                iterator = self.stream_generate_tokens(prompt, int(max_tokens), temp, float(top_p), stop_at_tool_boundary=stop_at_tool_boundary, cancelled=cancelled)
                try:
                    for piece in iterator:
                        pieces.append(str(piece))
                        if progress_callback is not None:
                            progress_callback(idx, count, getattr(self, "last_generation_tokens", None),
                                              time.perf_counter() - started, finished=False)
                        if (detector is not None and detector.feed(piece)) or (callable(cancelled) and cancelled()):
                            break
                finally:
                    close = getattr(iterator, "close", None)
                    if callable(close):
                        close()
                result_text = (detector.text if detector is not None else "".join(pieces)).strip()
                observed = getattr(self, "last_generation_tokens", None)
                if type(observed) is int and observed >= 0:
                    tokens = observed
            else:
                result = self._request(
                    "/completion",
                    {
                        "prompt": prompt,
                        "n_predict": int(max_tokens),
                        "temperature": temp,
                        "top_p": float(top_p),
                        "stream": False,
                        **({"cache_prompt": False} if getattr(self, "_best_work_inference_fork", False) else {}),
                    },
                    timeout=1800,
                )
                result_text = str(result.get("content") or result.get("text") or "").strip()
                count_value = result.get("tokens_predicted")
                if type(count_value) is int and count_value >= 0:
                    tokens = count_value
            out.append(result_text)
            if progress_callback is not None:
                progress_callback(idx + 1, count, tokens, time.perf_counter() - started)
        return out

    def stream_generate_tokens(
        self, prompt: str, max_tokens: int = 1536, temperature: float = 0.65,
        top_p: float = 0.92, stop_at_tool_boundary: bool = False, cancelled=None,
    ) -> Generator[str, None, None]:
        if not self._base_url:
            return
        from core.generation_metrics import TokenMeter, observe_server_event
        from core.tool_boundary_stop import ToolBoundaryStopper
        meter = TokenMeter(self, "bitnet-server")
        detector = ToolBoundaryStopper() if stop_at_tool_boundary else None
        payload = {"prompt": prompt, "n_predict": int(max_tokens),
                   "temperature": float(temperature), "top_p": float(top_p),
                   "stream": True, "return_tokens": True, "timings_per_token": True}
        if getattr(self, "_best_work_inference_fork", False):
            payload["cache_prompt"] = False
        req = urllib.request.Request(self._base_url + "/completion",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
                                     method="POST")
        response = urllib.request.urlopen(req, timeout=1800)
        try:
            for raw in response:
                line = raw.decode("utf-8", errors="strict").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                item = json.loads(body)
                observe_server_event(meter, item)
                piece = str(item.get("content") or item.get("text") or "")
                stop_here = False
                if detector is not None:
                    offset = len(detector.text)
                    stop_here = detector.feed(piece)
                    piece = detector.text[offset:]
                if piece:
                    yield piece
                if stop_here or item.get("stop") is True or (callable(cancelled) and cancelled()):
                    break
        finally:
            response.close()
            meter.finish()

    def calculate_token_entropy(self, prompt: str) -> float:
        # bitnet.cpp server does not expose stable full-vocabulary logits through
        # its public server contract. Keep routing functional without an extra fake
        # model pass.
        del prompt
        return 0.32

    def training_ready(self) -> bool:
        """True only when the declared BF16 lineage can be really rebuilt for bitnet.cpp."""
        if not self.training_base_model_id or not self.is_loaded:
            return False
        try:
            from core.bitnet_rebuild_trainer import BitNetRebuildTrainer
            trainer = BitNetRebuildTrainer(
                base_model_id=self.training_base_model_id,
                runtime_root=str(self._root()),
                deploy_root=str(self.training_root),
            )
            if getattr(self, "learning_implementation", None) == "legacy":
                prepared = trainer.can_prepare_legacy()
            else:
                prepared = trainer.can_prepare()
            if not prepared:
                return False
            if self._selected_tensor_types is None:
                from core.gguf_metadata import inspect_gguf
                self._selected_tensor_types = inspect_gguf(self._selected_model_file)["quantization_types"]
            return bool(trainer.selected_quantization(self._selected_tensor_types))
        except Exception:
            return False

    supports_learning_modes = True

    def train_mini_batch(
        self,
        adapters: Any,
        data: List[Dict[str, str]],
        fisher_matrix: Any = None,
        lambda_ewc: float = 0.0,
        learning_rate: float = 1e-4,
        steps: int = 3,
        save_path: Optional[str] = None,
        learning_mode=None, **options,
    ):
        """Select the original completion trainer or the managed learning objectives."""
        from core.learning_implementation import resolve_learning_mode
        mode = resolve_learning_mode(self, learning_mode, options)
        return BitNetCppReasoningBackend._train_mini_batch(
            self, adapters, data, fisher_matrix, lambda_ewc, learning_rate, steps, save_path,
            learning_mode=mode, **options,
        )

    def _train_mini_batch(
        self,
        adapters: Any,
        data: List[Dict[str, str]],
        fisher_matrix: Any = None,
        lambda_ewc: float = 0.0,
        learning_rate: float = 1e-4,
        steps: int = 3,
        save_path: Optional[str] = None,
        *, learning_mode,
        cancel_event=None, validation_callback=None, learning_metadata=None, anchor_texts=None,
        reward_weighting="group_relative", entropic_kl_target=None, reference_kl_coef=0.0,
        objective='causal_lm', supervise_prompt=False, max_window_tokens=512, row_weights=None,
        context_text=None, max_context_tokens=2048, **_kwargs,
    ):
        """Train the BF16 lineage and validate the rebuilt deployment in bitnet.cpp."""
        from core.bitnet_rebuild_trainer import BitNetRebuildTrainer
        from core.native_context_learning import update_deployment
        if not self.training_ready():
            raise RuntimeError("The selected deployment requires a matching local BF16 training parent and compatible BitNet converter")
        trainer = BitNetRebuildTrainer(base_model_id=self.training_base_model_id, runtime_root=str(self._root()),
                                       deploy_root=str(self.training_root))
        return update_deployment(
            self, trainer, data, learning_mode=learning_mode, learning_rate=learning_rate, steps=steps,
            fisher_matrix=fisher_matrix, lambda_ewc=lambda_ewc, cancel_event=cancel_event,
            validation_callback=validation_callback, learning_metadata=learning_metadata, anchor_texts=anchor_texts,
            reward_weighting=reward_weighting, entropic_kl_target=entropic_kl_target, reference_kl_coef=reference_kl_coef,
            objective=objective, supervise_prompt=supervise_prompt, row_weights=row_weights, max_window_tokens=max_window_tokens,
            context_text=context_text, max_context_tokens=max_context_tokens,
        )

    def _check_deployment_format(self, deployment):
        from core.gguf_metadata import inspect_gguf
        original = inspect_gguf(self._selected_model_file, verify_layout=True)
        trial = inspect_gguf(deployment, verify_layout=True)
        for key in ("architecture", "vocab_size", "embedding_length", "layer_count", "tensor_layout_sha256", "tokenizer_layout_sha256"):
            if original[key] != trial[key]:
                raise RuntimeError("BitNet rebuild differs from the selected model's " + key)

    def _verify_training_parent(self, trainer):
        from core.gguf_metadata import inspect_gguf
        from core.training_lineage import verify_parent
        trainer.quantization = trainer.selected_quantization(inspect_gguf(self._selected_model_file)["quantization_types"])
        def tokens(text):
            return self._request("/tokenize", {"content": text, "add_special": False, "parse_special": False})["tokens"]
        return verify_parent(self, trainer, self._selected_model_file, tokens, self.training_root / "learned_memory")

    def score_learning_probe(self, prompt, completion):
        from core.native_context_learning import score_server_probe
        from core.native_speed import runtime_lock
        with runtime_lock(self):
            return score_server_probe(self, prompt, completion)

    def supports_media_input(self, kind: str) -> bool:
        del kind
        return False
