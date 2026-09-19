"""App-launched cross-platform bridge for the canonical 4,014-item eval.

This module is intentionally dormant for normal CLI runs. When SMARTAI_APP_EVAL_CONFIG
is set by the desktop Eval window, it replaces only model/runtime plumbing:
- dataset/scoring/prompt/install order stay owned by master_4000_eval_suite.py;
- Apple MLX keeps the existing benchmark implementation;
- GGUF/Prism, BitNet, and Transformers/controller use their production app backends;
- Phase 3 uses the backend's real persistent train_mini_batch path and fails closed
  when that backend cannot really train.

No synthetic generation, fake parameter drift, or hidden model substitution is used.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from config.settings import get_settings
from core.pro_engine import ProReasoningEngine
from core.temperature_policy import EVAL_N1_TEMPERATURE
from core.training_memory import process_rss_mb, release_training_memory
from eval._master_4000_base import BenchmarkDatasetProvider, EvaluationCheckpointManager
from memory.knowledge_graph import RelationalKnowledgeGraph
from run_studio_complete import (
    EngineSettings,
    FastMCPDispatcher,
    GramSchmidtOGPProjector,
    POSIXHardenedSandbox,
    SymbolicMCTSSearchEngine,
)


APP_EVAL_CONFIG_ENV = "SMARTAI_APP_EVAL_CONFIG"
APP_EVAL_PAUSE_ENV = "SMARTAI_APP_EVAL_PAUSE_FILE"
APP_EVAL_CANCEL_ENV = "SMARTAI_APP_EVAL_CANCEL_FILE"


def _config_path() -> Optional[Path]:
    raw = str(os.environ.get(APP_EVAL_CONFIG_ENV, "") or "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def _load_config() -> Dict[str, Any]:
    path = _config_path()
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("App eval config must contain one JSON object")
    return data


def _control_wait() -> None:
    cancel = str(os.environ.get(APP_EVAL_CANCEL_ENV, "") or "").strip()
    pause = str(os.environ.get(APP_EVAL_PAUSE_ENV, "") or "").strip()
    if cancel and os.path.exists(cancel):
        raise KeyboardInterrupt
    announced = False
    while pause and os.path.exists(pause):
        if cancel and os.path.exists(cancel):
            raise KeyboardInterrupt
        if not announced:
            print("[APP EVAL] Paused at safe boundary.", flush=True)
            announced = True
        time.sleep(0.20)
    if announced:
        print("[APP EVAL] Resumed.", flush=True)


def _resolve_cached_path(info: Dict[str, Any]) -> Optional[str]:
    raw = str(info.get("model_path") or "").strip()
    gguf_file = str(info.get("gguf_file") or "").strip()
    if raw and os.path.exists(raw):
        if os.path.isdir(raw) and gguf_file:
            candidate = os.path.join(raw, gguf_file)
            if os.path.isfile(candidate):
                return candidate
        return raw

    repo_id = str(info.get("repo_id") or "").strip()
    if not repo_id:
        return raw or None
    try:
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo_id, local_files_only=True)
        if gguf_file:
            candidate = os.path.join(local, gguf_file)
            if os.path.isfile(candidate):
                return candidate
        return local
    except Exception:
        return raw or repo_id


class AppEvalEngineAdapter:
    """Compatibility surface expected by the existing eval modules."""

    def __init__(self, config: Dict[str, Any], run_dir: str):
        self.config = dict(config)
        info = dict(config.get("model_info") or {})
        self.model_info = info
        self.backend_key = ""

        # This process is eval-only, so give MLX a run-local adapter without
        # changing the normal app's persistent adapter or native runtime/cache.
        settings = get_settings()
        settings.lora_adapter_path = os.path.join(
            run_dir, "backend_state", "mlx", "adapters.safetensors"
        )
        settings.database_path = os.path.join(run_dir, "memory.db")
        self.pro = ProReasoningEngine(
            settings=settings,
            lora_adapter_path=settings.lora_adapter_path,
        )

        model_path = _resolve_cached_path(info)
        result = self.pro.load_model(
            str(info.get("name") or info.get("short_name") or "Bonsai 2 27B Ternary"),
            model_path=model_path,
            model_info=info,
        )
        if str(result.get("status") or "") != "loaded":
            raise RuntimeError(
                "App eval could not load the selected text model: "
                + str(result.get("error") or result.get("message") or result)
            )

        self.backend_key = str(self.pro.active_backend or "").lower()
        self.backend = self._active_backend()
        if self.backend is None or getattr(self.backend, "model", None) is None:
            raise RuntimeError("App eval loaded no real text backend")

        eval_settings = EngineSettings(enable_awake_ogp_daemon=False)
        eval_settings.db_path = os.path.join(run_dir, "memory.db")
        eval_settings.mlx_model_path = str(
            info.get("repo_id") or model_path or "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
        )
        self.settings = eval_settings
        self.kg = RelationalKnowledgeGraph(eval_settings.db_path)
        self.sandbox = POSIXHardenedSandbox(
            timeout_sec=eval_settings.sandbox_timeout_seconds,
            max_memory_mb=eval_settings.sandbox_max_memory_mb,
        )
        self.mcp = FastMCPDispatcher(self.sandbox, self.kg)
        self.mcts = SymbolicMCTSSearchEngine(self.sandbox)
        self.ogp_projector = GramSchmidtOGPProjector(
            tolerance=eval_settings.ogp_ortho_tolerance
        )
        self.moe_manager = None
        self.moe_router = None
        self.ogp_daemon = None

    def _active_backend(self):
        key = str(self.pro.active_backend or "").lower()
        if key == "mlx":
            return getattr(self.pro, "mlx_backend", None) or getattr(self.pro, "mlx_engine", None)
        if key == "gguf":
            return getattr(self.pro, "gguf_backend", None)
        if key == "bitnet":
            return getattr(self.pro, "bitnet_backend", None)
        if key == "controller":
            return getattr(self.pro, "controller_backend", None)
        for name in ("controller_backend", "gguf_backend", "bitnet_backend", "mlx_backend"):
            value = getattr(self.pro, name, None)
            if value is not None and getattr(value, "model", None) is not None:
                return value
        return None

    @property
    def model(self):
        return getattr(self.backend, "model", None)

    @model.setter
    def model(self, value):
        if getattr(self, "backend", None) is not None:
            self.backend.model = value

    @property
    def tokenizer(self):
        return getattr(self.backend, "tokenizer", None)

    @tokenizer.setter
    def tokenizer(self, value):
        if getattr(self, "backend", None) is not None:
            self.backend.tokenizer = value

    def unload_model(self):
        try:
            self.pro.unload_model()
        finally:
            release_training_memory(self.backend)


def _logical_model_key(engine: AppEvalEngineAdapter) -> str:
    info = engine.model_info
    return "|".join(
        [
            str(engine.backend_key),
            str(info.get("repo_id") or ""),
            str(info.get("gguf_file") or ""),
            str(info.get("runtime_family") or ""),
        ]
    )


def install(runtime_module, phase4_module, cls) -> None:
    """Install only when launched from the desktop Eval panel."""
    if _config_path() is None or getattr(cls, "_app_cross_platform_bridge_installed", False):
        return

    config = _load_config()
    run_dir = str(config.get("run_dir") or os.getcwd())
    os.makedirs(run_dir, exist_ok=True)

    original_init = cls.__init__
    original_fast = cls._fast_generate
    original_branch_generate = phase4_module._generate_branches_same_model
    original_entropy = phase4_module._normalized_entropy
    original_pro_backend = phase4_module._pro_backend
    original_assert_same = phase4_module._assert_same_model
    original_phase3 = phase4_module._run_phase3_consolidation
    original_restore = phase4_module._restore_rsi_adapter
    original_context_limit = runtime_module._model_context_limit

    def _backend_context_limit(engine) -> Optional[int]:
        if not isinstance(engine, AppEvalEngineAdapter):
            return original_context_limit(engine)

        backend = getattr(engine, "backend", None)
        values = []
        for value in (
            getattr(backend, "n_ctx", None),
            (engine.model_info or {}).get("max_context"),
        ):
            try:
                parsed = int(value)
            except Exception:
                continue
            if 1024 <= parsed <= 10_000_000:
                values.append(parsed)

        model = getattr(backend, "model", None)
        n_ctx_fn = getattr(model, "n_ctx", None)
        if callable(n_ctx_fn):
            try:
                parsed = int(n_ctx_fn())
                if 1024 <= parsed <= 10_000_000:
                    values.append(parsed)
            except Exception:
                pass

        native = original_context_limit(engine)
        if native is not None:
            try:
                values.append(int(native))
            except Exception:
                pass
        return min(values) if values else None

    def _clamp_output_to_backend(self, prompt: str, requested: int) -> int:
        requested = max(1, int(requested))
        limit = _backend_context_limit(self.engine)
        if limit is None:
            return requested
        prompt_tokens = _token_count(self.engine.tokenizer, prompt)
        remaining = int(limit) - int(prompt_tokens)
        if remaining <= 0:
            raise RuntimeError(
                f"Formatted eval prompt requires {prompt_tokens:,} tokens but "
                f"{self.engine.backend_key} context is {limit:,}; refusing truncation."
            )
        return max(1, min(requested, remaining))

    def app_eval_init(self, max_duration_hours: float = 72.0):
        # Do not call the legacy constructor: it eagerly owns an MLX-only engine.
        self.max_duration_seconds = float(max_duration_hours) * 3600.0
        self.engine = AppEvalEngineAdapter(config, run_dir)
        self.provider = BenchmarkDatasetProvider()
        self.checkpoint_mgr = EvaluationCheckpointManager(
            os.path.join(run_dir, "eval_checkpoint_4000.json")
        )
        self.telemetry_file = os.path.join(run_dir, "telemetry_stream.jsonl")
        self._eval_target_model_label = str(
            self.engine.model_info.get("name")
            or self.engine.model_info.get("short_name")
            or "Bonsai 2 27B Ternary"
        )
        self._app_eval_backend_key = self.engine.backend_key
        self._app_eval_logical_model_key = _logical_model_key(self.engine)
        backend_adapter = str(
            getattr(self.engine.backend, "adapter_path", "") or ""
        )
        if backend_adapter:
            phase4_module.RSI_ADAPTER_PATH = backend_adapter
        os.makedirs(run_dir, exist_ok=True)
        print(
            f"[APP EVAL] model={self._eval_target_model_label} "
            f"backend={self._app_eval_backend_key} RAM={process_rss_mb():.0f} MB",
            flush=True,
        )

    def _backend(self):
        return self.engine.backend

    def _token_count(tokenizer, text: str) -> int:
        try:
            return len(tokenizer.encode(str(text)))
        except Exception:
            return max(1, len(str(text)) // 4)

    def _single_generate(self, prompt: str, max_tokens: int) -> str:
        _control_wait()
        backend = _backend(self)
        if backend is None:
            raise RuntimeError("App eval backend disappeared")
        pieces = []
        started = time.perf_counter()
        safe_max_tokens = _clamp_output_to_backend(self, prompt, max_tokens)
        try:
            iterator = backend.stream_generate_tokens(
                prompt,
                max_tokens=safe_max_tokens,
                temperature=float(EVAL_N1_TEMPERATURE),
                top_p=0.92,
            )
            for chunk in iterator:
                text = str(chunk or "")
                if text:
                    pieces.append(text)
                    print(text, end="", flush=True)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"{self.engine.backend_key} eval generation failed: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            print("", flush=True)
        output = "".join(pieces)
        if not output and str(self.engine.backend_key or "").lower() == "gguf":
            # Generic llama.cpp historically swallowed stream exceptions. Probe the
            # same backend once non-streaming so eval can distinguish empty EOS from
            # an actual runtime failure without fabricating an answer.
            fallback = backend.generate_branches(
                prompt,
                branch_count=1,
                max_tokens=safe_max_tokens,
                temperature=float(EVAL_N1_TEMPERATURE),
                top_p=0.92,
            )
            value = str(fallback[0] if fallback else "")
            if value.startswith("⚠️ GGUF generation error:"):
                raise RuntimeError(value)
            if value:
                output = value
                print(value, flush=True)

        elapsed = max(0.001, time.perf_counter() - started)
        generated = _token_count(self.engine.tokenizer, output) if output else 0
        self.last_output_tokens = generated
        self.live_generated_tokens = generated
        self.last_generation_seconds = elapsed
        self.last_tok_per_sec = generated / elapsed if generated else 0.0
        self.last_generation_error = None
        return output

    def generic_fast(self, prompt, max_tokens=16384, stream=False):
        del stream
        key = str(getattr(self, "_app_eval_backend_key", "") or "")
        if key == "mlx":
            return original_fast(self, prompt, max_tokens=max_tokens, stream=False)

        # Preserve Phase-4 Pro semantics from phase4_pro_rsi for non-MLX runtimes.
        if str(getattr(self, "_current_phase", "")).startswith("Phase 4"):
            identity = int(getattr(self, "_rsi_model_identity", id(self.engine.model)))
            logical_assert(self, identity, "Phase 4 Pro generation")
            router = phase4_module._pro_router(self)
            entropy = generic_entropy(self, prompt)
            split = str(getattr(self, "_current_split", ""))
            item = getattr(self, "_phase4_current_item", {}) or {}
            has_tests = phase4_module._has_answer_blind_verifier(split)
            mode, branch_count = router.route(entropy, has_test_cases=has_tests)
            temperatures = phase4_module.get_ladder_temperatures(branch_count)

            started = time.perf_counter()
            branches = generic_branches(
                self,
                prompt,
                temperatures,
                max_tokens=max_tokens,
                top_p=0.92,
            )
            if not branches:
                raise RuntimeError("Phase-4 Pro branch generation returned no candidates")
            winner, winning_idx, verified, selection = phase4_module._choose_without_ground_truth(
                self, split, item, branches
            )
            elapsed = max(0.001, time.perf_counter() - started)
            all_tokens = sum(_token_count(self.engine.tokenizer, value) for value in branches)
            selected_tokens = _token_count(self.engine.tokenizer, winner)
            self.last_output_tokens = selected_tokens
            self.live_generated_tokens = selected_tokens
            self.last_generation_seconds = elapsed
            self.last_tok_per_sec = all_tokens / elapsed if all_tokens else 0.0
            self.last_generation_error = None
            self._last_phase4_pro_meta = {
                "mode": mode,
                "entropy": float(entropy),
                "branch_count": len(branches),
                "temperatures": temperatures,
                "winning_branch": winning_idx + 1,
                "verified": verified,
                "selection": selection,
                "all_branch_tokens": all_tokens,
            }
            return winner

        return _single_generate(self, prompt, int(max_tokens))

    def generic_branches(self, formatted_prompt, temperatures, max_tokens, top_p=0.92):
        key = str(getattr(self, "_app_eval_backend_key", "") or "")
        if key == "mlx":
            return original_branch_generate(
                self, formatted_prompt, temperatures, max_tokens=max_tokens, top_p=top_p
            )
        _control_wait()
        backend = _backend(self)
        if backend is None:
            raise RuntimeError("App eval backend disappeared")
        temps = list(temperatures or [0.65])
        started = time.perf_counter()
        values = []
        # Generate one branch at a time so Pause/Cancel can take effect between
        # branches without changing branch temperatures or selection semantics.
        for temp in temps:
            _control_wait()
            safe_max_tokens = _clamp_output_to_backend(
                self, formatted_prompt, max_tokens
            )
            one = backend.generate_branches(
                formatted_prompt,
                branch_count=1,
                max_tokens=safe_max_tokens,
                temperature=float(temp),
                top_p=float(top_p),
            )
            if not one:
                raise RuntimeError("App eval backend returned no branch")
            value = str(one[0] or "")
            if value.startswith("⚠️ GGUF generation error:"):
                raise RuntimeError(value)
            values.append(value)
        elapsed = max(0.001, time.perf_counter() - started)
        total_tokens = sum(_token_count(self.engine.tokenizer, value) for value in values)
        self.last_tok_per_sec = total_tokens / elapsed if total_tokens else 0.0
        self.last_generation_seconds = elapsed
        self.last_output_tokens = _token_count(self.engine.tokenizer, values[0]) if values else 0
        for idx, value in enumerate(values, 1):
            print(f"\n[APP EVAL branch {idx}/{len(values)}]\n{value}", flush=True)
        return values

    def generic_entropy(self, prompt: str) -> float:
        key = str(getattr(self, "_app_eval_backend_key", "") or "")
        if key == "mlx":
            return original_entropy(self, prompt)
        backend = _backend(self)
        try:
            raw = float(backend.calculate_token_entropy(prompt))
        except Exception:
            raw = 0.35
        if raw > 1.0:
            raw = min(1.0, raw / 12.0)
        return max(0.0, min(1.0, raw))

    def generic_pro_backend(self):
        key = str(getattr(self, "_app_eval_backend_key", "") or "")
        if key == "mlx":
            return original_pro_backend(self)
        return _backend(self)

    def logical_assert(self, _identity: int, where: str):
        key = str(getattr(self, "_app_eval_backend_key", "") or "")
        if key == "mlx":
            return original_assert_same(self, _identity, where)
        backend = _backend(self)
        if backend is None or getattr(backend, "model", None) is None:
            raise RuntimeError(f"{where}: active app eval backend/model is no longer loaded")
        expected = str(getattr(self, "_app_eval_logical_model_key", "") or "")
        current = _logical_model_key(self.engine)
        if expected and current != expected:
            raise RuntimeError(
                f"{where}: logical model/backend changed ({expected!r} -> {current!r})"
            )

    def generic_restore(self) -> bool:
        key = str(getattr(self, "_app_eval_backend_key", "") or "")
        if key == "mlx":
            return original_restore(self)
        backend = _backend(self)
        path = str(getattr(backend, "adapter_path", "") or "")
        return bool(path and os.path.exists(path))

    def generic_phase3(self) -> Dict[str, Any]:
        key = str(getattr(self, "_app_eval_backend_key", "") or "")
        if key == "mlx":
            return original_phase3(self)

        _control_wait()
        backend = _backend(self)
        if backend is None:
            raise RuntimeError("Phase 3 has no active backend")
        ready = getattr(backend, "training_ready", None)
        if callable(ready) and not bool(ready()):
            raise RuntimeError(
                f"Phase 3 requires a real persistent training path; backend {key!r} is inference-only here"
            )
        trainer = getattr(backend, "train_mini_batch", None)
        if not callable(trainer):
            raise RuntimeError(f"Phase 3 backend {key!r} exposes no real train_mini_batch")

        memories = list(phase4_module._fetch_benchmark_training_memories(self))
        if not memories:
            print("[APP EVAL] Phase 3: no eligible Learn/RSI memories.", flush=True)
            return {"updated": False, "memories": 0, "fallback_updates": 0, "persisted": False}

        data = [
            {
                "prompt": str(row.get("prompt") or ""),
                "completion": str(row.get("completion") or ""),
            }
            for row in memories
        ]
        print(
            f"[APP EVAL] Phase 3: backend={key} memories={len(data)} "
            f"RAM={process_rss_mb():.0f} MB",
            flush=True,
        )
        started = time.perf_counter()
        try:
            meta, drift = trainer(
                adapters=getattr(backend, "adapters", {}) or {},
                data=data,
                fisher_matrix=None,
                lambda_ewc=0.0,
                learning_rate=1e-4,
                steps=1,
                save_path=getattr(backend, "adapter_path", None),
            )
        finally:
            # Backends already clean their own successful path; this outer guard
            # also covers early exceptions/custom controller failures.
            release_training_memory(backend)
        elapsed = max(0.001, time.perf_counter() - started)
        drift = float(drift)
        if drift <= 0.0:
            raise RuntimeError("Phase 3 backend training returned zero parameter drift")

        learn_ids = []
        rsi_ids = []
        for row in memories:
            if row.get("id") is None:
                continue
            if str(row.get("memory_kind") or "") == "rsi_self":
                rsi_ids.append(int(row["id"]))
            else:
                learn_ids.append(int(row["id"]))
        if learn_ids:
            self.engine.kg.mark_consolidated(learn_ids)
        if rsi_ids:
            marker = getattr(self.engine.kg, "mark_rsi_self_memories_consolidated", None)
            if callable(marker):
                marker(rsi_ids)

        adapter_path = str(getattr(backend, "adapter_path", "") or "")
        phase4_module.RSI_ADAPTER_PATH = adapter_path or phase4_module.RSI_ADAPTER_PATH
        persisted = bool(adapter_path and os.path.exists(adapter_path))
        if not persisted:
            raise RuntimeError(
                f"Phase 3 updated {key} weights but no persisted adapter/model artifact was found"
            )
        print(
            f"[APP EVAL] Phase 3 complete: {len(data)}/{len(data)} memories | "
            f"||ΔW||2={drift:.8f} | {elapsed:.1f}s | RAM={process_rss_mb():.0f} MB",
            flush=True,
        )
        return {
            "updated": True,
            "memories": len(data),
            "fallback_updates": 0,
            "persisted": True,
            "real_trainable_delta_l2": drift,
            "trainable_parameters_touched": int(
                (meta or {}).get("trainable_parameters_touched", 0)
            ) if isinstance(meta, dict) else 0,
        }

    runtime_module._model_context_limit = _backend_context_limit
    cls.__init__ = app_eval_init
    cls._fast_generate = generic_fast
    phase4_module._generate_branches_same_model = generic_branches
    phase4_module._normalized_entropy = generic_entropy
    phase4_module._pro_backend = generic_pro_backend
    phase4_module._assert_same_model = logical_assert
    phase4_module._restore_rsi_adapter = generic_restore
    phase4_module._run_phase3_consolidation = generic_phase3
    cls._app_cross_platform_bridge_installed = True
