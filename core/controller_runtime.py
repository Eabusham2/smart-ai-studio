"""Extensible controller-model runtime adapters.

This layer is intentionally metadata-driven.  A controller model may be a plain
text LM or a multimodal language model; generation stays in ProReasoningEngine,
while the adapter owns architecture-specific loading/inference/perception.

Existing MLX-LM/GGUF/BitNet paths remain the default fast paths.  This module is
used only when model metadata requests a specialized controller_runtime.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import re
import shutil
import time
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Generator, List, Optional, Tuple
from core.training_memory import release_training_memory


_RUNTIME_FAMILY_RESOLVERS: Dict[str, Any] = {}


def register_runtime_family(name: str, resolver) -> None:
    """Register an architecture-family -> concrete runtime resolver."""
    key = str(name or "").strip().lower()
    if not key or not callable(resolver):
        raise ValueError("runtime family registration requires a name and callable resolver")
    _RUNTIME_FAMILY_RESOLVERS[key] = resolver


def resolve_controller_runtime(model_info: Dict[str, Any], model_path: str) -> str:
    """Resolve explicit runtime metadata, then architecture family, then format."""
    info = dict(model_info or {})
    explicit = str(info.get("controller_runtime") or "").strip().lower()
    if explicit and explicit != "auto":
        return explicit

    family = str(info.get("runtime_family") or "").strip().lower()
    resolver = _RUNTIME_FAMILY_RESOLVERS.get(family)
    if resolver is not None:
        resolved = str(resolver(info, model_path) or "").strip().lower()
        if resolved:
            return resolved

    blob = " ".join(
        str(x or "").lower()
        for x in (
            model_path,
            info.get("repo_id"),
            info.get("precision"),
            family,
        )
    )
    if "gguf" in blob:
        return "gguf"
    if "bitnet" in blob:
        return "bitnet"
    modalities = {str(x).lower() for x in info.get("input_modalities") or ["text"]}
    if "mlx" in blob:
        return "mlx_vlm" if modalities - {"text"} else "mlx_lm"
    return "auto"


def _bonsai2_runtime(info: Dict[str, Any], model_path: str) -> str:
    blob = " ".join(str(x or "").lower() for x in (model_path, info.get("repo_id"), info.get("name")))
    if "gguf" in blob:
        return "gguf"
    if "jang" in blob:
        return "jang_vlm"
    return "mlx_repo_vlm"


register_runtime_family("bonsai2_hadamard", _bonsai2_runtime)


def resolve_local_snapshot(identifier: str) -> str:
    """Resolve a local path or an already-downloaded HF snapshot without networking."""
    value = os.path.abspath(os.path.expanduser(identifier)) if os.path.exists(os.path.expanduser(identifier)) else identifier
    if os.path.exists(value):
        return value
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id=identifier, local_files_only=True)
    except Exception:
        return identifier


def resolve_gguf_artifacts(identifier: str, model_info: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[str]]:
    """Resolve exact pinned GGUF language/projector artifacts, failing closed for built-ins."""
    info = model_info or {}
    root = resolve_local_snapshot(identifier)
    explicit_model = str(info.get("gguf_file") or "").strip()
    explicit_mmproj = str(info.get("mmproj_file") or "").strip()
    mmproj_repo_id = str(info.get("mmproj_repo_id") or "").strip()

    def exact_artifact(directory, relative_name):
        relative = PurePosixPath(relative_name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in relative_name or ":" in relative_name:
            raise RuntimeError(f"Invalid selected GGUF artifact path: {relative_name}")
        candidate = Path(directory).joinpath(*relative.parts)
        if candidate.is_file():
            return candidate
        # Legacy catalog entries store a basename. A unique match is required;
        # a relative path is never resolved as a different nested file.
        if len(relative.parts) == 1:
            matches = [p for p in Path(directory).rglob(relative.name) if p.is_file()]
            if len(matches) > 1:
                raise RuntimeError(f"Ambiguous GGUF artifact: {relative_name}. Select its full relative path.")
            if matches:
                return matches[0]
        return None

    if os.path.isfile(root):
        model_file = Path(root)
        if explicit_model and model_file.name != PurePosixPath(explicit_model).name:
            raise RuntimeError(
                f"Pinned GGUF mismatch: expected {explicit_model}, got {model_file.name}"
            )
    elif os.path.isdir(root):
        ggufs = [p for p in Path(root).rglob("*.gguf") if p.is_file()]
        if not ggufs:
            raise RuntimeError(f"No GGUF files found in cached snapshot for {identifier}")

        if explicit_model:
            selected = exact_artifact(root, explicit_model)
            if selected is None:
                raise RuntimeError(
                    f"Selected GGUF is missing: {explicit_model}. "
                    "Refusing to substitute another quant."
                )
            model_file = selected
            for required in info.get("gguf_files") or ():
                if exact_artifact(root, required) is None:
                    raise RuntimeError(f"Selected GGUF shard is missing: {required}")
        else:
            projectors = [
                p for p in ggufs
                if any(marker in p.name.lower() for marker in ("mmproj", "projector", "vision"))
            ]
            language = [p for p in ggufs if p not in projectors]
            preference = str(info.get("gguf_preference") or "").lower().strip()
            preferred = [p for p in language if preference and preference in p.name.lower()]
            pool = preferred or language
            if not pool:
                raise RuntimeError(f"No language-model GGUF found for {identifier}")
            model_file = max(pool, key=lambda p: p.stat().st_size)
    else:
        raise RuntimeError(
            f"GGUF repository is not downloaded locally: {identifier}. "
            "Use Grab/Install before loading."
        )

    mmproj: Optional[Path] = None
    if explicit_mmproj:
        projector_root_id = mmproj_repo_id or identifier
        if not mmproj_repo_id:
            projector_root = str(info.get("path") or (Path(root).parent if os.path.isfile(root) else root))
        else:
            projector_root = resolve_local_snapshot(projector_root_id)
        if os.path.isfile(projector_root):
            candidate = Path(projector_root)
            if candidate.name == PurePosixPath(explicit_mmproj).name:
                mmproj = candidate
        elif os.path.isdir(projector_root):
            mmproj = exact_artifact(projector_root, explicit_mmproj)
        if mmproj is None:
            raise RuntimeError(
                f"Pinned multimodal projector is missing: {explicit_mmproj} "
                f"from {projector_root_id}. Refusing a mismatched projector."
            )
    elif not info.get("gguf_files"):
        search_root = Path(root).parent if os.path.isfile(root) else Path(root)
        projectors = [
            p for p in search_root.rglob("*.gguf")
            if p.is_file() and any(marker in p.name.lower() for marker in ("mmproj", "projector", "vision"))
        ]
        if projectors:
            mmproj = max(projectors, key=lambda p: p.stat().st_size)

    return str(model_file), str(mmproj) if mmproj else None


class UniversalControllerBackend:
    """Protocol adapter for specialized text/multimodal controller runtimes."""

    def __init__(self, model_path: str, model_info: Dict[str, Any]):
        self.model_path = str(model_path)
        self.model_info = dict(model_info or {})
        self.runtime = str(self.model_info.get("controller_runtime") or "auto").lower()
        self.input_modalities = {
            str(x).lower() for x in (self.model_info.get("input_modalities") or ["text"])
        }
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.config = None
        self.runtime_module = None
        self.is_loaded = False
        self.is_mlx_available = False
        self.can_train = False
        self.adapters: Dict[str, Any] = {}
        self.adapter_path: Optional[str] = None
        self._generator = None
        self._apply_chat_template = None
        self._chat_config = None
        self.memory_placement: Dict[str, Any] = {}
        self._offload_directory: Optional[str] = None

    def load_model(self) -> bool:
        if self.runtime in ("mlx_vlm", "mlx-vlm"):
            return self._load_mlx_vlm()
        if self.runtime in ("jang_vlm", "jang-vlm"):
            return self._load_jang_vlm()
        if self.runtime in ("mlx_repo_vlm", "mlx-repo-vlm", "repo_mlx_vlm"):
            return self._load_repo_mlx_vlm()
        if self.runtime in ("transformers_auto", "transformers", "hf_transformers"):
            if self.model is not None or self._offload_directory:
                self.unload_model()
                if self._offload_directory:
                    raise RuntimeError("Previous Transformers offload files are still in use; retry unloading before loading another model.")
            try:
                return self._load_transformers_auto()
            except Exception as exc:
                failure = str(exc)
            # Leave the exception scope first: its traceback can retain the model
            # and Accelerate hook cycles. Release those before deleting mapped
            # files, especially on Windows, and reset partial processor state.
            self.unload_model()
            raise RuntimeError(failure) from None
        raise RuntimeError(f"Unsupported specialized controller runtime: {self.runtime}")

    def _load_mlx_vlm(self) -> bool:
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template

        source = resolve_local_snapshot(self.model_path)
        loaded = load(source)
        if not isinstance(loaded, tuple) or len(loaded) < 2:
            raise RuntimeError("mlx-vlm loader did not return model + processor")
        self.model, self.processor = loaded[:2]
        self.config = getattr(self.model, "config", None)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self._generator = generate
        self._apply_chat_template = apply_chat_template
        self.is_loaded = self.model is not None and self.processor is not None
        self.is_mlx_available = self.is_loaded
        return self.is_loaded

    def _load_jang_vlm(self) -> bool:
        # JANG-Hadamard has input-side rotations and mixed packed/float weights.
        # The generic MLX affine QuantizedLinear route raises uint32/float32
        # errors (or silently generates incorrect output) for this checkpoint.
        try:
            from vmlx_engine.utils.jang_loader import load_jang_vlm_model
        except ImportError as exc:
            raise RuntimeError(
                "This CRACK JANG-Hadamard model requires the compatible vMLX "
                "loader (vmlx >= 1.6.84 on Python 3.11+ / Apple Silicon). "
                "The stock MLX and older generic JANG loader cannot safely "
                "execute this mixed packed checkpoint."
            ) from exc
        source = resolve_local_snapshot(self.model_path)
        loaded = load_jang_vlm_model(source)
        if not isinstance(loaded, tuple) or len(loaded) < 2:
            raise RuntimeError("JANG VLM loader did not return model + processor")
        self.model, self.processor = loaded[:2]
        self.config = loaded[2] if len(loaded) > 2 else getattr(self.model, "config", None)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        self._generator = generate
        self._apply_chat_template = apply_chat_template
        self.is_loaded = self.model is not None and self.processor is not None
        self.is_mlx_available = self.is_loaded
        return self.is_loaded

    def _load_repo_mlx_vlm(self) -> bool:
        source = resolve_local_snapshot(self.model_path)
        if not os.path.isdir(source):
            raise RuntimeError("Specialized MLX-VLM runtime requires the downloaded model snapshot")

        runtime_dir = Path(source) / str(self.model_info.get("runtime_dir") or "runtime")
        module_name = str(self.model_info.get("runtime_module") or "vision_artifact")
        loader_name = str(self.model_info.get("runtime_loader") or "load_vl_model")
        chat_config_name = str(self.model_info.get("runtime_chat_config") or "chat_config")
        module_file = runtime_dir / (module_name.replace(".", os.sep) + ".py")
        if not module_file.is_file():
            raise RuntimeError(f"Bundled runtime module not found: {module_file}")

        unique_name = "_smart_ai_runtime_" + re.sub(r"[^a-zA-Z0-9_]", "_", module_name)
        spec = importlib.util.spec_from_file_location(unique_name, module_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import bundled runtime: {module_file}")
        module = importlib.util.module_from_spec(spec)
        from core.repository_runtime_imports import repository_runtime_imports
        old_path = list(sys.path)
        try:
            sys.path.insert(0, str(runtime_dir))
            # vision_artifact imports `Packed` from its sibling runtime.py,
            # not from an unrelated installed module named `runtime`.
            with repository_runtime_imports(runtime_dir):
                spec.loader.exec_module(module)
                loader = getattr(module, loader_name, None)
                if not callable(loader):
                    raise RuntimeError(f"Bundled runtime has no callable {loader_name}()")
                if module_name == "vision_artifact":
                    from core.mlx_repo_config_compat import compatible_qwen35_configs
                    with compatible_qwen35_configs():
                        loaded = loader(source)
                else:
                    loaded = loader(source)
        finally:
            sys.path[:] = old_path
        if not isinstance(loaded, tuple) or len(loaded) < 2:
            raise RuntimeError("Bundled runtime loader did not return model + processor")
        self.model, self.processor = loaded[:2]
        self.config = loaded[2] if len(loaded) > 2 else getattr(self.model, "config", None)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.runtime_module = module
        self._chat_config = getattr(module, chat_config_name, None)

        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        self._generator = generate
        self._apply_chat_template = apply_chat_template
        self.is_loaded = self.model is not None and self.processor is not None
        self.is_mlx_available = self.is_loaded

        # A repository runtime may opt into the existing real MLX trainer only if
        # it actually exposes the required training methods. Never fake capability.
        self.can_train = all(
            callable(getattr(self, name, None))
            for name in ("train_mini_batch", "compute_mlx_fisher")
        )
        return self.is_loaded

    def _load_transformers_auto(self) -> bool:
        import torch
        import transformers
        from core.transformers_placement import (
            describe_transformers_placement, estimate_transformers_kv_bytes, transformers_memory_limits,
        )
        from config.paths import get_portable_data_dir
        import hashlib

        from core.learning_state import cached_model_source
        source = str(cached_model_source(self.model_path, self.model_info))
        revision = (self.model_info.get("resolved_revision") or self.model_info.get("repo_revision")
                    or self.model_info.get("source_revision") or self.model_info.get("revision"))
        source_options = {"revision": revision} if revision and not os.path.exists(source) else {}
        context_tokens = int(getattr(self, "context_tokens", 0) or self.model_info.get("default_context") or 32768)
        kv_bytes = estimate_transformers_kv_bytes(source, context_tokens)
        budgets = transformers_memory_limits(torch, getattr(self, "speed_memory_limit_gb", None), kv_bytes=kv_bytes)
        offload_root = Path(get_portable_data_dir()) / "runtime" / "transformers-offload"
        offload_root.mkdir(parents=True, exist_ok=True)
        source_key = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:16]
        self._offload_directory = tempfile.mkdtemp(prefix=f"{source_key}-", dir=offload_root)
        processor = None
        try:
            processor = transformers.AutoProcessor.from_pretrained(source, trust_remote_code=True, **source_options)
        except Exception:
            processor = None
        if processor is None:
            processor = transformers.AutoTokenizer.from_pretrained(source, trust_remote_code=True, **source_options)

        requested = str(self.model_info.get("transformers_auto_class") or "").strip()
        candidates = [requested] if requested else []
        candidates += [
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForSpeechSeq2Seq",
            "AutoModelForCausalLM",
            "AutoModelForSeq2SeqLM",
        ]
        model = None
        errors = []
        seen = set()
        for class_name in candidates:
            if not class_name or class_name in seen:
                continue
            seen.add(class_name)
            cls = getattr(transformers, class_name, None)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(
                    source,
                    torch_dtype="auto",
                    device_map="auto",
                    max_memory=budgets,
                    offload_folder=self._offload_directory,
                    offload_state_dict=True,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True, **source_options,
                )
                break
            except Exception as exc:
                errors.append(f"{class_name}: {exc}")
        if model is None:
            raise RuntimeError("No compatible Transformers AutoModel loader succeeded: " + " | ".join(errors[-3:]))

        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.config = getattr(model, "config", None)
        self.memory_placement = describe_transformers_placement(model, budgets)
        self.memory_placement.update({"context_tokens": context_tokens, "kv_bytes_estimate": kv_bytes})

        # Persistent controller LoRA lives beside other portable app state. Reload it
        # transactionally on every model load so learned chat state survives restarts.
        key = hashlib.sha256(str(self.model_path).encode("utf-8")).hexdigest()[:16]
        eval_root = str(os.getenv("SMARTAI_CONTROLLER_ADAPTER_ROOT", "") or "").strip()
        from core.learning_state import bind_learning_state
        legacy_root = os.path.abspath(eval_root) if eval_root else os.path.join(get_portable_data_dir(), "controller_lora")
        legacy_path = os.path.join(legacy_root, key)
        state_root = os.path.join(legacy_root, "learned_memory")
        store = bind_learning_state(self, model_path=source, model_info=self.model_info,
                                    state_root=state_root, adapter_format="peft")
        self._loaded_learning_checkpoint = self.adapter_path = store.current_checkpoint()
        model_scope = getattr(getattr(store, "scope", None), "kind", "model") == "model"
        if model_scope and self.adapter_path is None and os.path.isfile(os.path.join(legacy_path, "adapter_config.json")):
            selected = self.model_info.get("legacy_adapter_path")
            if not selected or os.path.realpath(selected) != os.path.realpath(legacy_path):
                raise RuntimeError("Existing controller LoRA has no base-model fingerprint; refusing to omit or apply it. "
                                   "Explicitly select the legacy adapter for this base to migrate it: " + legacy_path)
            self.adapter_path = legacy_path
        if self.adapter_path:
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(
                    model, self.adapter_path,
                    is_trainable=not self.memory_placement["inference_only"],
                    device_map="auto", max_memory=budgets,
                    offload_folder=self._offload_directory,
                )
            except Exception as exc:
                raise RuntimeError(f"The saved adapter could not be restored; refusing to omit it: {exc}") from exc
            # PEFT may dispatch again after adding adapter tensors. Report its
            # actual final map and retain inference-only protection if it spills.
            self.memory_placement = describe_transformers_placement(model, budgets)
            self.memory_placement.update({"context_tokens": context_tokens, "kv_bytes_estimate": kv_bytes})

        model.eval()
        self.model = model
        self.is_loaded = True
        self.is_mlx_available = False
        self.can_train = self.training_ready()
        return True

    def training_ready(self) -> bool:
        """True only when this loaded controller has a real persistent update path."""
        if self.model is None or self.tokenizer is None:
            return False
        if getattr(self, "memory_placement", {}).get("inference_only"):
            return False
        if self.runtime not in ("transformers_auto", "transformers", "hf_transformers"):
            return bool(self.can_train)
        if getattr(self.config, "is_encoder_decoder", False):
            return False
        try:
            from peft import PeftModel
            return isinstance(self.model, PeftModel) or bool(self._transformers_lora_targets(self.model))
        except Exception:
            return False

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        text = "\n".join(str(m.get("content", "")) for m in messages or [])
        tok = self.tokenizer
        try:
            encode = getattr(tok, "encode", None)
            if callable(encode):
                return len(encode(text))
            result = tok(text, return_tensors=None)
            ids = result.get("input_ids") if isinstance(result, dict) else getattr(result, "input_ids", None)
            if ids is not None:
                return len(ids[0] if ids and isinstance(ids[0], (list, tuple)) else ids)
        except Exception:
            pass
        return max(1, len(text) // 4)

    @staticmethod
    def _transformers_lora_targets(model) -> List[str]:
        import torch
        from transformers.pytorch_utils import Conv1D
        suffixes = {
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
            "query_key_value", "dense_h_to_4h", "dense_4h_to_h",
            "c_attn", "c_proj",
        }
        blocked = (
            "vision", "visual", "image", "audio", "speech",
            "projector", "mm_projector", "vision_tower", "mtp",
        )
        targets: List[str] = []
        for name, module in model.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if leaf not in suffixes or not isinstance(module, (torch.nn.Linear, Conv1D)):
                continue
            low = name.lower()
            if any(marker in low for marker in blocked):
                continue
            targets.append(name)
        return sorted(set(targets))

    def _ensure_transformers_lora(self):
        if self.runtime not in ("transformers_auto", "transformers", "hf_transformers"):
            raise RuntimeError("Controller runtime has no generic PEFT training bridge")
        if not self.training_ready():
            raise RuntimeError("PEFT/Torch training runtime is unavailable for this controller")

        from peft import LoraConfig, PeftModel, get_peft_model
        if isinstance(self.model, PeftModel):
            return self.model

        targets = self._transformers_lora_targets(self.model)
        if not targets:
            raise RuntimeError(
                "No compatible language projection modules were found for controller LoRA; "
                "refusing to adapt vision/audio towers or fabricate support."
            )
        cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=targets,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, cfg)
        return self.model

    def train_mini_batch(
        self,
        adapters: Any,
        data: List[Dict[str, str]],
        fisher_matrix: Any = None,
        lambda_ewc: float = 0.0,
        learning_rate: float = 1e-4,
        steps: int = 3,
        save_path: Optional[str] = None,
        cancel_event=None, validation_callback=None, learning_metadata=None, anchor_texts=None,
        reward_weighting="group_relative", entropic_kl_target=None, reference_kl_coef=0.0,
        objective='causal_lm', supervise_prompt=True, max_window_tokens=256, row_weights=None,
        context_text=None, max_context_tokens=2048, **_kwargs,
    ):
        """Validate the live PEFT candidate before publishing its checkpoint generation."""
        import torch
        from core.torch_context_learning import check_cancelled, default_validation, lora_parameters, update_lora, validate_candidate, verify_saved_lora

        if objective == "reward_policy":
            from core.reward_adaptation import reward_advantages
            reward_advantages(data, reward_weighting=reward_weighting, entropic_kl_target=entropic_kl_target)
        store = self.learning_state
        txn = store.begin(metadata=learning_metadata)
        before = {}
        base_trainability = []
        original_model = self.model
        model = None
        original_path = self.adapter_path
        try:
            if store.current_checkpoint() != self._loaded_learning_checkpoint:
                raise RuntimeError("Reload the latest committed controller checkpoint before learning")
            check_cancelled(cancel_event)
            if validation_callback is None:
                validation_callback = default_validation(self, data, objective=objective, reward_weighting=reward_weighting, entropic_kl_target=entropic_kl_target)
            if not getattr(original_model, "peft_config", None):
                base_trainability = [(p, p.requires_grad) for p in original_model.parameters()]
            model = self._ensure_transformers_lora()
            named = lora_parameters(model)
            before = {name: param.detach().cpu().clone() for name, param in named.items()}
            metrics, drift = update_lora(
                model, self.tokenizer, data, learning_rate=learning_rate, steps=steps, max_tokens=max_window_tokens,
                reward_weighting=reward_weighting, entropic_kl_target=entropic_kl_target, reference_kl_coef=reference_kl_coef,
                objective=objective, supervise_prompt=supervise_prompt, row_weights=row_weights,
                context_text=context_text, max_context_tokens=max_context_tokens,
                fisher_matrix=fisher_matrix, lambda_ewc=lambda_ewc, cancel_event=cancel_event, before=before,
                anchor_texts=anchor_texts, output_directory=txn.directory,
                previous_directory=Path(self._loaded_learning_checkpoint).parent if self._loaded_learning_checkpoint else None,
            )
            evidence = validate_candidate(self, validation_callback, cancel_event)
            tmp = txn.directory / "peft"
            model.save_pretrained(tmp, safe_serialization=True, save_embedding_layers=False)
            verify_saved_lora(model, tmp)
            check_cancelled(cancel_event)
            metrics["adapter_format"] = "peft-lora"
            txn.commit("peft", metadata={"metrics": metrics, "validation": evidence, "drift_l2": drift}, cancel_event=cancel_event)
            return dict(metrics), drift
        except BaseException:
            if before and not txn.committed:
                with torch.no_grad():
                    for name, param in named.items():
                        param.copy_(before[name].to(device=param.device, dtype=param.dtype))
                self.adapter_path = original_path
            if not txn.committed and model is not None and model is not original_model:
                # PEFT injects layers into the original module in-place. Undo
                # that first injection as well as its factors on a rejected trial.
                self.model = model.unload()
                for p, was_trainable in base_trainability:
                    p.requires_grad_(was_trainable)
            raise
        finally:
            if txn.committed:
                # Publication is authoritative even if interrupted during
                # post-commit cleanup. The live tensors are already validated.
                self._loaded_learning_checkpoint = self.adapter_path = str(tmp)
                self.adapters = metrics
                self.last_learning_metrics = dict(metrics)
            txn.abort()
            before.clear()
            if model is not None:
                model.eval()
            release_training_memory(self)

    def score_learning_probe(self, prompt, completion):
        from core.native_speed import runtime_lock
        from core.torch_context_learning import score_torch_probe
        with runtime_lock(self):
            return score_torch_probe(self.model, self.tokenizer, prompt, completion)

    @staticmethod
    def _text_from_generation(value: Any) -> str:
        if isinstance(value, str):
            return value
        text = getattr(value, "text", None)
        if isinstance(text, str):
            return text
        return str(value or "")

    def _mlx_prompt(self, prompt: str, image_count: int = 0) -> str:
        if not callable(self._apply_chat_template):
            return prompt
        kwargs = {}
        if callable(self._chat_config):
            try:
                kwargs["config"] = self._chat_config(self.config)
            except Exception:
                pass
        try:
            if "config" in kwargs:
                return self._apply_chat_template(
                    self.processor,
                    kwargs["config"],
                    prompt,
                    num_images=image_count,
                )
            return self._apply_chat_template(
                self.processor,
                self.config,
                prompt,
                num_images=image_count,
            )
        except Exception:
            return prompt

    def _generate_mlx(self, prompt: str, images: Optional[List[str]], max_tokens: int, temperature: float) -> str:
        if not callable(self._generator):
            raise RuntimeError("MLX-VLM generator unavailable")
        prepared = self._mlx_prompt(prompt, len(images or []))
        started = time.perf_counter()
        value = self._generator(
            self.model,
            self.processor,
            prepared,
            images or [],
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        self.last_generation_seconds = max(1e-9, time.perf_counter() - started)
        # Use the backend's actual generation counters when supplied; do not
        # derive token counts from a number of words or emitted text chunks.
        count = getattr(value, "generation_tokens", None)
        self.last_generation_tokens = count if type(count) is int and count >= 0 else None
        measured = getattr(value, "generation_tps", None)
        self.last_tok_per_sec = (
            float(measured) if type(measured) in (float, int)
            and math.isfinite(measured) and measured > 0 else None
        )
        return self._text_from_generation(value).strip()

    def _sample_video_frames(self, path: str, max_frames: int = 8) -> List[str]:
        """Extract a small ordered frame set so any image-capable controller can inspect video."""
        import imageio.v3 as iio
        from PIL import Image

        frames = []
        try:
            meta = iio.immeta(path)
            total = int(meta.get("nframes") or meta.get("n_images") or 0)
        except Exception:
            total = 0

        arrays = []
        try:
            if total > 0:
                indices = sorted({int(round(i * max(0, total - 1) / max(1, max_frames - 1))) for i in range(max_frames)})
                for idx in indices:
                    try:
                        arrays.append(iio.imread(path, index=idx))
                    except Exception:
                        pass
            else:
                for idx, frame in enumerate(iio.imiter(path)):
                    if idx >= max_frames:
                        break
                    arrays.append(frame)
        except Exception as exc:
            raise RuntimeError(f"Could not decode video frames: {exc}") from exc

        if not arrays:
            raise RuntimeError("Video contained no decodable frames")

        temp_dir = tempfile.mkdtemp(prefix="smart-ai-video-")
        for idx, array in enumerate(arrays):
            out = os.path.join(temp_dir, f"frame-{idx:02d}.jpg")
            Image.fromarray(array).convert("RGB").save(out, quality=90)
            frames.append(out)
        return frames

    def _generate_transformers(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        media: Optional[Dict[str, Any]] = None,
    ) -> str:
        import torch
        kwargs: Dict[str, Any] = {"text": prompt, "return_tensors": "pt"}
        media = media or {}
        if "image" in media:
            from PIL import Image
            kwargs["images"] = Image.open(media["image"]).convert("RGB")
        if "audio" in media:
            import soundfile as sf
            audio, sr = sf.read(media["audio"])
            kwargs["audio"] = audio
            kwargs["sampling_rate"] = sr
        if "video" in media:
            import imageio.v3 as iio
            frames = []
            for idx, frame in enumerate(iio.imiter(media["video"])):
                if idx >= 8:
                    break
                frames.append(frame)
            if not frames:
                raise RuntimeError("Video contained no decodable frames")
            kwargs["videos"] = [frames]
        try:
            inputs = self.processor(**kwargs)
        except TypeError:
            # Some multimodal processors expose video as a sequence of images.
            videos = kwargs.pop("videos", None)
            if videos:
                kwargs["images"] = videos[0]
            inputs = self.processor(**kwargs)
        from core.transformers_placement import transformers_input_device
        device = transformers_input_device(self.model)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)
        do_sample = float(temperature) > 0.0
        started = time.perf_counter()
        with torch.no_grad():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                do_sample=do_sample,
                temperature=max(float(temperature), 1e-5) if do_sample else None,
            )
        elapsed = max(1e-9, time.perf_counter() - started)
        self.last_generation_seconds = elapsed
        try:
            prompt_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
            prompt_length = int(prompt_ids.shape[-1]) if prompt_ids is not None else 0
            self.last_generation_tokens = max(0, int(ids.shape[-1]) - prompt_length)
            self.last_tok_per_sec = (self.last_generation_tokens / elapsed
                                     if self.last_generation_tokens > 0 else None)
        except (AttributeError, TypeError, ValueError):
            self.last_generation_tokens = None
            self.last_tok_per_sec = None
        decoder = getattr(self.processor, "batch_decode", None) or getattr(self.tokenizer, "batch_decode", None)
        if callable(decoder):
            return str(decoder(ids, skip_special_tokens=True)[0]).strip()
        return str(ids)

    def _generate(self, prompt: str, max_tokens: int, temperature: float, media: Optional[Dict[str, Any]] = None) -> str:
        if self.runtime.startswith("mlx") or self.runtime in ("jang_vlm", "jang-vlm", "repo_mlx_vlm"):
            images = [media["image"]] if media and media.get("image") else []
            return self._generate_mlx(prompt, images, max_tokens, temperature)
        if self.runtime in ("transformers_auto", "transformers", "hf_transformers"):
            return self._generate_transformers(prompt, max_tokens, temperature, media)
        raise RuntimeError(f"Generation unavailable for runtime {self.runtime}")

    def stream_generate_tokens(
        self, prompt: str, max_tokens: int = 1536, temperature: float = 0.65,
        top_p: float = 0.92, stop_at_tool_boundary: bool = False, cancelled=None,
    ) -> Generator[str, None, None]:
        if self.runtime.startswith("mlx") or self.runtime in ("jang_vlm", "jang-vlm", "repo_mlx_vlm"):
            from mlx_vlm.generate import stream_generate
            from core.generation_metrics import TokenMeter, observe_mlx_response
            from core.tool_boundary_stop import ToolBoundaryStopper
            meter = TokenMeter(self, "controller-mlx")
            detector = ToolBoundaryStopper() if stop_at_tool_boundary else None
            iterator = stream_generate(self.model, self.processor,
                                       prompt=self._mlx_prompt(prompt, 0), image=[],
                                       max_tokens=int(max_tokens), temperature=float(temperature), top_p=float(top_p))
            try:
                for response in iterator:
                    if callable(cancelled) and cancelled():
                        return
                    observe_mlx_response(meter, response)
                    piece = str(getattr(response, "text", ""))
                    stop_here = False
                    if detector is not None:
                        offset = len(detector.text)
                        stop_here = detector.feed(piece)
                        piece = detector.text[offset:]
                    if piece:
                        yield piece
                    if stop_here:
                        return
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
                meter.finish()
            return
        if self.runtime in ("transformers_auto", "transformers", "hf_transformers"):
            from core.transformers_stream import stream_transformers
            from core.transformers_placement import transformers_input_device
            inputs = self.processor(text=prompt, return_tensors="pt")
            device = transformers_input_device(self.model)
            if device is not None and hasattr(inputs, "to"):
                inputs = inputs.to(device)
            do_sample = float(temperature) > 0
            options = {"max_new_tokens": int(max_tokens), "do_sample": do_sample,
                       "temperature": max(float(temperature), 1e-5) if do_sample else None}
            # Preserve this controller's original sampling policy: its prior
            # generate() did not override top_p with the Pro value.
            yield from stream_transformers(self.model, self.tokenizer, inputs, options,
                                          owner=self, stop_at_tool_boundary=stop_at_tool_boundary,
                                          cancelled=cancelled)
            return
        raise RuntimeError(f"Generation unavailable for runtime {self.runtime}")

    def generate_branches(
        self, prompt: str, branch_count: int = 1, max_tokens: int = 1536,
        temperature: Any = 0.65, top_p: float = 0.92, progress_callback=None,
        stop_at_tool_boundary: bool = False, cancelled=None,
    ) -> List[str]:
        out = []
        count = max(1, int(branch_count))
        for idx in range(count):
            if callable(cancelled) and cancelled():
                return out
            t = float(temperature[idx % len(temperature)]) if isinstance(temperature, (list, tuple)) else float(temperature)
            started = time.perf_counter()
            if stop_at_tool_boundary or cancelled is not None:
                pieces = []
                iterator = self.stream_generate_tokens(prompt, max_tokens, t, top_p,
                                                       stop_at_tool_boundary=stop_at_tool_boundary, cancelled=cancelled)
                try:
                    for piece in iterator:
                        pieces.append(piece)
                        if progress_callback is not None:
                            progress_callback(idx, count, getattr(self, "last_generation_tokens", None),
                                              time.perf_counter() - started, finished=False)
                        if callable(cancelled) and cancelled():
                            break
                finally:
                    close = getattr(iterator, "close", None)
                    if callable(close):
                        close()
                out.append("".join(pieces))
            else:
                out.append(self._generate(prompt, max_tokens, t))
            if progress_callback is not None:
                observed = getattr(self, "last_generation_tokens", None)
                progress_callback(idx + 1, count,
                                  observed if type(observed) is int and observed >= 0 else None,
                                  time.perf_counter() - started)
        return out

    def calculate_token_entropy(self, prompt: str) -> float:
        del prompt
        # Specialized runtimes often do not expose stable logits APIs.  Returning a
        # neutral normalized entropy keeps routing functional without a second forward.
        return 0.35

    def supports_media_input(self, kind: str) -> bool:
        kind = str(kind or "").lower()
        if kind not in self.input_modalities:
            return False
        if self.runtime.startswith("mlx") or self.runtime in ("jang_vlm", "jang-vlm", "repo_mlx_vlm"):
            return kind in {"image", "video"}
        if self.runtime in ("transformers_auto", "transformers", "hf_transformers"):
            return kind in {"image", "audio", "video"}
        return False

    def review_media_input(self, path: str, kind: str, prompt: str = "") -> Dict[str, Any]:
        kind = str(kind or "").lower()
        if not self.supports_media_input(kind):
            return {
                "perception_available": False,
                "reason": f"{self.runtime} cannot ingest {kind} with this model.",
            }
        if not os.path.isfile(path):
            raise ValueError("Media input file is missing")

        task = str(prompt or "").strip() or f"Describe the supplied {kind}."
        review = (
            task
            + "\nReturn JSON only with keys description, score, reasoning. "
              "score must be 0 to 100 and must judge only the supplied media."
        )
        cleanup_dir = None
        try:
            if kind == "video" and (self.runtime.startswith("mlx") or self.runtime in ("jang_vlm", "jang-vlm")):
                frames = self._sample_video_frames(path)
                cleanup_dir = os.path.dirname(frames[0]) if frames else None
                prepared = self._mlx_prompt(review, len(frames))
                value = self._generator(
                    self.model,
                    self.processor,
                    prepared,
                    frames,
                    max_tokens=768,
                    temperature=0.0,
                )
                text = self._text_from_generation(value).strip()
            else:
                media = {kind: path}
                text = self._generate(review, 768, 0.0, media)
        finally:
            if cleanup_dir:
                import shutil
                shutil.rmtree(cleanup_dir, ignore_errors=True)
        match = re.search(r"\{[\s\S]*\}", text)
        parsed: Dict[str, Any] = {}
        if match:
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                parsed = {}
        score = parsed.get("score")
        result = {
            "perception_available": True,
            "description": str(parsed.get("description") or text),
            "analysis": str(parsed.get("reasoning") or text),
        }
        if isinstance(score, (int, float)):
            result["score"] = max(0.0, min(100.0, float(score)))
        else:
            result["reason"] = "Model inspected the media but did not return a numeric self-grade."
        return result

    def _clear_offload_files(self) -> None:
        directory = getattr(self, "_offload_directory", None)
        if not directory:
            return
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass
        except OSError:
            # A mapped file can still be closing on Windows. Retain the owned
            # directory so a later unload can retry; never remove model sources.
            return
        self._offload_directory = None

    def unload_model(self) -> None:
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.config = None
        self.runtime_module = None
        self.is_loaded = False
        self.is_mlx_available = False
        self.can_train = False
        try:
            import mlx.core as mx
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass
        if getattr(self, "_offload_directory", None):
            release_training_memory()
            self._clear_offload_files()
        self.memory_placement = {}
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# PEFT optimization, teacher-forced probes, serving and unload share one reentrant lock.
from core.native_speed import install_native_runtime_lock
install_native_runtime_lock(UniversalControllerBackend)
