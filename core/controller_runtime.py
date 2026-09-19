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
import os
import re
import sys
import tempfile
from pathlib import Path
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

    if os.path.isfile(root):
        model_file = Path(root)
        if explicit_model and model_file.name != explicit_model:
            raise RuntimeError(
                f"Pinned GGUF mismatch: expected {explicit_model}, got {model_file.name}"
            )
    elif os.path.isdir(root):
        ggufs = [p for p in Path(root).rglob("*.gguf") if p.is_file()]
        if not ggufs:
            raise RuntimeError(f"No GGUF files found in cached snapshot for {identifier}")

        if explicit_model:
            candidates = [p for p in ggufs if p.name == explicit_model]
            if not candidates:
                raise RuntimeError(
                    f"Pinned ternary GGUF is missing: {explicit_model}. "
                    "Refusing to substitute another quant."
                )
            model_file = candidates[0]
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
        projector_root = resolve_local_snapshot(projector_root_id)
        if os.path.isfile(projector_root):
            candidate = Path(projector_root)
            if candidate.name == explicit_mmproj:
                mmproj = candidate
        elif os.path.isdir(projector_root):
            candidates = [
                p for p in Path(projector_root).rglob("*.gguf")
                if p.is_file() and p.name == explicit_mmproj
            ]
            if candidates:
                mmproj = candidates[0]
        if mmproj is None:
            raise RuntimeError(
                f"Pinned multimodal projector is missing: {explicit_mmproj} "
                f"from {projector_root_id}. Refusing a mismatched projector."
            )
    else:
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

    def load_model(self) -> bool:
        if self.runtime in ("mlx_vlm", "mlx-vlm"):
            return self._load_mlx_vlm()
        if self.runtime in ("jang_vlm", "jang-vlm"):
            return self._load_jang_vlm()
        if self.runtime in ("mlx_repo_vlm", "mlx-repo-vlm", "repo_mlx_vlm"):
            return self._load_repo_mlx_vlm()
        if self.runtime in ("transformers_auto", "transformers", "hf_transformers"):
            return self._load_transformers_auto()
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
        from jang_tools.loader import load_jang_vlm_model
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
        old_path = list(sys.path)
        try:
            sys.path.insert(0, str(runtime_dir))
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = old_path

        loader = getattr(module, loader_name, None)
        if not callable(loader):
            raise RuntimeError(f"Bundled runtime has no callable {loader_name}()")
        loaded = loader(source)
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

        source = resolve_local_snapshot(self.model_path)
        processor = None
        try:
            processor = transformers.AutoProcessor.from_pretrained(source, trust_remote_code=True)
        except Exception:
            processor = None
        if processor is None:
            processor = transformers.AutoTokenizer.from_pretrained(source, trust_remote_code=True)

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
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
                    device_map="auto",
                    trust_remote_code=True,
                )
                break
            except Exception as exc:
                errors.append(f"{class_name}: {exc}")
        if model is None:
            raise RuntimeError("No compatible Transformers AutoModel loader succeeded: " + " | ".join(errors[-3:]))

        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.config = getattr(model, "config", None)

        # Persistent controller LoRA lives beside other portable app state. Reload it
        # transactionally on every model load so learned chat state survives restarts.
        try:
            from config.paths import get_portable_data_dir
            import hashlib
            key = hashlib.sha256(str(self.model_path).encode("utf-8")).hexdigest()[:16]
            eval_root = str(os.getenv("SMARTAI_CONTROLLER_ADAPTER_ROOT", "") or "").strip()
            self.adapter_path = (
                os.path.join(os.path.abspath(eval_root), key)
                if eval_root
                else os.path.join(get_portable_data_dir(), "controller_lora", key)
            )
            adapter_cfg = os.path.join(self.adapter_path, "adapter_config.json")
            if os.path.isfile(adapter_cfg):
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, self.adapter_path, is_trainable=True)
        except Exception:
            # Inference remains available if the optional PEFT runtime is unavailable;
            # training_ready() will correctly report False.
            pass

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
        if self.runtime not in ("transformers_auto", "transformers", "hf_transformers"):
            return bool(self.can_train)
        try:
            import torch  # noqa: F401
            import peft  # noqa: F401
            return True
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
        suffixes = {
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
            "query_key_value", "dense_h_to_4h", "dense_4h_to_h",
            "c_attn", "c_proj",
        }
        blocked = (
            "vision", "visual", "image", "audio", "speech",
            "projector", "mm_projector", "vision_tower",
        )
        targets: List[str] = []
        for name, module in model.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if leaf not in suffixes or not isinstance(module, torch.nn.Linear):
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
        **_kwargs,
    ):
        """Real transactional completion-only PEFT update for Transformers controllers."""
        del adapters, fisher_matrix, lambda_ewc, save_path
        import math
        import shutil
        import torch

        model = self._ensure_transformers_lora()
        tok = self.tokenizer
        rows = [
            item for item in (data or [])
            if str(item.get("prompt") or "").strip()
            and str(item.get("completion") or "").strip()
        ]
        if not rows:
            raise RuntimeError("Controller trainer received no prompt/completion pairs")

        before = {
            name: param.detach().float().cpu().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        params = [param for param in model.parameters() if param.requires_grad]
        if not params:
            raise RuntimeError("Controller LoRA exposes no trainable parameters")

        optimizer = torch.optim.AdamW(params, lr=float(learning_rate))
        model.train()
        tmp = ""
        backup = ""
        had_adapter = bool(self.adapter_path and os.path.isdir(self.adapter_path))

        try:
            trained_rows = 0
            for _ in range(max(1, int(steps))):
                for item in rows:
                    prompt = str(item["prompt"]).strip()
                    completion = str(item["completion"]).strip()
                    messages = [{"role": "user", "content": prompt}]
                    prefix = None
                    apply_template = getattr(tok, "apply_chat_template", None)
                    if callable(apply_template):
                        try:
                            prefix = apply_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                        except Exception:
                            prefix = None
                    if not prefix:
                        prefix = (
                            f"<|im_start|>user\n{prompt}<|im_end|>\n"
                            f"<|im_start|>assistant\n"
                        )
                    full = prefix + completion
                    if "<|im_start|>" in prefix:
                        full += "<|im_end|>"

                    encoded = tok(
                        full,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    )
                    prefix_ids = tok(
                        prefix,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    )["input_ids"]
                    try:
                        device = next(param.device for param in params)
                    except StopIteration:
                        raise RuntimeError("Controller LoRA lost its trainable parameters")

                    encoded = {key: value.to(device) for key, value in encoded.items()}
                    labels = encoded["input_ids"].clone()
                    prompt_len = min(int(prefix_ids.shape[1]), int(labels.shape[1]))
                    labels[:, :prompt_len] = -100

                    optimizer.zero_grad(set_to_none=True)
                    output = model(**encoded, labels=labels)
                    loss = getattr(output, "loss", None)
                    if loss is None or not torch.isfinite(loss):
                        raise RuntimeError("Controller LoRA training produced a non-finite loss")
                    loss.backward()
                    optimizer.step()
                    trained_rows += 1

            if trained_rows <= 0:
                raise RuntimeError("Controller LoRA found no trainable prompt/completion rows")

            total = 0.0
            touched = 0
            for name, param in model.named_parameters():
                if not param.requires_grad or name not in before:
                    continue
                delta = param.detach().float().cpu() - before[name]
                total += float((delta * delta).sum().item())
                touched += int(param.numel())
            drift = math.sqrt(max(total, 0.0))
            if drift <= 0.0 or touched <= 0:
                raise RuntimeError("Controller LoRA completed but measured zero parameter change")

            if not self.adapter_path:
                raise RuntimeError("Controller adapter persistence path is unavailable")
            os.makedirs(os.path.dirname(self.adapter_path), exist_ok=True)
            tmp = self.adapter_path + ".next"
            backup = self.adapter_path + ".previous"
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(backup, ignore_errors=True)

            model.save_pretrained(tmp, safe_serialization=True)
            if had_adapter:
                os.replace(self.adapter_path, backup)
            os.replace(tmp, self.adapter_path)
            tmp = ""
            if os.path.isdir(backup):
                shutil.rmtree(backup, ignore_errors=True)
            backup = ""

            self.adapters = {
                "trainable_parameters_touched": touched,
                "adapter_format": "peft-lora",
            }
            return dict(self.adapters), float(drift)

        except BaseException:
            # Restore live trainable tensors first, then restore the persisted
            # adapter directory if the filesystem transaction had started.
            try:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        snapshot = before.get(name)
                        if snapshot is None or not param.requires_grad:
                            continue
                        param.copy_(
                            snapshot.to(device=param.device, dtype=param.dtype)
                        )
            except Exception:
                pass

            if tmp and os.path.isdir(tmp):
                shutil.rmtree(tmp, ignore_errors=True)

            if backup and os.path.isdir(backup):
                if os.path.isdir(self.adapter_path):
                    shutil.rmtree(self.adapter_path, ignore_errors=True)
                os.replace(backup, self.adapter_path)
                backup = ""
            elif not had_adapter and self.adapter_path and os.path.isdir(self.adapter_path):
                # A brand-new failed transaction must not leave an unaccepted
                # persisted adapter behind.
                shutil.rmtree(self.adapter_path, ignore_errors=True)
            raise

        finally:
            if tmp and os.path.isdir(tmp):
                shutil.rmtree(tmp, ignore_errors=True)
            if backup and os.path.isdir(backup):
                # Reaching finally with a backup means the previous adapter is the
                # authoritative state unless the normal success path removed it.
                if not os.path.isdir(self.adapter_path):
                    try:
                        os.replace(backup, self.adapter_path)
                    except Exception:
                        pass
                elif os.path.isdir(backup):
                    shutil.rmtree(backup, ignore_errors=True)

            try:
                optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
            before.clear()
            params.clear()
            encoded = None
            prefix_ids = None
            labels = None
            output = None
            loss = None
            try:
                del optimizer
            except Exception:
                pass
            model.eval()
            release_training_memory(self)

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
        value = self._generator(
            self.model,
            self.processor,
            prepared,
            images or [],
            max_tokens=int(max_tokens),
            temperature=float(temperature),
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
        device = getattr(self.model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)
        do_sample = float(temperature) > 0.0
        with torch.no_grad():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                do_sample=do_sample,
                temperature=max(float(temperature), 1e-5) if do_sample else None,
            )
        decoder = getattr(self.processor, "batch_decode", None) or getattr(self.tokenizer, "batch_decode", None)
        if callable(decoder):
            return str(decoder(ids, skip_special_tokens=True)[0]).strip()
        return str(ids)

    def _generate(self, prompt: str, max_tokens: int, temperature: float, media: Optional[Dict[str, Any]] = None) -> str:
        if self.runtime.startswith("mlx"):
            images = [media["image"]] if media and media.get("image") else []
            return self._generate_mlx(prompt, images, max_tokens, temperature)
        if self.runtime in ("transformers_auto", "transformers", "hf_transformers"):
            return self._generate_transformers(prompt, max_tokens, temperature, media)
        raise RuntimeError(f"Generation unavailable for runtime {self.runtime}")

    def stream_generate_tokens(
        self,
        prompt: str,
        max_tokens: int = 1536,
        temperature: float = 0.65,
        top_p: float = 0.92,
    ) -> Generator[str, None, None]:
        del top_p
        text = self._generate(prompt, max_tokens, temperature)
        for match in re.finditer(r"\S+\s*", text):
            yield match.group(0)

    def generate_branches(
        self,
        prompt: str,
        branch_count: int = 1,
        max_tokens: int = 1536,
        temperature: Any = 0.65,
        top_p: float = 0.92,
    ) -> List[str]:
        del top_p
        out = []
        for idx in range(max(1, int(branch_count))):
            t = float(temperature[idx % len(temperature)]) if isinstance(temperature, (list, tuple)) else float(temperature)
            out.append(self._generate(prompt, max_tokens, t))
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
        if self.runtime.startswith("mlx") or self.runtime in ("jang_vlm", "jang-vlm"):
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

    def unload_model(self) -> None:
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.config = None
        self.runtime_module = None
        self.is_loaded = False
        self.is_mlx_available = False
        try:
            import mlx.core as mx
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
