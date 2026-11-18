"""Train a cumulative PEFT LoRA on the selected GGUF's verified HF parent.

The export uses llama.cpp's LoRA converter. The serving backend evaluates the
converted candidate over the original quantized base before publishing it.
"""
from __future__ import annotations

from functools import lru_cache
import os
import shutil
import sys
from pathlib import Path
from core.training_memory import release_training_memory
from core.legacy_torch_learning import LegacyCompletionTrainer, set_legacy_padding_token, update_legacy_lora
from core.training_lineage import local_training_parent, run_conversion


def _find_or_clone_llama_cpp(tool_root: str, cancel_event=None) -> Path:
    explicit = os.getenv("LLAMA_CPP_DIR", "").strip()
    candidates = [
        Path(explicit) if explicit else None,
        Path(tool_root) / "llama.cpp",
        Path.cwd() / "llama.cpp",
        Path.cwd().parent / "llama.cpp",
    ]
    for candidate in candidates:
        if candidate and (candidate / "convert_lora_to_gguf.py").is_file():
            return candidate

    git = shutil.which("git")
    if not git:
        raise RuntimeError(
            "GGUF LoRA conversion requires llama.cpp's official converter. "
            "Install git or set LLAMA_CPP_DIR to an existing llama.cpp checkout."
        )

    target = Path(tool_root) / "llama.cpp"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not (target / "convert_lora_to_gguf.py").is_file():
        raise RuntimeError("Existing llama.cpp directory is missing its LoRA converter: " + str(target))

    proc = run_conversion(
        [git, "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", str(target)],
        cancel_event=cancel_event,
    )
    if proc.returncode != 0 or not (target / "convert_lora_to_gguf.py").is_file():
        raise RuntimeError(
            "Could not obtain llama.cpp LoRA converter: "
            + (proc.stderr or proc.stdout or "git clone failed").strip()
        )
    return target


@lru_cache(maxsize=128)
def _check_nf4_kernel(device, input_width, output_tile):
    """Exercise eval prefill and the input gradient with the same NF4 settings as training."""
    import bitsandbytes as bnb
    import torch

    # Meta construction avoids consuming the user's RNG. Preserve the real
    # input width and output width modulo 32 for bnb's tiled CPU packing path.
    layer = bnb.nn.Linear4bit(input_width, output_tile, bias=False, compute_dtype=torch.bfloat16,
                              compress_statistics=True, quant_type="nf4", device="meta")
    values = torch.arange(output_tile * input_width, dtype=torch.float32).remainder_(7).sub_(3)
    layer.weight = bnb.nn.Params4bit(values.reshape(output_tile, input_width), requires_grad=False,
                                     compress_statistics=True, quant_type="nf4")
    layer = layer.to(device)
    x = torch.ones((2, input_width), device=device, dtype=torch.bfloat16)
    layer.eval()
    with torch.no_grad():
        output = layer(x)
        if output.shape != (2, output_tile) or not torch.isfinite(output).all().item():
            raise ValueError("NF4 prefill produced invalid output")
    with torch.enable_grad():
        x.requires_grad_()
        gradient, = torch.autograd.grad(layer(x).float().sum(), x)
        if gradient.shape != x.shape or not torch.isfinite(gradient).all().item():
            raise ValueError("NF4 backward produced an invalid input gradient")


def _check_parent_nf4(source, device):
    import bitsandbytes as bnb
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.pytorch_utils import Conv1D

    target = torch.device("cuda", device) if type(device) is int else torch.device(device)
    if target.type not in getattr(bnb, "supported_torch_devices", {"cuda"}):
        raise ValueError(f"bitsandbytes does not support {target.type}")
    config = AutoConfig.from_pretrained(source, local_files_only=True, trust_remote_code=True)
    with init_empty_weights(include_buffers=True):
        skeleton = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    head = skeleton.get_output_embeddings()
    shapes = set()
    for module in skeleton.modules():
        if module is head:
            continue
        if type(module) is torch.nn.Linear:
            inputs, outputs = module.in_features, module.out_features
        elif isinstance(module, Conv1D):
            inputs, outputs = module.weight.shape
        else:
            continue
        # Exercise a representative pair of output tiles without allocating
        # a dense copy of a 27B projection to decide whether it fits.
        tile = min(outputs, 32 + (outputs - 1) % 32 + 1)
        shapes.add((inputs, tile))
    if not shapes:
        raise ValueError("The training parent has no bitsandbytes linear projections")
    for inputs, tile in sorted(shapes):
        _check_nf4_kernel(target, inputs, tile)


class GGUFLoRATrainer(LegacyCompletionTrainer):
    def __init__(
        self,
        *,
        model_path: str,
        base_model_id: str,
        adapter_root: str,
        rank: int = 32,
        alpha: int = 64,
        memory_limit_gb=None,
    ):
        self.model_path = os.path.abspath(str(model_path))
        self.base_model_id = str(base_model_id or "").strip()
        if not self.base_model_id:
            raise ValueError("GGUF LoRA trainer requires a matching base_model_id")
        self.adapter_root = os.path.abspath(adapter_root)
        self.training_source = None
        self.peft_dir = os.path.join(self.adapter_root, "peft")
        self.gguf_adapter_path = os.path.join(self.adapter_root, "adapter.gguf")
        self.cancel_event = None
        self._legacy_training = False
        self.memory_limit_gb = memory_limit_gb
        self.rank = int(rank)
        self.alpha = int(alpha)

    @staticmethod
    def _training_backend_available() -> bool:
        try:
            import torch
            import bitsandbytes  # noqa: F401
        except Exception:
            return False

        if torch.cuda.is_available():
            # CUDA and ROCm both surface through torch.cuda.
            return True
        xpu = getattr(torch, "xpu", None)
        if xpu is not None:
            try:
                if xpu.is_available():
                    return True
            except Exception:
                pass
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None:
            try:
                if mps.is_available():
                    return True
            except Exception:
                pass

        # Current bitsandbytes also has a CPU backend. A 27B 4-bit training graph
        # still needs substantial host RAM, so fail closed on small systems.
        try:
            import psutil
            return (psutil.virtual_memory().total / (1024 ** 3)) >= 24.0
        except Exception:
            return False

    def _training_budgets(self, torch, legacy):
        from core.transformers_placement import transformers_memory_limits

        budgets = transformers_memory_limits(torch, self.memory_limit_gb)
        xpu = getattr(torch, "xpu", None)
        if legacy and not torch.cuda.is_available() and xpu is not None and xpu.is_available():
            # The pre-PR QLoRA loader let Accelerate place weights on Intel
            # accelerators. Measure each device before admitting a training graph.
            for index in range(xpu.device_count()):
                try:
                    free, total = xpu.mem_get_info(index)
                except (AttributeError, RuntimeError, OSError):
                    continue
                usable = int(free) - max(1024**3, int(total * 0.10))
                if usable >= 256 * 1024**2:
                    budgets[f"xpu:{index}"] = usable
        return budgets

    def _training_placement(self, source, *, legacy=None):
        import torch

        weights = list(Path(source).glob("*.safetensors")) or list(Path(source).glob("pytorch_model*.bin"))
        weight_bytes = sum(path.stat().st_size for path in weights)
        if not weight_bytes:
            raise RuntimeError("The selected training parent contains no weight data")
        if legacy is None:
            legacy = self._legacy_training
        budgets = self._training_budgets(torch, legacy)
        # Prefer a resident CPU or Metal parent whenever it fits. A published
        # device tag alone does not establish support for a GQA projection's
        # width, double quantization, or the backward kernel on this machine.
        rejected = []
        prefer_nf4 = legacy and self._training_backend_available()
        accelerators = {device: size for device, size in budgets.items()
                        if isinstance(device, int) or str(device).startswith("xpu:")}
        required = weight_bytes * 0.6 + 512 * 1024**2
        if prefer_nf4 and len(accelerators) > 1 and sum(accelerators.values()) > required:
            # The original QLoRA loader used Accelerate's auto map. Preserve
            # sharded training without claiming an untested NF4 device.
            try:
                for device in accelerators:
                    _check_parent_nf4(source, device)
            except (AssertionError, AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
                rejected.append(f"multi-device NF4: {error}")
            else:
                return "auto", "nf4"
        for device in sorted(budgets, key=lambda d: d == "cpu"):
            available = budgets[device]
            resident = available > weight_bytes * 2 + 512 * 1024**2
            if device in {"cpu", "mps"} and resident and not prefer_nf4:
                return device, "resident"
            if available > required:
                try:
                    _check_parent_nf4(source, device)
                except (AssertionError, AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
                    rejected.append(f"{device}: {error}")
                else:
                    return device, "nf4"
            if resident:
                return device, "resident"
        reason = "The declared training parent does not fit available device memory"
        if rejected:
            reason += ". NF4 unavailable (" + ", ".join(rejected) + ")"
        raise RuntimeError(reason)

    def can_prepare(self, *, legacy=False) -> bool:
        if not os.path.isfile(self.model_path) or not self.base_model_id:
            return False
        try:
            import transformers  # noqa: F401
            import peft  # noqa: F401
            source = local_training_parent(self.base_model_id)
            self._training_placement(source, legacy=legacy)
        except Exception:
            return False
        return bool(shutil.which("git") or (Path(os.getenv("LLAMA_CPP_DIR", "")) / "convert_lora_to_gguf.py").is_file())

    def can_prepare_legacy(self):
        """An explicit legacy update can fetch its declared training parent."""
        try:
            local_training_parent(self.base_model_id)
        except RuntimeError:
            pass
        else:
            return self.can_prepare(legacy=True)
        if not os.path.isfile(self.model_path) or not self._training_backend_available():
            return False
        try:
            import peft  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        return bool(shutil.which("git") or (Path(os.getenv("LLAMA_CPP_DIR", "")) / "convert_lora_to_gguf.py").is_file())

    @staticmethod
    def _legacy_lora_targets(base):
        import torch

        candidate_suffixes = {
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
        }
        target_modules = []
        for module_name, module in base.named_modules():
            leaf = module_name.rsplit(".", 1)[-1]
            if leaf not in candidate_suffixes:
                continue
            lowered = module_name.lower()
            if any(marker in lowered for marker in ("visual", "vision", "mtp")):
                continue
            # Keep only decoder/language paths. Full conditional-generation
            # checkpoints expose these under language_model/model.layers.
            if ".layers." not in module_name and not module_name.startswith("model.layers."):
                continue
            if isinstance(module, torch.nn.Linear):
                target_modules.append(module_name)

        return target_modules

    def _load_trainable_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

        source = self.training_source or local_training_parent(self.base_model_id)
        device, self.training_mode = self._training_placement(source)
        options = {"device_map": {"": device}, "torch_dtype": "auto"}
        if device == "auto":
            budgets = self._training_budgets(torch, self._legacy_training)
            # Accelerate accepts integer indices for accelerator memory caps.
            limits = {int(d.split(":")[1]) if isinstance(d, str) and d.startswith("xpu:") else d: size
                      for d, size in budgets.items()}
            options.update(device_map="auto", max_memory={**limits, "cpu": 0})
        if self.training_mode == "nf4":
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, local_files_only=True)
        if self._legacy_training:
            set_legacy_padding_token(tokenizer)
        base = AutoModelForCausalLM.from_pretrained(
            source, low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True, **options,
        )
        base.config.use_cache = False
        if self.training_mode == "nf4":
            if self._legacy_training and hasattr(base, "gradient_checkpointing_enable"):
                try:
                    base.gradient_checkpointing_enable()
                except (RuntimeError, ValueError, NotImplementedError):
                    pass
            base = prepare_model_for_kbit_training(base)

        if os.path.isfile(os.path.join(self.peft_dir, "adapter_config.json")):
            model = PeftModel.from_pretrained(base, self.peft_dir, is_trainable=True)
        else:
            from core.controller_runtime import UniversalControllerBackend
            target_modules = self._legacy_lora_targets(base) if self._legacy_training else []
            if not target_modules:
                target_modules = UniversalControllerBackend._transformers_lora_targets(base)
            if not target_modules:
                raise RuntimeError("GGUF training parent exposes no compatible language projections")

            cfg = LoraConfig(
                r=self.rank,
                lora_alpha=self.alpha,
                target_modules=sorted(set(target_modules)),
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(base, cfg)

        model.train()
        return model, tokenizer

    def _convert_to_gguf(self, peft_dir: str, output_path: str) -> str:
        llama_root = _find_or_clone_llama_cpp(os.path.dirname(self.adapter_root), self.cancel_event)
        converter = llama_root / "convert_lora_to_gguf.py"
        os.makedirs(self.adapter_root, exist_ok=True)
        if os.path.exists(output_path):
            os.remove(output_path)

        proc = run_conversion(
            [
                sys.executable,
                str(converter),
                "--base",
                self.training_source or local_training_parent(self.base_model_id),
                "--outtype",
                "f16",
                "--outfile",
                output_path,
                peft_dir,
            ],
            cwd=str(llama_root), cancel_event=self.cancel_event,
        )
        if proc.returncode != 0 or not os.path.isfile(output_path):
            raise RuntimeError(
                "llama.cpp LoRA conversion failed: "
                + (proc.stderr or proc.stdout or "no adapter output").strip()
            )
        return output_path

    def train(self, data, *, learning_rate=1e-4, steps=3, max_length=512, learning_mode=None, **options):
        """Historical calls use per-example AdamW. Managed calls can use scoped objectives and replay."""
        from core.learning_implementation import resolve_learning_mode

        if "fisher_matrix" in options or "lambda_ewc" in options:
            if learning_mode == "legacy":
                raise ValueError("Standalone Fisher regularization requires the managed learning mode")
            learning_mode = learning_mode or "managed"
        routing = options
        if learning_mode == "legacy":
            # A serving backend stages even historical updates before hot-reload.
            routing = {key: value for key, value in options.items()
                       if key not in {"output_directory", "previous_peft", "cancel_event"}}
        mode = resolve_learning_mode(self, learning_mode, routing)
        return self._train(data, learning_mode=mode, learning_rate=learning_rate, steps=steps,
                           max_length=max_length, **options)

    def _train(
        self, data, *, learning_mode, output_directory=None, previous_peft=None, learning_rate=1e-4,
        steps=3, max_length=512, fisher_matrix=None, lambda_ewc=0.0, cancel_event=None, anchor_texts=None,
        reward_weighting="group_relative", entropic_kl_target=None, reference_kl_coef=0.0,
        objective='causal_lm', supervise_prompt=False, row_weights=None, context_text=None, max_context_tokens=None,
    ):
        """Build a staged deployment or publish the standalone cumulative checkpoint."""
        from core.torch_context_learning import check_cancelled, update_lora, verify_saved_lora
        from core.native_context_learning import native_training_output

        destination = Path(self.gguf_adapter_path)
        with native_training_output(self, output_directory, destination, previous_peft, cancel_event, data, objective) as stage:
            output, previous_peft, published, data = stage
            self.cancel_event = cancel_event
            self._legacy_training = learning_mode == "legacy"
            peft_path = output / "peft"
            if previous_peft is not None and not (Path(previous_peft) / "adapter_config.json").is_file():
                raise RuntimeError("The active deployment is missing its cumulative PEFT checkpoint")
            self.peft_dir = previous_peft or peft_path
            check_cancelled(cancel_event)
            model, tokenizer = self._load_trainable_model()
            try:
                if learning_mode == "legacy":
                    metrics, drift = update_legacy_lora(
                        model, tokenizer, data, trainer=self, learning_rate=learning_rate, steps=steps,
                        max_length=max_length, output_directory=output, cancel_event=cancel_event,
                    )
                else:
                    metrics, drift = update_lora(
                        model, tokenizer, data, learning_rate=learning_rate, steps=steps, max_tokens=max_length,
                        fisher_matrix=fisher_matrix, lambda_ewc=lambda_ewc, cancel_event=cancel_event, anchor_texts=anchor_texts,
                        previous_directory=Path(previous_peft).parent if previous_peft else None, output_directory=output,
                        reward_weighting=reward_weighting, entropic_kl_target=entropic_kl_target, reference_kl_coef=reference_kl_coef,
                        objective=objective, supervise_prompt=supervise_prompt, row_weights=row_weights,
                        context_text=context_text, max_context_tokens=max_context_tokens,
                    )
                check_cancelled(cancel_event)
                model.save_pretrained(peft_path, safe_serialization=True, save_embedding_layers=False)
                verify_saved_lora(model, peft_path)
                tokenizer.save_pretrained(peft_path)
                model = tokenizer = None
                release_training_memory()
                check_cancelled(cancel_event)
                deployment = output / "adapter.gguf"
                self._convert_to_gguf(str(peft_path), str(deployment))
                adapter_format = "gguf-lora"
                check_cancelled(cancel_event)
                metrics.update({"base_model_id": self.base_model_id, "adapter_format": adapter_format,
                                "parent_training_mode": self.training_mode})
                return metrics, drift, metrics["trainable_parameters_touched"], str(published)
            finally:
                model = tokenizer = None
                release_training_memory()
