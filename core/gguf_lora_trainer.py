"""Train a cumulative PEFT LoRA on the selected GGUF's verified HF parent.

The export uses llama.cpp's LoRA converter. The serving backend evaluates the
converted candidate over the original quantized base before publishing it.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from core.training_memory import release_training_memory
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


class GGUFLoRATrainer:
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
        self.peft_dir = None
        self.cancel_event = None
        self.memory_limit_gb = memory_limit_gb
        self.rank = int(rank)
        self.alpha = int(alpha)

    def _training_placement(self, source):
        import torch
        from core.transformers_placement import transformers_memory_limits

        weights = list(Path(source).glob("*.safetensors")) or list(Path(source).glob("pytorch_model*.bin"))
        weight_bytes = sum(path.stat().st_size for path in weights)
        if not weight_bytes:
            raise RuntimeError("The selected training parent contains no weight data")
        budgets = transformers_memory_limits(torch, self.memory_limit_gb)
        try:
            import bitsandbytes
            nf4_devices = getattr(bitsandbytes, "supported_torch_devices", {"cuda"})
        except (ImportError, OSError, RuntimeError):
            nf4_devices = set()
        # Keep BF16/FP16 learning available when a resident HF parent fits but
        # bitsandbytes has no kernels for this host. Neither choice touches the
        # GGUF inference file. Reserve headroom beyond frozen weight storage.
        devices = sorted(budgets, key=lambda d: d == "cpu")
        for device in devices:
            kind = "cuda" if type(device) is int else str(device)
            available = budgets[device]
            if kind in nf4_devices and available > weight_bytes * 0.6 + 512 * 1024**2:
                return device, "nf4"
            if available > weight_bytes * 2 + 512 * 1024**2:
                return device, "resident"
        raise RuntimeError("The declared training parent does not fit available device memory")

    def can_prepare(self) -> bool:
        if not os.path.isfile(self.model_path) or not self.base_model_id:
            return False
        try:
            import transformers  # noqa: F401
            import peft  # noqa: F401
            self._training_placement(local_training_parent(self.base_model_id))
        except Exception:
            return False
        return bool(shutil.which("git") or (Path(os.getenv("LLAMA_CPP_DIR", "")) / "convert_lora_to_gguf.py").is_file())

    def _load_trainable_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

        source = self.training_source or local_training_parent(self.base_model_id)
        device, self.training_mode = self._training_placement(source)
        options = {"device_map": {"": device}, "torch_dtype": "auto"}
        if self.training_mode == "nf4":
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(
            source, low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True, **options,
        )
        base.config.use_cache = False
        if self.training_mode == "nf4":
            base = prepare_model_for_kbit_training(base)

        if os.path.isfile(os.path.join(self.peft_dir, "adapter_config.json")):
            model = PeftModel.from_pretrained(base, self.peft_dir, is_trainable=True)
        else:
            from core.controller_runtime import UniversalControllerBackend
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

    def train(
        self, data, *, output_directory, previous_peft=None, learning_rate=1e-4,
        steps=3, max_length=256, fisher_matrix=None, lambda_ewc=0.0, cancel_event=None, anchor_texts=None,
        reward_weighting="group_relative", entropic_kl_target=None, reference_kl_coef=0.0,
        objective='causal_lm', supervise_prompt=True, row_weights=None,
    ):
        """Build the complete candidate inside the caller's unpublished generation."""
        from core.torch_context_learning import check_cancelled, update_lora, verify_saved_lora

        output = Path(output_directory)
        self.cancel_event = cancel_event
        peft_path = output / "peft"
        if previous_peft is not None and not (Path(previous_peft) / "adapter_config.json").is_file():
            raise RuntimeError("The active deployment is missing its cumulative PEFT checkpoint")
        self.peft_dir = previous_peft or peft_path
        check_cancelled(cancel_event)
        model, tokenizer = self._load_trainable_model()
        try:
            metrics, drift = update_lora(
                model, tokenizer, data, learning_rate=learning_rate, steps=steps, max_tokens=max_length,
                fisher_matrix=fisher_matrix, lambda_ewc=lambda_ewc, cancel_event=cancel_event, anchor_texts=anchor_texts,
                previous_directory=Path(previous_peft).parent if previous_peft else None, output_directory=output,
                reward_weighting=reward_weighting, entropic_kl_target=entropic_kl_target, reference_kl_coef=reference_kl_coef,
                objective=objective, supervise_prompt=supervise_prompt, row_weights=row_weights,
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
            return metrics, drift, metrics["trainable_parameters_touched"], str(deployment)
        finally:
            model = tokenizer = None
            release_training_memory()
