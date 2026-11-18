"""Train a cumulative PEFT adapter on the declared BitNet BF16 parent.

The conversion uses the selected deployment's verified ternary encoding. A
staged rebuild must match the original tensor layout and tokenizer, load in
bitnet.cpp and pass behavioral validation before it becomes current.
"""
from __future__ import annotations

import ast
import os
import re
import sys
import tempfile
from pathlib import Path
from core.training_memory import release_training_memory
from core.training_lineage import local_training_parent, run_conversion
from typing import List, Optional


class BitNetRebuildTrainer:
    def __init__(
        self,
        *,
        base_model_id: str,
        runtime_root: str,
        deploy_root: str,
        rank: int = 16,
        alpha: int = 32,
    ):
        self.base_model_id = str(base_model_id or "").strip()
        if not self.base_model_id:
            raise ValueError("BitNet training requires a BF16 training-base model id")
        self.runtime_root = Path(runtime_root)
        self.deploy_root = Path(deploy_root)
        self.training_source = None
        self.cancel_event = None
        self.quantization = None
        self.peft_dir = self.deploy_root / "peft"
        self.learned_model_path = self.deploy_root / "learned-i2_s.gguf"
        self.rank = int(rank)
        self.alpha = int(alpha)

    def _bitnet_source(self) -> Optional[Path]:
        candidates = [
            self.runtime_root / "BitNet",
            Path(os.getenv("BITNET_CPP_DIR", "").strip()) if os.getenv("BITNET_CPP_DIR", "").strip() else None,
            Path.cwd() / "BitNet",
        ]
        for candidate in candidates:
            if candidate and (candidate / "utils" / "convert-hf-to-gguf-bitnet.py").is_file():
                return candidate
        return None

    @staticmethod
    def _quantizer(source: Path) -> Optional[Path]:
        candidates = [
            source / "build/bin/llama-quantize",
            source / "build/bin/llama-quantize.exe",
            source / "build/bin/Release/llama-quantize.exe",
            source / "build/bin/Release/llama-quantize",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def selected_quantization(self, observed_types):
        """Resolve the exact type IDs from the installed converter's own enum."""
        source = self._bitnet_source()
        if source is None:
            raise RuntimeError("Local BitNet converter is unavailable")
        for root in (source / "3rdparty" / "llama.cpp", source):
            constants = root / "gguf-py" / "gguf" / "constants.py"
            if not constants.is_file():
                continue
            for node in ast.parse(constants.read_text(encoding="utf-8")).body:
                if not isinstance(node, ast.ClassDef) or node.name != "GGMLQuantizationType":
                    continue
                matches = set()
                for entry in node.body:
                    if not isinstance(entry, ast.Assign) or not isinstance(entry.value, ast.Constant):
                        continue
                    for key in entry.targets:
                        if (isinstance(key, ast.Name) and key.id in {"TL1", "TL2", "I2_S"}
                                and entry.value.value in observed_types):
                            matches.add(key.id.lower())
                if len(matches) == 1:
                    return matches.pop()
        raise RuntimeError("The converter cannot prove the selected deployment's ternary encoding")

    @staticmethod
    def _training_runtime_available() -> bool:
        try:
            import torch
            import transformers  # noqa: F401
            import peft  # noqa: F401
        except Exception:
            return False

        if torch.cuda.is_available():
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

        # CPU fine-tuning is allowed only with enough host memory to avoid turning
        # automatic chat consolidation into an OOM/restart loop.
        try:
            import psutil
            return (psutil.virtual_memory().total / (1024 ** 3)) >= 16.0
        except Exception:
            return False

    def can_prepare(self) -> bool:
        try:
            local_training_parent(self.base_model_id)
        except (ValueError, OSError, RuntimeError):
            return False
        if not self.base_model_id or not self._training_runtime_available():
            return False
        source = self._bitnet_source()
        if source is None:
            return False
        return True

    @staticmethod
    def _target_modules(model) -> List[str]:
        import torch

        suffixes = {
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "c_attn", "c_proj",
        }
        blocked = ("vision", "visual", "image", "audio", "speech", "projector", "mm_projector")
        targets: List[str] = []
        for name, module in model.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if leaf not in suffixes:
                continue
            lowered = name.lower()
            if any(marker in lowered for marker in blocked):
                continue
            # HF's trainable AutoBitLinear inherits nn.Linear; packed deployment
            # BitLinear does not. This deliberately refuses frozen packed modules.
            if isinstance(module, torch.nn.Linear):
                targets.append(name)
        return sorted(set(targets))

    def _load_trainable_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, PeftModel, get_peft_model

        if not self._training_runtime_available():
            raise RuntimeError("No supported runtime is available for real BitNet BF16 fine-tuning")

        source = self.training_source or local_training_parent(self.base_model_id)
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=True)

        device_map = "auto"
        dtype = torch.bfloat16
        try:
            if not torch.cuda.is_available() and not (
                getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            ):
                # BF16 CPU kernels are not universal; float32 is slower but correct.
                dtype = torch.float32
        except Exception:
            dtype = torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass

        if (self.peft_dir / "adapter_config.json").is_file():
            model = PeftModel.from_pretrained(model, str(self.peft_dir), is_trainable=True)
        else:
            targets = self._target_modules(model)
            if not targets:
                raise RuntimeError(
                    "BitNet BF16 training checkpoint exposes no PEFT-supported language "
                    "linear targets; refusing to fabricate trainability for packed BitLinear weights."
                )
            cfg = LoraConfig(
                r=self.rank,
                lora_alpha=self.alpha,
                target_modules=targets,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, cfg)

        model.train()
        return model, tokenizer

    def _convert_merged_checkpoint(self, merged_dir: str, output_tmp: str) -> None:
        source = self._bitnet_source()
        if source is None:
            raise RuntimeError("Microsoft BitNet conversion source is unavailable")

        converter = source / "utils" / "convert-hf-to-gguf-bitnet.py"
        if self.quantization is None:
            raise RuntimeError("Select and verify the source BitNet tensor encoding before conversion")
        help_result = run_conversion([sys.executable, str(converter), "--help"], cwd=str(source),
                                     cancel_event=self.cancel_event)
        supported = re.search(r"--outtype\s+\{([^}]+)\}", help_result.stdout or "")
        if supported and self.quantization in supported[1].split(","):
            run = run_conversion([sys.executable, str(converter), merged_dir, "--outtype", self.quantization,
                                  "--outfile", output_tmp], cwd=str(source), cancel_event=self.cancel_event)
            if run.returncode or not Path(output_tmp).is_file():
                raise RuntimeError("Native BitNet conversion failed: " + (run.stderr or run.stdout)[-2000:])
            return
        if self.quantization != "i2_s":
            raise RuntimeError("The installed converter does not support rebuilding the selected " + self.quantization)
        quantizer = self._quantizer(source)
        if quantizer is None:
            raise RuntimeError("Official BitNet quantizer is unavailable")

        low_base = self.base_model_id.lower()
        # Microsoft's 2B BF16 master needs the official preprocessing step before
        # conversion. Run the helper's exact underlying scripts ourselves so Windows
        # can use build/bin/Release/llama-quantize.exe via _quantizer().
        if low_base.startswith("microsoft/bitnet-b1.58-2b-4t"):
            preprocess = source / "utils" / "preprocess-huggingface-bitnet.py"
            converter_ms = source / "utils" / "convert-ms-to-gguf-bitnet.py"
            model_file = Path(merged_dir) / "model.safetensors"
            backup_file = Path(merged_dir) / "model.safetensors.pre-bitnet"
            processed_file = Path(merged_dir) / "model.safetensors.processed"
            f32 = Path(merged_dir) / "ggml-model-f32-bitnet.gguf"

            if not preprocess.is_file() or not converter_ms.is_file() or not model_file.is_file():
                raise RuntimeError(
                    "Official Microsoft BitNet preprocess/converter inputs are unavailable"
                )

            proc = run_conversion(
                [
                    sys.executable,
                    str(preprocess),
                    "--input", str(model_file),
                    "--output", str(processed_file),
                ],
                cwd=str(source), cancel_event=self.cancel_event,
            )
            if proc.returncode != 0 or not processed_file.is_file():
                raise RuntimeError(
                    "Microsoft BitNet preprocessing failed: "
                    + (proc.stderr or proc.stdout or "preprocessed weights were not produced").strip()
                )

            os.replace(str(model_file), str(backup_file))
            os.replace(str(processed_file), str(model_file))
            try:
                proc = run_conversion(
                    [
                        sys.executable,
                        str(converter_ms),
                        merged_dir,
                        "--vocab-type", "bpe",
                        "--outtype", "f32",
                        "--concurrency", "1",
                        "--outfile", str(f32),
                    ],
                    cwd=str(source), cancel_event=self.cancel_event,
                )
                if proc.returncode != 0 or not f32.is_file():
                    raise RuntimeError(
                        "Microsoft BitNet BF16→GGUF conversion failed: "
                        + (proc.stderr or proc.stdout or "f32 GGUF was not produced").strip()
                    )
            finally:
                try:
                    if model_file.is_file():
                        model_file.unlink()
                    if backup_file.is_file():
                        os.replace(str(backup_file), str(model_file))
                except OSError:
                    pass

            proc = run_conversion(
                [str(quantizer), str(f32), output_tmp, "I2_S", "1"],
                cwd=str(source), cancel_event=self.cancel_event,
            )
            if proc.returncode != 0 or not os.path.isfile(output_tmp):
                raise RuntimeError(
                    "Microsoft BitNet I2_S quantization failed: "
                    + (proc.stderr or proc.stdout or "learned I2_S GGUF was not produced").strip()
                )
            return

        # Other official/community BitNet families use the generic converter path
        # when their architecture is registered by Microsoft's converter.
        converter = source / "utils" / "convert-hf-to-gguf-bitnet.py"
        if not converter.is_file():
            raise RuntimeError("Official BitNet HF converter is unavailable")

        proc = run_conversion(
            [sys.executable, str(converter), merged_dir, "--outtype", "f32"],
            cwd=str(source), cancel_event=self.cancel_event,
        )
        f32 = Path(merged_dir) / "ggml-model-f32.gguf"
        if proc.returncode != 0 or not f32.is_file():
            raise RuntimeError(
                "BitNet BF16→GGUF conversion failed: "
                + (proc.stderr or proc.stdout or "f32 GGUF was not produced").strip()
            )

        proc = run_conversion(
            [str(quantizer), str(f32), output_tmp, "I2_S", "1"],
            cwd=str(source), cancel_event=self.cancel_event,
        )
        if proc.returncode != 0 or not os.path.isfile(output_tmp):
            raise RuntimeError(
                "BitNet I2_S quantization failed: "
                + (proc.stderr or proc.stdout or "learned I2_S GGUF was not produced").strip()
            )

    def train(
        self, data, *, output_directory=None, previous_peft=None, learning_rate=1e-4,
        steps=3, max_length=512, fisher_matrix=None, lambda_ewc=0.0, cancel_event=None, anchor_texts=None,
        reward_weighting="group_relative", entropic_kl_target=None, reference_kl_coef=0.0,
        objective='causal_lm', supervise_prompt=False, row_weights=None, context_text=None, max_context_tokens=None,
    ):
        """Build a staged deployment or publish the standalone cumulative checkpoint."""
        from core.torch_context_learning import check_cancelled, update_lora, verify_saved_lora
        from core.native_context_learning import native_training_output

        if output_directory is None and self.quantization is None:
            self.quantization = "i2_s"  # Historical standalone rebuild format.
        destination = self.deploy_root / f"learned-{self.quantization}.gguf"
        with native_training_output(self, output_directory, destination, previous_peft, cancel_event, data, objective) as stage:
            output, previous_peft, published, data = stage
            self.cancel_event = cancel_event
            peft_path = output / "peft"
            if previous_peft is not None and not (Path(previous_peft) / "adapter_config.json").is_file():
                raise RuntimeError("The active deployment is missing its cumulative PEFT checkpoint")
            self.peft_dir = Path(previous_peft or peft_path)
            check_cancelled(cancel_event)
            model, tokenizer = self._load_trainable_model()
            try:
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
                with tempfile.TemporaryDirectory(prefix="bitnet-merge-", dir=output) as merged_dir:
                    merged = model.merge_and_unload(safe_merge=True)
                    merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="100GB")
                    tokenizer.save_pretrained(merged_dir)
                    del merged
                    model = tokenizer = None
                    release_training_memory()
                    check_cancelled(cancel_event)
                    deployment = output / f"learned-{self.quantization}.gguf"
                    self._convert_merged_checkpoint(merged_dir, str(deployment))
                adapter_format = "peft-merged-bitnet-" + self.quantization
                check_cancelled(cancel_event)
                metrics.update({"base_model_id": self.base_model_id, "adapter_format": adapter_format})
                return metrics, drift, metrics["trainable_parameters_touched"], str(published)
            finally:
                model = tokenizer = None
                release_training_memory()
