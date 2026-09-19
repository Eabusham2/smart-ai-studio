"""Persistent BitNet learning bridge.

bitnet.cpp is an inference runtime, not a trainer. For supported model families this
module updates a real PEFT LoRA against the declared BF16 training sibling, merges
that learned state into a temporary Hugging Face checkpoint, then uses Microsoft's
official BitNet conversion pipeline to rebuild an I2_S deployment GGUF. The caller
hot-reloads bitnet.cpp only after the converted learned model exists.

Unsupported architectures fail closed; no synthetic drift or fake training state is
ever reported.
"""
from __future__ import annotations

import gc
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from core.training_memory import release_training_memory
from typing import Any, Dict, List, Optional, Tuple


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
        if not self.base_model_id or not self._training_runtime_available():
            return False
        source = self._bitnet_source()
        if source is None:
            return False
        if self._quantizer(source) is None:
            return False
        return True

    @staticmethod
    def _format_training_example(tokenizer, prompt: str, completion: str) -> Tuple[str, str]:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        prefix_messages = [{"role": "user", "content": prompt}]
        try:
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prefix = tokenizer.apply_chat_template(
                prefix_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return str(prefix), str(full_text)
        except Exception:
            prefix = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            return prefix, prefix + completion + "<|im_end|>"

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

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})

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
            self.base_model_id,
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

    @staticmethod
    def _snapshot_trainable(model) -> Dict[str, Any]:
        return {
            name: param.detach().float().cpu().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @staticmethod
    def _drift_l2(before: Dict[str, Any], model) -> Tuple[float, int]:
        total = 0.0
        touched = 0
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in before:
                continue
            delta = param.detach().float().cpu() - before[name]
            total += float((delta * delta).sum().item())
            touched += int(param.numel())
        return math.sqrt(max(total, 0.0)), touched

    def _convert_merged_checkpoint(self, merged_dir: str, output_tmp: str) -> None:
        source = self._bitnet_source()
        if source is None:
            raise RuntimeError("Microsoft BitNet conversion source is unavailable")

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

            proc = subprocess.run(
                [
                    sys.executable,
                    str(preprocess),
                    "--input", str(model_file),
                    "--output", str(processed_file),
                ],
                cwd=str(source),
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0 or not processed_file.is_file():
                raise RuntimeError(
                    "Microsoft BitNet preprocessing failed: "
                    + (proc.stderr or proc.stdout or "preprocessed weights were not produced").strip()
                )

            os.replace(str(model_file), str(backup_file))
            os.replace(str(processed_file), str(model_file))
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(converter_ms),
                        merged_dir,
                        "--vocab-type", "bpe",
                        "--outtype", "f32",
                        "--concurrency", "1",
                        "--outfile", str(f32),
                    ],
                    cwd=str(source),
                    capture_output=True,
                    text=True,
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

            proc = subprocess.run(
                [str(quantizer), str(f32), output_tmp, "I2_S", "1"],
                cwd=str(source),
                capture_output=True,
                text=True,
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

        proc = subprocess.run(
            [sys.executable, str(converter), merged_dir, "--outtype", "f32"],
            cwd=str(source),
            capture_output=True,
            text=True,
        )
        f32 = Path(merged_dir) / "ggml-model-f32.gguf"
        if proc.returncode != 0 or not f32.is_file():
            raise RuntimeError(
                "BitNet BF16→GGUF conversion failed: "
                + (proc.stderr or proc.stdout or "f32 GGUF was not produced").strip()
            )

        proc = subprocess.run(
            [str(quantizer), str(f32), output_tmp, "I2_S", "1"],
            cwd=str(source),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not os.path.isfile(output_tmp):
            raise RuntimeError(
                "BitNet I2_S quantization failed: "
                + (proc.stderr or proc.stdout or "learned I2_S GGUF was not produced").strip()
            )

    def train(
        self,
        data: List[Dict[str, str]],
        *,
        learning_rate: float = 1e-4,
        steps: int = 3,
        max_length: int = 512,
    ) -> Tuple[Dict[str, Any], float, int, str]:
        import torch

        rows = [
            item for item in (data or [])
            if str(item.get("prompt") or "").strip()
            and str(item.get("completion") or "").strip()
        ]
        if not rows:
            raise RuntimeError("BitNet trainer received no prompt/completion pairs")

        self.deploy_root.mkdir(parents=True, exist_ok=True)
        model, tokenizer = self._load_trainable_model()
        before = self._snapshot_trainable(model)
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("BitNet PEFT model exposes no trainable parameters")
        optimizer = torch.optim.AdamW(params, lr=float(learning_rate))

        tmp_peft = tempfile.mkdtemp(prefix="bitnet-peft-next-", dir=str(self.deploy_root))
        tmp_merged = tempfile.mkdtemp(prefix="bitnet-merged-", dir=str(self.deploy_root))
        tmp_deploy = str(self.learned_model_path) + ".tmp"
        peft_backup = str(self.peft_dir) + ".previous"

        try:
            for _ in range(max(1, int(steps))):
                for item in rows:
                    prompt = str(item["prompt"]).strip()
                    completion = str(item["completion"]).strip()
                    prefix, full_text = self._format_training_example(tokenizer, prompt, completion)
                    encoded = tokenizer(
                        full_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max(32, int(max_length)),
                    )
                    prefix_ids = tokenizer(
                        prefix,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max(32, int(max_length)),
                    )["input_ids"]

                    try:
                        device = next(p.device for p in model.parameters() if p.requires_grad)
                    except StopIteration as exc:
                        raise RuntimeError("BitNet adapter exposes no trainable parameters") from exc

                    encoded = {k: v.to(device) for k, v in encoded.items()}
                    labels = encoded["input_ids"].clone()
                    prompt_len = min(int(prefix_ids.shape[1]), int(labels.shape[1]))
                    labels[:, :prompt_len] = -100

                    optimizer.zero_grad(set_to_none=True)
                    out = model(**encoded, labels=labels)
                    loss = out.loss
                    if loss is None or not torch.isfinite(loss):
                        raise RuntimeError("BitNet adapter training produced a non-finite loss")
                    loss.backward()
                    optimizer.step()

            drift, touched = self._drift_l2(before, model)
            if drift <= 0.0 or touched <= 0:
                raise RuntimeError("BitNet training completed but measured zero parameter change")

            model.save_pretrained(tmp_peft, safe_serialization=True)
            tokenizer.save_pretrained(tmp_peft)

            # Merge the cumulative PEFT state into the BF16 master checkpoint, then
            # convert that real updated model back into BitNet's deployment format.
            if not hasattr(model, "merge_and_unload"):
                raise RuntimeError("BitNet PEFT model cannot merge its learned adapter")
            merged = model.merge_and_unload()
            # Microsoft convert-helper-bitnet.py expects one model.safetensors.
            merged.save_pretrained(
                tmp_merged,
                safe_serialization=True,
                max_shard_size="100GB",
            )
            tokenizer.save_pretrained(tmp_merged)

            # Microsoft's converter is file-based. Release both the PEFT graph and
            # merged BF16 model before conversion so their memory cannot overlap.
            try:
                optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
            before.clear()
            del optimizer
            del merged
            del model
            model = None
            release_training_memory()

            self._convert_merged_checkpoint(tmp_merged, tmp_deploy)

            if os.path.isdir(peft_backup):
                shutil.rmtree(peft_backup, ignore_errors=True)
            if self.peft_dir.is_dir():
                os.replace(str(self.peft_dir), peft_backup)
            os.replace(tmp_peft, str(self.peft_dir))
            tmp_peft = ""

            os.replace(tmp_deploy, str(self.learned_model_path))
            if os.path.isdir(peft_backup):
                shutil.rmtree(peft_backup, ignore_errors=True)

            return {
                "trainable_parameters_touched": int(touched),
                "base_model_id": self.base_model_id,
                "adapter_format": "peft-merged-bitnet-i2_s",
            }, float(drift), int(touched), str(self.learned_model_path)
        except Exception:
            if os.path.isdir(peft_backup) and not self.peft_dir.is_dir():
                os.replace(peft_backup, str(self.peft_dir))
            raise
        finally:
            if tmp_peft and os.path.isdir(tmp_peft):
                shutil.rmtree(tmp_peft, ignore_errors=True)
            shutil.rmtree(tmp_merged, ignore_errors=True)
            try:
                if os.path.isfile(tmp_deploy):
                    os.remove(tmp_deploy)
            except OSError:
                pass
            try:
                optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
            try:
                before.clear()
            except Exception:
                pass
            try:
                del optimizer
            except Exception:
                pass
            try:
                del tokenizer
            except Exception:
                pass
            try:
                if model is not None:
                    del model
            except Exception:
                pass
            release_training_memory()
