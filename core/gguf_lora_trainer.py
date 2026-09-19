"""Real persistent learning bridge for GGUF inference models.

GGUF inference weights remain frozen/quantized. Learning updates a PEFT LoRA sidecar
against the matching trainable Hugging Face architecture, converts that adapter with
llama.cpp's official convert_lora_to_gguf.py, and the GGUF backend hot-loads the
resulting adapter over the exact selected Bonsai GGUF.

For Bonsai-2 this is mathematically compatible with Prism's Hadamard runtime: Prism
applies LoRA to the original activation path and adds it after the rotated ternary
base matmul. No fake parameter drift is reported.
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


def _find_or_clone_llama_cpp(tool_root: str) -> Path:
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
        shutil.rmtree(target, ignore_errors=True)

    proc = subprocess.run(
        [git, "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", str(target)],
        capture_output=True,
        text=True,
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
    ):
        self.model_path = os.path.abspath(str(model_path))
        self.base_model_id = str(base_model_id or "").strip()
        if not self.base_model_id:
            raise ValueError("GGUF LoRA trainer requires a matching base_model_id")
        self.adapter_root = os.path.abspath(adapter_root)
        self.peft_dir = os.path.join(self.adapter_root, "peft")
        self.gguf_adapter_path = os.path.join(self.adapter_root, "adapter.gguf")
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

    def can_prepare(self) -> bool:
        """Report training readiness only when a real QLoRA backend is available."""
        if not os.path.isfile(self.model_path) or not self.base_model_id:
            return False
        try:
            import transformers  # noqa: F401
            import peft  # noqa: F401
            import bitsandbytes  # noqa: F401
        except Exception:
            return False
        if not self._training_backend_available():
            return False
        if not shutil.which("git") and not os.getenv("LLAMA_CPP_DIR", "").strip():
            return False
        return True

    def _load_trainable_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

        if not self._training_backend_available():
            raise RuntimeError(
                "The GGUF model can infer on this device, but this machine does not "
                "currently have enough supported 4-bit training capability for a real "
                "27B QLoRA parameter update."
            )

        try:
            import bitsandbytes  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "GGUF parameter learning requires bitsandbytes for 4-bit QLoRA."
            ) from exc

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=quant,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        base.config.use_cache = False
        if hasattr(base, "gradient_checkpointing_enable"):
            try:
                base.gradient_checkpointing_enable()
            except Exception:
                pass
        base = prepare_model_for_kbit_training(base)

        if os.path.isfile(os.path.join(self.peft_dir, "adapter_config.json")):
            model = PeftModel.from_pretrained(base, self.peft_dir, is_trainable=True)
        else:
            # Qwen3.8 is hybrid: most layers are GatedDeltaNet and the rest are
            # full attention. Discover the exact language-layer projections so LoRA
            # covers both families without accidentally adapting the vision tower.
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

            if not target_modules:
                raise RuntimeError(
                    "Qwen3.8 GGUF QLoRA found no compatible language projection modules"
                )

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

    @staticmethod
    def _snapshot_trainable(model) -> Dict[str, Any]:
        snap: Dict[str, Any] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                snap[name] = param.detach().float().cpu().clone()
        return snap

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

    def _convert_to_gguf(self, peft_dir: str, output_path: str) -> str:
        llama_root = _find_or_clone_llama_cpp(os.path.dirname(self.adapter_root))
        converter = llama_root / "convert_lora_to_gguf.py"
        os.makedirs(self.adapter_root, exist_ok=True)
        if os.path.exists(output_path):
            os.remove(output_path)

        proc = subprocess.run(
            [
                sys.executable,
                str(converter),
                "--base-model-id",
                self.base_model_id,
                "--outtype",
                "f16",
                "--outfile",
                output_path,
                peft_dir,
            ],
            cwd=str(llama_root),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not os.path.isfile(output_path):
            raise RuntimeError(
                "llama.cpp LoRA conversion failed: "
                + (proc.stderr or proc.stdout or "no adapter output").strip()
            )
        return output_path

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
            raise RuntimeError("GGUF LoRA trainer received no prompt/completion pairs")

        os.makedirs(self.adapter_root, exist_ok=True)
        model, tokenizer = self._load_trainable_model()
        before = self._snapshot_trainable(model)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=float(learning_rate),
        )

        tmp_peft = tempfile.mkdtemp(prefix="peft-next-", dir=self.adapter_root)
        tmp_gguf = self.gguf_adapter_path + ".tmp"
        backup_peft = self.peft_dir + ".previous"

        try:
            for _ in range(max(1, int(steps))):
                for item in rows:
                    prompt = str(item["prompt"]).strip()
                    completion = str(item["completion"]).strip()
                    prefix, full_text = self._format_training_example(
                        tokenizer, prompt, completion
                    )
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
                        device = next(
                            p.device for p in model.parameters()
                            if p.requires_grad
                        )
                    except StopIteration:
                        raise RuntimeError("GGUF QLoRA model exposes no trainable LoRA parameters")

                    encoded = {k: v.to(device) for k, v in encoded.items()}
                    labels = encoded["input_ids"].clone()
                    prompt_len = min(int(prefix_ids.shape[1]), int(labels.shape[1]))
                    labels[:, :prompt_len] = -100

                    optimizer.zero_grad(set_to_none=True)
                    out = model(**encoded, labels=labels)
                    loss = out.loss
                    if loss is None or not torch.isfinite(loss):
                        raise RuntimeError("GGUF QLoRA training produced a non-finite loss")
                    loss.backward()
                    optimizer.step()

            drift, touched = self._drift_l2(before, model)
            if drift <= 0.0 or touched <= 0:
                raise RuntimeError("GGUF QLoRA completed but measured zero parameter change")

            model.save_pretrained(tmp_peft, safe_serialization=True)
            tokenizer.save_pretrained(tmp_peft)

            # Conversion is file-based. Do not keep the 27B QLoRA graph resident
            # while the converter allocates its own buffers.
            try:
                optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
            before.clear()
            encoded = None
            prefix_ids = None
            labels = None
            out = None
            loss = None
            del optimizer
            del model
            model = None
            tokenizer = None
            release_training_memory()

            self._convert_to_gguf(tmp_peft, tmp_gguf)

            # Do not make a new state live until both the PEFT checkpoint and converted
            # GGUF adapter exist. The caller separately backs up the currently live
            # adapter and reloads llama.cpp before accepting this update.
            if os.path.isdir(backup_peft):
                shutil.rmtree(backup_peft, ignore_errors=True)
            if os.path.isdir(self.peft_dir):
                os.replace(self.peft_dir, backup_peft)
            os.replace(tmp_peft, self.peft_dir)
            tmp_peft = ""

            os.replace(tmp_gguf, self.gguf_adapter_path)
            if os.path.isdir(backup_peft):
                shutil.rmtree(backup_peft, ignore_errors=True)

            return {
                "trainable_parameters_touched": touched,
                "base_model_id": self.base_model_id,
                "adapter_format": "gguf-lora",
            }, float(drift), touched, self.gguf_adapter_path
        except BaseException:
            if os.path.isdir(backup_peft) and not os.path.isdir(self.peft_dir):
                os.replace(backup_peft, self.peft_dir)
            raise
        finally:
            if tmp_peft and os.path.isdir(tmp_peft):
                shutil.rmtree(tmp_peft, ignore_errors=True)
            try:
                if os.path.isfile(tmp_gguf):
                    os.remove(tmp_gguf)
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
