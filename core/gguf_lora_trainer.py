"""Real GGUF learning bridge.

GGUF base weights are inference-only/quantized.  Learning therefore updates a
persistent PEFT LoRA sidecar against the matching Hugging Face base architecture,
converts that adapter with llama.cpp's official convert_lora_to_gguf.py, and then
llama.cpp loads the learned adapter on top of the original GGUF.

No parameter drift is fabricated: success requires a measurable LoRA update,
successful GGUF-LoRA conversion, and successful llama.cpp reload.
"""
from __future__ import annotations

import gc
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_name(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value or "model"))
    return out.strip("_") or "model"


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

    target = Path(tool_root) / "llama.cpp"
    target.parent.mkdir(parents=True, exist_ok=True)
    git = shutil.which("git")
    if not git:
        raise RuntimeError(
            "GGUF LoRA training needs llama.cpp's official converter. "
            "Install git or set LLAMA_CPP_DIR to a llama.cpp checkout."
        )
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
        base_model_id: str,
        adapter_root: str,
        rank: int = 32,
        alpha: int = 64,
    ):
        self.base_model_id = str(base_model_id or "").strip()
        if not self.base_model_id:
            raise ValueError("GGUF LoRA trainer requires a matching base_model_id")
        self.adapter_root = os.path.abspath(adapter_root)
        self.peft_dir = os.path.join(self.adapter_root, "peft")
        self.gguf_adapter_path = os.path.join(self.adapter_root, "adapter.gguf")
        self.rank = int(rank)
        self.alpha = int(alpha)

    def _load_trainable_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, PeftModel, get_peft_model

        if not torch.cuda.is_available():
            raise RuntimeError(
                "GGUF inference works on CPU/CUDA, but 27B GGUF parameter learning "
                "requires a CUDA QLoRA training runtime. No fake CPU update was performed."
            )

        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "GGUF parameter learning requires bitsandbytes for 4-bit QLoRA. "
                "Install the gguf training dependencies."
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
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=quant,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        from peft import prepare_model_for_kbit_training
        base = prepare_model_for_kbit_training(base)

        if os.path.isfile(os.path.join(self.peft_dir, "adapter_config.json")):
            model = PeftModel.from_pretrained(base, self.peft_dir, is_trainable=True)
        else:
            cfg = LoraConfig(
                r=self.rank,
                lora_alpha=self.alpha,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(base, cfg)
        model.train()
        return model, tokenizer

    @staticmethod
    def _snapshot_trainable(model) -> Dict[str, Any]:
        snap = {}
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

    def _convert_to_gguf(self) -> str:
        llama_root = _find_or_clone_llama_cpp(os.path.dirname(self.adapter_root))
        converter = llama_root / "convert_lora_to_gguf.py"
        tmp_out = self.gguf_adapter_path + ".tmp"
        os.makedirs(self.adapter_root, exist_ok=True)
        if os.path.exists(tmp_out):
            os.remove(tmp_out)

        proc = subprocess.run(
            [
                sys.executable,
                str(converter),
                "--base-model-id",
                self.base_model_id,
                "--outtype",
                "f16",
                "--outfile",
                tmp_out,
                self.peft_dir,
            ],
            cwd=str(llama_root),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not os.path.isfile(tmp_out):
            raise RuntimeError(
                "llama.cpp LoRA conversion failed: "
                + (proc.stderr or proc.stdout or "no adapter output").strip()
            )
        os.replace(tmp_out, self.gguf_adapter_path)
        return self.gguf_adapter_path

    def train(
        self,
        data: List[Dict[str, str]],
        *,
        learning_rate: float = 1e-4,
        steps: int = 3,
        max_length: int = 512,
    ) -> Tuple[Dict[str, Any], float, int, str]:
        import torch

        model, tokenizer = self._load_trainable_model()
        before = self._snapshot_trainable(model)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=float(learning_rate),
        )

        rows = [
            item for item in (data or [])
            if str(item.get("prompt") or "").strip() and str(item.get("completion") or "").strip()
        ]
        if not rows:
            raise RuntimeError("GGUF LoRA trainer received no prompt/completion pairs")

        try:
            for _ in range(max(1, int(steps))):
                for item in rows:
                    prompt = str(item["prompt"]).strip()
                    completion = str(item["completion"]).strip()
                    prefix = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                    text = prefix + completion + "<|im_end|>"
                    encoded = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max(32, int(max_length)),
                    )
                    prompt_ids = tokenizer(
                        prefix,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max(32, int(max_length)),
                    )["input_ids"]
                    device = next((p.device for p in model.parameters() if p.requires_grad), None)
                    if device is not None:
                        encoded = {k: v.to(device) for k, v in encoded.items()}
                    labels = encoded["input_ids"].clone()
                    prompt_len = min(int(prompt_ids.shape[1]), int(labels.shape[1]))
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

            os.makedirs(self.peft_dir, exist_ok=True)
            model.save_pretrained(self.peft_dir)
            tokenizer.save_pretrained(self.peft_dir)
            adapter_path = self._convert_to_gguf()
            return {"trainable_parameters_touched": touched}, float(drift), touched, adapter_path
        finally:
            del model
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
