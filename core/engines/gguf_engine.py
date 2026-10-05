"""
GGUF / Llama.cpp Cross-Platform Reasoning Backend.
Provides high-performance GGUF model execution across NVIDIA CUDA, Apple Metal, and Multi-Core CPU,
with multimodal vision projector support (nanoLLaVA / CLIP).
"""

import math
import base64
import io
import json
import mimetypes
import re
import os
import platform
import shutil
import sys
import time
from typing import Any, Dict, Generator, List, Optional, Tuple


class GGUFReasoningBackend:
    def __init__(
        self,
        model_path: str,
        mmproj_path: Optional[str] = None,
        n_gpu_layers: int = -1,
        n_ctx: int = 32768,
        verbose: bool = False,
        adapter_root: Optional[str] = None,
        training_base_model_id: Optional[str] = None,
    ):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.verbose = verbose
        self.training_base_model_id = str(training_base_model_id or "").strip()
        eval_adapter_root = str(os.getenv("SMARTAI_GGUF_ADAPTER_ROOT", "") or "").strip()
        if adapter_root:
            self.adapter_root = os.path.abspath(adapter_root)
        elif eval_adapter_root:
            self.adapter_root = os.path.abspath(eval_adapter_root)
        else:
            from config.paths import get_portable_data_dir
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(self.model_path) or "gguf")
            self.adapter_root = os.path.join(get_portable_data_dir(), "gguf_lora", stem)
        self.adapter_path = os.path.join(self.adapter_root, "adapter.gguf")
        self.adapters: Dict[str, Any] = {}

        self.model = None
        self.chat_handler = None
        self.is_gguf_available = False

        self._check_llama_cpp()

    def _check_llama_cpp(self):
        try:
            import llama_cpp
            self.is_gguf_available = True
        except ImportError:
            self.is_gguf_available = False

    def load_model(self) -> bool:
        """Initializes Llama.cpp GGUF instance with GPU layer offloading and optional vision clip handler."""
        if not self.is_gguf_available:
            return False

        if not os.path.exists(self.model_path):
            return False

        try:
            from llama_cpp import Llama

            # Vision / multimodal projector initialization. Prefer MTMD because
            # Bonsai-2 carries an embedded Qwen-family multimodal chat template.
            if self.mmproj_path and os.path.exists(self.mmproj_path):
                self.chat_handler = None
                try:
                    from llama_cpp.llama_chat_format import MTMDChatHandler
                    self.chat_handler = MTMDChatHandler(
                        clip_model_path=self.mmproj_path,
                        verbose=self.verbose,
                    )
                except Exception:
                    try:
                        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                        self.chat_handler = Qwen25VLChatHandler(
                            clip_model_path=self.mmproj_path,
                            verbose=self.verbose,
                        )
                    except Exception:
                        try:
                            from llama_cpp.llama_chat_format import Llava15ChatHandler
                            self.chat_handler = Llava15ChatHandler(
                                clip_model_path=self.mmproj_path,
                                verbose=self.verbose,
                            )
                        except Exception:
                            self.chat_handler = None

            self.model = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                chat_handler=self.chat_handler,
                lora_path=self.adapter_path if os.path.isfile(self.adapter_path) else None,
                no_perf=True,
                verbose=self.verbose
            )
            return True
        except Exception:
            self.model = None
            return False

    def training_ready(self) -> bool:
        """Return True only when the exact-GGUF LoRA trainer can be prepared."""
        if not self.is_gguf_available or not os.path.isfile(self.model_path):
            return False
        try:
            from core.gguf_lora_trainer import GGUFLoRATrainer
            trainer = GGUFLoRATrainer(
                model_path=self.model_path,
                base_model_id=self.training_base_model_id,
                adapter_root=self.adapter_root,
            )
            return bool(trainer.can_prepare())
        except Exception:
            return False

    @property
    def tokenizer(self):
        """Compatibility tokenizer facade for the shared Learn/awake trainer contract."""
        return self.model

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        if self.model is None:
            return max(1, sum(len(str(m.get("content", ""))) for m in messages or []) // 4)
        text = "\n".join(str(m.get("content", "")) for m in messages or [])
        try:
            return len(self.model.tokenize(text.encode("utf-8")))
        except Exception:
            return max(1, len(text) // 4)

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
        """Train a real LoRA directly on this exact GGUF, then hot-reload llama.cpp."""
        del adapters, fisher_matrix, lambda_ewc, save_path
        from core.gguf_lora_trainer import GGUFLoRATrainer

        # Preserve the currently working learned state until the new update has
        # both trained and reloaded successfully.
        backup_path = self.adapter_path + ".previous"
        peft_dir = os.path.join(self.adapter_root, "peft")
        peft_backup = peft_dir + ".previous-runtime"
        had_adapter = os.path.isfile(self.adapter_path)
        had_peft = os.path.isdir(peft_dir)
        if had_adapter:
            shutil.copy2(self.adapter_path, backup_path)
        if os.path.isdir(peft_backup):
            shutil.rmtree(peft_backup, ignore_errors=True)
        if had_peft:
            shutil.copytree(peft_dir, peft_backup)

        success = False
        self.unload_model()
        trainer = GGUFLoRATrainer(
            model_path=self.model_path,
            base_model_id=self.training_base_model_id,
            adapter_root=self.adapter_root,
        )
        try:
            meta, drift, _touched, adapter_path = trainer.train(
                data,
                learning_rate=float(learning_rate),
                steps=max(1, int(steps)),
            )
            self.adapter_path = adapter_path
            self.adapters = dict(meta or {})
            if not self.load_model():
                raise RuntimeError("GGUF LoRA trained successfully but llama.cpp failed to reload it")
            result = (dict(self.adapters), float(drift))
            success = True
            return result
        except BaseException:
            success = False
            # Roll back the adapter atomically and restore inference. Cancellation
            # (KeyboardInterrupt) is a transaction failure too.
            # failed consolidation as the live parameter state.
            if os.path.isfile(backup_path):
                os.replace(backup_path, self.adapter_path)
            elif not had_adapter:
                try:
                    os.remove(self.adapter_path)
                except OSError:
                    pass

            if os.path.isdir(peft_dir):
                shutil.rmtree(peft_dir, ignore_errors=True)
            if os.path.isdir(peft_backup):
                os.replace(peft_backup, peft_dir)
            elif not had_peft:
                shutil.rmtree(peft_dir, ignore_errors=True)

            self.load_model()
            raise
        finally:
            if success:
                try:
                    os.remove(backup_path)
                except OSError:
                    pass
                if os.path.isdir(peft_backup):
                    shutil.rmtree(peft_backup, ignore_errors=True)

    def supports_media_input(self, kind: str) -> bool:
        """Bonsai GGUF accepts images and sampled-video frames when its projector is loaded."""
        return (
            str(kind or "").lower() in {"image", "video"}
            and self.model is not None
            and self.chat_handler is not None
        )

    @staticmethod
    def _image_data_url_from_bytes(raw: bytes, mime: str = "image/jpeg") -> str:
        return "data:" + mime + ";base64," + base64.b64encode(raw).decode("ascii")

    def _video_frame_data_urls(self, path: str, max_frames: int = 8) -> List[str]:
        import imageio.v3 as iio
        from PIL import Image

        arrays = []
        try:
            meta = iio.immeta(path)
            total = int(meta.get("nframes") or meta.get("n_images") or 0)
        except Exception:
            total = 0

        if total > 0:
            indices = sorted({
                int(round(i * max(0, total - 1) / max(1, max_frames - 1)))
                for i in range(max_frames)
            })
            for idx in indices:
                try:
                    arrays.append(iio.imread(path, index=idx))
                except Exception:
                    pass
        else:
            try:
                for idx, frame in enumerate(iio.imiter(path)):
                    if idx >= max_frames:
                        break
                    arrays.append(frame)
            except Exception:
                pass

        if not arrays:
            raise RuntimeError("Video contained no decodable frames")

        urls: List[str] = []
        for array in arrays:
            buf = io.BytesIO()
            Image.fromarray(array).convert("RGB").save(buf, format="JPEG", quality=90)
            urls.append(self._image_data_url_from_bytes(buf.getvalue(), "image/jpeg"))
        return urls

    def review_media_input(self, path: str, kind: str, prompt: str = "") -> Dict[str, Any]:
        kind = str(kind or "").lower()
        if not self.supports_media_input(kind):
            return {
                "perception_available": False,
                "reason": f"GGUF backend cannot ingest {kind} input with the currently loaded projector.",
            }
        if not os.path.isfile(path):
            raise ValueError("Media input file is missing")

        task = str(prompt or "").strip() or f"Describe what is actually visible in this {kind}."
        review_prompt = (
            task
            + "\nReturn JSON only with keys description, score, reasoning. "
              "score must be a number from 0 to 100 representing how well the supplied media satisfies the request. "
              "Do not infer unseen content."
        )

        content: List[Dict[str, Any]] = [{"type": "text", "text": review_prompt}]
        if kind == "image":
            if os.path.getsize(path) > 64 * 1024 * 1024:
                raise ValueError("Image review input is limited to 64 MiB")
            with open(path, "rb") as handle:
                raw = handle.read()
            mime = mimetypes.guess_type(path)[0] or "image/jpeg"
            content.append({
                "type": "image_url",
                "image_url": {"url": self._image_data_url_from_bytes(raw, mime)},
            })
        else:
            for frame_url in self._video_frame_data_urls(path):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": frame_url},
                })

        result = self.model.create_chat_completion(
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=768,
        )
        text = str(result.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
        match = re.search(r"\{[\s\S]*\}", text)
        parsed: Dict[str, Any] = {}
        if match:
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                parsed = {}
        score = parsed.get("score")
        response = {
            "perception_available": True,
            "description": str(parsed.get("description") or text),
            "analysis": str(parsed.get("reasoning") or text),
        }
        if isinstance(score, (int, float)):
            response["score"] = max(0.0, min(100.0, float(score)))
        else:
            response["reason"] = "Model inspected the media but did not return a numeric self-grade."
        return response


    def generate_branches(
        self,
        prompt: str,
        branch_count: int = 1,
        max_tokens: int = 1536,
        temperature: Any = 0.75,
        top_p: float = 0.92,
        image_bytes: Optional[bytes] = None
    ) -> List[str]:
        """Generates candidate reasoning branches using GGUF model."""
        if self.model is None:
            return []

        branches = []
        for b_idx in range(branch_count):
            t_val = float(temperature[b_idx % len(temperature)]) if isinstance(temperature, (list, tuple)) else float(temperature)
            try:
                output = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=t_val,
                    top_p=top_p,
                    stop=["<|im_end|>", "</s>", "\nuser", "\nHuman:"]
                )
                text = output["choices"][0]["text"].strip()
                branches.append(text)
            except Exception as e:
                branches.append(f"⚠️ GGUF generation error: {str(e)}")

        return branches

    def stream_generate_tokens(
        self,
        prompt: str,
        max_tokens: int = 1536,
        temperature: float = 0.75,
        top_p: float = 0.92,
        image_bytes: Optional[bytes] = None
    ) -> Generator[str, None, None]:
        """Yields live streamed tokens from GGUF inference."""
        if self.model is None:
            return

        try:
            stream = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                stop=["<|im_end|>", "</s>", "\nuser", "\nHuman:"]
            )
            for chunk in stream:
                token = chunk["choices"][0]["text"]
                if token:
                    yield token
        except Exception:
            return

    def calculate_token_entropy(self, prompt: str) -> float:
        """Calculates Shannon entropy from GGUF next-token logits."""
        if self.model is None:
            return 0.35

        try:
            # Evaluate prompt and extract final logits
            tokens = self.model.tokenize(prompt.encode("utf-8"))
            if not tokens:
                return 0.35
            self.model.eval(tokens[-32:])
            logits = self.model._scores[-1]

            # Softmax & Entropy
            max_l = max(logits)
            exp_l = [math.exp(x - max_l) for x in logits]
            sum_exp = sum(exp_l)
            probs = [p / sum_exp for p in exp_l if p > 0]
            entropy = -sum(p * math.log(p) for p in probs if p > 1e-12)
            max_ent = math.log(max(2, len(logits)))
            return round(entropy / max_ent, 4)
        except Exception:
            return 0.35

    def unload_model(self):
        """Releases all GGUF weights and context memory."""
        if self.model is not None:
            try:
                del self.model
            except Exception:
                pass
            self.model = None
            self.chat_handler = None
            import gc
            gc.collect()
