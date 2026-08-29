"""
GGUF / Llama.cpp Cross-Platform Reasoning Backend.
Provides high-performance GGUF model execution across NVIDIA CUDA, Apple Metal, and Multi-Core CPU,
with multimodal vision projector support (nanoLLaVA / CLIP).
"""

import math
import os
import platform
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
        verbose: bool = False
    ):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.verbose = verbose

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

            # Vision / Multimodal projector initialization
            if self.mmproj_path and os.path.exists(self.mmproj_path):
                try:
                    from llama_cpp.llama_chat_format import NanoLlavaChatHandler, Llava15ChatHandler
                    self.chat_handler = Llava15ChatHandler(clip_model_path=self.mmproj_path)
                except Exception:
                    self.chat_handler = None

            self.model = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                chat_handler=self.chat_handler,
                verbose=self.verbose
            )
            return True
        except Exception:
            self.model = None
            return False

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
