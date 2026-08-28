"""
Pro Reasoning Engine.
Manages test-time compute scaling, entropy routing, parallel Best-of-N rollouts,
Process Reward Model (PRM) trajectory scoring, and deterministic RLVR sandbox verification.
Supports PyTorch (CUDA / MPS / CPU), Apple Silicon native MLX, and Mock backends.
"""

import math
import os
import platform
import random
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from config.settings import Settings, get_settings
from core.entropy_router import EntropyRouter
from core.hf_downloader import is_model_cached_locally
from core.mlx_engine import MLXReasoningBackend
from core.platform import get_auto_context_window_size
from core.speculative_engine import SpeculativeEngine
from core.verifier import GroundTruthVerifier, VerificationResult


def parse_reasoning_and_response(raw_text: str) -> Tuple[Optional[str], str]:
    """Separates internal thinking process from final assistant response."""
    if not raw_text:
        return None, ""
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    match = think_pattern.search(raw_text)
    if match:
        thinking = match.group(1).strip()
        final = think_pattern.sub("", raw_text).strip()
        return (thinking if thinking else None), (final if final else raw_text)

    # Heuristic for unstructured chain-of-thought
    if "Here's a thinking process:" in raw_text:
        parts = raw_text.split("Here's a thinking process:", 1)
        subparts = re.split(r"(?:\n\n(?=[A-Z#*])|Draft the response.*?:\s*)", parts[1], maxsplit=1)
        if len(subparts) == 2:
            return subparts[0].strip(), subparts[1].strip()

    return None, raw_text


class ProReasoningEngine:
    def __init__(
        self,
        base_model_path: Optional[str] = None,
        lora_adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        backend: Optional[str] = None,
        settings: Optional[Settings] = None
    ):
        self.settings = settings or get_settings()
        self.base_model_path = base_model_path or self.settings.base_model_path
        self.lora_adapter_path = lora_adapter_path or self.settings.lora_adapter_path
        self.device = device or self.settings.device
        self.backend = backend or self.settings.backend

        self.router = EntropyRouter(
            low_threshold=self.settings.entropy_low_threshold,
            high_threshold=self.settings.entropy_high_threshold,
            instant_branches=self.settings.instant_branch_count,
            pro_branches_mid=8,
            pro_branches_high=self.settings.pro_branch_count_default
        )

        self.verifier = GroundTruthVerifier(
            sandbox_timeout=self.settings.sandbox_timeout_seconds,
            use_docker=self.settings.use_docker_sandbox,
            docker_image=self.settings.docker_image
        )

        self.mlx_backend = None
        self.model = None
        self.base_model = None
        self.tokenizer = None
        self.active_model_name = None
        self._model_lock = threading.Lock()

        self.speculative_engine = SpeculativeEngine(
            target_model=None,
            tokenizer=None,
            mode=self.settings.speculative_mode,
            max_draft_tokens=self.settings.speculative_tokens
        )

    def load_model(self, model_name: str) -> Dict[str, Any]:
        """
        Enforces strict single-model mutual exclusion with zero memory leak:
        Completely purges any loaded model from VRAM and RAM before loading the new model.
        """
        with self._model_lock:
            self.unload_model()  # Strictly unload previous model first
            self.active_model_name = model_name

            if "qwen" in model_name.lower():
                target_path = self.settings.flash_model_path
            elif "dolphin" in model_name.lower():
                target_path = "cognitivecomputations/dolphin-2.9.2-qwen2-7b"
            else:
                target_path = self.settings.mlx_model_path

            # If model is not cached locally, do not attempt heavy unquantized load
            if not is_model_cached_locally(target_path) and not os.path.exists(target_path):
                return {
                    "status": "not_downloaded",
                    "model": model_name,
                    "path": target_path,
                    "message": f"Weights for {model_name} ({target_path}) are not downloaded yet."
                }

            try:
                # 1. Native Apple Silicon MLX (Quantized BitLinear / 4-bit)
                if platform.system() == "Darwin" and platform.machine() == "arm64":
                    self.mlx_backend = MLXReasoningBackend(
                        model_path=target_path,
                        adapter_path=self.lora_adapter_path
                    )
                    if self.mlx_backend.load_model():
                        return {
                            "status": "loaded",
                            "model": model_name,
                            "backend": "mlx",
                            "path": target_path
                        }
                    else:
                        return {"status": "not_downloaded", "model": model_name, "path": target_path}

                # 2. CUDA on Linux / Windows only (strictly GPU memory bounds)
                if self.device == "cuda":
                    import torch
                    if torch.cuda.is_available():
                        from transformers import AutoModelForCausalLM, AutoTokenizer
                        self.tokenizer = AutoTokenizer.from_pretrained(target_path, trust_remote_code=True)
                        self.model = AutoModelForCausalLM.from_pretrained(
                            target_path,
                            torch_dtype=torch.bfloat16,
                            device_map="auto",
                            trust_remote_code=True
                        )
                        self.model.eval()
                        return {"status": "loaded", "model": model_name, "backend": "cuda", "path": target_path}

                return {"status": "not_downloaded", "model": model_name, "path": target_path}

            except Exception as e:
                return {"status": "error", "model": model_name, "error": str(e), "path": target_path}

    def unload_model(self) -> Dict[str, Any]:
        """
        Completely purges all model instances, tokenizers, and framework caches from memory.
        Ensures strictly zero lingering VRAM or RAM footprint.
        """
        old_model = self.active_model_name

        if hasattr(self, "model") and self.model is not None:
            try:
                del self.model
            except Exception:
                pass
            self.model = None

        if hasattr(self, "base_model") and self.base_model is not None:
            try:
                del self.base_model
            except Exception:
                pass
            self.base_model = None

        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            try:
                del self.tokenizer
            except Exception:
                pass
            self.tokenizer = None

        if hasattr(self, "mlx_backend") and self.mlx_backend is not None:
            try:
                if hasattr(self.mlx_backend, "model") and self.mlx_backend.model is not None:
                    del self.mlx_backend.model
                    self.mlx_backend.model = None
                if hasattr(self.mlx_backend, "tokenizer") and self.mlx_backend.tokenizer is not None:
                    del self.mlx_backend.tokenizer
                    self.mlx_backend.tokenizer = None
                del self.mlx_backend
            except Exception:
                pass
            self.mlx_backend = None

        if hasattr(self, "speculative_engine") and self.speculative_engine is not None:
            self.speculative_engine.target_model = None
            self.speculative_engine.tokenizer = None

        import gc
        gc.collect(2)

        # Clear PyTorch CUDA / MPS (Metal Performance Shaders) cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass

        # Clear Apple MLX cache
        try:
            import mlx.core as mx
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass

        # Trigger macOS OS memory zone pressure relief
        if sys.platform == "darwin":
            try:
                import ctypes
                libc = ctypes.CDLL("libc.dylib")
                if hasattr(libc, "malloc_zone_pressure_relief"):
                    libc.malloc_zone_pressure_relief(0, 0)
            except Exception:
                pass

        self.active_model_name = None
        return {"status": "unloaded", "previous_model": old_model}

    def calculate_token_entropy(self, prompt: str) -> float:
        """Evaluates model next-token Shannon entropy across supported backends."""
        if self.mlx_backend and self.mlx_backend.is_mlx_available:
            return self.mlx_backend.calculate_token_entropy(prompt)

        if self.model is None:
            return self.router.estimate_prompt_entropy_heuristic(prompt)

        try:
            import torch
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.device in ("cuda", "mps") and hasattr(inputs, "to"):
                inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]
                return self.router.compute_entropy_from_logits(logits)
        except Exception:
            return self.router.estimate_prompt_entropy_heuristic(prompt)

    def _format_prompt_with_history(self, prompt: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Packs previous conversational turns within context window without warning pollution."""
        max_context_tokens = get_auto_context_window_size()
        reserved_gen_tokens = self.settings.max_new_tokens or 1536
        prompt_est_tokens = len(prompt.split()) * 2
        history_token_budget = max_context_tokens - reserved_gen_tokens - prompt_est_tokens
        history_char_budget = history_token_budget * 4

        if not history or len(history) <= 1:
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        retained_turns = []
        curr_chars = 0
        for item in reversed(history[:-1]):
            role = item.get("role", "user")
            content = item.get("content", "").strip()
            # Prevent poisoning model with previous warning messages
            if not content or "weights are not currently loaded" in content or "Click '⬇️ Download" in content:
                continue
            turn_str = f"<|im_start|>{role}\n{content}<|im_end|>\n"
            if curr_chars + len(turn_str) > history_char_budget:
                break
            retained_turns.append(turn_str)
            curr_chars += len(turn_str)

        retained_turns.reverse()
        context_str = "".join(retained_turns)
        context_str += f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return context_str

    def stream_solve(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.75,
        top_p: float = 0.92,
        cancel_event: Optional[Any] = None
    ):
        """Yields live tokens in real time from the active MLX model."""
        if not (self.mlx_backend and self.mlx_backend.is_mlx_available):
            return

        formatted_prompt = self._format_prompt_with_history(prompt, history)
        for token in self.mlx_backend.stream_generate_tokens(
            prompt=formatted_prompt,
            max_tokens=self.settings.max_new_tokens,
            temperature=temperature,
            top_p=top_p
        ):
            if cancel_event and cancel_event.is_set():
                break
            yield token

    def generate_parallel_branches(self, prompt: str, branch_count: int = 16, history: Optional[List[Dict[str, str]]] = None) -> List[str]:
        """Samples candidate reasoning rollouts using auto-scaled context window based on host RAM."""
        formatted_prompt = self._format_prompt_with_history(prompt, history)

        # 1. Apple Silicon Native MLX Inference
        if self.mlx_backend and self.mlx_backend.is_mlx_available:
            branches = self.mlx_backend.generate_branches(
                prompt=formatted_prompt,
                branch_count=branch_count,
                max_tokens=self.settings.max_new_tokens,
                temperature=self.settings.search_temperature,
                top_p=self.settings.search_top_p
            )
            if branches:
                return branches

        # 2. PyTorch CUDA / MPS / CPU Inference
        if self.model is not None and self.tokenizer is not None:
            try:
                import torch
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
                if self.device in ("cuda", "mps") and hasattr(inputs, "to"):
                    inputs = inputs.to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.settings.max_new_tokens,
                        num_return_sequences=branch_count,
                        do_sample=True,
                        temperature=self.settings.search_temperature,
                        top_p=self.settings.search_top_p,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                prompt_len = inputs["input_ids"].shape[1]
                return [self.tokenizer.decode(out[prompt_len:], skip_special_tokens=True) for out in outputs]
            except Exception as e:
                return [f"⚠️ Error executing neural generation: {str(e)}"]

        m_name = self.active_model_name or "Active Model"
        return [
            f"⚠️ **{m_name}** weights are not currently loaded into memory.\n\n"
            f"• Please click **'⬇️ Download from HuggingFace'** or **'⚡ Load Model'** in the top bar to load the neural weights into Apple Silicon unified memory."
        ]

    def solve(
        self,
        prompt: str,
        test_cases: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        cancel_event: Optional[Any] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Main reasoning loop with conversational history retention:
        1. Evaluates entropy & decides compute budget.
        2. Executes Instant pass or Pro parallel search.
        3. Validates against ground-truth sandbox / SymPy assertions.
        4. Calculates surprise metric for episodic memory.
        """
        start_time = time.perf_counter()
        has_tests = bool(test_cases and test_cases.strip())
        entropy = self.calculate_token_entropy(prompt)
        mode, branch_count = self.router.route(entropy, has_test_cases=has_tests)

        # Path 1: Instant Single-Pass Mode
        if branch_count == 1 and not has_tests:
            candidates = self.generate_parallel_branches(prompt, branch_count=1, history=history)
            raw_response = candidates[0]
            thinking_text, clean_response = parse_reasoning_and_response(raw_response)
            response = clean_response or raw_response
            exec_time = (time.perf_counter() - start_time) * 1000

            metadata = {
                "mode": mode,
                "backend": "mlx" if (self.mlx_backend and self.mlx_backend.is_mlx_available) else ("torch" if self.model is not None else "live_unloaded"),
                "entropy": entropy,
                "branch_count": 1,
                "verified": False,
                "verified_reward": 0.0,
                "surprise_score": 0.05,
                "winning_branch": 0,
                "thinking_text": thinking_text,
                "raw_branches": candidates,
                "execution_time_ms": exec_time,
                "verifier_details": "Single-pass execution without test assertions.",
                "speculative": self.speculative_engine.get_telemetry()
            }
            return response, metadata

        # Path 2: Pro Multi-Branch Search
        candidates = self.generate_parallel_branches(prompt, branch_count=branch_count, history=history)
        winning_branch = 0
        winning_response = candidates[0]
        verified = False
        verified_reward = 0.0
        verifier_details = ""

        # Step 3: Ground-Truth Verification (RLVR)
        if has_tests:
            first_branch_passed = False
            for idx, candidate in enumerate(candidates):
                # Check cancellation between branches
                if cancel_event and cancel_event.is_set():
                    break
                code_segment = self.verifier.extract_code_block(candidate)
                if not code_segment:
                    code_segment = candidate

                v_res = self.verifier.verify_in_sandbox(code_segment, test_cases)
                if idx == 0 and v_res.passed:
                    first_branch_passed = True

                if v_res.passed:
                    winning_branch = idx
                    winning_response = candidate
                    verified = True
                    verified_reward = 1.0
                    verifier_details = f"Sandbox verification passed on branch #{idx+1} ({v_res.execution_time_ms:.1f}ms)"
                    break

            # Calculate Surprise Score
            if verified:
                if first_branch_passed:
                    surprise_score = 0.10 + 0.10 * entropy
                else:
                    surprise_score = 0.50 + 0.40 * (winning_branch / max(1, branch_count - 1)) + 0.10 * entropy
            else:
                surprise_score = 0.80
                verifier_details = "All search branches failed sandbox assertions."
        else:
            winner, count, ratio = self.verifier.consensus_voting(candidates)
            winning_response = winner
            surprise_score = max(0.1, 1.0 - ratio)
            verifier_details = f"Consensus agreement: {count}/{len(candidates)} branches ({ratio*100:.1f}%)"

        exec_time = (time.perf_counter() - start_time) * 1000
        thinking_text, clean_winning = parse_reasoning_and_response(winning_response)
        final_resp = clean_winning or winning_response

        metadata = {
            "mode": mode,
            "backend": "mlx" if (self.mlx_backend and self.mlx_backend.is_mlx_available) else ("torch" if self.model is not None else "live_unloaded"),
            "entropy": round(entropy, 4),
            "branch_count": len(candidates),
            "verified": verified,
            "verified_reward": verified_reward,
            "surprise_score": round(surprise_score, 4),
            "winning_branch": winning_branch,
            "thinking_text": thinking_text,
            "raw_branches": candidates,
            "execution_time_ms": exec_time,
            "verifier_details": verifier_details,
            "speculative": self.speculative_engine.get_telemetry()
        }
        return final_resp, metadata
