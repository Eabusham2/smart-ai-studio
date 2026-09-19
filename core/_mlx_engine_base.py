"""
Apple Silicon Native MLX Reasoning Backend.
Provides optimized inference, parallel candidate rollouts, token entropy calculations,
and LoRA sleep consolidation specifically tailored for macOS Apple Silicon unified memory.
"""

import math
import os
import platform
from typing import Any, Dict, List, Optional, Tuple


class MLXReasoningBackend:
    def __init__(
        self,
        model_path: str = "orcarouter/Qwen3.8-27B-Uncensored-MLX",
        adapter_path: Optional[str] = None
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.adapters = {"lora_layer_0": 0.0}
        self.is_mlx_available = False

    def load_model(self) -> bool:
        """
        Loads MLX neural model into Apple Silicon unified memory with dynamic RAM scaling.
        Memory allocation starts lean and dynamically expands on demand as context & rollouts grow.
        """
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            self.is_mlx_available = False
            return False

        if not os.path.exists(self.model_path) and os.getenv("OFFLINE", "0") == "1":
            self.is_mlx_available = False
            return False

        try:
            import mlx.core as mx
            import mlx.nn as nn
            import mlx_lm
            from core.memory_watchdog import SystemMemoryWatchdog

            # Configure dynamic elastic Metal headroom based on available host RAM
            SystemMemoryWatchdog.adjust_dynamic_metal_headroom()

            self.model, self.tokenizer = mlx_lm.load(
                self.model_path,
                adapter_path=self.adapter_path if self.adapter_path and os.path.exists(self.adapter_path) else None
            )
            self.is_mlx_available = True
            return True
        except Exception as e:
            self.is_mlx_available = False
            return False

    def get_memory_breakdown(self) -> Dict[str, Any]:
        """Provides granular live memory metrics for MLX model and Apple Silicon Metal buffers."""
        from core.memory_watchdog import SystemMemoryWatchdog
        return SystemMemoryWatchdog.get_detailed_memory_breakdown()

    def calculate_token_entropy(self, prompt: str) -> float:
        """Computes next-token Shannon entropy using native MLX arrays."""
        if not self.is_mlx_available or self.model is None:
            return 0.45

        try:
            import mlx.core as mx
            import math

            tokens = self.tokenizer.encode(prompt)
            if not tokens:
                return 0.45
            input_ids = mx.array([tokens])
            logits = self.model(input_ids)
            last_logits = logits[:, -1, :]

            # Compute Shannon entropy: H = -sum(p * log(p))
            probs = mx.softmax(last_logits, axis=-1)
            log_probs = mx.log(probs + 1e-12)
            entropy_val = -mx.sum(probs * log_probs, axis=-1).item()
            if math.isnan(entropy_val) or math.isinf(entropy_val):
                return 0.45
            return float(entropy_val)
        except Exception:
            return 0.45

    def generate_branches(
        self,
        prompt: str,
        branch_count: int = 16,
        max_tokens: int = 512,
        temperature: Any = 0.75,
        top_p: float = 0.92
    ) -> List[str]:
        """Generates candidate reasoning branches via MLX with strict memory reclamation."""
        if not self.is_mlx_available or self.model is None:
            return []

        try:
            import gc
            import mlx.core as mx
            import mlx_lm

            try:
                from mlx_lm.sample_utils import make_sampler
                has_make_sampler = True
            except Exception:
                has_make_sampler = False

            branches = []
            count = max(1, min(branch_count, 16))

            for b_idx in range(count):
                # Always clear Metal cache and collect garbage before each branch rollout
                gc.collect(1)
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()

                # Determine branch-specific temperature from ladder if provided
                if isinstance(temperature, (list, tuple)):
                    t_val = float(temperature[b_idx % len(temperature)])
                else:
                    t_val = float(temperature)

                try:
                    kwargs = {"max_tokens": min(max_tokens, 1024), "verbose": False}
                    if has_make_sampler:
                        try:
                            kwargs["sampler"] = make_sampler(temp=t_val, top_p=top_p)
                        except Exception:
                            kwargs["temp"] = t_val
                            kwargs["top_p"] = top_p
                    else:
                        kwargs["temp"] = t_val
                        kwargs["top_p"] = top_p

                    response = mlx_lm.generate(
                        self.model,
                        self.tokenizer,
                        prompt=prompt,
                        **kwargs
                    )
                except Exception:
                    try:
                        response = mlx_lm.generate(
                            self.model,
                            self.tokenizer,
                            prompt=prompt,
                            max_tokens=min(max_tokens, 1024),
                            temp=t_val,
                            verbose=False
                        )
                    except Exception:
                        response = mlx_lm.generate(
                            self.model,
                            self.tokenizer,
                            prompt=prompt,
                            max_tokens=min(max_tokens, 1024),
                            verbose=False
                        )

                branches.append(response)

                # Reclaim Metal cache after each branch
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()

            return branches
        except Exception as e:
            logger.error(f"MLX generate_branches failed: {e}")
            return []

    def stream_generate_tokens(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.75,
        top_p: float = 0.92
    ):
        """Yields live tokens in real-time as they are synthesized on Apple Silicon."""
        if not self.is_mlx_available or self.model is None:
            return

        try:
            import mlx.core as mx
            import mlx_lm

            sampler = None
            try:
                from mlx_lm.sample_utils import make_sampler
                sampler = make_sampler(temp=temperature, top_p=top_p)
            except Exception:
                pass

            kwargs = {"max_tokens": min(max_tokens, 1024)}
            if sampler is not None:
                kwargs["sampler"] = sampler
            else:
                kwargs["temp"] = temperature
                kwargs["top_p"] = top_p

            for response in mlx_lm.stream_generate(self.model, self.tokenizer, prompt=prompt, **kwargs):
                if hasattr(response, "text"):
                    yield response.text
                elif isinstance(response, str):
                    yield response
                elif hasattr(response, "token"):
                    yield self.tokenizer.decode([response.token])

            # Reclaim Metal cache after generation
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception as e:
            logger.error(f"MLX stream_generate error: {e}")
            try:
                ans = mlx_lm.generate(self.model, self.tokenizer, prompt=prompt, max_tokens=min(max_tokens, 512), verbose=False)
                if ans:
                    yield ans
            except Exception:
                pass

    def inject_lora_adapters(self, r: int = 8, scale: float = 2.0) -> Dict[str, Any]:
        """
        Attaches trainable LoRA adapter tensors to base model linear projection layers
        while freezing all foundation weights.
        """
        if not self.is_mlx_available or self.model is None:
            raise RuntimeError("Cannot inject LoRA adapters: MLX model is not loaded.")

        import mlx.core as mx
        import mlx.nn as nn
        import mlx.utils
        from mlx_lm.tuner.lora import LoRALinear

        self.model.freeze()
        lora_count = 0

        target_layers = getattr(self.model, "layers", None)
        if target_layers is None and hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            target_layers = self.model.model.layers

        if target_layers:
            for layer in target_layers:
                # MLP down_proj
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                    layer.mlp.down_proj = LoRALinear.from_base(layer.mlp.down_proj, r=r, scale=scale)
                    layer.mlp.down_proj.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
                    lora_count += 1
                # Attention projections
                if hasattr(layer, "self_attn"):
                    if hasattr(layer.self_attn, "q_proj"):
                        layer.self_attn.q_proj = LoRALinear.from_base(layer.self_attn.q_proj, r=r, scale=scale)
                        layer.self_attn.q_proj.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
                        lora_count += 1
                    if hasattr(layer.self_attn, "v_proj"):
                        layer.self_attn.v_proj = LoRALinear.from_base(layer.self_attn.v_proj, r=r, scale=scale)
                        layer.self_attn.v_proj.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
                        lora_count += 1
                elif hasattr(layer, "linear_attn") and hasattr(layer.linear_attn, "out_proj"):
                    layer.linear_attn.out_proj = LoRALinear.from_base(layer.linear_attn.out_proj, r=r, scale=scale)
                    layer.linear_attn.out_proj.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
                    lora_count += 1

        trainable_params = dict(mlx.utils.tree_flatten(self.model.trainable_parameters()))
        self.adapters = trainable_params
        return trainable_params

    def compute_mlx_fisher(self, anchor_texts: List[str]) -> Dict[str, Any]:
        """Compute diagonal Fisher while releasing each anchor's autograd state immediately."""
        if not self.is_mlx_available or self.model is None or self.tokenizer is None:
            return {}

        import gc
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.utils

        trainable_params = dict(mlx.utils.tree_flatten(self.model.trainable_parameters()))
        if not trainable_params:
            self.inject_lora_adapters(r=8)
            trainable_params = dict(mlx.utils.tree_flatten(self.model.trainable_parameters()))
        if not trainable_params:
            return {}

        fisher_matrix = {k: mx.zeros_like(v) for k, v in trainable_params.items()}
        mx.eval(*fisher_matrix.values())

        def loss_fn(model, inputs, targets):
            logits = model(inputs)
            logits = logits[:, :-1, :]
            targets = targets[:, 1:]
            return mx.mean(nn.losses.cross_entropy(logits, targets))

        grad_fn = mx.grad(loss_fn)
        valid_anchors = 0
        was_training = bool(getattr(self.model, "training", False))

        try:
            self.model.train()
            for text in anchor_texts:
                tokens = self.tokenizer.encode(text)
                if len(tokens) < 2:
                    continue

                inputs = mx.array([tokens])
                grads = grad_fn(self.model, inputs, inputs)
                flat_grads = dict(mlx.utils.tree_flatten(grads))
                if flat_grads:
                    mx.eval(*flat_grads.values())

                for key, grad in flat_grads.items():
                    if key not in fisher_matrix:
                        continue
                    grad = mx.stop_gradient(grad)
                    value = fisher_matrix[key] + (grad * grad)
                    mx.eval(value)
                    fisher_matrix[key] = mx.stop_gradient(value)

                valid_anchors += 1
                grads = None
                flat_grads = None
                inputs = None
                gc.collect(2)
                try:
                    mx.clear_cache()
                except Exception:
                    pass

            if valid_anchors > 0:
                inv = 1.0 / float(valid_anchors)
                for key in list(fisher_matrix):
                    value = fisher_matrix[key] * inv
                    mx.eval(value)
                    fisher_matrix[key] = mx.stop_gradient(value)
            return fisher_matrix
        finally:
            if not was_training:
                try:
                    self.model.eval()
                except Exception:
                    pass
            gc.collect(2)
            try:
                mx.clear_cache()
            except Exception:
                pass


    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Calculates total token length of conversation turns."""
        if not messages:
            return 0
        if self.tokenizer is not None:
            try:
                if hasattr(self.tokenizer, "apply_chat_template"):
                    formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
                    return len(self.tokenizer.encode(formatted_text))
                else:
                    text = "\n".join(m.get("content", "") for m in messages)
                    return len(self.tokenizer.encode(text))
            except Exception:
                pass
        # Fallback estimation
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return max(1, total_chars // 4)

    def train_mini_batch(
        self,
        adapters: Any,
        data: List[Dict[str, str]],
        fisher_matrix: Optional[Dict[str, Any]] = None,
        reference_weights: Optional[Dict[str, Any]] = None,
        lambda_ewc: float = 400.0,
        learning_rate: float = 1e-4,
        steps: int = 3,
        save_path: Optional[str] = None
    ) -> Tuple[Dict[str, Any], float]:
        """Run a real transactional MLX AdamW/EWC LoRA update."""
        del adapters
        if not self.is_mlx_available or self.model is None or self.tokenizer is None:
            raise RuntimeError("MLX training requires the currently loaded real model and tokenizer")

        import gc
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        import mlx.utils

        trainable_params = dict(mlx.utils.tree_flatten(self.model.trainable_parameters()))
        if not trainable_params:
            self.inject_lora_adapters(r=8)
            trainable_params = dict(mlx.utils.tree_flatten(self.model.trainable_parameters()))
        if not trainable_params:
            raise RuntimeError("MLX training exposes no trainable adapter parameters")

        def _clone(value):
            try:
                return mx.copy(value)
            except Exception:
                return value + mx.zeros_like(value)

        rollback_params = {key: _clone(value) for key, value in trainable_params.items()}
        mx.eval(*rollback_params.values())

        w_initial_flat = mx.concat(
            [mx.reshape(value, (-1,)) for value in rollback_params.values()]
        )
        mx.eval(w_initial_flat)

        if reference_weights is None:
            reference_weights = dict(rollback_params)

        optimizer = optim.AdamW(learning_rate=learning_rate)
        tmp_save_path = None
        was_training = bool(getattr(self.model, "training", False))

        def ewc_loss_fn(model, inputs, targets):
            logits = model(inputs)
            logits = logits[:, :-1, :]
            targets = targets[:, 1:]
            ce_loss = mx.mean(nn.losses.cross_entropy(logits, targets))

            if fisher_matrix and reference_weights and lambda_ewc > 0:
                penalty = mx.array(0.0)
                current_params = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
                for key, weight in current_params.items():
                    if key in fisher_matrix and key in reference_weights:
                        diff = weight - reference_weights[key]
                        penalty = penalty + mx.sum(fisher_matrix[key] * (diff ** 2))
                ce_loss = ce_loss + (lambda_ewc / 2.0) * penalty
            return ce_loss

        loss_and_grad_fn = nn.value_and_grad(self.model, ewc_loss_fn)

        try:
            self.model.train()
            trained_rows = 0
            for _step in range(max(1, int(steps))):
                for item in data:
                    prompt_text = item.get("prompt", "")
                    completion_text = item.get("completion", "")
                    full_text = (
                        f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
                        f"<|im_start|>assistant\n{completion_text}<|im_end|>"
                    )
                    tokens = self.tokenizer.encode(full_text)
                    if len(tokens) < 2:
                        continue

                    inputs = mx.array([tokens])
                    loss, grads = loss_and_grad_fn(self.model, inputs, inputs)
                    optimizer.update(self.model, grads)
                    mx.eval(self.model.parameters(), optimizer.state)
                    trained_rows += 1

                    loss = None
                    grads = None
                    inputs = None
                    gc.collect(1)
                    try:
                        mx.clear_cache()
                    except Exception:
                        pass

            if trained_rows <= 0:
                raise RuntimeError("MLX training found no valid prompt/completion token rows")

            updated_params = dict(mlx.utils.tree_flatten(self.model.trainable_parameters()))
            w_final_flat = mx.concat(
                [mx.reshape(value, (-1,)) for value in updated_params.values()]
            )
            mx.eval(w_final_flat)
            param_drift = float(mx.linalg.norm(w_final_flat - w_initial_flat).item())
            if param_drift <= 0.0:
                raise RuntimeError("MLX training completed but measured zero parameter change")

            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                tmp_save_path = save_path + ".next.safetensors"
                try:
                    if os.path.exists(tmp_save_path):
                        os.remove(tmp_save_path)
                except OSError:
                    pass
                mx.save_safetensors(tmp_save_path, updated_params)
                os.replace(tmp_save_path, save_path)
                tmp_save_path = None

            self.adapters = updated_params
            return updated_params, param_drift

        except BaseException:
            # A failed update/persist must not remain live. Restore only the LoRA
            # trainables captured before the transaction; foundation weights stay frozen.
            try:
                self.model.update(
                    mlx.utils.tree_unflatten(list(rollback_params.items()))
                )
                mx.eval(self.model.parameters())
                self.adapters = dict(
                    mlx.utils.tree_flatten(self.model.trainable_parameters())
                )
            finally:
                if tmp_save_path:
                    try:
                        if os.path.exists(tmp_save_path):
                            os.remove(tmp_save_path)
                    except OSError:
                        pass
            raise
        finally:
            if not was_training:
                try:
                    self.model.eval()
                except Exception:
                    pass
            rollback_params.clear()
            gc.collect(2)
            try:
                mx.clear_cache()
            except Exception:
                pass

