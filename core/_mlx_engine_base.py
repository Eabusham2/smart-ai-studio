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
    supports_learning_modes = True

    def __init__(
        self,
        model_path: str = "orcarouter/Qwen3.8-27B-Uncensored-MLX",
        adapter_path: Optional[str] = None
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.adapters = {}
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

    def _learning_mode_for_call(self, requested, options):
        from core.learning_implementation import resolve_learning_mode
        from core.learning_state import check_hybrid_store

        mode = resolve_learning_mode(self, requested, options)
        check_hybrid_store(getattr(self, "learning_state", None), mode)
        return mode

    def _uses_original_projections(self, mode):
        # Packed VLMs require the native LoRA wrappers and objectives introduced in PR17.
        return mode in {"legacy", "hybrid"} and getattr(self, "_vlm_runtime", None) not in {"mlx_repo_vlm", "jang_vlm"}

    def inject_lora_adapters(self, r: int = 8, scale: float = 2.0, layout=None, target_paths=None,
                             *, learning_mode=None) -> Dict[str, Any]:
        """Install projections for the selected learner or reconstruct an exact checkpoint layout."""
        if not self.is_mlx_available or self.model is None:
            raise RuntimeError("Cannot inject LoRA adapters: MLX model is not loaded.")
        mode = self._learning_mode_for_call(learning_mode, {})
        if self._uses_original_projections(mode) and layout is None and target_paths is None:
            from core.mlx_legacy_learning import inject_adapters
            return inject_adapters(self, r, scale)
        from core.mlx_learning import ensure_adapters
        return ensure_adapters(self, r, scale, layout, target_paths=target_paths)

    def compute_mlx_fisher(self, anchor_texts: List[str], *, learning_mode=None, **options):
        """Compute the original whole-anchor or managed window-weighted Fisher diagonal."""
        if not self.is_mlx_available or self.model is None or self.tokenizer is None:
            return {}
        mode = self._learning_mode_for_call(learning_mode, options)
        if mode == "legacy" and self._uses_original_projections(mode):
            from core.mlx_legacy_learning import fisher_diagonal
            return fisher_diagonal(self, anchor_texts, **options)
        from core.mlx_learning import bounded_tokens, fisher_diagonal
        max_tokens = bounded_tokens(self, options.pop("max_window_tokens", 128))
        if mode == "hybrid" and self._uses_original_projections(mode):
            from core.mlx_hybrid_learning import fisher_diagonal
        return fisher_diagonal(self, anchor_texts, max_tokens, **options)

    def score_learning_probe(self, prompt, completion, *, max_window_tokens=128):
        """Mean teacher-forced NLL on the supplied completion, using the current live weights."""
        if not self.is_mlx_available or self.model is None or self.tokenizer is None:
            raise RuntimeError("MLX learning probe requires the loaded language model and tokenizer")
        from core.mlx_learning import bounded_tokens, score_rows
        rows = [{"prompt": prompt, "completion": completion}]
        was_training = bool(getattr(self.model, "training", False))
        try:
            self.model.eval()
            return score_rows(self.model, self.tokenizer, rows, bounded_tokens(self, max_window_tokens))
        finally:
            if was_training:
                self.model.train()

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
        save_path: Optional[str] = None,
        *,
        learning_mode=None,
        **options,
    ) -> Tuple[Dict[str, Any], float]:
        """Select full-row legacy training or the managed windowed objectives."""
        mode = self._learning_mode_for_call(learning_mode, options)
        return self._update_adapters(adapters, data, fisher_matrix, reference_weights, lambda_ewc,
                                     learning_rate, steps, save_path, mode=mode, **options)

    def _update_adapters(
        self, adapters, data, fisher_matrix, reference_weights, lambda_ewc, learning_rate, steps, save_path,
        *, mode,
        max_window_tokens=128,
        supervise_prompt=True,
        objective="causal_lm",
        reward_weighting="group_relative", entropic_kl_target=None, reference_kl_coef=0.0,
        row_weights=None,
        context_text=None,
        max_context_tokens=2048,
        anchor_texts=None,
        retention_data=None,
        retention_tolerance=0.1,
        cancel_event=None,
        validation_callback=None,
        learning_metadata=None,
    ) -> Tuple[Dict[str, Any], float]:
        """Update only live adapters and publish one validated, recoverable generation."""
        del adapters
        if not self.is_mlx_available or self.model is None or self.tokenizer is None:
            raise RuntimeError("MLX training requires the currently loaded real model and tokenizer")

        import mlx.core as mx
        import mlx.optimizers as optim
        from core import mlx_legacy_learning
        from mlx.utils import tree_unflatten
        from core.mlx_adapter_persistence import persist_candidate, resume_optimizer
        from core.mlx_learning import (
            bounded_tokens, cancel_if_requested, check_finite, ensure_adapters,
            flat_trainables, optimize_windows, score_rows,
        )

        original_rows = self._uses_original_projections(mode) and objective == "causal_lm"
        legacy = mode == "legacy" and original_rows
        if (not math.isfinite(learning_rate) or learning_rate <= 0
                or not math.isfinite(lambda_ewc) or (not legacy and (lambda_ewc < 0 or int(steps) < 1))):
            raise ValueError("Invalid MLX learning rate, EWC coefficient or step count")
        if not math.isfinite(retention_tolerance) or retention_tolerance < 0:
            raise ValueError("Invalid retention loss tolerance")
        max_tokens = None if legacy else bounded_tokens(self, max_window_tokens)
        cancel_if_requested(cancel_event)
        if objective not in {"causal_lm", "reward_policy", "query_only"}:
            raise ValueError("Unknown test-time training objective")
        if objective != "causal_lm" and row_weights is not None:
            raise ValueError("Policy and query objectives supply their own weights")
        if objective == "reward_policy":
            from core.reward_adaptation import reward_advantages
            reward_advantages(data, reward_weighting=reward_weighting, entropic_kl_target=entropic_kl_target)
        rows = tuple(data)
        if objective != "causal_lm" and validation_callback is None:
            from core.torch_context_learning import default_validation
            validation_callback = default_validation(self, rows, objective=objective, reward_weighting=reward_weighting, entropic_kl_target=entropic_kl_target)
        installed = []
        old_adapters = self.adapters
        rollback_params = {}
        old_training = bool(getattr(self.model, "training", False))
        old_metrics = getattr(self, "last_learning_metrics", None)
        store = getattr(self, "learning_state", None)
        metadata = dict(learning_metadata or {})
        txn = None

        try:
            if store is not None:
                txn = store.begin(update_id=metadata.get("update_id"), metadata=metadata, inherit_metadata=mode != "legacy")
                if txn.parent != getattr(self, "_mlx_parent_generation", None):
                    raise RuntimeError("MLX live weights differ from the committed learning generation; reload the model")
            if original_rows:
                params = mlx_legacy_learning.inject_adapters(self, installed=installed)
                if not params:
                    raise RuntimeError("MLX training exposes no trainable adapter parameters")
            else:
                params = ensure_adapters(self, installed=installed)
            # Native MLX updates replace immutable leaves. Keep the evaluated
            # references to roll back without copying any foundation tensors.
            rollback_params = dict(params)
            check_finite(params.values(), "starting adapter weights")
            reference = reference_weights if reference_weights is not None else rollback_params
            self.model.eval()
            if anchor_texts and fisher_matrix is None and lambda_ewc > 0:
                if mode == "hybrid" and not original_rows:
                    from core.mlx_learning import fisher_diagonal
                    fisher_matrix = fisher_diagonal(self, anchor_texts, max_tokens, cancel_event)
                else:
                    fisher_matrix = self.compute_mlx_fisher(
                        anchor_texts, learning_mode=mode, max_window_tokens=max_tokens, cancel_event=cancel_event)
                if not fisher_matrix:
                    raise ValueError("No learnable causal targets in Fisher anchors")
            fisher_tensors = 0
            if original_rows and fisher_matrix:
                # The original EWC loop applies only the intersection of supplied keys.
                fisher_matrix = {key: value for key, value in fisher_matrix.items() if key in params and key in reference}
                reference = {key: reference[key] for key in fisher_matrix}
            if fisher_matrix and lambda_ewc > 0:
                if (not fisher_matrix.keys() <= params.keys()
                        or reference_weights is not None and reference.keys() != fisher_matrix.keys()):
                    raise ValueError("MLX EWC tensors do not match active adapter keys")
                for key in fisher_matrix:
                    value = params[key]
                    if value.shape != reference[key].shape or value.shape != fisher_matrix[key].shape:
                        raise ValueError(f"MLX EWC shape mismatch for {key}")
                check_finite([*fisher_matrix.values(), *(reference[key] for key in fisher_matrix)], "EWC tensors")
                if any(bool(mx.any(value < 0).item()) for value in fisher_matrix.values()):
                    raise ValueError("MLX Fisher diagonal contains negative entries")
                fisher_tensors = sum(bool(mx.any(value > 0).item()) for value in fisher_matrix.values())

            def penalty(model):
                if not fisher_tensors:
                    return 0.0
                weights = flat_trainables(model)
                if original_rows:
                    total = mx.array(0.0)
                    for key, fisher in fisher_matrix.items():
                        total = total + mx.sum(fisher * (weights[key] - reference[key]) ** 2)
                    return (lambda_ewc / 2.0) * total
                loss = 0.0
                for key, fisher in fisher_matrix.items():
                    diff = weights[key].astype(mx.float32) - reference[key].astype(mx.float32)
                    loss = loss + (0.5 * lambda_ewc) * mx.sum(fisher * mx.square(diff))
                return loss

            retention_before = None
            if retention_data:
                retention_before = score_rows(self.model, self.tokenizer, retention_data, max_tokens, cancel_event)
            if original_rows:
                optimizer = optim.AdamW(learning_rate=learning_rate)
                resumed_optimizer = False
                if legacy:
                    details, window_count, token_count = mlx_legacy_learning.optimize_rows(
                        self, rows, optimizer, penalty, steps)
                else:
                    from core.mlx_hybrid_learning import optimize_rows
                    details, window_count, token_count = optimize_rows(
                        self, rows, optimizer, penalty, int(steps), max_tokens, supervise_prompt, row_weights, cancel_event)
                    self.model.eval()
            else:
                optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.0)
                resumed_optimizer = resume_optimizer(self, optimizer, txn, learning_rate)
                details, window_count, token_count = optimize_windows(
                    self, rows, objective=objective, row_weights=row_weights, supervise_prompt=supervise_prompt,
                    reward_weighting=reward_weighting, entropic_kl_target=entropic_kl_target, reference_kl_coef=reference_kl_coef,
                    optimizer=optimizer, penalty=penalty, steps=int(steps), max_tokens=max_tokens,
                    context_text=context_text, max_context_tokens=bounded_tokens(self, max_context_tokens), cancel_event=cancel_event)
            candidate = flat_trainables(self.model)
            squared = []
            changed = []
            for key, weight in candidate.items():
                diff = weight.astype(mx.float32) - rollback_params[key].astype(mx.float32)
                squared.append(mx.sum(mx.square(diff)))
                changed.append(mx.sum(diff != 0))
            drift = math.sqrt(float(mx.sum(mx.stack(squared)).item()))
            touched = sum(int(value.item()) for value in changed)
            if not math.isfinite(drift) or drift <= 0.0:
                raise RuntimeError("MLX training completed but measured zero parameter change or non-finite drift")
            head_only = set(candidate) == {"lm_head.lora_a", "lm_head.lora_b"}
            metrics = {
                **details,
                "learning_mode": "managed" if mode == "legacy" and not legacy else mode,
                "adapter_delta_l2": drift,
                "trainable_parameters_touched": touched,
                "trained_windows": window_count,
                "supervised_tokens": token_count,
                "ewc_used": fisher_tensors > 0,
                "fisher_tensors": fisher_tensors,
                "optimizer_resumed": resumed_optimizer,
                "adapter_target": "language_output_head" if head_only else "language_projections",
            }
            if mode == "hybrid":
                metrics["training_algorithm"] = ("original_row_adamw" if original_rows else
                                                 "managed_windows" if objective == "causal_lm" else f"managed_{objective}")
            if getattr(self, "_packed_learning_fallback", None):
                metrics["packed_projection_fallback"] = self._packed_learning_fallback
            if retention_data:
                after = score_rows(self.model, self.tokenizer, retention_data, max_tokens, cancel_event)
                metrics.update(retention_loss_before=retention_before, retention_loss_after=after)
                if after > retention_before + retention_tolerance:
                    raise RuntimeError("MLX candidate exceeded retention loss tolerance")
            cancel_if_requested(cancel_event)
            self.adapters = candidate
            if validation_callback is not None:
                verdict = validation_callback(self)
                accepted = verdict.get("accepted") if isinstance(verdict, dict) else verdict
                if accepted is not True:
                    raise RuntimeError("MLX candidate rejected by learning validation")
                if isinstance(verdict, dict):
                    metrics["validation"] = verdict
            cancel_if_requested(cancel_event)
            if bool(getattr(self.model, "training", False)) != old_training:
                self.model.train(old_training)
            self.last_learning_metrics = metrics
            persist_candidate(self, candidate, save_path, txn, metadata, metrics, cancel_event, optimizer)
            return candidate, drift
        except BaseException:
            if txn is not None and txn.committed:
                # The pointer is authoritative even if an interrupt arrives
                # after publication. Keep the validated live tensors and receipt.
                self.adapter_path = str(txn.directory / "adapters.safetensors")
                self._last_managed_checkpoint = self.adapter_path
                self._mlx_parent_generation = txn.generation
                raise
            try:
                if rollback_params:
                    self.model.update(tree_unflatten(list(rollback_params.items())))
                    mx.eval(*rollback_params.values())
            finally:
                for parent, name, original in reversed(installed):
                    setattr(parent, name, original)
                self.adapters = old_adapters if installed or not rollback_params else rollback_params
            self.last_learning_metrics = old_metrics
            raise
        finally:
            if txn is not None:
                txn.abort()  # Idempotent after publication.
            if bool(getattr(self.model, "training", False)) != old_training:
                self.model.train(old_training)
