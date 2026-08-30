import math
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

MLX_AVAILABLE = False
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils
    MLX_AVAILABLE = True
except ImportError:
    pass


class GramSchmidtOGPProjector:
    def __init__(self, tolerance: float = 1e-5):
        self.tolerance = tolerance
        self.anchor_basis_vectors: List[Any] = []

    @staticmethod
    def flatten_gradients(grads_dict: Dict[str, Any]) -> Tuple[Any, List[Tuple[str, Tuple[int, ...]]]]:
        flat_list = []
        shapes = []
        sorted_keys = sorted(grads_dict.keys())
        for k in sorted_keys:
            val = grads_dict[k]
            shapes.append((k, val.shape))
            if MLX_AVAILABLE and isinstance(val, mx.array):
                flat_list.append(val.reshape(-1).astype(mx.float32))
        if MLX_AVAILABLE and flat_list:
            return mx.concatenate(flat_list), shapes
        return None, shapes

    @staticmethod
    def unflatten_gradients(flat_vec: Any, shapes: List[Tuple[str, Tuple[int, ...]]]) -> Dict[str, Any]:
        unflat = {}
        offset = 0
        if MLX_AVAILABLE and isinstance(flat_vec, mx.array):
            for k, shape in shapes:
                numel = 1
                for dim in shape:
                    numel *= dim
                unflat[k] = flat_vec[offset : offset + numel].reshape(shape)
                offset += numel
        return unflat

    def register_anchor_gradient(self, flat_anchor_grad: Any):
        if not MLX_AVAILABLE or flat_anchor_grad is None:
            return
        v = flat_anchor_grad.astype(mx.float32)
        for basis in self.anchor_basis_vectors:
            proj = (mx.sum(v * basis) / (mx.sum(basis * basis) + 1e-12)) * basis
            v = v - proj
        norm_sq = float(mx.sum(v * v).item())
        if norm_sq > self.tolerance:
            unit_basis = v / math.sqrt(norm_sq)
            mx.eval(unit_basis)
            self.anchor_basis_vectors.append(unit_basis)

    def project_gradient(self, flat_task_grad: Any) -> Any:
        if not MLX_AVAILABLE or flat_task_grad is None or not self.anchor_basis_vectors:
            return flat_task_grad
        g_proj = flat_task_grad.astype(mx.float32)
        for basis in self.anchor_basis_vectors:
            coeff = mx.sum(g_proj * basis)
            g_proj = g_proj - (coeff * basis)
        mx.eval(g_proj)
        return g_proj

    def verify_orthogonality(self, flat_proj_grad: Any) -> float:
        if not self.anchor_basis_vectors or flat_proj_grad is None:
            return 0.0
        max_overlap = 0.0
        for basis in self.anchor_basis_vectors:
            overlap = abs(float(mx.sum(flat_proj_grad * basis).item()))
            if overlap > max_overlap:
                max_overlap = overlap
        return max_overlap


class ProjectedSleepConsolidationDaemon(threading.Thread):
    def __init__(self, moe_manager: Any, ogp_projector: GramSchmidtOGPProjector,
                 kg: Any, tokenizer: Any, settings: Any, stream_lock: Optional[threading.Lock] = None):
        super().__init__(daemon=True, name="OGP-Sleep-Daemon")
        self.moe_manager = moe_manager
        self.ogp_projector = ogp_projector
        self.kg = kg
        self.tokenizer = tokenizer
        self.settings = settings
        self.stream_lock = stream_lock or threading.Lock()
        self.running = True
        self.total_consolidations = 0
        self.last_loss = 0.0
        self.last_ortho_overlap = 0.0
        self.queue_length = 0

    def run(self):
        while self.running:
            time.sleep(2.0)
            try:
                items = self.kg.fetch_unconsolidated_high_surprise(
                    min_surprise=self.settings.min_surprise_threshold,
                    limit=32
                )
                self.queue_length = len(items)
                if self.queue_length >= self.settings.min_batch_queue_size:
                    self._consolidate_batch(items[:self.settings.min_batch_queue_size])
            except Exception:
                pass

    def _consolidate_batch(self, items: List[Dict[str, Any]]):
        if not MLX_AVAILABLE or self.moe_manager.model is None or not items:
            return
        model = self.moe_manager.model

        def loss_fn(m, tokens):
            logits = m(tokens)[:, :-1, :].astype(mx.float32)
            targets = tokens[:, 1:]
            return mx.mean(nn.losses.cross_entropy(logits, targets))

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        processed_ids = []

        for item in items:
            text = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n{item['completion']}<|im_end|>"
            toks = self.tokenizer.encode(text)
            if len(toks) > 1:
                inp = mx.array([toks[:min(len(toks), 64)]])
                with self.stream_lock:
                    loss_val, raw_grads = loss_and_grad_fn(model, inp)
                    flat_grads, shapes = self.ogp_projector.flatten_gradients(dict(mlx.utils.tree_flatten(raw_grads)))
                    
                    # Gradient-Variance Adaptive Learning Rate Scheduling
                    # eta_t = eta_0 / sqrt(1 + Var(grad))
                    grad_var = float(mx.var(flat_grads).item()) if flat_grads is not None else 0.0
                    adaptive_lr = self.settings.base_learning_rate / math.sqrt(1.0 + grad_var)
                    optimizer = optim.AdamW(learning_rate=adaptive_lr)

                    proj_flat = self.ogp_projector.project_gradient(flat_grads)
                    self.last_ortho_overlap = self.ogp_projector.verify_orthogonality(proj_flat)
                    proj_tree = self.ogp_projector.unflatten_gradients(proj_flat, shapes)
                    optimizer.update(model, mlx.utils.tree_unflatten(list(proj_tree.items())))
                    mx.eval(model.parameters(), optimizer.state)
                self.last_loss = float(loss_val.item())
                processed_ids.append(item["id"])
                self.total_consolidations += 1

        self.moe_manager.adapters_buffer_b = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
        self.moe_manager.swap_buffers_atomic()
        self.kg.mark_consolidated(processed_ids)
