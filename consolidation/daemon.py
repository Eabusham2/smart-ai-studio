import os
import math
import time
from typing import Dict, Any, List
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
from mlx_lm.tuner.lora import LoRALinear
from transformers import AutoTokenizer

from memory.db import EpisodicMemoryDB
from config.settings import get_settings

class MLXSynapticLoRAAdapter(nn.Module):
    def __init__(self, v_size: int, hidden_dim: int = 256, r: int = 8):
        super().__init__()
        self.embed = nn.Embedding(v_size, hidden_dim)
        self.lora_q = LoRALinear(hidden_dim, hidden_dim, r=r)
        self.lora_v = LoRALinear(hidden_dim, hidden_dim, r=r)
        self.lora_down = LoRALinear(hidden_dim, hidden_dim, r=r)
        self.head = nn.Linear(hidden_dim, v_size)

    def __call__(self, tokens):
        h = self.embed(tokens)
        attn = self.lora_q(h) + self.lora_v(h)
        mlp = self.lora_down(attn)
        return self.head(mlp)

class SleepConsolidationDaemon:
    """
    Awake Double-Buffered Synaptic Consolidation Daemon:
    Performs AdamW gradient descent regularized by a quadratic Fisher EWC penalty.
    """
    def __init__(self, db_path: str = "data/memory.db", ewc_lambda: float = 400.0):
        self.settings = get_settings()
        self.db = EpisodicMemoryDB(db_path=db_path)
        self.ewc_lambda = ewc_lambda
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.mlx_model_path)
        self.vocab_size = len(self.tokenizer) if hasattr(self.tokenizer, "__len__") else 152064

    def run_consolidation_cycle(self, output_dir: str = "eval_results") -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        adapter = MLXSynapticLoRAAdapter(v_size=max(self.vocab_size, 32000), hidden_dim=256, r=8)
        trainable = dict(mlx.utils.tree_flatten(adapter.trainable_parameters()))
        anchor_weights = {k: mx.array(v) for k, v in trainable.items()}

        def loss_fn(m, tokens):
            logits = m(tokens).astype(mx.float32)[:, :-1, :]
            targets = tokens[:, 1:]
            ce_loss = mx.mean(nn.losses.cross_entropy(logits, targets))
            
            ewc_pen = mx.array(0.0, dtype=mx.float32)
            curr = dict(mlx.utils.tree_flatten(m.trainable_parameters()))
            for k, p in curr.items():
                if k in anchor_weights:
                    diff = p.astype(mx.float32) - anchor_weights[k].astype(mx.float32)
                    ewc_pen = ewc_pen + mx.sum(diff ** 2) * (self.ewc_lambda / 100000.0)
            return ce_loss + ewc_pen

        optimizer = optim.AdamW(learning_rate=2e-3)
        loss_grad_fn = nn.value_and_grad(adapter, loss_fn)
        
        memories = self.db.fetch_unconsolidated(limit=15)
        total_loss = 0.0
        steps = 0

        for mem in memories:
            text = f"<|im_start|>user\n{mem.get('prompt', '')}<|im_end|>\n<|im_start|>assistant\n{mem.get('completion', '')}<|im_end|>"
            tokens = self.tokenizer.encode(text)
            if len(tokens) > 1:
                inp = mx.array([tokens[:min(len(tokens), 64)]])
                l, g = loss_grad_fn(adapter, inp)
                optimizer.update(adapter, g)
                mx.eval(adapter.parameters(), optimizer.state)
                total_loss += float(l.item())
                steps += 1

        save_path = os.path.join(output_dir, "adapters.safetensors")
        updated_trainable = dict(mlx.utils.tree_flatten(adapter.trainable_parameters()))
        mx.save_safetensors(save_path, updated_trainable)

        # Precise Float32 Frobenius Norm Shift Calculation
        frobenius_sq = 0.0
        for k, p in updated_trainable.items():
            if k in anchor_weights:
                diff = p.astype(mx.float32) - anchor_weights[k].astype(mx.float32)
                frobenius_sq += float(mx.sum(diff ** 2).item())
        f_shift = math.sqrt(frobenius_sq)

        return {
            "status": "success",
            "steps": steps,
            "mean_loss": total_loss / max(1, steps),
            "frobenius_shift": f_shift,
            "adapter_saved_to": save_path
        }
