import os
import time
import json
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
from mlx_lm import load, generate
from mlx_lm.tuner.lora import LoRALinear
from huggingface_hub import snapshot_download

def inject_lora(model, r=8):
    for i, layer in enumerate(model.model.layers):
        # Inject LoRA into Q and V projections
        if hasattr(layer.self_attn, "q_proj"):
            layer.self_attn.q_proj = LoRALinear.from_base(layer.self_attn.q_proj, r=r)
        if hasattr(layer.self_attn, "v_proj"):
            layer.self_attn.v_proj = LoRALinear.from_base(layer.self_attn.v_proj, r=r)
        # Inject into MLP down_proj
        if hasattr(layer.mlp, "down_proj"):
            layer.mlp.down_proj = LoRALinear.from_base(layer.mlp.down_proj, r=r)

def run_pipeline(is_dry_run=False):
    repo_id = "orcarouter/Qwen3.8-27B-Uncensored-MLX"
    print(f"[*] Downloading 2-bit weights from {repo_id}...")
    local_dir = snapshot_download(repo_id, allow_patterns=["2-bit/*", "*.jinja"])
    model_path = os.path.join(local_dir, "2-bit")
    
    print(f"[*] Loading live model from {model_path}...")
    model, tokenizer = load(model_path)
    
    # 1. Evaluate a live benchmark split
    print("[*] Evaluating live benchmark problem...")
    prompt = "Write a python function to compute the 10th fibonacci number."
    response = generate(model, tokenizer, prompt=prompt, max_tokens=20 if is_dry_run else 100, verbose=True)
    print(f"[✓] Inference complete. Output:\n{response}")
    
    # 2. RLVR Error Correction
    print("\n[*] Running RLVR error-correction rollout...")
    rlvr_prompt = f"The previous code failed because it was too slow. Optimize it.\nPrevious code:\n{response}"
    response_optimized = generate(model, tokenizer, prompt=rlvr_prompt, max_tokens=20 if is_dry_run else 100, verbose=True)
    print(f"[✓] RLVR iteration complete. Output:\n{response_optimized}")
    
    # 3. Real MLX LoRA Gradient Backpropagation
    print("\n[*] Preparing MLX LoRA gradients...")
    model.freeze()
    
    # Inject LoRA Adapters
    inject_lora(model)
    
    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        logits = logits[:, :-1, :]
        targets = targets[:, 1:]
        loss = nn.losses.cross_entropy(logits, targets)
        return mx.mean(loss)

    optimizer = optim.AdamW(learning_rate=1e-5)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    inputs = mx.array([tokenizer.encode("Hello world! This is a real training step.")])
    targets = inputs 
    
    print("[*] Executing real MLX gradient backpropagation...")
    loss, grads = loss_and_grad_fn(model, inputs, targets)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    
    print(f"[✓] Backprop successful! Loss: {loss.item():.4f}")
    
    print("[*] Saving adapters.safetensors...")
    os.makedirs("eval_results", exist_ok=True)
    
    # Save only trainable parameters (the LoRA adapters)
    trainable_params = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
    mx.save_safetensors("eval_results/adapters.safetensors", trainable_params)
    
    with open("eval_results/pipeline_log.json", "w") as f:
        json.dump({"loss": float(loss.item()), "status": "success"}, f)
    print("[✓] Pipeline dry-run complete!")

if __name__ == "__main__":
    run_pipeline()
