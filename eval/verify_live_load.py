import os
import psutil
import time
from mlx_lm import load, generate
from huggingface_hub import snapshot_download

def verify_live_load():
    repo_id = "orcarouter/Qwen3.8-27B-Uncensored-MLX"
    print(f"[*] Downloading 2-bit weights from {repo_id} (skipping other quants)...")
    local_dir = snapshot_download(repo_id, allow_patterns=["2-bit/*", "*.jinja"])
    model_path = os.path.join(local_dir, "2-bit")
    
    print(f"[*] Loading live model from {model_path} into Unified Memory...")
    
    mem_before = psutil.Process().memory_info().rss / (1024**3)
    t0 = time.time()
    
    model, tokenizer = load(model_path)
    load_time = time.time() - t0
    
    mem_after = psutil.Process().memory_info().rss / (1024**3)
    
    print(f"[✓] Model loaded in {load_time:.2f}s")
    print(f"[!] Resident Memory (RSS): {mem_after:.2f} GB (Delta: +{mem_after - mem_before:.2f} GB)")
    
    prompt = "Below is a mathematical fact:\n"
    print(f"\n[*] Generating real tokens for prompt: {prompt.strip()}")
    
    t0 = time.time()
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=20, 
        verbose=True
    )
    gen_time = time.time() - t0
    
    print(f"\n[✓] Generated response in {gen_time:.2f}s")
    print(f"--- OUTPUT ---\n{response}\n--------------")
    print("SUCCESS: Physical MLX execution verified.")

if __name__ == "__main__":
    verify_live_load()
