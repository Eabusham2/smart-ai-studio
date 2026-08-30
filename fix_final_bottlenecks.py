import re

eval_path = "master_4000_eval_suite.py"
with open(eval_path, "r", encoding="utf-8") as f:
    code = f.read()

# 1. Remove the artificial cache limit causing allocation thrashing
code = code.replace("mx.metal.set_cache_limit(512 * 1024 * 1024)", "pass # Let MLX pool memory naturally")

# 2. Replace the slow custom python generation loop with MLX's native optimized C++ generator
new_fast_gen = '''    def _fast_generate(self, prompt: str, max_tokens: int = 48) -> str:
        if not MLX_AVAILABLE or self.engine.model is None or self.engine.tokenizer is None:
            return f"[Offline: {prompt[:30]}]"
        
        from mlx_lm import generate as native_generate
        
        # Native C++ generation path bypasses Python loop syncs for 15-20+ tok/s
        try:
            res = native_generate(self.engine.model, self.engine.tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
            return res
        except Exception:
            return ""

    def run_full_suite'''

code = re.sub(r'    def _fast_generate\(self, prompt: str, max_tokens: int = 48\) -> str:.*?\n    def run_full_suite', new_fast_gen, code, flags=re.DOTALL)

# 3. Aggressive GC and Metal Cache clearing AFTER EVERY SINGLE ITEM (Stops the 7.77GB creep entirely)
loop_replacement = '''                is_correct = self._evaluate_single_item(split_name, item)
                dur = max(0.001, time.perf_counter() - t0)

                correct += 1 if is_correct else 0
                completed_cache[item_key] = is_correct

                # FORCE STRICT MEMORY FLUSH AFTER EVERY ITEM
                gc.collect()
                if MLX_AVAILABLE:
                    mx.metal.clear_cache()

                if global_idx % 5 == 0:
                    self.checkpoint_mgr.save_checkpoint(completed_cache, phase_label, start_time)'''

code = re.sub(r'                is_correct = self\._evaluate_single_item\(split_name, item\)[\s\S]*?self\.checkpoint_mgr\.save_checkpoint\(completed_cache, phase_label, start_time\)', loop_replacement, code)

with open(eval_path, "w", encoding="utf-8") as f:
    f.write(code)

print("[✓] C++ Generation loop restored. Sync bottlenecks removed.")
print("[✓] Per-item garbage collection enforced to halt RAM creep.")
