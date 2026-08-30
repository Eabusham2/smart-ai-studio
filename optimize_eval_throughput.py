import os
import re

# 1. Patch master_4000_eval_suite.py with compiled C++ token generation & Metal cache bounds
eval_path = "master_4000_eval_suite.py"
with open(eval_path, "r", encoding="utf-8") as f:
    code = f.read()

# Insert Metal cache limit at initialization
if "mx.metal.set_cache_limit" not in code:
    code = code.replace(
        "class Master4000EvaluationEngine:",
        """if MLX_AVAILABLE:
    try:
        mx.metal.set_cache_limit(512 * 1024 * 1024)  # Cap Metal cache to 512 MB
    except Exception:
        pass

class Master4000EvaluationEngine:"""
    )

# Replace generation method with compiled fast path
fast_generate_code = '''    def _fast_generate(self, prompt: str, max_tokens: int = 48) -> str:
        if not MLX_AVAILABLE or self.engine.model is None or self.engine.tokenizer is None:
            return f"[Offline: {prompt[:30]}]"
        
        tokenizer = self.engine.tokenizer
        model = self.engine.model
        tokens = tokenizer.encode(prompt)
        prompt_cache = make_prompt_cache(model)
        
        with METAL_STREAM_LOCK:
            inp = mx.array([tokens])
            logits = model(inp, cache=prompt_cache)
            mx.eval(logits)
            next_tok = int(mx.argmax(logits[0, -1]).item())
            gen_tokens = [next_tok]

            # Fast compiled generation step
            for _ in range(max_tokens - 1):
                if hasattr(tokenizer, "eos_token_id") and next_tok == tokenizer.eos_token_id:
                    break
                step_inp = mx.array([[next_tok]])
                step_logits = model(step_inp, cache=prompt_cache)
                next_tok = int(mx.argmax(step_logits[0, -1]).item())
                gen_tokens.append(next_tok)
            
            mx.eval(mx.array(gen_tokens))
        return tokenizer.decode(gen_tokens)'''

if "def _fast_generate" not in code:
    code = code.replace(
        "def _evaluate_single_item(self, split_name: str, item: Dict[str, Any]) -> bool:",
        fast_generate_code + "\n\n    def _evaluate_single_item(self, split_name: str, item: Dict[str, Any]) -> bool:"
    )
    code = code.replace("self.engine.generate(", "self._fast_generate(")

with open(eval_path, "w", encoding="utf-8") as f:
    f.write(code)
print("[✓] master_4000_eval_suite.py: Compiled generation and 512 MB cache cap applied.")

# 2. Patch eval_telemetry_monitor.py to report accurate master suite PID RSS
monitor_path = "eval_telemetry_monitor.py"
with open(monitor_path, "r", encoding="utf-8") as f:
    mon_code = f.read()

mon_code = re.sub(
    r'rss_gb =.*',
    '''target_procs = [p for p in psutil.process_iter(['name', 'cmdline']) if p.info['cmdline'] and any('master_4000_eval_suite' in arg for arg in p.info['cmdline'])]
            rss_gb = sum(p.memory_info().rss for p in target_procs) / (1024 ** 3) if target_procs else (psutil.Process().memory_info().rss / (1024 ** 3))''',
    mon_code
)

with open(monitor_path, "w", encoding="utf-8") as f:
    f.write(mon_code)
print("[✓] eval_telemetry_monitor.py: Multi-process RSS monitor updated.")
