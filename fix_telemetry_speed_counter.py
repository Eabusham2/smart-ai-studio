import os
import re

eval_path = "master_4000_eval_suite.py"
with open(eval_path, "r", encoding="utf-8") as f:
    code = f.read()

# 1. Update _fast_generate to measure actual token throughput (tok/sec)
old_gen_pattern = r'    def _fast_generate\(self, prompt: str, max_tokens: int = 48\) -> str:.*?\n        except Exception:\n            return ""'

new_gen_impl = '''    def _fast_generate(self, prompt: str, max_tokens: int = 48) -> str:
        if not MLX_AVAILABLE or self.engine.model is None or self.engine.tokenizer is None:
            self.last_tok_per_sec = 0.0
            return f"[Offline: {prompt[:30]}]"
        
        from mlx_lm import generate as native_generate
        try:
            t0 = time.perf_counter()
            res = native_generate(self.engine.model, self.engine.tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
            dur = max(0.001, time.perf_counter() - t0)
            tok_len = max(1, len(self.engine.tokenizer.encode(res)))
            self.last_tok_per_sec = tok_len / dur
            return res
        except Exception:
            self.last_tok_per_sec = 0.0
            return ""'''

code = re.sub(old_gen_pattern, new_gen_impl, code, flags=re.DOTALL)

# 2. Initialize self.last_tok_per_sec in __init__ if missing
if "self.last_tok_per_sec" not in code:
    code = code.replace(
        'self.telemetry_file = "eval_results/telemetry_stream.jsonl"',
        'self.telemetry_file = "eval_results/telemetry_stream.jsonl"\n        self.last_tok_per_sec = 0.0'
    )

# 3. Use actual measured speed in telemetry stream
code = code.replace(
    'tok_speed = 32.0 / dur',
    'tok_speed = getattr(self, \'last_tok_per_sec\', 15.0)'
)

with open(eval_path, "w", encoding="utf-8") as f:
    f.write(code)

# 4. Clear old telemetry stream log so monitor restarts fresh
telemetry_log = "eval_results/telemetry_stream.jsonl"
if os.path.exists(telemetry_log):
    os.remove(telemetry_log)

print("[✓] master_4000_eval_suite.py updated with live token speed measurement.")
print("[✓] Telemetry history stream cleared for fresh counter restart.")
