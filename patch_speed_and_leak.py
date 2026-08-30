eval_path = "master_4000_eval_suite.py"

with open(eval_path, "r", encoding="utf-8") as f:
    code = f.read()

# Replace _fast_generate with leak-free, compiled C++ generation
new_fast_gen = '''    def _fast_generate(self, prompt: str, max_tokens: int = 48) -> str:
        if not MLX_AVAILABLE or self.engine.model is None or self.engine.tokenizer is None:
            return f"[Offline: {prompt[:30]}]"
        
        tokenizer = self.engine.tokenizer
        model = self.engine.model
        tokens = tokenizer.encode(prompt)
        prompt_cache = make_prompt_cache(model)
        
        gen_tokens = []
        with METAL_STREAM_LOCK:
            inp = mx.array([tokens])
            logits = model(inp, cache=prompt_cache)
            next_tok = mx.argmax(logits[0, -1])
            gen_tokens.append(int(next_tok.item()))

            for _ in range(max_tokens - 1):
                if hasattr(tokenizer, "eos_token_id") and gen_tokens[-1] == tokenizer.eos_token_id:
                    break
                step_logits = model(mx.array([[gen_tokens[-1]]]), cache=prompt_cache)
                next_tok = mx.argmax(step_logits[0, -1])
                gen_tokens.append(int(next_tok.item()))
            
            # Explicit cleanup to prevent Metal memory creep
            del prompt_cache
            del inp
            del logits
            del step_logits
            
        return tokenizer.decode(gen_tokens)'''

# Replace in file
import re
code = re.sub(r'    def _fast_generate\(self, prompt: str, max_tokens: int = 48\) -> str:[\s\S]*?return tokenizer\.decode\(gen_tokens\)', new_fast_gen, code)

# Ensure per-item gc collect
code = code.replace(
    'is_correct = self._evaluate_single_item(split_name, item)',
    '''is_correct = self._evaluate_single_item(split_name, item)
                # Free transient Metal buffers per item
                if MLX_AVAILABLE and global_idx % 2 == 0:
                    mx.metal.clear_cache()'''
)

with open(eval_path, "w", encoding="utf-8") as f:
    f.write(code)

print("[✓] Patched master_4000_eval_suite.py: Cleaned memory leak & streamlined Metal stream.")
