eval_path = "master_4000_eval_suite.py"

with open(eval_path, "r", encoding="utf-8") as f:
    code = f.read()

# Ensure make_prompt_cache is imported under MLX_AVAILABLE
target_import = """if MLX_AVAILABLE:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils
    from mlx_lm.models.cache import make_prompt_cache"""

code = code.replace(
"""if MLX_AVAILABLE:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils""",
target_import
)

with open(eval_path, "w", encoding="utf-8") as f:
    f.write(code)

print("[✓] Added make_prompt_cache to master_4000_eval_suite.py")
