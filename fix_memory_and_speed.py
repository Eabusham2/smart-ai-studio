import re

# 1. Patch master_4000_eval_suite.py to clear Metal cache and garbage collect every 5 items
eval_path = "master_4000_eval_suite.py"
with open(eval_path, "r", encoding="utf-8") as f:
    code = f.read()

# Add Metal cache purge & gc call inside the evaluation loop
if "mx.metal.clear_cache()" not in code:
    code = code.replace(
        "if global_idx % 5 == 0:",
        """if global_idx % 5 == 0:
                    gc.collect()
                    if MLX_AVAILABLE:
                        mx.metal.clear_cache()"""
    )
    with open(eval_path, "w", encoding="utf-8") as f:
        f.write(code)
    print("[✓] Patched master_4000_eval_suite.py: Added periodic Metal cache purging.")

# 2. Patch eval_telemetry_monitor.py to report true Python process RSS
monitor_path = "eval_telemetry_monitor.py"
with open(monitor_path, "r", encoding="utf-8") as f:
    mon_code = f.read()

mon_code = re.sub(
    r'rss_gb = psutil\.Process\(\)\.memory_info\(\)\.rss / \(1024 \*\* 3\)',
    '''# Sum RSS of all active evaluation python processes
            eval_procs = [p for p in psutil.process_iter(['name', 'cmdline']) if p.info['cmdline'] and any('master_4000_eval_suite' in arg for arg in p.info['cmdline'])]
            rss_gb = sum(p.memory_info().rss for p in eval_procs) / (1024 ** 3) if eval_procs else (psutil.Process().memory_info().rss / (1024 ** 3))''',
    mon_code
)

with open(monitor_path, "w", encoding="utf-8") as f:
    f.write(mon_code)
print("[✓] Patched eval_telemetry_monitor.py: Fixed multi-process RSS telemetry tracking.")

