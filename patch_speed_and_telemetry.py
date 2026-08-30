import re

# 1. Fix telemetry monitor to inspect the eval runner PID
monitor_path = "eval_telemetry_monitor.py"
with open(monitor_path, "r") as f:
    code = f.read()

# Replace process RSS lookup to find master_4000_eval_suite process
fixed_code = code.replace(
    'rss_gb = psutil.Process().memory_info().rss / (1024 ** 3)',
    '''target_proc = next((p for p in psutil.process_iter(['name', 'cmdline']) if p.info['cmdline'] and 'master_4000_eval_suite.py' in ' '.join(p.info['cmdline'])), None)
            rss_gb = (target_proc.memory_info().rss / (1024 ** 3)) if target_proc else (psutil.Process().memory_info().rss / (1024 ** 3))'''
)

with open(monitor_path, "w") as f:
    f.write(fixed_code)

print("✓ Telemetry monitor patched to track master runner process RSS.")
