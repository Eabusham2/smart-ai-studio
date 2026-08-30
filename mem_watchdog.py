import os
import time
import psutil

RAM_LIMIT_GB = 12.0
LIMIT_BYTES = RAM_LIMIT_GB * (1024**3)

print(f"[*] Memory Watchdog Active: Monitoring Python processes (Cap: {RAM_LIMIT_GB} GB)...")

try:
    while True:
        current_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
            try:
                name = proc.info['name'] or ''
                cmdline = ' '.join(proc.info['cmdline'] or [])
                
                if ('python' in name.lower() or 'python' in cmdline.lower()) and proc.pid != current_pid:
                    mem_rss = proc.info['memory_info'].rss
                    mem_gb = mem_rss / (1024**3)
                    
                    if mem_rss > LIMIT_BYTES:
                        print(f"\n[!] LIMIT EXCEEDED: PID {proc.pid} is using {mem_gb:.2f} GB RAM.")
                        print(f"[*] Sending SIGKILL to PID {proc.pid} ({name})...")
                        proc.kill()
                        print("[+] Process terminated successfully.")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
                
        time.sleep(1.0)
except KeyboardInterrupt:
    print("\n[*] Watchdog stopped.")
