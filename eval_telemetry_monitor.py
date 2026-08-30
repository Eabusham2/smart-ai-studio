import json
import os
import platform
import psutil
import time
import sys
from datetime import datetime, timedelta

def run_telemetry_dashboard(log_file="eval_results/telemetry_stream.jsonl", refresh_rate=1.0):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    start_time = time.time()
    
    print("\033[2J\033[H", end="") # Clear terminal screen
    print("=" * 95)
    print("📡 SMART AI STUDIO: 4,000+ ITEM EVALUATION TELEMETRY MONITOR (72-HR BUDGET)")
    print(f"│ Target Host: {platform.node()} ({platform.machine()}) | OS: {platform.system()} {platform.release()}")
    print("=" * 95)

    last_line_read = 0
    latest_metrics = {
        "item_idx": 0,
        "total_items": 4014,
        "split": "Initializing",
        "pass_rate": 0.0,
        "tok_per_sec": 0.0,
        "ram_gb": 0.0,
        "lif_spikes": 0,
        "speculative_hit_rate": 0.0,
        "ortho_overlap": 0.0,
        "phase": "Phase 1: Baseline"
    }

    try:
        while True:
            # Read latest telemetry stream lines
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    if len(lines) > last_line_read:
                        for line in lines[last_line_read:]:
                            try:
                                data = json.loads(line.strip())
                                latest_metrics.update(data)
                            except Exception:
                                pass
                        last_line_read = len(lines)

            # System resources
            ram_info = psutil.virtual_memory()
            target_proc = next((p for p in psutil.process_iter(['name', 'cmdline']) if p.info['cmdline'] and 'master_4000_eval_suite.py' in ' '.join(p.info['cmdline'])), None)
            target_procs = [p for p in psutil.process_iter(['name', 'cmdline']) if p.info['cmdline'] and any('master_4000_eval_suite' in arg for arg in p.info['cmdline'])]
            rss_gb = sum(p.memory_info().rss for p in target_procs) / (1024 ** 3) if target_procs else (psutil.Process().memory_info().rss / (1024 ** 3))
            cpu_pct = psutil.cpu_percent()
            
            elapsed = time.time() - start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            
            # Progress calculation
            curr_idx = latest_metrics.get("item_idx", 0)
            total_items = max(1, latest_metrics.get("total_items", 4014))
            progress_pct = (curr_idx / total_items) * 100.0
            
            # ETA calculation
            if curr_idx > 0 and elapsed > 5:
                rate = curr_idx / elapsed
                remaining_sec = (total_items - curr_idx) / rate
                eta_str = str(timedelta(seconds=int(remaining_sec)))
            else:
                eta_str = "--:--:--"

            # Render ASCII Terminal Telemetry Dashboard
            print("\033[H", end="") # Cursor home
            print("=" * 95)
            print(f"📡 EVALUATION TELEMETRY MONITOR | Phase: \033[96m{latest_metrics.get('phase', 'Running')}\033[0m")
            print(f"│ Elapsed: {elapsed_str} | ETA (72h Cap): {eta_str} | Overall Progress: {curr_idx}/{total_items} ({progress_pct:.2f}%)")
            print("=" * 95)
            
            # Progress Bar
            bar_len = 50
            filled = int(bar_len * (curr_idx / total_items))
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"Progress: [{bar}] {progress_pct:.1f}%")
            print("-" * 95)
            
            # Metric Columns
            print(f"│ Active Benchmark Split    : \033[93m{latest_metrics.get('split', 'N/A')}\033[0m")
            print(f"│ Running Split Accuracy    : \033[92m{latest_metrics.get('pass_rate', 0.0):.2f}%\033[0m")
            print(f"│ Generation Speed          : {latest_metrics.get('tok_per_sec', 0.0):.1f} tok/s")
            print(f"│ Speculative Trie Hit Rate : {latest_metrics.get('speculative_hit_rate', 0.0):.1f}%")
            print(f"│ LIF Action Spikes         : {latest_metrics.get('lif_spikes', 0)}")
            print(f"│ OGP Orthogonal Overlap    : {latest_metrics.get('ortho_overlap', 0.0):.2e} (<g, m_j> = 0)")
            print("-" * 95)
            print(f"│ Host Physical RAM Footprint: {rss_gb:.2f} GB / {ram_info.total / (1024**3):.1f} GB (System: {ram_info.percent}%)")
            print(f"│ CPU Compute Activity      : {cpu_pct:.1f}%")
            print(f"│ Checkpoint Heartbeat File : eval_results/eval_checkpoint_4000.json")
            print("=" * 95)
            print("Press Ctrl+C to stop telemetry viewer (Evaluation runner continues independently).")
            
            time.sleep(refresh_rate)
    except KeyboardInterrupt:
        print("\n[Telemetry Monitor Exited]")

if __name__ == "__main__":
    run_telemetry_dashboard()
