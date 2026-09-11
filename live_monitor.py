import os
import sys
import time
import json
import psutil

TOTAL_ITEMS = 56
STREAM_FILE = "eval_results/raw_eval_stream.jsonl"

def get_python_ram_gb():
    total_rss = 0.0
    for p in psutil.process_iter(['name', 'cmdline', 'memory_info']):
        try:
            cmd = " ".join(p.info['cmdline'] or [])
            if "run_master" in cmd and p.pid != os.getpid():
                total_rss += p.info['memory_info'].rss / (1024 ** 3)
        except Exception:
            continue
    return total_rss

print("\033[2J\033[H", end="")

while True:
    records = []
    if os.path.exists(STREAM_FILE):
        with open(STREAM_FILE, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass

    completed = len(records)
    pct = min(100.0, (completed / TOTAL_ITEMS) * 100.0)
    
    # Progress Bar
    bar_width = 32
    filled = int(bar_width * (pct / 100.0))
    bar = "█" * filled + "░" * (bar_width - filled)

    # Calculate metrics from real physical trace data
    if records:
        last_item = records[-1]
        current_split = last_item.get("split", "Running...")
        last_tps = last_item.get("tok_per_sec", 0.0)
        last_tokens = last_item.get("tokens_generated", 0)
        last_latency = last_item.get("latency_s", 0.0)
        last_status = "PASS (1.0)" if last_item.get("verified") else "FAIL (0.0)"
        
        # Recent moving average throughput & latency (last 5 items)
        recent_window = records[-5:]
        avg_lat = sum(r.get("latency_s", 0.0) for r in recent_window) / len(recent_window)
        avg_tps_recent = sum(r.get("tok_per_sec", 0.0) for r in recent_window) / len(recent_window)
        
        remaining_items = max(0, TOTAL_ITEMS - completed)
        eta_sec = int(remaining_items * avg_lat)
        eta_str = f"{eta_sec // 60}m {eta_sec % 60:02d}s"
    else:
        current_split = "Initializing Phase 1..."
        last_tps = 0.0
        last_tokens = 0
        last_latency = 0.0
        last_status = "Waiting..."
        avg_tps_recent = 0.0
        eta_str = "Calculating..."

    ram_gb = get_python_ram_gb()

    output = f"""\033[H
====================================================================
  🚀 SMART AI STUDIO: ACCURATE TELEMETRY & METAL MONITOR
====================================================================
  • Progress       : [{bar}] {pct:.1f}% ({completed}/{TOTAL_ITEMS} items)
  • Stabilized ETA : {eta_str} remaining  |  Moving Avg TPS: {avg_tps_recent:.1f} tok/s
  • Resident RAM   : {ram_gb:.2f} GB / 13.00 GB (Watchdog Active)
--------------------------------------------------------------------
  CURRENT ITEM PROFILE:
  • Active Split   : {current_split}
  • Last Inference : {last_tps:.2f} tok/s ({last_tokens} tokens in {last_latency:.2f}s)
  • Outcome        : {last_status}
--------------------------------------------------------------------
  (Press Ctrl+C to exit monitor — pipeline runs uninterrupted)
"""
    sys.stdout.write(output)
    sys.stdout.flush()

    if completed >= TOTAL_ITEMS and os.path.exists("eval_results/.complete"):
        print("\n[✓] Evaluation completed successfully!")
        break

    time.sleep(1.0)
