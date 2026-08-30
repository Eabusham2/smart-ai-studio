import json
import os
import psutil
import sys
import time
from collections import defaultdict

STREAM_FILE = "eval_results/raw_eval_stream.jsonl"
REPORT_FILE = "eval_results/ULTIMATE_MASTER_EVAL_REPORT.md"
SENTINEL_FILE = "eval_results/.complete"
TOTAL_ITEMS = 864  # 432 Baseline + 432 Post

def format_bar(completed: int, total: int, width: int = 28) -> str:
    pct = min(1.0, max(0.0, completed / max(1, total)))
    filled = int(width * pct)
    return "█" * filled + "░" * (width - filled)

def render_dashboard():
    while True:
        completed = 0
        total_tokens = 0
        total_gen_time = 0.0
        
        baseline_passed = 0
        baseline_total = 0
        post_passed = 0
        post_total = 0
        
        split_stats = defaultdict(lambda: {"baseline_pass": 0, "baseline_tot": 0, "post_pass": 0, "post_tot": 0})
        recent_events = []

        if os.path.exists(STREAM_FILE):
            try:
                with open(STREAM_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            completed += 1
                            tokens = record.get("tokens_generated", 0)
                            lat = record.get("latency_s", 0.0)
                            total_tokens += tokens
                            total_gen_time += lat
                            
                            split_raw = record.get("split", "Unknown")
                            is_post = split_raw.startswith("Post-")
                            clean_split = split_raw.replace("Baseline-", "").replace("Post-", "")
                            
                            passed = bool(record.get("verified", False))
                            if is_post:
                                post_total += 1
                                if passed:
                                    post_passed += 1
                                split_stats[clean_split]["post_tot"] += 1
                                if passed:
                                    split_stats[clean_split]["post_pass"] += 1
                            else:
                                baseline_total += 1
                                if passed:
                                    baseline_passed += 1
                                split_stats[clean_split]["baseline_tot"] += 1
                                if passed:
                                    split_stats[clean_split]["baseline_pass"] += 1

                            recent_events.append(record)
                        except Exception:
                            pass
            except Exception:
                pass

        try:
            current_proc = psutil.Process()
            rss_gb = current_proc.memory_info().rss / (1024 ** 3)
        except Exception:
            rss_gb = 0.0

        avg_tps = total_tokens / max(0.001, total_gen_time)
        pct = min(100.0, (completed / TOTAL_ITEMS) * 100.0)
        progress_bar = format_bar(completed, TOTAL_ITEMS, width=30)

        is_done = os.path.exists(SENTINEL_FILE) or (completed >= TOTAL_ITEMS and os.path.exists(REPORT_FILE))
        if is_done:
            stage_str = "PHASE 5: Complete (Report Synthesized)"
            status_badge = "✔ COMPLETE"
        elif completed < 432:
            stage_str = f"PHASE 1: Full Zero-Shot Baseline ({completed}/432 items)"
            status_badge = "⚡ ACTIVE"
        elif completed == 432:
            stage_str = "PHASE 2 & 3: Ingestion & LoRA EWC Sleep Consolidation"
            status_badge = "🧠 TRAINING"
        else:
            stage_str = f"PHASE 4: Post-Consolidation Full Evaluation ({completed - 432}/432 items)"
            status_badge = "⚡ ACTIVE"

        b_rate = (baseline_passed / max(1, baseline_total)) * 100.0 if baseline_total else 0.0
        p_rate = (post_passed / max(1, post_total)) * 100.0 if post_total else 0.0
        delta = p_rate - b_rate

        buf = []
        buf.append("\033[2J\033[H")
        buf.append("=" * 78)
        buf.append("  🚀 SMART AI STUDIO: REAL-TIME 864-ITEM MASTER PIPELINE MONITOR")
        buf.append("=" * 78)
        buf.append(f"  • Status : {status_badge:<12} | Resident RAM: {rss_gb:.2f} GB | Throughput: {avg_tps:5.1f} tok/s")
        buf.append(f"  • Stage  : {stage_str}")
        buf.append(f"  • Total  : [{progress_bar}] {pct:5.1f}% ({completed}/{TOTAL_ITEMS} items)")
        buf.append("-" * 78)
        buf.append(f"  • Baseline Pass Rate : {baseline_passed:3d}/{baseline_total:<3d} ({b_rate:5.1f}%)")
        buf.append(f"  • Post-Training Rate : {post_passed:3d}/{post_total:<3d} ({p_rate:5.1f}%) [ΔScore: {'+' if delta >= 0 else ''}{delta:5.1f}%]")
        buf.append("-" * 78)

        if split_stats:
            buf.append(f"  {'BENCHMARK SPLIT':<28} | {'BASELINE':<12} | {'POST-CONSOLIDATION':<18}")
            buf.append("  " + "-" * 74)
            for split_name in sorted(split_stats.keys()):
                st = split_stats[split_name]
                b_str = f"{st['baseline_pass']}/{st['baseline_tot']}" if st['baseline_tot'] else "..."
                p_str = f"{st['post_pass']}/{st['post_tot']}" if st['post_tot'] else "..."
                buf.append(f"  {split_name:<28} | {b_str:<12} | {p_str:<18}")
            buf.append("-" * 78)

        buf.append("  RECENT TRACES:")
        if recent_events:
            for ev in recent_events[-4:]:
                split_lbl = ev.get("split", "Item")[:26]
                toks = ev.get("tokens_generated", 0)
                lat = ev.get("latency_s", 0.0)
                tps = ev.get("tok_per_sec", 0.0)
                status_txt = "\033[92mPASS\033[0m" if ev.get("verified") else "\033[91mFAIL\033[0m"
                buf.append(f"    • {split_lbl:<26} | {toks:3d} toks | {lat:5.2f}s ({tps:4.1f} tps) | {status_txt}")
        else:
            buf.append("    (Waiting for pipeline initialization...)")

        buf.append("=" * 78)
        buf.append("  (Press Ctrl+C to exit monitor — background pipeline execution continues)")

        sys.stdout.write("\n".join(buf) + "\n")
        sys.stdout.flush()

        if is_done:
            print(f"\n[✓] Master Evaluation Finished. Final Report: {REPORT_FILE}\n")
            break

        time.sleep(1.0)

if __name__ == "__main__":
    try:
        render_dashboard()
    except KeyboardInterrupt:
        print("\n[!] Monitor exited.")
