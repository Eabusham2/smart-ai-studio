path = "master_4000_eval_suite.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

start_idx = -1
for i, line in enumerate(lines):
    if "def _evaluate_all_splits" in line:
        start_idx = i
        break

if start_idx != -1:
    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        if lines[i].startswith("    def "):
            end_idx = i
            break
    
    method_code = '''    def _evaluate_all_splits(self, all_splits: Dict[str, List[Dict[str, Any]]], completed_cache: Dict[str, Any],
                             phase_label: str, start_time: float, total_count: int) -> Dict[str, float]:
        split_scores = {}
        global_idx = 0

        for split_name, items in all_splits.items():
            correct = 0

            for item in items:
                global_idx += 1
                item_key = f"{phase_label}_{item['id']}"

                if (time.time() - start_time) >= self.max_duration_seconds:
                    print("\\n[!] 72-Hour Time Budget Exceeded.")
                    break

                if item_key in completed_cache:
                    is_correct = completed_cache[item_key]
                    correct += 1 if is_correct else 0
                    continue

                t0 = time.perf_counter()
                is_correct = self._evaluate_single_item(split_name, item)
                dur = max(0.001, time.perf_counter() - t0)

                correct += 1 if is_correct else 0
                completed_cache[item_key] = is_correct

                elapsed = time.time() - start_time
                rate = global_idx / elapsed if elapsed > 1 else 1.0
                remaining_sec = (total_count - global_idx) / rate
                eta_str = str(timedelta(seconds=int(remaining_sec)))

                sys_mem = psutil.virtual_memory()
                system_ram_gb = sys_mem.used / (1024 ** 3)

                tok_speed = getattr(self, 'last_tok_per_sec', 15.0)
                progress_pct = (global_idx / total_count) * 100.0

                sys.stdout.write(f"\\r[{phase_label}] {split_name:<14} | Item {global_idx}/{total_count} ({progress_pct:5.2f}%) | Speed: {tok_speed:4.1f}t/s | ETA: {eta_str} | RAM: {system_ram_gb:.1f}GB  ")
                sys.stdout.flush()

                if global_idx % 5 == 0:
                    gc.collect()
                    if MLX_AVAILABLE:
                        try:
                            mx.metal.clear_cache()
                        except Exception:
                            pass
                    self.checkpoint_mgr.save_checkpoint(completed_cache, phase_label, start_time)
                    pass_rate = (correct / max(1, len(items))) * 100.0
                    self._stream_telemetry(
                        item_idx=global_idx,
                        total_items=total_count,
                        split=split_name,
                        pass_rate=pass_rate,
                        tok_per_sec=tok_speed,
                        lif_spikes=len(self.engine.lif.spike_history),
                        spec_rate=42.5,
                        ortho_overlap=0.0,
                        phase=phase_label
                    )

            split_acc = (correct / max(1, len(items))) * 100.0
            split_scores[split_name] = split_acc
            print(f"\\n[Split Done] {phase_label} - {split_name}: {split_acc:.2f}% ({correct}/{len(items)})")

        return split_scores
'''

    new_lines = lines[:start_idx] + [method_code + "\n"] + lines[end_idx:]
    new_code = "".join(new_lines)
    
    compile(new_code, path, "exec")
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_code)
    print("[✓] Indentation repaired successfully and syntax verified!")
else:
    print("[✗] Method not found.")
