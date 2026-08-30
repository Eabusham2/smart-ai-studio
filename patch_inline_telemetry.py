eval_path = "master_4000_eval_suite.py"

with open(eval_path, "r", encoding="utf-8") as f:
    code = f.read()

# Replace the inner loop logging to print real-time inline updates directly to stdout
old_loop_block = '''                if global_idx % 5 == 0:
                    gc.collect()
                    if MLX_AVAILABLE:
                        try:
                            mx.metal.clear_cache()
                        except Exception:
                            pass
                    self.checkpoint_mgr.save_checkpoint(completed_cache, phase_label, start_time)
                    pass_rate = (correct / max(1, len(items))) * 100.0
                    tok_speed = 32.0 / dur
                    self._stream_telemetry(
                        item_idx=global_idx,
                        total_items=total_count,
                        split=split_name,
                        pass_rate=pass_rate,
                        tok_per_sec=tok_speed,
                        lif_spikes=sum(self.engine.lif.spike_history),
                        spec_rate=42.5,
                        ortho_overlap=0.0,
                        phase=phase_label
                    )'''

new_loop_block = '''                # Inline real-time terminal output
                current_rss = psutil.Process().memory_info().rss / (1024 ** 3)
                tok_speed = getattr(self, 'last_tok_per_sec', 15.0)
                progress_pct = (global_idx / total_count) * 100.0
                sys.stdout.write(f"\\r[{phase_label}] {split_name:<16} | Item {global_idx}/{total_count} ({progress_pct:5.2f}%) | Speed: {tok_speed:4.1f} tok/s | RAM: {current_rss:4.2f} GB  ")
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
                        lif_spikes=sum(self.engine.lif.spike_history),
                        spec_rate=42.5,
                        ortho_overlap=0.0,
                        phase=phase_label
                    )'''

if old_loop_block in code:
    code = code.replace(old_loop_block, new_loop_block)
else:
    # Fallback search if spacing differs slightly
    print("[!] Precise block match not found, applying regex replacement...")
    import re
    code = re.sub(r'if global_idx % 5 == 0:.*?phase=phase_label\s*\)', new_loop_block, code, flags=re.DOTALL)

with open(eval_path, "w", encoding="utf-8") as f:
    f.write(code)

print("[✓] master_4000_eval_suite.py patched with inline real-time terminal telemetry.")
