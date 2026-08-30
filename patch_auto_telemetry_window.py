eval_path = "master_4000_eval_suite.py"

with open(eval_path, "r", encoding="utf-8") as f:
    code = f.read()

# Add automatic terminal window spawn for telemetry in Master4000EvaluationEngine.__init__
auto_telemetry_init = '''        self.telemetry_file = "eval_results/telemetry_stream.jsonl"
        self.last_tok_per_sec = 0.0
        os.makedirs("eval_results", exist_ok=True)
        
        # Automatically launch live telemetry monitor window on macOS
        try:
            cwd = os.getcwd()
            script_path = os.path.join(cwd, "eval_telemetry_monitor.py")
            subprocess.Popen([
                "osascript", "-e",
                f'tell application "Terminal" to do script "cd {cwd} && python3 {script_path}"'
            ])
        except Exception:
            pass'''

code = code.replace(
    'self.telemetry_file = "eval_results/telemetry_stream.jsonl"\n        self.last_tok_per_sec = 0.0',
    auto_telemetry_init
)

with open(eval_path, "w", encoding="utf-8") as f:
    f.write(code)

print("[✓] Patched master_4000_eval_suite.py to auto-launch the telemetry monitor window.")
