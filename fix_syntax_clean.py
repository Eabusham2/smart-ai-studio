eval_path = "master_4000_eval_suite.py"

with open(eval_path, "r", encoding="utf-8") as f:
    code = f.read()

# Locate and replace the broken print statement or method cleanly
old_print = 'print(f"\\n│ [{phase_label}] {split_name:<28} : {split_acc:>6.2f}% ({correct}/{len(items)})")'
new_print = 'print(f"\\n│ [{phase_label}] {{split_name:<28}} : {{split_acc:>6.2f}}% ({{correct}}/{{len(items)}})")'

if old_print in code:
    code = code.replace(old_print, new_print)
else:
    # Alternative pattern match
    import re
    code = re.sub(r'print\(f"\\n│.*?split_acc.*?print\(f"\\n│.*?\)', 'print(f"\\n│ [{phase_label}] {split_name:<28} : {split_acc:>6.2f}% ({correct}/{len(items)})")', code)

with open(eval_path, "w", encoding="utf-8") as f:
    f.write(code)

print("[✓] Syntax corrected successfully.")
