"""Final benchmark wiring fixes layered over master_4000_runtime.

Does not change the V5 fused generation loop, EOS behavior, or 4096-token ceiling.
It restores model-driven evaluation for every model benchmark and fixes live RAM/ETA display.
"""
from __future__ import annotations

import collections
import json
import re
import threading
import time
from datetime import timedelta

import psutil

from run_studio_complete import MLX_AVAILABLE

if MLX_AVAILABLE:
    import mlx.core as mx


def _chat(tokenizer, user, system):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _clean_output(text):
    if not text:
        return ""
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    elif "<think>" in text:
        boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
        if boxed:
            return boxed[-1].strip()
        code = re.findall(r"```(?:python|diff|patch)?\s*(.*?)\s*```", text, re.S | re.I)
        if code:
            return code[-1].strip()
        lines = [x.strip() for x in text.replace("<think>", "").splitlines() if x.strip()]
        return lines[-1] if lines else ""
    code = re.findall(r"```(?:python|diff|patch)?\s*(.*?)\s*```", text, re.S | re.I)
    return code[-1].strip() if code else text.strip()


def _parse_json_object(text):
    cleaned = _clean_output(text)
    candidates = [cleaned]
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    return None


def _mlx_active_memory_bytes():
    if not MLX_AVAILABLE:
        return 0
    for owner in (mx, getattr(mx, "metal", None)):
        if owner is None:
            continue
        fn = getattr(owner, "get_active_memory", None)
        if callable(fn):
            try:
                value = int(fn())
                if value > 0:
                    return value
            except Exception:
                pass
    return 0


def _real_ram_gb():
    """Prefer MLX active unified-memory accounting for the model; fall back to process RSS."""
    active = _mlx_active_memory_bytes()
    rss = psutil.Process().memory_info().rss
    return (active if active > 0 else rss) / (1024 ** 3)


def install(cls, system_prompt):
    original_fast_generate = cls._fast_generate

    def generate(self, user_message):
        prompt = _chat(self.engine.tokenizer, user_message, system_prompt)
        out = original_fast_generate(self, prompt, max_tokens=4096)
        self.last_raw_out = out
        return out

    def evaluate_one(self, split, item):
        # Baseline prompt wording recovered from the user's working pre-checkout file.
        if "HumanEval" in split or "LiveCodeBench" in split:
            out = generate(
                self,
                f"{item['prompt']}\n\nComplete the Python function above. Use scratchpad only for logic outline. "
                "Output ONLY the valid executable Python code wrapped in ```python ... ```.",
            )
            code = _clean_output(out)
            result = self.engine.sandbox.execute_python_code(item["prompt"] + "\n" + code, item["test"])
            return bool(result.passed)

        if "DeepSWE" in split:
            repo_text = "\n\n".join(
                f"### {path}\n```\n{body}\n```" for path, body in item["repo_files"].items()
            )
            out = generate(
                self,
                "Repair the repository so the test command passes. Output ONLY the unified diff patch.\n\n"
                f"Repository files:\n{repo_text}\n\nTest command: {item['test_cmd']}",
            )
            patch = _clean_output(out)
            return bool(self.engine.sandbox.verify_git_diff_patch(item["repo_files"], patch, item["test_cmd"]).passed)

        if any(x in split for x in ("RSI", "SelfImprovement", "SelfCorrection", "Branch", "Consolidation")):
            out = generate(
                self,
                f"{item['prompt']}\n\nAnalyze and rectify flaws on scratchpad. State the optimized result directly.",
            )
            cleaned = _clean_output(out)
            if item.get("test"):
                result = self.engine.sandbox.execute_python_code(
                    item.get("prompt", "") + "\n" + cleaned, item["test"]
                )
                return bool(result.passed)
            expected = str(item.get("expected", "")).strip()
            return bool(expected and (expected.lower() in cleaned.lower() or expected.lower() in out.lower()))

        if any(x in split for x in ("DialogueRecall", "LearningFacts", "FactRetention")):
            # Do not let the external KG short-circuit the benchmark: the model must answer.
            out = generate(self, f"{item['prompt']}\nState the exact recalled entity or fact directly.")
            expected = str(item.get("expected", item.get("expected_keyword", ""))).strip()
            return bool(expected and expected.lower() in out.lower())

        if any(x in split for x in ("GSM8K", "MATH", "AIME")):
            out = generate(
                self,
                f"{item['prompt']}\n\nSolve this problem using a minimal scratchpad. "
                "State the final answer inside \\boxed{answer}.",
            )
            expected = str(item["expected"]).strip()
            boxed = re.findall(r"\\boxed\{([^}]+)\}", out)
            cleaned = _clean_output(out)
            return bool(
                (boxed and boxed[-1].strip() == expected)
                or expected in cleaned
                or expected.replace(" ", "") in cleaned.replace(" ", "")
                or expected in out
            )

        if "TensorGraphDSL" in split:
            out = generate(
                self,
                f"{item['prompt']}\n"
                "DSL Rules:\n"
                "- `arr >>~fold(k)`: Rotates list left by k positions.\n"
                "- `arr <#>scale(s)`: Multiplies each element by scalar s.\n"
                "- `arr1 @fuse arr2`: Element-wise addition.\n"
                "Calculate on scratchpad and output the final numeric list [x, y, ...] directly.",
            )
            try:
                expected_value = self.engine.sandbox.evaluate_dsl_expression(item["dsl_expr"])
            except Exception:
                return False
            if expected_value is None:
                return False
            expected = str(expected_value).strip()
            cleaned = _clean_output(out)
            return expected in cleaned or expected.replace(" ", "") in cleaned.replace(" ", "") or expected in out

        if "ZebraLogic" in split or "HLE" in split:
            out = generate(self, f"{item['prompt']}\nDeduce the solution directly. State the final answer on the last line.")
            expected = str(item.get("expected", item.get("expected_token", ""))).strip()
            cleaned = _clean_output(out)
            return bool(expected and (expected.lower() in cleaned.lower() or expected.lower() in out.lower()))

        if "BFCL" in split:
            out = generate(
                self,
                f"{item['prompt']}\nReturn ONLY one JSON object with keys `name` and `arguments`, "
                "using exactly the requested tool name and argument values.",
            )
            value = _parse_json_object(out)
            if not isinstance(value, dict):
                return False
            if isinstance(value.get("function"), dict):
                value = value["function"]
            name = value.get("name") or value.get("tool")
            args = value.get("arguments") or value.get("args")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            return name == item.get("expected_tool") and args == item.get("expected_args")

        # GPQA, MMLU-Pro, AutonomousEvolution and all other model-scored splits.
        out = generate(self, f"{item['prompt']}\nState only the final answer directly.")
        expected = str(item.get("expected", item.get("expected_token", ""))).strip()
        return bool(expected and expected.lower() in out.lower())

    def completed(cache, key):
        if key in cache:
            return True
        for prefix in ("__v2done__:", "__eyad_v3_done__:", "__eyad_v5_done__:"):
            if cache.get(prefix + key) is True:
                return True
        return False

    def fmt_status(phase, split_name, overall, total, speed, eta):
        pct = 100.0 * overall / max(1, total)
        return (
            f"[{phase}] {split_name:<22} | Item {overall}/{total} ({pct:5.2f}%) | "
            f"Speed: {speed:4.1f}t/s | ETA: {eta} | RAM: {_real_ram_gb():.1f}GB"
        )

    def evaluate_all(self, splits, cache, phase, start, total):
        self.time_budget_exhausted = False
        recent = collections.deque(maxlen=20)
        cached = sum(completed(cache, f"{phase}_{item['id']}") for items in splits.values() for item in items)
        remaining = total - cached
        overall = cached
        ran = 0
        scores = {}

        print(f"[*] Checkpoint Loaded: {cached}/{total} items already completed ({cached/total*100:.1f}%).")
        print(f"[*] Tasks to execute: {remaining}")

        for split_name, items in splits.items():
            split_correct = sum(cache.get(f"{phase}_{item['id']}") is True for item in items)
            pending = [item for item in items if not completed(cache, f"{phase}_{item['id']}")]
            if not pending:
                scores[split_name] = 100.0 * split_correct / max(1, len(items))
                print(f"[Split Done] {phase} - {split_name}: {scores[split_name]:.2f}% ({split_correct}/{len(items)})")
                continue

            for item in pending:
                key = f"{phase}_{item['id']}"
                if time.time() - start >= self.max_duration_seconds:
                    self.time_budget_exhausted = True
                    self.checkpoint_mgr.save_checkpoint(cache, phase, start)
                    print("[!] 72-hour active budget reached; checkpoint saved.")
                    return scores

                item_start = time.perf_counter()
                stop_status = threading.Event()

                def status_worker():
                    while not stop_status.wait(10.0):
                        speed = float(getattr(self, "last_tok_per_sec", 0.0) or 0.0)
                        left = max(0, remaining - ran)
                        if recent:
                            avg = sum(recent) / len(recent)
                            age = time.perf_counter() - item_start
                            eta_seconds = max(0, int(avg * left - min(age, avg)))
                            eta = str(timedelta(seconds=eta_seconds))
                        else:
                            eta = "calculating"
                        print(fmt_status(phase, split_name, overall, total, speed, eta), flush=True)

                thread = threading.Thread(target=status_worker, daemon=True)
                thread.start()
                try:
                    ok = bool(self._evaluate_single_item(split_name, item))
                except KeyboardInterrupt:
                    stop_status.set()
                    self.checkpoint_mgr.save_checkpoint(cache, phase, start)
                    print("[✓] Ctrl+C: checkpoint saved.")
                    raise
                finally:
                    stop_status.set()

                duration = max(0.001, time.perf_counter() - item_start)
                recent.append(duration)
                cache[key] = ok
                cache["__v2done__:" + key] = True
                cache["__eyad_v3_done__:" + key] = True
                overall += 1
                ran += 1
                if ok:
                    split_correct += 1

                left = max(0, remaining - ran)
                eta = str(timedelta(seconds=max(0, int((sum(recent) / len(recent)) * left))))
                speed = float(getattr(self, "last_tok_per_sec", 0.0) or 0.0)
                print(fmt_status(phase, split_name, overall, total, speed, eta), flush=True)

                if ran % 5 == 0:
                    self.checkpoint_mgr.save_checkpoint(cache, phase, start)

            self.checkpoint_mgr.save_checkpoint(cache, phase, start)
            scores[split_name] = 100.0 * split_correct / max(1, len(items))
            print(f"[Split Done] {phase} - {split_name}: {scores[split_name]:.2f}% ({split_correct}/{len(items)})")

        return scores

    cls._evaluate_single_item = evaluate_one
    cls._evaluate_all_splits = evaluate_all
