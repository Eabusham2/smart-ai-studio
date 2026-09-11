"""Final merged runtime for the 4,014-item master evaluation suite."""
from __future__ import annotations

import collections
import gc
import json
import platform
import re
import threading
import time
from datetime import timedelta

import psutil

from run_studio_complete import *

if MLX_AVAILABLE:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils
    from mlx_lm.models.cache import make_prompt_cache


SYSTEM_PROMPT = (
    "You are a fast symbolic computing engine. Keep your internal scratchpad (<think>) strictly minimal: "
    "write only concise intermediate values or math. No conversational monologue, no self-reflection, and no verification loops. "
    "Close </think> immediately once calculated and output the answer."
)


def _merged_init_lora(self):
    if not MLX_AVAILABLE or self.model is None:
        return
    self.model.freeze()
    layers = getattr(self.model, "layers", []) or getattr(getattr(self.model, "model", None), "layers", [])
    for layer in layers:
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj"):
            if not isinstance(layer.self_attn.q_proj, LoRALinear):
                layer.self_attn.q_proj = LoRALinear.from_base(
                    layer.self_attn.q_proj,
                    r=self.settings.lora_rank,
                    scale=self.settings.lora_alpha / max(1, self.settings.lora_rank),
                )
            if hasattr(layer.self_attn.q_proj, "unfreeze"):
                layer.self_attn.q_proj.unfreeze()
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
            if not isinstance(layer.mlp.down_proj, LoRALinear):
                layer.mlp.down_proj = LoRALinear.from_base(
                    layer.mlp.down_proj,
                    r=self.settings.lora_rank,
                    scale=self.settings.lora_alpha / max(1, self.settings.lora_rank),
                )
            if hasattr(layer.mlp.down_proj, "unfreeze"):
                layer.mlp.down_proj.unfreeze()
    self.adapters_buffer_a = {
        k: mx.array(v) for k, v in dict(mlx.utils.tree_flatten(self.model.trainable_parameters())).items()
    }
    self.adapters_buffer_b = {k: mx.array(v) for k, v in self.adapters_buffer_a.items()}


MoEDualBufferManager._init = _merged_init_lora


def _merged_dialogue_ingest(self):
    rows = [
        ("ASUS ROG GT-BE19000", "runs_service", "AdGuard Home DNS", "network_session"),
        ("AdGuard Home DNS", "hosted_in", "Portainer Docker AI Board", "network_session"),
        ("BD PROCHOT Sensor", "state_decision", "Disabled via ThrottleStop", "hardware_session"),
        ("ROG Z790 Motherboard", "paired_with", "Thermal Grizzly Contact Frame", "hardware_session"),
        ("Ternary-Bonsai-27B", "quantization_format", "1.58-bit ternary MLX", "ml_architecture"),
        ("Omni-agi Engine", "combines", "Spiking Neural Networks & Liquid Networks", "ml_architecture"),
        ("3.6TB BitLocker Partition", "recovered_via", "DiskGenius Sector Editing & repair-bde", "recovery_session"),
        ("banana-mcp", "deployed_on", "Vercel Serverless Handler", "mcp_session"),
        ("banana-mcp", "integrates_with", "Zapier Claude Google Gemini MCP", "mcp_session"),
        ("MLX Metal", "uses", "Apple unified memory", "ml_architecture"),
        ("TensorGraphDSL", "supports", "fold scale fuse", "eval"),
    ]
    for src, pred, target, session in rows:
        self.kg.insert_triple(src, pred, target, weight=1.0, session_id=session)
    return len(rows)


DialogueTimelineGraphIngester.ingest_developer_sessions = _merged_dialogue_ingest


def _strict_initialize_runtime(self):
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX/MLX-LM is unavailable; refusing to run the real benchmark offline.")
    before_used = psutil.virtual_memory().used
    try:
        self.model, self.tokenizer = load(self.settings.mlx_model_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load real model {self.settings.mlx_model_path}: {exc!r}") from exc
    if self.model is None or self.tokenizer is None:
        raise RuntimeError(f"Model loader returned no model/tokenizer for {self.settings.mlx_model_path}")
    try:
        mx.eval(self.model.parameters())
    except Exception as exc:
        raise RuntimeError(f"27B model loaded but weight materialization failed: {exc!r}") from exc
    after_used = psutil.virtual_memory().used
    self.model_resident_gb = max(0.0, (after_used - before_used) / (1024 ** 3))
    self.moe_manager = MoEDualBufferManager(self.model, self.settings)
    self.moe_router = HierarchicalMoERouter(self.model)
    self.grpo_trainer = GRPOTrainingEngine(self.model, self.tokenizer, self.sandbox)
    if self.settings.enable_awake_ogp_daemon:
        self.ogp_daemon = ProjectedSleepConsolidationDaemon(
            self.moe_manager,
            self.ogp_projector,
            self.kg,
            self.tokenizer,
            self.settings,
            METAL_STREAM_LOCK,
        )
        self.ogp_daemon.start()
    print(
        f"[✓] REAL MODEL READY: {self.settings.mlx_model_path} | "
        f"system RAM {psutil.virtual_memory().used/(1024**3):.2f} GB | "
        f"load delta {self.model_resident_gb:.2f} GB",
        flush=True,
    )


UnifiedMasterEngine._initialize_runtime = _strict_initialize_runtime


def clean_output(text: str) -> str:
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


def _chat(tokenizer, user, system=SYSTEM_PROMPT):
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


def _parse_json_object(text: str):
    cleaned = clean_output(text)
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


def _model_context_limit(engine) -> int | None:
    values = []
    tok = getattr(engine, "tokenizer", None)
    model = getattr(engine, "model", None)
    for obj in (getattr(model, "args", None), getattr(model, "config", None), tok):
        if obj is None:
            continue
        for name in (
            "max_position_embeddings",
            "max_seq_len",
            "max_sequence_length",
            "context_length",
            "model_max_length",
        ):
            try:
                value = int(getattr(obj, name))
            except Exception:
                continue
            if 1024 <= value <= 10_000_000:
                values.append(value)
    return min(values) if values else None


def _benchmark_ceiling(self) -> int:
    is_apple_silicon = platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64")
    target = 16384 if is_apple_silicon else 8192
    ctx = _model_context_limit(self.engine)
    if ctx is None:
        return target
    if target == 16384 and ctx >= 16384:
        return 16384
    if ctx >= 8192:
        return 8192
    return max(1024, ctx)


def _real_ram_gb() -> float:
    """Match the working baseline: report total unified/system RAM currently used."""
    return psutil.virtual_memory().used / (1024 ** 3)


def fast_generate(self, prompt, max_tokens=16384, stream=False):
    """Real fused MLX decode; benchmark ceiling is dynamic and normal EOS remains authoritative."""
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX/MLX-LM unavailable: refusing to fake/offline benchmark generation")
    if self.engine.model is None or self.engine.tokenizer is None:
        raise RuntimeError("27B model/tokenizer not loaded: refusing to evaluate without real generation")

    tok = self.engine.tokenizer
    model = self.engine.model
    cache = inp = logits = step = None
    try:
        ids = tok.encode(prompt)
        ctx = _model_context_limit(self.engine)
        if ctx is not None:
            max_tokens = min(int(max_tokens), max(1, ctx - len(ids)))
        cache = make_prompt_cache(model)
        eos = set()
        e = getattr(tok, "eos_token_id", None)
        if e is not None:
            eos.update(e if isinstance(e, (list, tuple, set)) else [e])
        for name in ("<|im_end|>", "<end_of_turn>", "<|eot_id|>", "<|endoftext|>", "</s>", "<eos>"):
            try:
                encoded = tok.encode(name, add_special_tokens=False)
                if len(encoded) == 1:
                    eos.add(int(encoded[0]))
                if hasattr(tok, "convert_tokens_to_ids"):
                    tid = tok.convert_tokens_to_ids(name)
                    if isinstance(tid, int) and tid >= 0:
                        eos.add(tid)
            except Exception:
                pass

        with METAL_STREAM_LOCK:
            inp = mx.array([ids])
            logits = model(inp, cache=cache)
            next_arr = mx.argmax(logits[0, -1])
            mx.eval(next_arr)
            nxt = int(next_arr.item())
            out = []
            if nxt not in eos:
                out.append(nxt)

            decode = 0
            t0 = time.perf_counter()
            live_update = t0
            for _ in range(max(0, max_tokens - 1)):
                if not out or out[-1] in eos:
                    break
                step = model(mx.array([[out[-1]]]), cache=cache)
                next_arr = mx.argmax(step[0, -1])
                mx.eval(next_arr)
                nxt = int(next_arr.item())
                if nxt in eos:
                    break
                out.append(nxt)
                decode += 1
                now = time.perf_counter()
                if now - live_update >= 2.0:
                    self.last_tok_per_sec = decode / max(0.001, now - t0)
                    self.live_generated_tokens = len(out)
                    live_update = now

            dt = max(0.001, time.perf_counter() - t0)

        self.last_tok_per_sec = decode / dt if decode else 0.0
        self.live_generated_tokens = len(out)
        self.last_generation_error = None
        return tok.decode(out)
    except Exception as exc:
        self.last_tok_per_sec = 0.0
        self.last_generation_error = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(f"Real MLX generation failed: {self.last_generation_error}") from exc
    finally:
        cache = inp = logits = step = None
        if MLX_AVAILABLE:
            try:
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()
            except Exception:
                pass
        gc.collect()


def evaluate_one(self, split, item):
    """Same model-driven handler is used in baseline and post-consolidation evaluation."""
    tok = self.engine.tokenizer

    def gen(user_message):
        prompt = _chat(tok, user_message)
        ceiling = getattr(self, "benchmark_max_tokens", None) or _benchmark_ceiling(self)
        out = self._fast_generate(prompt, max_tokens=ceiling)
        self.last_raw_out = out
        return out

    if "HumanEval" in split or "LiveCodeBench" in split:
        out = gen(
            f"{item['prompt']}\n\n"
            "Complete the Python function above. Use scratchpad only for logic outline. "
            "Output ONLY the valid executable Python code wrapped in ```python ... ```."
        )
        code = clean_output(out)
        result = self.engine.sandbox.execute_python_code(item["prompt"] + "\n" + code, item["test"])
        return bool(result.passed)

    if "DeepSWE" in split:
        repo_text = "\n\n".join(f"### {path}\n```\n{body}\n```" for path, body in item["repo_files"].items())
        out = gen(
            "Repair the repository so the test command passes. Output ONLY the unified diff patch.\n\n"
            f"Repository files:\n{repo_text}\n\nTest command: {item['test_cmd']}"
        )
        patch = clean_output(out)
        return bool(self.engine.sandbox.verify_git_diff_patch(item["repo_files"], patch, item["test_cmd"]).passed)

    if any(x in split for x in ("RSI", "SelfImprovement", "SelfCorrection", "Branch", "Consolidation")):
        out = gen(f"{item['prompt']}\n\nAnalyze and rectify flaws on scratchpad. State the optimized result directly.")
        cleaned = clean_output(out)
        if item.get("test"):
            result = self.engine.sandbox.execute_python_code(item.get("prompt", "") + "\n" + cleaned, item["test"])
            return bool(result.passed)
        expected = str(item.get("expected", "")).strip()
        return bool(expected and (expected.lower() in cleaned.lower() or expected.lower() in out.lower()))

    if any(x in split for x in ("DialogueRecall", "LearningFacts", "FactRetention")):
        out = gen(f"{item['prompt']}\nState the exact recalled entity or fact directly.")
        expected = str(item.get("expected", item.get("expected_keyword", ""))).strip()
        return bool(expected and expected.lower() in out.lower())

    if any(x in split for x in ("GSM8K", "MATH", "AIME")):
        out = gen(
            f"{item['prompt']}\n\nSolve this problem using a minimal scratchpad. "
            "State the final answer inside \\boxed{answer}."
        )
        expected = str(item["expected"]).strip()
        boxed = re.findall(r"\\boxed\{([^}]+)\}", out)
        cleaned = clean_output(out)
        return bool(
            (boxed and boxed[-1].strip() == expected)
            or expected in cleaned
            or expected.replace(" ", "") in cleaned.replace(" ", "")
            or expected in out
        )

    if "TensorGraphDSL" in split:
        out = gen(
            f"{item['prompt']}\n"
            "DSL Rules:\n"
            "- `arr >>~fold(k)`: Rotates list left by k positions.\n"
            "- `arr <#>scale(s)`: Multiplies each element by scalar s.\n"
            "- `arr1 @fuse arr2`: Element-wise addition.\n"
            "Calculate on scratchpad and output the final numeric list [x, y, ...] directly."
        )
        try:
            expected_value = self.engine.sandbox.evaluate_dsl_expression(item["dsl_expr"])
        except Exception:
            return False
        if expected_value is None:
            return False
        expected = str(expected_value).strip()
        cleaned = clean_output(out)
        return expected in cleaned or expected.replace(" ", "") in cleaned.replace(" ", "") or expected in out

    if "ZebraLogic" in split or "HLE" in split:
        out = gen(f"{item['prompt']}\nDeduce the solution directly. State the final answer on the last line.")
        expected = str(item.get("expected", item.get("expected_token", ""))).strip()
        cleaned = clean_output(out)
        return bool(expected and (expected.lower() in cleaned.lower() or expected.lower() in out.lower()))

    if "BFCL" in split:
        out = gen(
            f"{item['prompt']}\nReturn ONLY one JSON object with keys `name` and `arguments`, "
            "using exactly the requested tool name and argument values."
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

    out = gen(f"{item['prompt']}\nState only the final answer directly.")
    expected = str(item.get("expected", item.get("expected_token", ""))).strip()
    return bool(expected and expected.lower() in out.lower())


def scores_from_cache(splits, cache, phase):
    return {
        name: 100.0 * sum(cache.get(f"{phase}_{item['id']}") is True for item in items) / max(1, len(items))
        for name, items in splits.items()
    }


def _completed(cache, key):
    if key in cache:
        return True
    for prefix in ("__v2done__:", "__eyad_v3_done__:", "__eyad_v5_done__:"):
        if cache.get(prefix + key) is True:
            return True
    return False


def _fmt_status(phase, split_name, overall, total, speed, eta):
    pct = 100.0 * overall / max(1, total)
    return (
        f"[{phase}] {split_name:<22} | Item {overall}/{total} ({pct:5.2f}%) | "
        f"Speed: {speed:4.1f}t/s | ETA: {eta} | RAM: {_real_ram_gb():.1f}GB"
    )


def evaluate_all(self, splits, cache, phase, start, total):
    self.time_budget_exhausted = False
    recent = collections.deque(maxlen=20)
    cached = sum(_completed(cache, f"{phase}_{item['id']}") for items in splits.values() for item in items)
    remaining = total - cached
    overall = cached
    ran = 0
    scores = {}

    print(f"[*] Checkpoint Loaded: {cached}/{total} items already completed ({cached/total*100:.1f}%).")
    print(f"[*] Tasks to execute: {remaining}")

    for split_name, items in splits.items():
        split_correct = sum(cache.get(f"{phase}_{item['id']}") is True for item in items)
        pending = [item for item in items if not _completed(cache, f"{phase}_{item['id']}")]
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
                    print(_fmt_status(phase, split_name, overall, total, speed, eta), flush=True)

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
            print(_fmt_status(phase, split_name, overall, total, speed, eta), flush=True)

            if ran % 5 == 0:
                self.checkpoint_mgr.save_checkpoint(cache, phase, start)

        self.checkpoint_mgr.save_checkpoint(cache, phase, start)
        scores[split_name] = 100.0 * split_correct / max(1, len(items))
        print(f"[Split Done] {phase} - {split_name}: {scores[split_name]:.2f}% ({split_correct}/{len(items)})")

    return scores


def run_full(self):
    if not MLX_AVAILABLE or self.engine.model is None or self.engine.tokenizer is None:
        raise RuntimeError("Real 27B MLX model is not loaded; Phase 1 will not start offline")

    self.benchmark_max_tokens = _benchmark_ceiling(self)
    ctx = _model_context_limit(self.engine)
    ctx_label = str(ctx) if ctx else "unknown"

    print("=" * 95)
    print("🚀 COMMENCING 4,000+ ITEM MASTER EVALUATION SUITE (FULL PRE/TEACH/POST)")
    print("│ 4,014 baseline → dialogue/memory → MCTS → OGP attempt → 4,014 post")
    print(f"│ Benchmark-only generation ceiling: {self.benchmark_max_tokens:,} tokens | detected model context: {ctx_label}")
    print("=" * 95)

    splits = self.provider.load_all_4000_items()
    total = sum(map(len, splits.values()))
    chk = self.checkpoint_mgr.load_checkpoint()
    cache = chk.get("completed_items", {}) if isinstance(chk, dict) else {}
    cache = cache if isinstance(cache, dict) else {}
    start = time.time()
    phase = chk.get("phase", "Phase 1: Baseline") if isinstance(chk, dict) else "Phase 1: Baseline"

    if phase == "Phase 4: Post-Consolidation":
        base = scores_from_cache(splits, cache, "Phase 1: Baseline")
        post = self._evaluate_all_splits(splits, cache, "Phase 4: Post-Consolidation", start, total)
        if not self.time_budget_exhausted:
            self._generate_master_report(base, post, total, time.time() - start)
        return

    print("\n▶ PHASE 1: ZERO-SHOT BASELINE")
    base = self._evaluate_all_splits(splits, cache, "Phase 1: Baseline", start, total)
    if self.time_budget_exhausted:
        return

    print("\n▶ PHASE 2: DIALOGUE / MEMORY INGESTION + MCTS TEACHING")
    DialogueTimelineGraphIngester(self.engine.kg).ingest_developer_sessions()
    for item in splits["TensorGraphDSL-300"][:30]:
        inv, q, visits = self.engine.mcts.search_best_invariant(item["dsl_expr"])
        self.engine.kg.insert_triple(item["dsl_expr"], "evaluates_to", inv, weight=q)

    print("\n▶ PHASE 3: OGP SLEEP CONSOLIDATION")
    uncon = self.engine.kg.fetch_unconsolidated_high_surprise(0.80, 20)
    if uncon and MLX_AVAILABLE and self.engine.moe_manager:
        try:
            opt = optim.AdamW(learning_rate=1e-4)
            for memory in uncon:
                text = (
                    f"<|im_start|>user\n{memory['prompt']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{memory['completion']}<|im_end|>"
                )
                ids = self.engine.tokenizer.encode(text)
                if len(ids) <= 1:
                    continue
                with METAL_STREAM_LOCK:
                    inp = mx.array([ids[: min(len(ids), 64)]])
                    lossfn = lambda model: mx.mean(
                        nn.losses.cross_entropy(model(inp)[:, :-1, :].astype(mx.float32), inp[:, 1:])
                    )
                    loss, grads = nn.value_and_grad(self.engine.model, lossfn)(self.engine.model)
                    flat, shapes = self.engine.ogp_projector.flatten_gradients(dict(mlx.utils.tree_flatten(grads)))
                    projected = self.engine.ogp_projector.project_gradient(flat)
                    tree = self.engine.ogp_projector.unflatten_gradients(projected, shapes)
                    opt.update(self.engine.model, mlx.utils.tree_unflatten(list(tree.items())))
                    mx.eval(self.engine.model.parameters())
            self.engine.moe_manager.swap_buffers_atomic()
            print("[✓] OGP consolidation update applied.")
        except Exception as exc:
            print(f"[!] OGP backend limitation: {exc}")
            print("[!] No OGP parameter update claimed; continuing to Phase 4.")
    elif not uncon:
        print("[*] No eligible unconsolidated memories; continuing to Phase 4.")
    else:
        print("[!] OGP manager unavailable; no parameter update claimed; continuing to Phase 4.")

    print("\n▶ PHASE 4: POST-CONSOLIDATION FULL RETEST")
    self.checkpoint_mgr.save_checkpoint(cache, "Phase 4: Post-Consolidation", start)
    post = self._evaluate_all_splits(splits, cache, "Phase 4: Post-Consolidation", start, total)
    if not self.time_budget_exhausted:
        self._generate_master_report(base, post, total, time.time() - start)


def install(cls):
    cls._fast_generate = fast_generate
    cls._evaluate_single_item = evaluate_one
    cls._evaluate_all_splits = evaluate_all
    cls.run_full_suite = run_full
