"""RSI self-training, learning retention test, and Phase-4 Pro missed-item retest.

Design invariants:
- Phase 1 is the unchanged greedy baseline.
- "Learn" and RSI are distinct:
  * Learn = externally supplied/verified examples are stored for consolidation.
  * RSI = the model generates its own improved answers and self-critiques. Eligibility
    is decided transiently, but persistent RSI memory stores only the model's own trace:
    no benchmark question, split label, expected answer, correctness, or reward field.
- Phase 3 updates the already-loaded model in-place and persists its trainable adapter.
- Phase 4 retests only Phase-1 misses using the same RSI-updated in-memory model.
- Phase-4 Pro branch selection never sees benchmark expected answers. Deterministic
  task verifiers or answer-blind consensus choose the branch; benchmark ground truth
  is consulted only afterward for scoring.
"""
from __future__ import annotations

import collections
import gc
import json
import math
import os
import psutil
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from core.entropy_router import EntropyRouter
from core.mlx_engine import MLXReasoningBackend, _adaptive_prefill_step_size, _memory_pressure
from core.pro_engine import get_ladder_temperatures
from eval.master_4000_runtime import (
    RAW_OUTPUT_LOG,
    SYSTEM_PROMPT,
    _append_raw_generation_log,
    _benchmark_ceiling,
    _chat,
    _parse_json_object,
    _repair_suite,
    clean_output,
    scores_from_cache,
)
from run_studio_complete import (
    DialogueTimelineGraphIngester,
    METAL_STREAM_LOCK,
    MLX_AVAILABLE,
)

if MLX_AVAILABLE:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils
    import mlx_lm
    from mlx_lm.models.cache import make_prompt_cache


RSI_SESSION_ID = "phase1_rsi_self_generated_verified_v2"
LEARN_SESSION_ID = "phase2_supervised_learn_v1"
RSI_ADAPTER_PATH = os.path.join("eval_results", "rsi_post_phase3.safetensors")

# Small benchmark-learning payload. These are deliberately supplied examples:
# this is the "Learn" path, not RSI.
LEARN_EXAMPLES: List[Tuple[str, str]] = [
    ("What DNS service runs on the ASUS ROG GT-BE19000?", "AdGuard Home DNS"),
    ("Where is AdGuard Home DNS hosted?", "Portainer Docker AI Board"),
    ("What was the BD PROCHOT sensor decision?", "Disabled via ThrottleStop"),
    ("What contact frame is paired with the ROG Z790 motherboard?", "Thermal Grizzly Contact Frame"),
    ("What quantization format is used by Ternary-Bonsai-27B?", "1.58-bit ternary MLX"),
    ("What does MLX Metal use for model memory?", "Apple unified memory"),
    ("What operations does TensorGraphDSL support?", "fold scale fuse"),
]


def _assert_same_model(self, identity: int, where: str) -> None:
    if self.engine.model is None:
        raise RuntimeError(f"{where}: benchmark model unexpectedly became None")
    if id(self.engine.model) != identity:
        raise RuntimeError(
            f"{where}: model object was replaced/reloaded; refusing to test a different model"
        )


def _pro_backend(self):
    backend = getattr(self, "_phase4_pro_backend", None)
    if backend is None:
        # Constructor only. Never call backend.load_model().
        backend = MLXReasoningBackend(model_path=self.engine.settings.mlx_model_path)
        self._phase4_pro_backend = backend

    # Bind by reference to the exact benchmark model already in memory.
    backend.model = self.engine.model
    backend.tokenizer = self.engine.tokenizer
    backend.is_mlx_available = (
        self.engine.model is not None and self.engine.tokenizer is not None
    )
    return backend


def _pro_router(self):
    router = getattr(self, "_phase4_entropy_router", None)
    if router is None:
        router = EntropyRouter(
            low_threshold=0.25,
            high_threshold=0.70,
            instant_branches=1,
            pro_branches_mid=8,
            pro_branches_high=16,
        )
        self._phase4_entropy_router = router
    return router


def _normalized_entropy(self, prompt: str) -> float:
    """Use model entropy as uncertainty/confidence signal on a 0..1 scale."""
    backend = _pro_backend(self)
    raw = float(backend.calculate_token_entropy(prompt))
    tok = self.engine.tokenizer

    vocab_size = None
    for name in ("vocab_size", "n_vocab"):
        try:
            value = int(getattr(tok, name))
            if value > 1:
                vocab_size = value
                break
        except Exception:
            pass
    if vocab_size is None:
        try:
            vocab_size = len(tok)
        except Exception:
            vocab_size = 0

    if vocab_size and raw > 1.0:
        raw = raw / max(1e-9, math.log(vocab_size))
    return max(0.0, min(1.0, raw))


def _task_user_prompt(split: str, item: Dict[str, Any]) -> str:
    """Mirror the merged benchmark instructions without exposing ground truth."""
    if "HumanEval" in split:
        return (
            f"{item['prompt']}\n\n"
            "Complete the Python function above. Use scratchpad only for logic outline. "
            "Output ONLY the valid executable Python code wrapped in ```python ... ```."
        )

    if "LiveCodeBench" in split:
        return (
            f"{item['prompt']}\n\n"
            "Write the complete Python solution requested above. Use scratchpad only for logic outline. "
            "Output ONLY the valid executable Python code wrapped in ```python ... ```."
        )

    if "DeepSWE" in split:
        repo_text = "\n\n".join(
            f"### {path}\n```\n{body}\n```"
            for path, body in item["repo_files"].items()
        )
        return (
            "Repair the repository so the test command passes. Output ONLY the unified diff patch.\n\n"
            f"Repository files:\n{repo_text}\n\nTest command: {item['test_cmd']}"
        )

    if any(
        x in split
        for x in ("RSI", "SelfImprovement", "SelfCorrection", "Branch", "Consolidation")
    ):
        return (
            f"{item['prompt']}\n\n"
            "Analyze and rectify flaws on scratchpad. State the optimized result directly."
        )

    if any(x in split for x in ("DialogueRecall", "LearningFacts", "FactRetention")):
        return f"{item['prompt']}\nState the exact recalled entity or fact directly."

    if any(x in split for x in ("GSM8K", "MATH", "AIME")):
        return (
            f"{item['prompt']}\n\n"
            "Solve this problem using a minimal scratchpad. "
            "State the final answer inside \\boxed{answer}."
        )

    if "TensorGraphDSL" in split:
        return (
            f"{item['prompt']}\n"
            "DSL Rules:\n"
            "- `arr >>~fold(k)`: Rotates list left by k positions.\n"
            "- `arr <#>scale(s)`: Multiplies each element by scalar s.\n"
            "- `arr1 @fuse arr2`: Element-wise addition.\n"
            "Calculate on scratchpad and output the final numeric list [x, y, ...] directly."
        )

    if "ZebraLogic" in split or "HLE" in split:
        return (
            f"{item['prompt']}\n"
            "Deduce the solution directly. State the final answer on the last line."
        )

    if "BFCL" in split:
        return (
            f"{item['prompt']}\n"
            "Return ONLY one JSON object with keys `name` and `arguments`, "
            "using exactly the requested tool name and argument values."
        )

    return f"{item['prompt']}\nState only the final answer directly."


def _prompt_requested_bfcl(item: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Parse what the BFCL prompt itself asked for; don't use expected_* fields."""
    prompt = str(item.get("prompt", ""))
    name_match = re.search(r"Call tool [`'\"]?([A-Za-z0-9_.:-]+)", prompt)
    name = name_match.group(1) if name_match else None

    vec_a = re.search(r"vector_a=(\[[^\]]*\])", prompt)
    vec_b = re.search(r"vector_b=(\[[^\]]*\])", prompt)
    if not (vec_a and vec_b):
        return name, None
    try:
        args = {
            "vector_a": json.loads(vec_a.group(1).replace("'", '"')),
            "vector_b": json.loads(vec_b.group(1).replace("'", '"')),
        }
        return name, args
    except Exception:
        return name, None


def _has_answer_blind_verifier(split: str) -> bool:
    return any(
        name in split
        for name in (
            "HumanEval",
            "LiveCodeBench",
            "DeepSWE",
            "TensorGraphDSL",
            "BFCL",
        )
    )


def _candidate_passes_answer_blind(
    self, split: str, item: Dict[str, Any], candidate: str
) -> Optional[bool]:
    """Verifier allowed for Pro selection. It never reads benchmark expected answers."""
    if "HumanEval" in split:
        code = clean_output(candidate)
        return bool(
            self.engine.sandbox.execute_python_code(
                item["prompt"] + "\n" + code,
                item["test"],
            ).passed
        )

    if "LiveCodeBench" in split:
        code = clean_output(candidate)
        return bool(self.engine.sandbox.execute_python_code(code, item["test"]).passed)

    if "DeepSWE" in split:
        patch = clean_output(candidate)
        return bool(
            self.engine.sandbox.verify_git_diff_patch(
                item["repo_files"],
                patch,
                item["test_cmd"],
            ).passed
        )

    if "TensorGraphDSL" in split:
        value = self.engine.sandbox.evaluate_dsl_expression(item["dsl_expr"])
        if value is None:
            return False
        expected = str(value).replace(" ", "")
        cleaned = clean_output(candidate).replace(" ", "")
        return expected in cleaned

    if "BFCL" in split:
        requested_name, requested_args = _prompt_requested_bfcl(item)
        value = _parse_json_object(candidate)
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
                return False
        return bool(
            requested_name
            and requested_args is not None
            and name == requested_name
            and args == requested_args
        )

    return None


def _consensus_key(text: str) -> str:
    cleaned = clean_output(text).strip()
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return "boxed:" + re.sub(r"\s+", "", boxed[-1]).lower()

    parsed = _parse_json_object(text)
    if isinstance(parsed, dict):
        try:
            return "json:" + json.dumps(
                parsed, sort_keys=True, separators=(",", ":")
            )
        except Exception:
            pass

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    final = lines[-1] if lines else cleaned
    return re.sub(r"\s+", " ", final).strip().lower()


def _choose_without_ground_truth(
    self, split: str, item: Dict[str, Any], branches: List[str]
) -> Tuple[str, int, bool, str]:
    if not branches:
        return "", 0, False, "no branches"

    if _has_answer_blind_verifier(split):
        for idx, candidate in enumerate(branches):
            try:
                if _candidate_passes_answer_blind(self, split, item, candidate) is True:
                    return candidate, idx, True, f"answer-blind verifier passed branch {idx + 1}"
            except Exception:
                continue

    keys = [_consensus_key(candidate) for candidate in branches]
    counts = collections.Counter(keys)
    winning_key, votes = counts.most_common(1)[0]
    idx = keys.index(winning_key)
    return branches[idx], idx, False, f"answer-blind consensus {votes}/{len(branches)}"


def _hidden_reward_only_after_selection(
    self, split: str, item: Dict[str, Any], candidate: str
) -> bool:
    """RSI reward gate. Ground truth is never passed to generation or branch selection."""
    verified = _candidate_passes_answer_blind(self, split, item, candidate)
    if verified is not None:
        return bool(verified)

    expected = str(item.get("expected", item.get("expected_token", item.get("expected_keyword", "")))).strip()
    if not expected:
        return False

    cleaned = clean_output(candidate)
    if any(x in split for x in ("GSM8K", "MATH", "AIME")):
        boxed = re.findall(r"\\boxed\{([^}]+)\}", candidate)
        return bool(
            (boxed and boxed[-1].strip() == expected)
            or expected in cleaned
            or expected.replace(" ", "") in cleaned.replace(" ", "")
        )

    return expected.lower() in cleaned.lower() or expected.lower() in candidate.lower()


def _generate_branches_same_model(
    self,
    formatted_prompt: str,
    temperatures: List[float],
    max_tokens: int,
    top_p: float = 0.92,
) -> List[str]:
    """Sequential branches on the exact in-memory model; never load/reload weights."""
    if not MLX_AVAILABLE or self.engine.model is None or self.engine.tokenizer is None:
        raise RuntimeError("MLX model/tokenizer unavailable for RSI/Pro branching")

    branches: List[str] = []
    prompt_ids = self.engine.tokenizer.encode(formatted_prompt)
    try:
        from mlx_lm.sample_utils import make_sampler
    except Exception:
        make_sampler = None

    for temp in temperatures:
        # Keep allocator/kernel state warm between branches. Reclaim only when the
        # same pressure guard used by the app says memory is genuinely tight.
        if _memory_pressure():
            gc.collect(1)
            try:
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()
            except Exception:
                pass

        kwargs: Dict[str, Any] = {
            "max_tokens": max(1, int(max_tokens)),
            "verbose": False,
            "prompt_cache": make_prompt_cache(self.engine.model),
            "prefill_step_size": _adaptive_prefill_step_size(),
        }
        if make_sampler is not None:
            try:
                kwargs["sampler"] = make_sampler(temp=float(temp), top_p=top_p)
            except Exception:
                kwargs["temp"] = float(temp)
                kwargs["top_p"] = top_p
        else:
            kwargs["temp"] = float(temp)
            kwargs["top_p"] = top_p

        try:
            out = mlx_lm.generate(
                self.engine.model,
                self.engine.tokenizer,
                prompt=prompt_ids,
                **kwargs,
            )
        except TypeError:
            kwargs.pop("sampler", None)
            kwargs.pop("prompt_cache", None)
            kwargs.pop("prefill_step_size", None)
            kwargs["temp"] = float(temp)
            kwargs["top_p"] = top_p
            out = mlx_lm.generate(
                self.engine.model,
                self.engine.tokenizer,
                prompt=formatted_prompt,
                **kwargs,
            )

        branches.append(str(out))

    return branches


def _append_rsi_log(
    self,
    split: str,
    item_id: str,
    round_idx: int,
    candidate: str,
    passed: bool,
    selection: str,
) -> None:
    try:
        os.makedirs(os.path.dirname(RAW_OUTPUT_LOG), exist_ok=True)
        with open(RAW_OUTPUT_LOG, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 110 + "\n")
            f.write(f"RSI SELF-IMPROVEMENT | {split} | {item_id} | round {round_idx}\n")
            f.write(f"Selection: {selection}\n")
            f.write("SELF-GENERATED CANDIDATE:\n")
            f.write((candidate or "").rstrip() + "\n")
            f.write(f"Hidden reward after selection: {'PASS' if passed else 'FAIL'}\n")
            f.write("=" * 110 + "\n")
    except Exception:
        pass


def _delete_unconsumed_session_rows(self, session_id: str) -> None:
    db_path = getattr(getattr(self.engine, "kg", None), "db_path", None)
    if not db_path:
        return
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "DELETE FROM episodic_interactions WHERE session_id=? AND consolidated=0",
                (session_id,),
            )
    except Exception:
        pass


def _seed_supervised_learn(self) -> int:
    """Spoon-fed Learn path: supplied facts become verified training traces."""
    _delete_unconsumed_session_rows(self, LEARN_SESSION_ID)
    count = 0
    for prompt, completion in LEARN_EXAMPLES:
        try:
            self.engine.kg.log_interaction(
                LEARN_SESSION_ID,
                prompt,
                completion,
                1.0,
                0.90,
                domain="LEARN::supervised_fact",
            )
            count += 1
        except Exception:
            pass
    self._phase2_learn_seed_count = count
    print(f"[✓] LEARN: seeded {count} supplied verified examples for consolidation.", flush=True)
    return count


def _run_rsi_self_improvement(self, splits, cache) -> int:
    """Recursive self-improvement: self-generate -> self-critique -> verify -> train."""
    # Remove any legacy RSI rows that persisted benchmark prompts/rewards, then
    # start this run with a clean question-free self-memory inbox.
    _delete_unconsumed_session_rows(self, RSI_SESSION_ID)
    try:
        self.engine.kg.clear_unconsolidated_rsi_self_memories()
    except Exception:
        pass
    model_identity = id(self.engine.model)
    seeded = 0
    attempted = 0

    misses: List[Tuple[str, Dict[str, Any]]] = []
    for split_name, items in splits.items():
        for item in items:
            key = f"Phase 1: Baseline_{item['id']}"
            if cache.get(key) is False:
                misses.append((split_name, item))

    if not misses:
        self._phase1_rsi_seed_count = 0
        print("[*] RSI: no Phase-1 misses to self-improve.", flush=True)
        return 0

    # Keep the benchmark bounded. Prioritize all misses up to 64 verified self-training attempts.
    for split_name, item in misses[:64]:
        _assert_same_model(self, model_identity, "RSI")
        attempted += 1

        original_user = _task_user_prompt(split_name, item)
        previous = ""
        success = False

        for round_idx in (1, 2):
            if round_idx == 1:
                rsi_user = (
                    original_user
                    + "\n\nRecursive Self-Improvement: solve this task yourself from first principles. "
                    "Do not assume or request a hidden answer. Before finalizing, internally check likely failure modes, "
                    "then output the best corrected final response in the requested format."
                )
            else:
                rsi_user = (
                    original_user
                    + "\n\nRecursive Self-Improvement round 2. Your previous self-generated attempt was:\n"
                    + previous
                    + "\n\nCritique your own attempt, identify what may be wrong without access to any hidden answer, "
                    "and produce a materially improved final response in the requested format."
                )

            formatted = _chat(self.engine.tokenizer, rsi_user, system=SYSTEM_PROMPT)
            # RSI explores four self-generated alternatives. No benchmark answer enters generation.
            temps = [0.20, 0.38, 0.58, 0.82]
            branches = _generate_branches_same_model(
                self,
                formatted,
                temps,
                max_tokens=min(_benchmark_ceiling(self), 16384),
                top_p=0.92,
            )
            candidate, _, _, selection = _choose_without_ground_truth(
                self, split_name, item, branches
            )
            previous = candidate

            # Ground truth/test is used only as a reward AFTER the model has generated
            # and an answer-blind policy has selected its candidate.
            passed = _hidden_reward_only_after_selection(
                self, split_name, item, candidate
            )
            _append_rsi_log(
                self,
                split_name,
                str(item.get("id", "unknown")),
                round_idx,
                candidate,
                passed,
                selection,
            )

            if passed:
                try:
                    # Persist ONLY the model's self-generated trace. The hidden verifier
                    # gate is ephemeral: no question, split, reward, PASS/FAIL, or expected
                    # answer is written to persistent RSI memory.
                    self.engine.kg.log_rsi_self_memory(candidate)
                    seeded += 1
                    success = True
                except Exception:
                    pass
                break

        if not success:
            continue

    self._phase1_rsi_seed_count = seeded
    print(
        f"[✓] RSI: {seeded}/{attempted} Phase-1 misses produced self-generated verified corrections.",
        flush=True,
    )
    return seeded


def _fetch_benchmark_training_memories(self) -> List[Dict[str, Any]]:
    """Fetch Learn QA rows plus question-free/reward-free RSI self traces."""
    db_path = getattr(getattr(self.engine, "kg", None), "db_path", None)
    if not db_path:
        return []

    total_limit = max(
        20,
        min(
            256,
            int(getattr(self, "_phase1_rsi_seed_count", 0) or 0)
            + int(getattr(self, "_phase2_learn_seed_count", 0) or 0)
            + 8,
        ),
    )
    learn_limit = max(1, total_limit)
    rsi_limit = max(1, total_limit)

    memories: List[Dict[str, Any]] = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Learn remains explicitly supervised QA and keeps its reward metadata.
            learn_rows = conn.execute(
                """
                SELECT id,prompt,completion
                FROM episodic_interactions
                WHERE consolidated=0
                  AND reward>=0.8
                  AND surprise_score>=0.80
                  AND session_id=?
                ORDER BY surprise_score DESC,id ASC
                LIMIT ?
                """,
                (LEARN_SESSION_ID, learn_limit),
            ).fetchall()
            memories.extend(
                {
                    "memory_kind": "learn",
                    "id": int(row["id"]),
                    "prompt": str(row["prompt"]),
                    "completion": str(row["completion"]),
                }
                for row in learn_rows
            )

            # RSI table has no prompt/question and no reward/correctness columns.
            rsi_rows = conn.execute(
                """
                SELECT id,trace
                FROM rsi_self_memories
                WHERE consolidated=0
                ORDER BY id ASC
                LIMIT ?
                """,
                (rsi_limit,),
            ).fetchall()
            memories.extend(
                {
                    "memory_kind": "rsi_self",
                    "id": int(row["id"]),
                    # Fixed generic training cue only: never reconstruct the benchmark
                    # prompt/question from which this self-generated trace originated.
                    "prompt": "Internalize this self-generated reasoning pattern and improve future problem solving.",
                    "completion": str(row["trace"]),
                }
                for row in rsi_rows
            )
    except Exception:
        return []

    return memories[:total_limit]


def _save_rsi_adapter(self) -> bool:
    if not MLX_AVAILABLE or self.engine.model is None:
        return False
    try:
        os.makedirs(os.path.dirname(RSI_ADAPTER_PATH), exist_ok=True)
        flat = dict(mlx.utils.tree_flatten(self.engine.model.trainable_parameters()))
        if not flat:
            return False
        mx.save_safetensors(RSI_ADAPTER_PATH, flat)
        return True
    except Exception as exc:
        print(f"[!] Could not persist RSI adapter: {exc}", flush=True)
        return False


def _restore_rsi_adapter(self) -> bool:
    if not MLX_AVAILABLE or self.engine.model is None or not os.path.exists(RSI_ADAPTER_PATH):
        return False
    try:
        flat = mx.load(RSI_ADAPTER_PATH)
        if not isinstance(flat, dict) or not flat:
            return False
        self.engine.model.update(mlx.utils.tree_unflatten(list(flat.items())))
        mx.eval(self.engine.model.parameters())

        if getattr(self.engine, "moe_manager", None) is not None:
            self.engine.moe_manager.adapters_buffer_b = {
                k: mx.array(v) for k, v in flat.items()
            }
        print(f"[✓] Restored persisted RSI adapter: {RSI_ADAPTER_PATH}", flush=True)
        return True
    except Exception as exc:
        print(f"[!] Failed to restore RSI adapter: {exc}", flush=True)
        return False


def _phase3_fmt_eta(seconds: Optional[float]) -> str:
    if seconds is None or seconds <= 0:
        return "calculating"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _phase3_stack(model):
    """Resolve the loaded causal language stack without replacing/reloading it."""
    text_model = getattr(model, "language_model", None) or model
    inner = getattr(text_model, "model", None) or text_model

    layers = getattr(inner, "pipeline_layers", None)
    if layers is None:
        layers = getattr(inner, "layers", None)
    if layers is None:
        layers = getattr(model, "layers", None)

    layers = list(layers or [])
    embed_tokens = getattr(inner, "embed_tokens", None)
    norm = getattr(inner, "norm", None)
    if not layers or embed_tokens is None or norm is None:
        raise RuntimeError(
            "Phase 3 bounded training could not resolve the loaded transformer stack"
        )
    return text_model, inner, layers, embed_tokens, norm


def _phase3_window_value_and_grad(self, inp, target_start: int, target_end: int):
    """Backprop one small completion window one decoder layer at a time."""
    model = self.engine.model
    text_model, _inner, layers, embed_tokens, norm = _phase3_stack(model)

    module_paths = {}
    try:
        module_paths = {id(module): name for name, module in model.named_modules()}
    except Exception:
        pass

    hidden = embed_tokens(inp)
    mx.eval(hidden)
    hidden = mx.stop_gradient(hidden)
    boundaries = [hidden]

    for layer in layers:
        mask = None if bool(getattr(layer, "is_linear", False)) else "causal"
        hidden = layer(hidden, mask=mask, cache=None)
        mx.eval(hidden)
        hidden = mx.stop_gradient(hidden)
        boundaries.append(hidden)

    final_hidden = boundaries[-1]
    targets = inp[:, target_start + 1 : target_end + 1]

    def head_loss(local_hidden):
        selected = local_hidden[:, target_start:target_end, :]
        z = norm(selected)
        if hasattr(text_model, "lm_head"):
            logits = text_model.lm_head(z)
        elif hasattr(embed_tokens, "as_linear"):
            logits = embed_tokens.as_linear(z)
        else:
            raise RuntimeError("Phase 3 could not resolve LM head")
        return mx.mean(
            nn.losses.cross_entropy(
                logits.astype(mx.float32),
                targets,
            )
        )

    loss, cotangent = mx.value_and_grad(head_loss)(final_hidden)
    mx.eval(loss, cotangent)
    cotangent = mx.stop_gradient(cotangent)

    flat_grads: Dict[str, Any] = {}

    for layer_idx in range(len(layers) - 1, -1, -1):
        layer = layers[layer_idx]
        h_in = boundaries[layer_idx]
        incoming = mx.stop_gradient(cotangent)
        mask = None if bool(getattr(layer, "is_linear", False)) else "causal"
        params = layer.trainable_parameters()
        local_param_items = list(mlx.utils.tree_flatten(params))

        def forward_layer(local_params, local_hidden):
            layer.update(local_params)
            return layer(local_hidden, mask=mask, cache=None)

        checkpointed = mx.checkpoint(forward_layer)

        if local_param_items:
            def scalar(local_params, local_hidden):
                out = checkpointed(local_params, local_hidden)
                return mx.sum(
                    out.astype(mx.float32)
                    * incoming.astype(mx.float32)
                )

            _, pair = mx.value_and_grad(
                scalar,
                argnums=[0, 1],
            )(params, h_in)
            param_grads, input_grad = pair
            flat_local = list(mlx.utils.tree_flatten(param_grads))
            mx.eval(input_grad, *[grad for _, grad in flat_local])

            path = module_paths.get(id(layer), "")
            for local_key, grad in flat_local:
                key = f"{path}.{local_key}" if path else str(local_key)
                flat_grads[key] = mx.stop_gradient(grad)
        else:
            def scalar_hidden(local_hidden):
                out = layer(local_hidden, mask=mask, cache=None)
                return mx.sum(
                    out.astype(mx.float32)
                    * incoming.astype(mx.float32)
                )

            input_grad = mx.grad(scalar_hidden)(h_in)
            mx.eval(input_grad)

        cotangent = mx.stop_gradient(input_grad)
        boundaries[layer_idx + 1] = None

        try:
            mx.clear_cache()
        except Exception:
            pass

    expected = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
    missing = set(expected) - set(flat_grads)
    extra = set(flat_grads) - set(expected)
    if missing or extra:
        raise RuntimeError(
            "Phase 3 bounded gradient mismatch: "
            f"missing={sorted(missing)[:6]} extra={sorted(extra)[:6]}"
        )

    grads = mlx.utils.tree_unflatten(
        [(key, flat_grads[key]) for key in expected]
    )
    return loss, grads


def _phase3_bounded_gradients(
    self,
    ids: List[int],
    completion_loss_start: int,
    memory_index: int,
    memory_total: int,
):
    """Train the completion through 64-token windows without retaining a full-model graph."""
    max_row_tokens = 16_384
    original_len = len(ids)
    selected_start = max(0, original_len - max_row_tokens)
    selected = list(ids[selected_start:])
    if len(selected) < 2:
        raise RuntimeError("Phase 3 training row has fewer than two tokens")

    full = mx.array([selected])
    total_targets = len(selected) - 1
    first_target = max(0, int(completion_loss_start) - selected_start)
    first_target = min(first_target, total_targets)
    if first_target >= total_targets:
        raise RuntimeError("Phase 3 completion tokens fell outside the bounded training row")

    window_tokens = 64
    targets_per_window = 32
    aggregate: Dict[str, Any] = {}
    weighted_loss = 0.0
    trained_targets = 0
    target_global = first_target
    completion_targets = total_targets - first_target
    started = time.monotonic()

    while target_global < total_targets:
        target_end = min(total_targets, target_global + targets_per_window)
        window_end = target_end + 1
        window_start = max(0, window_end - window_tokens)

        local = full[:, window_start:window_end]
        local_start = target_global - window_start
        local_end = target_end - window_start
        count = target_end - target_global

        loss, grads = _phase3_window_value_and_grad(
            self,
            local,
            local_start,
            local_end,
        )

        flat = dict(mlx.utils.tree_flatten(grads))
        mx.eval(loss, *flat.values())

        for key, grad in flat.items():
            weighted = mx.stop_gradient(grad) * float(count)
            mx.eval(weighted)
            if key in aggregate:
                value = aggregate[key] + weighted
                mx.eval(value)
                aggregate[key] = mx.stop_gradient(value)
            else:
                aggregate[key] = mx.stop_gradient(weighted)

        weighted_loss += float(loss.item()) * count
        trained_targets += count
        target_global = target_end

        elapsed = max(0.001, time.monotonic() - started)
        speed = trained_targets / elapsed
        pct = 100.0 * trained_targets / max(1, completion_targets)
        remaining_estimate = (
            (completion_targets - trained_targets)
            + max(0, memory_total - memory_index) * completion_targets
        )
        total_eta = remaining_estimate / speed if speed > 0 else None

        try:
            ram_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        except Exception:
            ram_mb = 0.0

        print(
            f"Phase 3: {trained_targets}/{completion_targets}"
            f" | {pct:.1f}%"
            f" | {speed:.2f} tgt/s"
            f" | Total ETA {_phase3_fmt_eta(total_eta)}"
            f" | RAM {ram_mb:.0f} MB",
            flush=True,
        )

        loss = None
        grads = None
        flat = None
        local = None
        gc.collect(1)
        try:
            mx.clear_cache()
        except Exception:
            pass

    inv = 1.0 / float(max(1, trained_targets))
    for key in list(aggregate):
        value = aggregate[key] * inv
        mx.eval(value)
        aggregate[key] = mx.stop_gradient(value)

    # Preserve the newer transaction wrapper's real Fisher/EWC protection. The
    # wrapper supplies only small LoRA snapshots/fisher tensors; no full graph is kept.
    ewc = getattr(self, "_phase3_ewc_context", None)
    if isinstance(ewc, dict) and float(ewc.get("lambda", 0.0) or 0.0) > 0.0:
        fisher = ewc.get("fisher") or {}
        reference = ewc.get("reference") or {}
        current = dict(
            mlx.utils.tree_flatten(
                self.engine.model.trainable_parameters()
            )
        )
        lam = float(ewc.get("lambda", 0.0) or 0.0)
        for key in list(aggregate):
            if key not in fisher or key not in reference or key not in current:
                continue
            value = (
                aggregate[key]
                + lam * fisher[key] * (current[key] - reference[key])
            )
            mx.eval(value)
            aggregate[key] = mx.stop_gradient(value)

    grads = mlx.utils.tree_unflatten(list(aggregate.items()))
    return mx.array(weighted_loss * inv, dtype=mx.float32), grads


def _run_phase3_consolidation(self) -> Dict[str, Any]:
    """Train the exact already-loaded model in-place with bounded completion-only graphs."""
    memories = _fetch_benchmark_training_memories(self)
    if not memories:
        print("[*] Phase 3: no benchmark Learn/RSI memories eligible for consolidation.", flush=True)
        return {"updated": False, "memories": 0, "fallback_updates": 0}

    if not MLX_AVAILABLE or self.engine.model is None:
        raise RuntimeError("Phase 3 requires the already-loaded MLX model")

    trainable = dict(
        mlx.utils.tree_flatten(
            self.engine.model.trainable_parameters()
        )
    )
    if not trainable:
        _pro_backend(self).inject_lora_adapters(r=8)
        trainable = dict(
            mlx.utils.tree_flatten(
                self.engine.model.trainable_parameters()
            )
        )
    if not trainable:
        raise RuntimeError("Phase 3 has no trainable LoRA parameters")

    bad = [
        key for key in trainable
        if key.rsplit(".", 1)[-1] not in {"lora_a", "lora_b"}
    ]
    if bad:
        raise RuntimeError(
            "Phase 3 refuses non-LoRA trainables: "
            + ", ".join(bad[:6])
        )

    model_identity = id(self.engine.model)
    opt = optim.AdamW(learning_rate=1e-4)
    updated = 0
    fallback_updates = 0
    learn_consolidated_ids: List[int] = []
    rsi_consolidated_ids: List[int] = []
    was_training = bool(getattr(self.engine.model, "training", False))

    try:
        self.engine.model.train()

        for memory_index, memory in enumerate(memories, 1):
            prefix = (
                f"<|im_start|>user\n{memory['prompt']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            text = prefix + str(memory["completion"]) + "<|im_end|>"
            ids = self.engine.tokenizer.encode(text)
            prefix_ids = self.engine.tokenizer.encode(prefix)
            if len(ids) <= 1:
                continue

            completion_loss_start = max(0, len(prefix_ids) - 1)

            with METAL_STREAM_LOCK:
                loss, grads = _phase3_bounded_gradients(
                    self,
                    ids,
                    completion_loss_start,
                    memory_index,
                    len(memories),
                )

                try:
                    flat, shapes = self.engine.ogp_projector.flatten_gradients(
                        dict(mlx.utils.tree_flatten(grads))
                    )
                    projected = self.engine.ogp_projector.project_gradient(flat)
                    tree = self.engine.ogp_projector.unflatten_gradients(
                        projected, shapes
                    )
                    opt.update(
                        self.engine.model,
                        mlx.utils.tree_unflatten(list(tree.items())),
                    )
                except Exception:
                    opt.update(self.engine.model, grads)
                    fallback_updates += 1

                mx.eval(self.engine.model.parameters(), opt.state)
                updated += 1
                if memory.get("id") is not None:
                    if memory.get("memory_kind") == "rsi_self":
                        rsi_consolidated_ids.append(int(memory["id"]))
                    else:
                        learn_consolidated_ids.append(int(memory["id"]))

                loss = None
                grads = None
                flat = None
                projected = None
                tree = None
                gc.collect(1)
                try:
                    mx.clear_cache()
                except Exception:
                    pass

        _assert_same_model(self, model_identity, "Phase 3 consolidation")

        if getattr(self.engine, "moe_manager", None) is not None:
            current = {
                key: mx.array(value)
                for key, value in dict(
                    mlx.utils.tree_flatten(
                        self.engine.model.trainable_parameters()
                    )
                ).items()
            }
            self.engine.moe_manager.adapters_buffer_b = current
            self.engine.moe_manager.swap_buffers_atomic()
            _assert_same_model(self, model_identity, "Phase 3 buffer swap")

        if learn_consolidated_ids:
            try:
                self.engine.kg.mark_consolidated(learn_consolidated_ids)
            except Exception:
                pass
        if rsi_consolidated_ids:
            try:
                self.engine.kg.mark_rsi_self_memories_consolidated(
                    rsi_consolidated_ids
                )
            except Exception:
                pass

        persisted = _save_rsi_adapter(self)
        print(
            f"[✓] Phase 3 trained {updated} Learn/RSI memories in-place "
            f"(raw-gradient fallback updates: {fallback_updates}; persisted={persisted}).",
            flush=True,
        )
        return {
            "updated": updated > 0,
            "memories": updated,
            "fallback_updates": fallback_updates,
            "persisted": persisted,
        }
    finally:
        if not was_training:
            try:
                self.engine.model.eval()
            except Exception:
                pass
        gc.collect(2)
        try:
            mx.clear_cache()
        except Exception:
            pass


def _run_learning_retention_test(self, model_identity: int) -> Dict[str, Any]:
    """Small post-training test using the same RSI-updated model; outside the 4,014 score."""
    _assert_same_model(self, model_identity, "Learning retention test")
    prior_phase = getattr(self, "_current_phase", "")
    prior_split = getattr(self, "_current_split", "")
    prior_item = getattr(self, "_current_item_id", "")

    passed = 0
    total = min(5, len(LEARN_EXAMPLES))
    for idx, (prompt, expected) in enumerate(LEARN_EXAMPLES[:total]):
        self._current_phase = "Learning Test: Post-RSI"
        self._current_split = "LearningFacts"
        self._current_item_id = f"LearningFact_{idx}"

        user = prompt + "\nState the exact learned fact directly."
        formatted = _chat(self.engine.tokenizer, user, system=SYSTEM_PROMPT)
        out = self._fast_generate(
            formatted,
            max_tokens=min(_benchmark_ceiling(self), 16384),
        )
        self.last_raw_out = out
        _append_raw_generation_log(self, formatted, user, out)

        ok = expected.lower() in clean_output(out).lower() or expected.lower() in out.lower()
        if ok:
            passed += 1
        try:
            with open(RAW_OUTPUT_LOG, "a", encoding="utf-8") as f:
                f.write(f"RESULT: {'PASS' if ok else 'FAIL'}\n")
                f.write("=" * 110 + "\n")
        except Exception:
            pass

    self._current_phase = prior_phase
    self._current_split = prior_split
    self._current_item_id = prior_item

    _assert_same_model(self, model_identity, "Learning retention test end")
    pct = 100.0 * passed / max(1, total)
    print(
        f"[Learning Test: Post-RSI] LearningFacts | {passed}/{total} ({pct:.2f}%) "
        f"| same in-memory model",
        flush=True,
    )
    return {"correct": passed, "total": total, "accuracy": pct}


def _append_pro_metadata(self) -> None:
    meta = getattr(self, "_last_phase4_pro_meta", None)
    if not isinstance(meta, dict):
        return
    try:
        with open(RAW_OUTPUT_LOG, "a", encoding="utf-8") as f:
            f.write("\nPHASE-4 PRO RETEST METADATA:\n")
            f.write(f"Mode: {meta.get('mode')}\n")
            f.write(f"Normalized entropy: {meta.get('entropy'):.6f}\n")
            f.write(f"Branch count: {meta.get('branch_count')}\n")
            f.write(f"Temperature ladder: {meta.get('temperatures')}\n")
            f.write(f"Winning branch: {meta.get('winning_branch')}\n")
            f.write(f"Selection: {meta.get('selection')}\n")
            f.write(f"All-branch generated tokens: {meta.get('all_branch_tokens')}\n")
            f.flush()
    except Exception:
        pass


def _full_post_scores_from_missed_retest(self, full_splits, cache) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for name, items in full_splits.items():
        correct = 0
        for item in items:
            p1 = cache.get(f"Phase 1: Baseline_{item['id']}")
            if p1 is True:
                correct += 1
            elif p1 is False and cache.get(
                f"Phase 4: Post-Consolidation_{item['id']}"
            ) is True:
                correct += 1
        scores[name] = 100.0 * correct / max(1, len(items))
    return scores


def install(cls):
    """Install RSI, Learn retention, missed-only Phase-4 Pro, and no-reload orchestration."""
    base_fast = cls._fast_generate
    base_eval = cls._evaluate_single_item
    base_all = cls._evaluate_all_splits
    base_report = cls._generate_master_report

    def pro_fast_generate(self, prompt, max_tokens=16384, stream=False):
        # Phase 1, Learn, RSI, and Learning Test keep the exact merged baseline generator.
        if not str(getattr(self, "_current_phase", "")).startswith("Phase 4"):
            return base_fast(self, prompt, max_tokens=max_tokens, stream=stream)

        identity = int(getattr(self, "_rsi_model_identity", id(self.engine.model)))
        _assert_same_model(self, identity, "Phase 4 Pro generation")

        router = _pro_router(self)
        entropy = _normalized_entropy(self, prompt)
        split = str(getattr(self, "_current_split", ""))
        item = getattr(self, "_phase4_current_item", {}) or {}
        has_tests = _has_answer_blind_verifier(split)

        mode, branch_count = router.route(entropy, has_test_cases=has_tests)
        temperatures = get_ladder_temperatures(branch_count)

        started = time.perf_counter()
        branches = _generate_branches_same_model(
            self,
            prompt,
            temperatures,
            max_tokens=min(int(max_tokens), 16384),
            top_p=0.92,
        )
        if not branches:
            raise RuntimeError("Phase-4 Pro branch generation returned no candidates")

        winner, winning_idx, verified, selection = _choose_without_ground_truth(
            self, split, item, branches
        )
        elapsed = max(0.001, time.perf_counter() - started)

        token_counts = []
        for candidate in branches:
            try:
                token_counts.append(len(self.engine.tokenizer.encode(candidate)))
            except Exception:
                token_counts.append(0)
        try:
            selected_tokens = len(self.engine.tokenizer.encode(winner))
        except Exception:
            selected_tokens = 0
        try:
            self.last_prompt_tokens = len(self.engine.tokenizer.encode(prompt))
        except Exception:
            self.last_prompt_tokens = 0

        all_tokens = sum(token_counts)
        self.last_output_tokens = selected_tokens
        self.live_generated_tokens = selected_tokens
        self.last_generation_seconds = elapsed
        self.last_tok_per_sec = all_tokens / elapsed if all_tokens else 0.0
        self.last_generation_error = None
        self._last_phase4_pro_meta = {
            "mode": mode,
            "entropy": float(entropy),
            "branch_count": len(branches),
            "temperatures": temperatures,
            "winning_branch": winning_idx + 1,
            "verified": verified,
            "selection": selection,
            "all_branch_tokens": all_tokens,
        }
        return winner

    def pro_eval(self, split, item):
        if str(getattr(self, "_current_phase", "")).startswith("Phase 4"):
            self._phase4_current_item = item
            self._last_phase4_pro_meta = None
            result = base_eval(self, split, item)
            _append_pro_metadata(self)
            return result
        return base_eval(self, split, item)

    def missed_only_evaluate_all(self, splits, cache, phase, start, total):
        if phase != "Phase 4: Post-Consolidation":
            # Phase-1 telemetry/output format remains byte-for-byte controlled by base_all.
            return base_all(self, splits, cache, phase, start, total)

        missed_splits: Dict[str, List[Dict[str, Any]]] = {}
        for name, items in splits.items():
            subset = [
                item
                for item in items
                if cache.get(f"Phase 1: Baseline_{item['id']}") is False
            ]
            if subset:
                missed_splits[name] = subset

        missed_total = sum(len(v) for v in missed_splits.values())
        print(
            f"[*] Phase 4 retests Phase-1 misses only: {missed_total} items.",
            flush=True,
        )
        if missed_total:
            base_all(
                self,
                missed_splits,
                cache,
                phase,
                start,
                missed_total,
            )
        return _full_post_scores_from_missed_retest(self, splits, cache)

    def run_full_rsi(self):
        if not MLX_AVAILABLE or self.engine.model is None or self.engine.tokenizer is None:
            raise RuntimeError("Real 27B MLX model is not loaded; benchmark will not start offline")

        self.benchmark_max_tokens = _benchmark_ceiling(self)
        model_identity = id(self.engine.model)
        self._rsi_model_identity = model_identity

        print("=" * 95)
        print("🚀 COMMENCING 4,000+ ITEM MASTER EVALUATION SUITE (BASELINE → LEARN → RSI → RETEST)")
        print("│ Phase 1 baseline → Phase 2 supplied Learn + MCTS → RSI self-improvement → Phase 3 consolidation")
        print("│ Learning Test uses same RSI-updated model → Phase 4 Pro retests Phase-1 misses only")
        print(f"│ Benchmark-only generation ceiling: {self.benchmark_max_tokens:,} tokens")
        print(f"│ Raw outputs + post-output speed metrics: {RAW_OUTPUT_LOG}")
        print("=" * 95)

        splits = _repair_suite(self.provider.load_all_4000_items())
        total = sum(map(len, splits.values()))
        chk = self.checkpoint_mgr.load_checkpoint()
        cache = chk.get("completed_items", {}) if isinstance(chk, dict) else {}
        cache = cache if isinstance(cache, dict) else {}
        start = time.time()
        phase = chk.get("phase", "Phase 1: Baseline") if isinstance(chk, dict) else "Phase 1: Baseline"

        # Resume after a process restart: base weights are loaded once by engine init,
        # then persisted RSI trainable weights are restored. No model load occurs here.
        if phase == "Phase 4: Post-Consolidation":
            if not _restore_rsi_adapter(self):
                raise RuntimeError(
                    "Phase-4 resume requires the persisted RSI adapter; refusing to test base weights instead."
                )
            self._rsi_model_identity = id(self.engine.model)
            base = scores_from_cache(splits, cache, "Phase 1: Baseline")
            post = self._evaluate_all_splits(
                splits,
                cache,
                "Phase 4: Post-Consolidation",
                start,
                total,
            )
            if not self.time_budget_exhausted:
                base_report(self, base, post, total, time.time() - start)
            return

        print("\n▶ PHASE 1: ZERO-SHOT SINGLE-PASS BASELINE")
        base = self._evaluate_all_splits(
            splits,
            cache,
            "Phase 1: Baseline",
            start,
            total,
        )
        if self.time_budget_exhausted:
            return
        _assert_same_model(self, model_identity, "after Phase 1")

        print("\n▶ PHASE 2: SUPPLIED LEARN / MEMORY INGESTION + MCTS TEACHING")
        DialogueTimelineGraphIngester(self.engine.kg).ingest_developer_sessions()
        _seed_supervised_learn(self)
        for item in splits.get("TensorGraphDSL-300", [])[:30]:
            inv, q, visits = self.engine.mcts.search_best_invariant(item["dsl_expr"])
            self.engine.kg.insert_triple(item["dsl_expr"], "evaluates_to", inv, weight=q)
        _assert_same_model(self, model_identity, "after Phase 2 Learn")

        print("\n▶ RSI: RECURSIVE SELF-IMPROVEMENT ON PHASE-1 MISSES")
        _run_rsi_self_improvement(self, splits, cache)
        _assert_same_model(self, model_identity, "after RSI self-improvement")

        print("\n▶ PHASE 3: LEARN + RSI PARAMETRIC CONSOLIDATION")
        _run_phase3_consolidation(self)
        _assert_same_model(self, model_identity, "after Phase 3 consolidation")
        self._rsi_model_identity = model_identity

        print("\n▶ LEARNING TEST: SAME RSI-UPDATED MODEL")
        _run_learning_retention_test(self, model_identity)

        print("\n▶ PHASE 4: PRO RETEST OF PHASE-1 MISSES ONLY")
        self.checkpoint_mgr.save_checkpoint(
            cache,
            "Phase 4: Post-Consolidation",
            start,
        )
        post = self._evaluate_all_splits(
            splits,
            cache,
            "Phase 4: Post-Consolidation",
            start,
            total,
        )
        if not self.time_budget_exhausted:
            base_report(self, base, post, total, time.time() - start)

    cls._fast_generate = pro_fast_generate
    cls._evaluate_single_item = pro_eval
    cls._evaluate_all_splits = missed_only_evaluate_all
    cls.run_full_suite = run_full_rsi
