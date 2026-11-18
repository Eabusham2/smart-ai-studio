#!/usr/bin/env python3
"""Real Bonsai 2/CRACK memory evaluation, isolated from the user's learned state.

Each stage executes in a fresh process. Source-free questions use the same
sampling policy before/after learning, and every previously taught fact is
retested. The ordinary 4,014-item evaluator and its scoring policy are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "bonsai2": {
        "name": "Bonsai 2 27B Ternary Multimodal",
        "repo_id": "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
        "runtime_family": "bonsai2_hadamard",
        "input_modalities": ["text", "image", "video"],
        "ternary": True,
    },
    "bonsai2-crack": {
        "name": "Bonsai 2 27B Ternary CRACK",
        "repo_id": "dealignai/Bonsai-2-27B-CRACK-Ternary-JANG",
        "runtime_family": "bonsai2_hadamard",
        "input_modalities": ["text", "image", "video"],
        "ternary": True,
    },
}
CONTROLS = [
    {"id": "arithmetic", "question": "What is 7 + 8? Reply with only the number.", "answer": "15"},
    {"id": "spelling", "question": "Spell the word cat backwards. Reply with only the reversed word.", "answer": "tac"},
    {"id": "ordering", "question": "Which is smaller: 3 or 9? Reply with only the number.", "answer": "3"},
]


def make_facts(seed: int, cycles: int, count: int) -> list[dict]:
    rng = random.Random(seed)
    first = ["violet", "copper", "silver", "amber", "cobalt", "scarlet", "indigo", "ivory"]
    second = ["fern", "harbor", "meadow", "comet", "willow", "orchid", "lantern", "maple"]
    facts = []
    for cycle in range(cycles):
        for index in range(count):
            key = "MEM-" + f"{rng.getrandbits(48):012X}"
            value = rng.choice(first) + " " + rng.choice(second)
            statement = f"The registered phrase for {key} is {value}."
            facts.append({"id": key, "cycle": cycle, "statement": statement, "answer": value,
                          "question": f"What phrase was registered for {key}? Reply with only the phrase."})
    return facts


def normalize_answer(text: str) -> str:
    visible = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    return " ".join(re.sub(r"[^\w\s]", " ", visible.casefold()).split())


def score_answers(outputs: list[dict]) -> dict:
    total = len(outputs)
    correct = sum(normalize_answer(item["response"]) == normalize_answer(item["answer"]) for item in outputs)
    return {"correct": correct, "total": total, "accuracy": correct / total if total else 0.0}


def require_apple_mlx() -> None:
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        raise RuntimeError("Real Bonsai 2 memory evaluation requires Apple Silicon and the installed MLX/publisher runtimes")


def _load_engine(job, target):
    sys.path.insert(0, str(REPO_ROOT))
    from config.paths import get_bundled_model_path
    from config.settings import get_settings
    from core.pro_engine import ProReasoningEngine
    from huggingface_hub import snapshot_download

    info = dict(TARGETS[target])
    selected = job.get("paths", {}).get(target) or get_bundled_model_path(info["repo_id"])
    if not selected:
        selected = snapshot_download(info["repo_id"], local_files_only=not job["allow_download"])
    if not Path(selected).is_dir():
        raise RuntimeError("The selected target snapshot is not available: " + str(selected))
    info["model_path"] = str(Path(selected).resolve())
    state_directory = os.environ.get("SMARTAI_DATA_DIR")
    if not state_directory:
        raise RuntimeError("The evaluation worker requires an isolated state directory")
    settings = get_settings(backend="mlx", learning_implementation="managed", use_mock=False, live_mode=True, auto_download=False,
                            lora_adapter_path=None,
                            database_path=str(Path(state_directory) / "evaluation.sqlite3"))
    engine = ProReasoningEngine(settings=settings)
    engine._context_budget_tokens = job["context_tokens"]
    result = engine.load_model(info["name"], model_path=info["model_path"], backend="mlx", model_info=info)
    if result.get("status") != "loaded" or result.get("backend") != "mlx":
        raise RuntimeError("Target model failed to load through its production runtime: " + json.dumps(result))
    return engine


def _answer(engine, question, answer_tokens, seed):
    backend = engine.mlx_backend
    def generate():
        import mlx.core as mx
        # Both publisher VLM loaders apply their own chat template in the
        # stream path. Give it one raw question, with no retrieval/history.
        prepared = backend._prepare_vlm_prompt(question, 0)
        rendered_count = len(backend._encode_prompt(prepared))
        raw_count = len(backend._encode_prompt(question))
        available = int(engine._context_budget_tokens) - rendered_count
        if available <= 0:
            raise ValueError("The evaluation question exceeds the selected context")
        cap = min(answer_tokens, available)
        previous = getattr(backend, "context_budget_tokens", None)
        # Production computes the allowance from raw_count before VLM rendering.
        backend.context_budget_tokens = raw_count + cap
        try:
            mx.random.seed(seed & 0xFFFFFFFF)
            return "".join(backend.stream_generate_tokens(
                question, max_tokens=cap, temperature=0.65, top_p=0.92)).strip()
        finally:
            backend.context_budget_tokens = previous

    return backend.run_on_mlx_owner(generate)


def worker(job_path: Path, target: str, stage: str, cycle: int, output: Path):
    require_apple_mlx()
    job = json.loads(job_path.read_text())
    os.environ["SMARTAI_DATA_DIR"] = str(job_path.parent / target / "app_state")
    engine = _load_engine(job, target)
    start = time.monotonic()
    result = {"target": target, "stage": stage, "cycle": cycle, "pid": os.getpid(),
              "runtime": getattr(engine.mlx_backend, "_vlm_runtime", None), "model": TARGETS[target]["repo_id"]}
    try:
        backend = engine.mlx_backend
        expected_runtime = "jang_vlm" if target == "bonsai2-crack" else "mlx_repo_vlm"
        if backend._vlm_runtime != expected_runtime:
            raise RuntimeError("The selected snapshot did not load with its publisher runtime")
        initial = backend.learning_state.current_manifest()
        if stage == "baseline" and (initial or backend.adapter_path):
            raise RuntimeError("Baseline requires clean weights without a learned adapter")
        if stage == "train":
            turns = []
            for fact in job["facts"]:
                if fact["cycle"] == cycle:
                    turns.extend([{"role": "user", "content": "Remember this new fact: " + fact["statement"]},
                                  {"role": "assistant", "content": fact["statement"]}])
            service = engine.awake_consolidator
            if not service.consolidate_chunk_sync(turns):
                raise RuntimeError("Production consolidation rejected the update: " + service.last_error)
            result["learning"] = service.last_result
        # No source history, memory DB search, or expected answer is passed to
        # generation. This is a held-out wording of each taught association.
        eligible = job["facts"] if stage == "baseline" else [f for f in job["facts"] if f["cycle"] <= cycle]
        outputs = []
        for item in [*eligible, *CONTROLS]:
            seed = job["seed"] + int(hashlib.sha256(item["id"].encode()).hexdigest()[:8], 16)
            outputs.append({**item, "response": _answer(engine, item["question"], job["answer_tokens"], seed)})
        result["facts"] = outputs[:len(eligible)]
        result["controls"] = outputs[len(eligible):]
        result["fact_score"] = score_answers(result["facts"])
        result["control_score"] = score_answers(result["controls"])
        manifest = engine.mlx_backend.learning_state.current_manifest()
        result["generation"] = (manifest or {}).get("generation")
        result["parent_generation"] = (manifest or {}).get("parent")
        if stage != "baseline":
            if not manifest or backend._mlx_parent_generation != manifest["generation"]:
                raise RuntimeError("The live MLX model does not match its committed checkpoint")
            if stage == "reload" and (backend.last_restored_adapter_provenance != "model-bound"
                    or Path(backend.last_restored_adapter_path) != Path(backend.learning_state.current_checkpoint())):
                raise RuntimeError("Fresh process did not restore the committed adapter into its live model")
            if stage == "train" and (initial or {}).get("generation") != manifest["parent"]:
                raise RuntimeError("Training did not extend the loaded parameter generation")
        result["identity_digest"] = engine.mlx_backend.learning_state.identity_digest
        result["elapsed_seconds"] = time.monotonic() - start
        result["status"] = "measured"
        output.write_text(json.dumps(result, indent=2, default=str, allow_nan=False))
    finally:
        engine.unload_model()


def _run_stage(job_file, target, stage, cycle):
    output = job_file.parent / target / f"{stage}-{cycle}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--job", str(job_file),
               "--model", target, "--stage", stage, "--cycle", str(cycle), "--worker-output", str(output)]
    with open(output.with_suffix(".log"), "w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode or not output.exists():
        raise RuntimeError(f"{target} {stage} failed; inspect {output.with_suffix('.log')}")
    return json.loads(output.read_text())


def evaluate_target(job_file, target, cycles, minimum_accuracy):
    baseline = _run_stage(job_file, target, "baseline", -1)
    stages = [baseline]
    if baseline["generation"] is not None:
        raise RuntimeError("Baseline already has a learned checkpoint")
    parent = None
    earlier_successes = set()
    regressions = []
    reload_mismatches = []
    for cycle in range(cycles):
        trained = _run_stage(job_file, target, "train", cycle)
        reloaded = _run_stage(job_file, target, "reload", cycle)
        stages.extend([trained, reloaded])
        if (not trained["generation"] or trained["pid"] == reloaded["pid"]
                or trained["generation"] != reloaded["generation"]):
            raise RuntimeError("Fresh-process checkpoint restoration was not established")
        if not (trained["identity_digest"] == reloaded["identity_digest"] == baseline["identity_digest"]):
            raise RuntimeError("Model identity changed during the evaluation")
        if trained["generation"] == parent or trained["parent_generation"] != parent:
            raise RuntimeError("Training did not publish a new descendant of the prior checkpoint")
        parent = trained["generation"]
        before = {item["id"]: normalize_answer(item["response"]) for item in trained["facts"]}
        after = {item["id"]: normalize_answer(item["response"]) for item in reloaded["facts"]}
        reload_mismatches.extend(key for key in before if before[key] != after[key])
        successful = {item["id"] for item in reloaded["facts"]
                      if normalize_answer(item["response"]) == normalize_answer(item["answer"])}
        regressions.extend({"cycle": cycle, "fact": key} for key in sorted(earlier_successes - successful))
        earlier_successes |= successful
    final = stages[-1]
    baseline_controls = {item["id"] for item in baseline["controls"]
                         if normalize_answer(item["response"]) == normalize_answer(item["answer"])}
    final_controls = {item["id"] for item in final["controls"]
                      if normalize_answer(item["response"]) == normalize_answer(item["answer"])}
    passed = (final["fact_score"]["accuracy"] >= minimum_accuracy
              and final["fact_score"]["accuracy"] > baseline["fact_score"]["accuracy"]
              and not regressions and not reload_mismatches and baseline_controls <= final_controls)
    return {"target": target, "passed": passed, "stages": stages,
            "regressions": regressions, "reload_mismatches": reload_mismatches,
            "lost_baseline_controls": sorted(baseline_controls - final_controls)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*TARGETS, "both"], default="both")
    parser.add_argument("--bonsai-path")
    parser.add_argument("--crack-path")
    parser.add_argument("--allow-download", action="store_true", help="Allow downloading missing selected snapshots")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--facts-per-cycle", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--answer-tokens", type=int, default=256)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--minimum-accuracy", type=float, default=0.75)
    parser.add_argument("--output-dir", default="eval_results/persistent_memory")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--job", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=["baseline", "train", "reload"], help=argparse.SUPPRESS)
    parser.add_argument("--cycle", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require_apple_mlx()
    if args.worker:
        worker(args.job, args.model, args.stage, args.cycle, args.worker_output)
        return
    if args.context_tokens < 2 or min(args.cycles, args.facts_per_cycle, args.answer_tokens) < 1 or not 0 < args.minimum_accuracy <= 1:
        parser.error("Positive cycle/fact/token counts and an accuracy in (0,1] are required")
    directory = Path(args.output_dir).resolve() / uuid4().hex
    directory.mkdir(parents=True)
    job = {"facts": make_facts(args.seed, args.cycles, args.facts_per_cycle), "seed": args.seed,
           "answer_tokens": args.answer_tokens, "context_tokens": args.context_tokens,
           "allow_download": args.allow_download,
           "paths": {key: str(Path(value).expanduser().resolve()) if value else None
                     for key, value in (("bonsai2", args.bonsai_path), ("bonsai2-crack", args.crack_path))}}
    job_file = directory / "job.json"
    job_file.write_text(json.dumps(job, indent=2))
    results = []
    for target in TARGETS if args.model == "both" else [args.model]:
        print(f"Measuring {target}: baseline, parameter updates, all-fact retests and fresh-process reloads", flush=True)
        try:
            results.append(evaluate_target(job_file, target, args.cycles, args.minimum_accuracy))
        except Exception as exc:
            results.append({"target": target, "passed": False, "error": str(exc)})
        (directory / "report.json").write_text(json.dumps({"results": results}, indent=2))
    print("Measured report: " + str(directory / "report.json"), flush=True)
    if not all(result["passed"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
