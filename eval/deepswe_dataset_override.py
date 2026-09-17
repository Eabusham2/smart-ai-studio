"""Use real SWE-bench Verified tasks for the 32K DeepSWE slot.

No synthetic repositories, fake tests, hand-written stand-ins, prompt compaction,
or gold-patch leakage are allowed here. We intersect SWE-bench Verified with the
published SWE-bench BM25 retrieval contexts, then take a deterministic random
sample of 50 real Verified instances. The 27K published context is preferred; if
this model's tokenizer says the fully formatted prompt would not leave room inside
the 32K context window, the published 13K BM25 context for the same Verified
instance is used instead. Scoring is delegated to the official SWE-bench Docker
harness against SWE-bench/SWE-bench_Verified.
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
from typing import Any, Dict, List


_VERIFIED_DATASET = "SWE-bench/SWE-bench_Verified"
_BM25_27K = "princeton-nlp/SWE-bench_bm25_27K"
_BM25_13K = "princeton-nlp/SWE-bench_bm25_13K"
_SAMPLE_SEED = 20260916
_SAMPLE_SIZE = 50
_CONTEXT_LIMIT = 32768
_MIN_OUTPUT_ROOM = 4096


def _all_rows(real_module, dataset: str) -> List[Dict[str, Any]]:
    q = urllib.parse.quote(dataset, safe="")
    meta = real_module._json_get(
        f"https://datasets-server.huggingface.co/splits?dataset={q}"
    )
    descriptors = meta.get("splits", [])
    ordered = sorted(
        descriptors,
        key=lambda d: (
            0 if str(d.get("split", "")).casefold() == "test" else 1,
            str(d.get("config", "")),
        ),
    )
    out: List[Dict[str, Any]] = []
    for desc in ordered:
        config = str(desc.get("config", "default"))
        split = str(desc.get("split", "test"))
        offset = 0
        while True:
            url = (
                "https://datasets-server.huggingface.co/rows?"
                f"dataset={q}&config={urllib.parse.quote(config, safe='')}"
                f"&split={urllib.parse.quote(split, safe='')}"
                f"&offset={offset}&length=100"
            )
            payload = real_module._json_get(url)
            rows = payload.get("rows", [])
            if not rows:
                break
            out.extend(w.get("row", w) for w in rows)
            offset += len(rows)
            if len(rows) < 100:
                break
        if out:
            break
    return out


def _instance_id(row: Dict[str, Any]) -> str:
    return str(row.get("instance_id") or row.get("id") or "").strip()


def _retrieval_text(row: Dict[str, Any]) -> str:
    return str(row.get("text") or row.get("prompt") or "").strip()


def _load_verified_50(real_module, cache_dir: str) -> List[Dict[str, Any]]:
    def build():
        verified_rows = _all_rows(real_module, _VERIFIED_DATASET)
        bm27_rows = _all_rows(real_module, _BM25_27K)
        bm13_rows = _all_rows(real_module, _BM25_13K)

        verified = {_instance_id(r): r for r in verified_rows if _instance_id(r)}
        bm27 = {
            _instance_id(r): _retrieval_text(r)
            for r in bm27_rows
            if _instance_id(r) and _retrieval_text(r)
        }
        bm13 = {
            _instance_id(r): _retrieval_text(r)
            for r in bm13_rows
            if _instance_id(r) and _retrieval_text(r)
        }

        ids = sorted(set(verified) & set(bm27))
        rng = random.Random(_SAMPLE_SEED)
        rng.shuffle(ids)
        ids = ids[:_SAMPLE_SIZE]
        if len(ids) != _SAMPLE_SIZE:
            raise RuntimeError(
                f"SWE-bench Verified/BM25 intersection produced only {len(ids)} usable instances"
            )

        items: List[Dict[str, Any]] = []
        for iid in ids:
            row = verified[iid]
            items.append({
                "id": f"REAL-SWEBENCH-VERIFIED-{iid}",
                "instance_id": iid,
                "prompt": bm27[iid],
                "prompt_27k": bm27[iid],
                "prompt_13k": bm13.get(iid, ""),
                "problem_statement": str(row.get("problem_statement") or "").strip(),
                "repo": str(row.get("repo") or "").strip(),
                "base_commit": str(row.get("base_commit") or "").strip(),
                "real_source": (
                    "SWE-bench/SWE-bench_Verified + "
                    "princeton-nlp/SWE-bench_bm25_27K/13K"
                ),
                "_real_kind": "swebench_verified",
                "sample_seed": _SAMPLE_SEED,
            })
        return items

    return real_module._cached(
        cache_dir,
        "swebench_verified_random50_bm25_27k13k",
        build,
    )


def _patch_only(text: str) -> str:
    text = str(text or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    blocks = re.findall(r"```(?:diff|patch)?\s*([\s\S]*?)```", text, re.I)
    if blocks:
        text = blocks[-1].strip()
    tagged = re.search(r"<patch>\s*([\s\S]*?)\s*</patch>", text, re.I)
    if tagged:
        text = tagged.group(1).strip()
    starts = [p for p in (text.find("diff --git "), text.find("--- a/")) if p >= 0]
    if starts:
        text = text[min(starts):]
    return text.strip()


def _published_prompt_for_model(self, item, runtime_module, system_message: str):
    tokenizer = self.engine.tokenizer
    suffix = (
        "\n\nResolve the issue using the repository context above. "
        "Output ONLY a unified diff patch; do not restate the issue or include prose."
    )
    candidates = [item.get("prompt_27k", ""), item.get("prompt_13k", "")]
    for context in candidates:
        if not str(context).strip():
            continue
        user = str(context).rstrip() + suffix
        formatted = runtime_module._chat(tokenizer, user, system=system_message)
        token_count = len(tokenizer.encode(formatted))
        if token_count <= _CONTEXT_LIMIT - _MIN_OUTPUT_ROOM:
            return user, formatted, token_count
    raise RuntimeError(
        f"{item.get('instance_id')}: neither published 27K nor 13K SWE-bench context "
        "fits the 32K model window with output room; refusing to compact or fabricate context"
    )


def _verify_official_swebench(item: Dict[str, Any], candidate: str) -> bool:
    if shutil.which("docker") is None:
        raise RuntimeError(
            "DeepSWE requires Docker for the official SWE-bench Verified evaluator"
        )
    try:
        import swebench  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "DeepSWE requires the official `swebench` Python package"
        ) from exc

    iid = str(item["instance_id"])
    patch = _patch_only(candidate)
    if not patch:
        return False

    with tempfile.TemporaryDirectory(prefix="smartai_swebench_") as root:
        predictions = os.path.join(root, "predictions.jsonl")
        run_id = "smartai-" + re.sub(r"[^A-Za-z0-9_.-]", "-", iid) + f"-{int(time.time())}"
        model_name = "smart-ai-studio"
        with open(predictions, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "instance_id": iid,
                "model_name_or_path": model_name,
                "model_patch": patch,
            }) + "\n")

        cmd = [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            _VERIFIED_DATASET,
            "--split",
            "test",
            "--predictions_path",
            predictions,
            "--max_workers",
            "1",
            "--run_id",
            run_id,
            "--instance_ids",
            iid,
            "--timeout",
            "900",
        ]
        proc = subprocess.run(
            cmd,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=1200,
        )
        report_path = os.path.join(root, f"{model_name}.{run_id}.json")
        if not os.path.exists(report_path):
            if proc.returncode != 0:
                raise RuntimeError(
                    "Official SWE-bench evaluator failed: "
                    + (proc.stderr or proc.stdout)[-3000:]
                )
            return False
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        return iid in set(report.get("resolved_ids", []))


def install(real_module, runtime_module, phase4_module, cls) -> None:
    """Install only the DeepSWE source/prompt/verifier override."""
    real_module._load_code_repair = lambda cache_dir: _load_verified_50(real_module, cache_dir)

    original_eval = cls._evaluate_single_item
    original_task_prompt = phase4_module._task_user_prompt
    original_blind = phase4_module._candidate_passes_answer_blind
    original_hidden = phase4_module._hidden_reward_only_after_selection

    def evaluate(self, split, item):
        if item.get("_real_kind") != "swebench_verified":
            return original_eval(self, split, item)
        user, formatted, prompt_tokens = _published_prompt_for_model(
            self, item, runtime_module, phase4_module.SYSTEM_PROMPT
        )
        max_tokens = max(1, _CONTEXT_LIMIT - prompt_tokens)
        out = self._fast_generate(formatted, max_tokens=max_tokens)
        self.last_raw_out = out
        runtime_module._append_raw_generation_log(self, formatted, user, out)
        return _verify_official_swebench(item, out)

    def task_prompt(split, item):
        if item.get("_real_kind") == "swebench_verified":
            # RSI branches use the published 13K context so their existing 16K
            # branch ceiling remains inside the same 32K total context contract.
            context = item.get("prompt_13k") or item.get("prompt_27k") or item.get("prompt", "")
            return (
                str(context).rstrip()
                + "\n\nResolve the issue. Output ONLY a unified diff patch; no prose."
            )
        return original_task_prompt(split, item)

    def blind(self, split, item, candidate):
        if item.get("_real_kind") == "swebench_verified":
            # Branch choice stays answer-blind and cheap; the official harness is
            # invoked once only after the candidate has been selected.
            return None
        return original_blind(self, split, item, candidate)

    def hidden(self, split, item, candidate):
        if item.get("_real_kind") == "swebench_verified":
            return _verify_official_swebench(item, candidate)
        return original_hidden(self, split, item, candidate)

    cls._evaluate_single_item = evaluate
    phase4_module._task_user_prompt = task_prompt
    phase4_module._candidate_passes_answer_blind = blind
    phase4_module._hidden_reward_only_after_selection = hidden
