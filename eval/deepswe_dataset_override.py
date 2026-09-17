"""Use a strong real debugging benchmark for the 32K DeepSWE slot.

SWE-bench Verified is the preferred repository-level flagship, but honest instance
verification requires full repository checkout/build context that is not naturally
contained in this evaluator's 32K prompt/verifier contract. Do not truncate,
compact, summarize, or leak gold patches to force it in. For this slot use the
published hard Python split of PNYX/DebugBench, which supplies buggy code plus
executable assertions and naturally fits the existing local repair verifier.

The user explicitly allowed the suite size to grow when two genuinely useful
choices cannot be resolved; this override does not add a duplicate because there
is a clear fit here. HumanEval remains a separate benchmark unchanged.
"""
from __future__ import annotations

import json
import urllib.parse
from typing import Any, Dict, List


def install(real_module) -> None:
    def load_debugbench_hard(cache_dir: str) -> List[Dict[str, Any]]:
        def build():
            dataset = "PNYX/debugbench_pnyx"
            q = urllib.parse.quote(dataset, safe="")
            meta = real_module._json_get(
                f"https://datasets-server.huggingface.co/splits?dataset={q}"
            )
            match = None
            for desc in meta.get("splits", []):
                if (
                    str(desc.get("config", "")).casefold() == "python3"
                    and str(desc.get("split", "")).casefold() == "hard"
                ):
                    match = desc
                    break
            if match is None:
                raise RuntimeError("PNYX/DebugBench has no python3/hard split")

            config = str(match.get("config", "python3"))
            split = str(match.get("split", "hard"))
            out: List[Dict[str, Any]] = []
            offset = 0
            while len(out) < 50:
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
                for wrapped in rows:
                    row = wrapped.get("row", wrapped)
                    question = str(row.get("question") or "").strip()
                    constraints = str(row.get("constraints") or "").strip()
                    buggy = str(row.get("buggy_code") or "").strip()
                    init = str(row.get("initialization_code") or "").strip()
                    tests = str(row.get("test_code") or "").strip()
                    if not question or not buggy or not tests:
                        continue
                    task = question
                    if constraints:
                        task += "\n\nConstraints:\n" + constraints
                    full_buggy = (init + "\n" + buggy).strip() if init else buggy
                    source_id = str(row.get("slug") or f"hard-{offset + len(out)}")
                    out.append({
                        "id": f"REAL-DEBUGBENCH-{source_id}-{len(out):03d}",
                        "prompt": task,
                        "buggy_code": full_buggy,
                        "test": tests,
                        "real_source": "PNYX/debugbench_pnyx:python3/hard",
                        "_real_kind": "code_repair",
                        "difficulty": "hard",
                    })
                    if len(out) == 50:
                        break
                offset += len(rows)
                if len(rows) < 100:
                    break
            if len(out) != 50:
                raise RuntimeError(
                    f"DebugBench hard Python expected 50 usable rows, got {len(out)}"
                )
            return out

        return real_module._cached(cache_dir, "debugbench_python_hard_50", build)

    real_module._load_code_repair = load_debugbench_hard
