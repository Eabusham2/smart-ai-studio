"""Tiny source override for flagship public benchmark variants."""
from __future__ import annotations


def install(real_module) -> None:
    def load_lcb_full(cache_dir: str):
        def mapper(row, config, split):
            prompt = row.get("question_content") or row.get("prompt") or row.get("question")
            if not isinstance(prompt, str) or not prompt.strip():
                return None
            cases = []
            for key in ("private_test_cases", "public_test_cases", "test_cases"):
                cases.extend(real_module._normalize_io_cases(row.get(key)))
            unique, seen = [], set()
            for pair in cases:
                if pair not in seen:
                    seen.add(pair)
                    unique.append(pair)
            if not unique:
                return None
            starter = str(row.get("starter_code") or "").strip()
            full_prompt = prompt.strip()
            if starter:
                full_prompt += "\n\nStarter code:\n```python\n" + starter + "\n```"
            rid = row.get("question_id") or row.get("id") or row.get("task_id") or "row"
            return {
                "id": f"REAL-LCB-{rid}",
                "prompt": full_prompt,
                "test": real_module._io_test_harness(unique),
                "io_case_count": len(unique),
                "real_source": "livecodebench/code_generation",
                "_real_kind": "livecodebench",
            }

        return real_module._cached(
            cache_dir,
            "livecodebench_full_100",
            lambda: real_module._hf_rows("livecodebench/code_generation", 100, mapper),
        )

    real_module._load_lcb = load_lcb_full
