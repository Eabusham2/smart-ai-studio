"""Narrow schema/source fixes for the real public benchmark adapter.

This module only replaces loaders whose upstream public schema differs from the
recovered adapter's assumptions. It does not change stages, training, prompt
policy, checkpointing, or evaluation order.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List


def _conversation_text(value: Any) -> str:
    parts: List[str] = []

    def walk(node):
        if isinstance(node, dict):
            content = node.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(value)
    return "\n".join(parts).strip()


def _jsonl(text: str) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def install(real_module) -> None:
    def load_humaneval(cache_dir: str):
        def mapper(row, config, split):
            task_id = row.get("task_id")
            prompt = row.get("prompt")
            test = row.get("test")
            entry = row.get("entry_point")
            if not all(isinstance(x, str) and x.strip() for x in (task_id, prompt, test, entry)):
                return None
            return {
                "id": f"REAL-{task_id}",
                "prompt": prompt,
                "test": test + f"\ncheck({entry})\n",
                "entry_point": entry,
                "real_source": "openai/openai_humaneval",
                "_real_kind": "humaneval",
            }

        return real_module._cached(
            cache_dir,
            "humaneval_164_hf",
            lambda: real_module._hf_rows("openai/openai_humaneval", 164, mapper),
        )

    def load_olympiad(cache_dir: str):
        def mapper(row, config, split):
            problem = row.get("question") or row.get("problem")
            answer = row.get("final_answer") or row.get("answer")
            if not isinstance(problem, str) or not problem.strip() or answer is None:
                return None
            vals = real_module._answers(answer)
            if not vals:
                return None
            return {
                "id": f"REAL-OLYMPIAD-{row.get('id', abs(hash(problem)))}",
                "prompt": problem,
                "expected": vals[0],
                "expected_answers": vals,
                "real_source": "Hothan/OlympiadBench:OE_TO_maths_en_COMP",
                "_real_kind": "math",
            }

        return real_module._cached(
            cache_dir,
            "olympiad_text_math_en_150",
            lambda: real_module._hf_rows(
                "Hothan/OlympiadBench",
                150,
                mapper,
                ("OE_TO_maths_en_COMP",),
            ),
        )

    def load_supergpqa(cache_dir: str):
        def mapper(row, config, split):
            question = row.get("question")
            opts = real_module._options(row.get("options") or row.get("choices"))
            answer = row.get("answer_letter", row.get("answer", row.get("label")))
            letter = real_module._choice_letter(answer, opts)
            if not isinstance(question, str) or len(opts) < 2 or not letter:
                return None
            context_bits = [
                str(row.get("discipline") or "").strip(),
                str(row.get("field") or "").strip(),
                str(row.get("subfield") or "").strip(),
            ]
            context = " / ".join(x for x in context_bits if x)
            return {
                "id": f"REAL-SuperGPQA-{row.get('uuid', row.get('id', abs(hash(question))))}",
                "prompt": real_module._choice_prompt(question, opts, f"[{context}]" if context else ""),
                "expected": letter,
                "real_source": "m-a-p/SuperGPQA",
                "_real_kind": "choice",
            }

        return real_module._cached(
            cache_dir,
            "supergpqa_400_schema_fixed",
            lambda: real_module._hf_rows("m-a-p/SuperGPQA", 400, mapper),
        )

    def load_livebench_reasoning(cache_dir: str):
        def mapper(row, config, split):
            turns = row.get("turns")
            if isinstance(turns, (list, tuple)):
                prompt = "\n\n".join(str(x).strip() for x in turns if str(x).strip())
            else:
                prompt = str(turns or "").strip()
            answer = row.get("ground_truth") or row.get("answer") or row.get("reference_answer")
            vals = real_module._answers(answer)
            if not prompt or not vals:
                return None
            return {
                "id": f"REAL-LIVEBENCH-{row.get('question_id', row.get('id', abs(hash(prompt))))}",
                "prompt": prompt,
                "expected": vals[0],
                "expected_answers": vals,
                "real_source": "livebench/reasoning",
                "_real_kind": "freeform",
            }

        return real_module._cached(
            cache_dir,
            "livebench_reasoning_100_schema_fixed",
            lambda: real_module._hf_rows("livebench/reasoning", 100, mapper),
        )

    def load_bfcl(cache_dir: str):
        base = (
            "https://huggingface.co/datasets/"
            "gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/"
        )

        def build():
            rows: List[Dict[str, Any]] = []
            for filename in ("BFCL_v3_exec_simple.json", "BFCL_v3_exec_multiple.json"):
                for row in _jsonl(real_module._text_get(base + filename)):
                    prompt = _conversation_text(row.get("question"))
                    tools = row.get("function") or row.get("functions") or row.get("tools")
                    truth = row.get("ground_truth")
                    parsed = real_module._parse_call(truth)
                    if not prompt or not parsed:
                        continue
                    name, args = parsed
                    rows.append({
                        "id": f"REAL-BFCL-{row.get('id', len(rows))}",
                        "prompt": prompt + "\n\nAvailable tools:\n" + json.dumps(tools, ensure_ascii=False),
                        "expected_tool": name,
                        "expected_args": args,
                        "real_source": f"gorilla-llm/Berkeley-Function-Calling-Leaderboard:{filename}",
                        "_real_kind": "bfcl",
                    })
                    if len(rows) == 200:
                        return rows
            if len(rows) < 200:
                raise RuntimeError(f"BFCL public exec sets produced only {len(rows)}/200 usable rows")
            return rows[:200]

        return real_module._cached(cache_dir, "bfcl_v3_exec_200", build)

    real_module._load_humaneval = load_humaneval
    real_module._load_olympiad = load_olympiad
    real_module._load_supergpqa = load_supergpqa
    real_module._load_livebench_reasoning = load_livebench_reasoning
    real_module._load_bfcl = load_bfcl
