"""Narrow schema/source fixes for the real public benchmark adapter.

This module only replaces loaders whose upstream public schema differs from the
recovered adapter's assumptions. It does not change stages, training, prompt
policy, checkpointing, or evaluation order.
"""
from __future__ import annotations

import base64
import json
import pickle
import zlib
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

            # First use execution-simple rows, which publish one exact callable
            # ground truth in the same JSONL record.
            for row in _jsonl(real_module._text_get(base + "BFCL_v3_exec_simple.json")):
                prompt = _conversation_text(row.get("question"))
                tools = row.get("function") or row.get("functions") or row.get("tools")
                parsed = real_module._parse_call(row.get("ground_truth"))
                if not prompt or not parsed:
                    continue
                name, args = parsed
                rows.append({
                    "id": f"REAL-BFCL-{row.get('id', len(rows))}",
                    "prompt": prompt + "\n\nAvailable tools:\n" + json.dumps(tools, ensure_ascii=False),
                    "expected_tool": name,
                    "expected_args": args,
                    "real_source": "gorilla-llm/Berkeley-Function-Calling-Leaderboard:BFCL_v3_exec_simple",
                    "_real_kind": "bfcl",
                })
                if len(rows) == 200:
                    return rows

            # Fill only with unambiguous single-call BFCL simple rows. Skip any
            # row that has multiple accepted values for an argument so the existing
            # exact JSON scorer cannot reject another officially-valid answer.
            question_rows = _jsonl(real_module._text_get(base + "BFCL_v3_simple.json"))
            answer_rows = {
                str(x.get("id")): x
                for x in _jsonl(real_module._text_get(base + "possible_answer/BFCL_v3_simple.json"))
            }
            for row in question_rows:
                answer_row = answer_rows.get(str(row.get("id")))
                truth = answer_row.get("ground_truth") if answer_row else None
                if not isinstance(truth, list) or len(truth) != 1 or not isinstance(truth[0], dict) or len(truth[0]) != 1:
                    continue
                name, arg_options = next(iter(truth[0].items()))
                if not isinstance(arg_options, dict):
                    continue
                exact_args = {}
                ambiguous = False
                for key, candidates in arg_options.items():
                    candidates = candidates if isinstance(candidates, list) else [candidates]
                    usable = [x for x in candidates if x not in ("", None)]
                    if len(usable) != 1:
                        ambiguous = True
                        break
                    exact_args[str(key)] = usable[0]
                if ambiguous:
                    continue
                prompt = _conversation_text(row.get("question"))
                tools = row.get("function") or row.get("functions") or row.get("tools")
                if not prompt or not tools:
                    continue
                rows.append({
                    "id": f"REAL-BFCL-{row.get('id', len(rows))}",
                    "prompt": prompt + "\n\nAvailable tools:\n" + json.dumps(tools, ensure_ascii=False),
                    "expected_tool": str(name),
                    "expected_args": exact_args,
                    "real_source": "gorilla-llm/Berkeley-Function-Calling-Leaderboard:BFCL_v3_simple",
                    "_real_kind": "bfcl",
                })
                if len(rows) == 200:
                    return rows

            raise RuntimeError(f"BFCL single-call public sets produced only {len(rows)}/200 unambiguous rows")

        return real_module._cached(cache_dir, "bfcl_v3_single_call_200", build)

    def normalize_io_cases(raw):
        value = real_module._as_json(raw)
        if isinstance(value, str):
            try:
                value = json.loads(
                    pickle.loads(
                        zlib.decompress(base64.b64decode(value.encode("utf-8")))
                    )
                )
            except Exception:
                return []
        if isinstance(value, dict):
            value = [value]
        pairs = []
        if isinstance(value, list):
            for case in value:
                case = real_module._as_json(case)
                if not isinstance(case, dict):
                    continue
                if str(case.get("testtype", "stdin")).lower() not in ("stdin", ""):
                    continue
                inp = case.get("input", case.get("stdin"))
                out = case.get("output", case.get("stdout"))
                if inp is not None and out is not None:
                    pairs.append((str(inp), str(out)))
        return pairs

    real_module._load_humaneval = load_humaneval
    real_module._load_olympiad = load_olympiad
    real_module._load_supergpqa = load_supergpqa
    real_module._load_livebench_reasoning = load_livebench_reasoning
    real_module._load_bfcl = load_bfcl
    real_module._normalize_io_cases = normalize_io_cases
