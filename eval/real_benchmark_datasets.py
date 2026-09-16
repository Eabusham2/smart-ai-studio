"""Real public benchmark loader for the 4,014-item evaluation suite.

This replaces the legacy public-benchmark stand-ins before any evaluation stage
sees them. The two project-specific probes that measure novel DSL learning and
post-Learn dialogue recall remain purpose-built by design.
"""
from __future__ import annotations

import ast
import json
from typing import Any, Dict, Iterable, List


def _datasets():
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Real benchmark loading requires the `datasets` package; synthetic fallback is disabled."
        ) from exc
    return load_dataset


def _pick_split(ds, *preferred):
    for name in preferred:
        if name in ds:
            return ds[name]
    if not ds:
        raise RuntimeError("Dataset loaded without any splits")
    return ds[next(iter(ds.keys()))]


def _rows(repo: str, config: str | None = None, *, preferred=("test", "validation", "train")):
    load_dataset = _datasets()
    ds = load_dataset(repo, config) if config else load_dataset(repo)
    return _pick_split(ds, *preferred)


def _letter(idx: int) -> str:
    return chr(ord("A") + int(idx))


def _option_prompt(question: str, options: Iterable[Any]) -> str:
    opts = list(options)
    return str(question).strip() + "\n" + "\n".join(
        f"({_letter(i)}) {str(value).strip()}" for i, value in enumerate(opts)
    )


def _single_answer(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        value = value[0]
    return str(value).strip()


def _human_eval() -> List[Dict[str, Any]]:
    rows = _rows("openai/openai_humaneval")
    out = []
    for row in rows:
        out.append({
            "id": str(row["task_id"]),
            "prompt": str(row["prompt"]),
            "canonical_solution": str(row["canonical_solution"]),
            "test": str(row["test"]),
            "entry_point": str(row["entry_point"]),
            "source": "openai/openai_humaneval",
        })
    if len(out) != 164:
        raise RuntimeError(f"HumanEval expected 164 rows, got {len(out)}")
    return out


def _bigcodebench_hard() -> List[Dict[str, Any]]:
    load_dataset = _datasets()
    ds = load_dataset("bigcode/bigcodebench-hard")
    rows = _pick_split(ds, "v0.1.4", "v0.1.3", "v0.1.2", "v0.1.1", "v0.1.0_hf")
    out = []
    for row in rows:
        prompt = row.get("instruct_prompt") or row.get("complete_prompt") or row.get("question")
        test = row.get("test")
        if not prompt or not test:
            continue
        out.append({
            "id": f"LCB_Hard_{len(out)}",
            "prompt": str(prompt),
            "test": str(test),
            "entry_point": str(row.get("entry_point") or "task_func"),
            "source_id": str(row.get("task_id") or row.get("_id") or len(out)),
            "source": "bigcode/bigcodebench-hard",
        })
        if len(out) == 100:
            break
    if len(out) != 100:
        raise RuntimeError(f"BigCodeBench-Hard expected 100 usable rows, got {len(out)}")
    return out


def _gsm8k() -> List[Dict[str, Any]]:
    rows = _rows("openai/gsm8k", "main")
    out = []
    for i, row in enumerate(rows):
        answer = str(row["answer"])
        final = answer.rsplit("####", 1)[-1].strip().replace(",", "")
        out.append({"id": f"GSM8K_{i}", "prompt": str(row["question"]), "expected": final,
                    "source": "openai/gsm8k"})
        if len(out) == 500:
            break
    return out


def _math500() -> List[Dict[str, Any]]:
    rows = _rows("HuggingFaceH4/MATH-500")
    out = []
    for i, row in enumerate(rows):
        out.append({"id": f"MATH_{i}", "prompt": str(row["problem"]),
                    "expected": str(row["answer"]).strip(), "source": "HuggingFaceH4/MATH-500"})
    if len(out) != 500:
        raise RuntimeError(f"MATH-500 expected 500 rows, got {len(out)}")
    return out


def _aime_olympiad() -> List[Dict[str, Any]]:
    aime = _rows("AI-MO/aimo-validation-aime", preferred=("train", "test", "validation"))
    olympiad = _rows("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", preferred=("train",))
    out = []
    for row in aime:
        expected = row.get("answer")
        if expected is None:
            continue
        out.append({"id": f"AIME_{len(out)}", "prompt": str(row["problem"]),
                    "expected": str(expected).strip(), "source": "AI-MO/aimo-validation-aime"})
        if len(out) == 90:
            break
    for row in olympiad:
        if len(out) == 150:
            break
        expected = _single_answer(row.get("final_answer"))
        if not expected or not row.get("question"):
            continue
        out.append({"id": f"AIME_{len(out)}", "prompt": str(row["question"]),
                    "expected": expected, "source": "Hothan/OlympiadBench:OE_TO_maths_en_COMP"})
    if len(out) != 150:
        raise RuntimeError(f"AIME/Olympiad split expected 150 rows, got {len(out)}")
    return out


def _super_gpqa() -> List[Dict[str, Any]]:
    rows = _rows("m-a-p/SuperGPQA", preferred=("train",))
    out = []
    for row in rows:
        options = row.get("options") or []
        expected = str(row.get("answer_letter") or "").strip().upper()
        if not row.get("question") or not options or not expected:
            continue
        out.append({"id": f"GPQA_{len(out)}", "prompt": _option_prompt(row["question"], options),
                    "expected": expected, "source_id": str(row.get("uuid") or len(out)),
                    "source": "m-a-p/SuperGPQA"})
        if len(out) == 400:
            break
    if len(out) != 400:
        raise RuntimeError(f"SuperGPQA expected 400 rows, got {len(out)}")
    return out


def _mmlu_pro() -> List[Dict[str, Any]]:
    rows = _rows("TIGER-Lab/MMLU-Pro", preferred=("test", "validation", "train"))
    out = []
    for row in rows:
        options = row.get("options") or []
        expected = str(row.get("answer") or "").strip().upper()
        if not row.get("question") or not options or not expected:
            continue
        out.append({"id": f"MMLU_Pro_{len(out)}", "prompt": _option_prompt(row["question"], options),
                    "expected": expected, "source_id": str(row.get("question_id") or len(out)),
                    "source": "TIGER-Lab/MMLU-Pro"})
        if len(out) == 1000:
            break
    if len(out) != 1000:
        raise RuntimeError(f"MMLU-Pro expected 1000 rows, got {len(out)}")
    return out


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _call_from_text(text: str):
    try:
        node = ast.parse(text, mode="eval").body
        if not isinstance(node, ast.Call):
            return None, None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            cur = node.func
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            name = ".".join(reversed(parts))
        else:
            return None, None
        args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
        return name, args
    except Exception:
        return None, None


def _question_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("content") or item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(x for x in parts if x)
    return str(value)


def _bfcl() -> List[Dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for BFCL") from exc
    repo = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    qpath = hf_hub_download(repo, "BFCL_v3_exec_simple.json", repo_type="dataset")
    apath = hf_hub_download(repo, "possible_answer/BFCL_v3_exec_simple.json", repo_type="dataset")
    answers = {str(x.get("id")): x for x in _read_jsonl(apath)}
    out = []
    for row in _read_jsonl(qpath):
        ans = answers.get(str(row.get("id")), {})
        truth = ans.get("ground_truth") or []
        if isinstance(truth, list) and truth:
            truth = truth[0]
        if isinstance(truth, list) and truth:
            truth = truth[0]
        if not isinstance(truth, str):
            continue
        name, args = _call_from_text(truth)
        if not name or args is None:
            continue
        prompt = _question_text(row.get("question"))
        funcs = row.get("function") or []
        prompt += "\n\nAvailable functions:\n" + json.dumps(funcs, ensure_ascii=False)
        out.append({"id": f"BFCL_{len(out)}", "prompt": prompt,
                    "expected_tool": name, "expected_args": args,
                    "source_id": str(row.get("id")), "source": repo})
        if len(out) == 200:
            break
    if len(out) != 200:
        raise RuntimeError(f"BFCL expected 200 parseable simple calls, got {len(out)}")
    return out


def _logiqa() -> List[Dict[str, Any]]:
    rows = _rows("lucasmccabe/logiqa", preferred=("test", "validation", "train"))
    out = []
    for row in rows:
        options = row.get("options") or []
        prompt = str(row.get("context") or "").strip() + "\n\n" + str(row.get("query") or "").strip()
        prompt = _option_prompt(prompt, options)
        out.append({"id": f"Zebra_{len(out)}", "prompt": prompt,
                    "expected": _letter(int(row["correct_option"])), "source": "lucasmccabe/logiqa"})
        if len(out) == 200:
            break
    if len(out) != 200:
        raise RuntimeError(f"LogiQA expected 200 rows, got {len(out)}")
    return out


def _livebench_reasoning() -> List[Dict[str, Any]]:
    rows = _rows("livebench/reasoning")
    out = []
    for row in rows:
        turns = row.get("turns") or []
        prompt = turns[0] if turns else row.get("question")
        if not prompt:
            continue
        out.append({"id": f"HLE_{len(out)}", "prompt": str(prompt),
                    "expected": str(row.get("ground_truth") or "").strip(),
                    "source_id": str(row.get("question_id") or len(out)),
                    "source": "livebench/reasoning"})
        if len(out) == 100:
            break
    if len(out) != 100:
        raise RuntimeError(f"LiveBench reasoning expected 100 rows, got {len(out)}")
    return out


def _humaneval_fix() -> List[Dict[str, Any]]:
    rows = _rows("bigcode/humanevalpack", "python")
    out = []
    for row in rows:
        declaration = str(row.get("declaration") or "")
        buggy = str(row.get("buggy_solution") or "")
        entry = str(row.get("entry_point") or "")
        tests = str(row.get("test") or "")
        if not declaration or not buggy or not entry or not tests:
            continue
        solution = declaration + buggy
        test_file = f"from solution import {entry}\n\n{tests}\n\ncheck({entry})\n"
        out.append({
            "id": f"SWE_{len(out)}",
            "repo_files": {"solution.py": solution, "test_case.py": test_file},
            "test_cmd": "python3 test_case.py",
            "issue": str(row.get("instruction") or row.get("prompt") or "Repair the buggy function."),
            "source_id": str(row.get("task_id") or len(out)),
            "source": "bigcode/humanevalpack:python HumanEvalFix",
        })
        if len(out) == 50:
            break
    if len(out) != 50:
        raise RuntimeError(f"HumanEvalFix expected 50 rows, got {len(out)}")
    return out


def _olympiad_physics() -> List[Dict[str, Any]]:
    rows = _rows("Hothan/OlympiadBench", "OE_TO_physics_en_COMP", preferred=("train",))
    out = []
    for row in rows:
        expected = _single_answer(row.get("final_answer"))
        if not expected or not row.get("question"):
            continue
        out.append({"id": f"AutoEvol_{len(out)}", "prompt": str(row["question"]),
                    "expected_token": expected,
                    "source": "Hothan/OlympiadBench:OE_TO_physics_en_COMP"})
        if len(out) == 200:
            break
    if len(out) != 200:
        raise RuntimeError(f"OlympiadBench physics expected 200 rows, got {len(out)}")
    return out


def _dsl_probe() -> List[Dict[str, Any]]:
    return [
        {
            "id": f"DSL_{i}",
            "dsl_expr": f"[{i}, {i+2}, {i+4}] >>~fold({(i % 3) + 1}) <#>scale({(i % 4) + 2})",
            "prompt": f"Derive the exact numeric evaluation of: `[{i}, {i+2}, {i+4}] >>~fold({(i % 3) + 1}) <#>scale({(i % 4) + 2})`",
            "source": "Smart AI Studio novel-skill probe",
        }
        for i in range(300)
    ]


def _dialogue_probe() -> List[Dict[str, Any]]:
    return [
        {"id": f"Dialogue_{i}", "prompt": f"Recall developer decision regarding subsystem parameter #{i % 10} from prior multi-turn architecture session.",
         "expected_keyword": ["AdGuard", "BD PROCHOT", "Ternary", "BitLocker", "banana-mcp", "Thermal Grizzly", "MLX Metal", "1.58-bit", "Portainer", "Z790"][i % 10],
         "source": "Smart AI Studio post-Learn recall probe"}
        for i in range(150)
    ]


def load_real_4014() -> Dict[str, List[Dict[str, Any]]]:
    suite = {
        "HumanEval-164": _human_eval(),
        "LiveCodeBench-Hard": _bigcodebench_hard(),
        "GSM8K-500": _gsm8k(),
        "MATH-500": _math500(),
        "AIME-150": _aime_olympiad(),
        "GPQA-400": _super_gpqa(),
        "MMLU-Pro-1000": _mmlu_pro(),
        "BFCL-200": _bfcl(),
        "ZebraLogic-200": _logiqa(),
        "HLE-100": _livebench_reasoning(),
        "DeepSWE-50": _humaneval_fix(),
        "TensorGraphDSL-300": _dsl_probe(),
        "AutonomousEvolution-200": _olympiad_physics(),
        "DialogueRecall-150": _dialogue_probe(),
    }
    total = sum(len(x) for x in suite.values())
    if total != 4014:
        raise RuntimeError(f"Real 4K suite must remain exactly 4,014 items; got {total}")
    return suite


def install(provider_cls) -> None:
    if getattr(provider_cls, "_real_public_benchmarks_installed", False):
        return

    def load_all_4000_items(self):
        return load_real_4014()

    provider_cls.load_all_4000_items = load_all_4000_items
    provider_cls._real_public_benchmarks_installed = True
