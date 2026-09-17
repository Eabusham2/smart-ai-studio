"""Narrow schema fixes for real public benchmark replacements.

This module does not change stage order, RSI, training, checkpointing, or model
behavior. It only makes the already-selected real datasets load/score according
to their published schemas.
"""
from __future__ import annotations

import base64
import json
import pickle
import re
import zlib


def _choice_aj(text: str):
    text = str(text or "")
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    for value in reversed(boxed):
        m = re.fullmatch(r"\s*\(?([A-Ja-j])\)?\s*", value)
        if m:
            return m.group(1).upper()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line = lines[-1] if lines else text.strip()
    m = re.fullmatch(r"\s*(?:\*\*)?\(?([A-Ja-j])\)?[\).:]?(?:\*\*)?\s*", line)
    if m:
        return m.group(1).upper()
    matches = re.findall(
        r"(?i)(?:final\s+answer|answer|choice|option)\s*(?:is\s*)?[:=\-]?\s*\(?([A-J])\)?\b",
        line,
    )
    return matches[-1].upper() if matches else None


def _flatten_question(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        if str(value.get("role", "")).lower() == "user" and value.get("content"):
            return str(value["content"]).strip()
        return "\n".join(_flatten_question(v) for v in value.values() if _flatten_question(v))
    if isinstance(value, list):
        parts = []
        for part in value:
            text = _flatten_question(part)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return ""


def install_sources(real_module, scoring_module) -> None:
    # HumanEval: use the published HF 164-row test split directly. This avoids
    # relying on the old uncompressed GitHub path, which no longer exists.
    def load_humaneval(cache_dir: str):
        def mapper(row, config, split):
            prompt = row.get("prompt")
            test = row.get("test")
            entry = row.get("entry_point")
            task_id = row.get("task_id")
            if not all(isinstance(x, str) and x.strip() for x in (prompt, test, entry, task_id)):
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
            lambda: real_module._hf_rows(
                "openai/openai_humaneval", 164, mapper, ("openai_humaneval",)
            ),
        )

    # OlympiadBench: text-only English competition math. Never drop an image and
    # pretend the text alone is the full question.
    def load_olympiad(cache_dir: str):
        def mapper(row, config, split):
            question = row.get("question")
            answers = real_module._answers(row.get("final_answer"))
            if not isinstance(question, str) or not question.strip() or not answers:
                return None
            if str(row.get("modality", "")).lower() not in ("text", "text-only", ""):
                return None
            return {
                "id": f"REAL-OLYMPIAD-{row.get('id', abs(hash(question)))}",
                "prompt": question,
                "expected": answers[0],
                "expected_answers": answers,
                "real_source": "Hothan/OlympiadBench:OE_TO_maths_en_COMP",
                "_real_kind": "math",
            }
        return real_module._cached(
            cache_dir,
            "olympiadbench_text_en_150",
            lambda: real_module._hf_rows(
                "Hothan/OlympiadBench", 150, mapper, ("OE_TO_maths_en_COMP",)
            ),
        )

    # SuperGPQA publishes answer_letter explicitly and has up to ten options.
    def load_supergpqa(cache_dir: str):
        def mapper(row, config, split):
            question = row.get("question")
            options = real_module._options(row.get("options"))
            letter = str(row.get("answer_letter") or "").strip().upper()
            if not isinstance(question, str) or len(options) < 2 or not re.fullmatch(r"[A-J]", letter):
                return None
            context = " / ".join(
                str(row.get(k) or "").strip()
                for k in ("discipline", "field", "subfield")
                if str(row.get(k) or "").strip()
            )
            return {
                "id": f"REAL-SuperGPQA-{row.get('uuid', abs(hash(question)))}",
                "prompt": real_module._choice_prompt(question, options, f"[{context}]" if context else ""),
                "expected": letter,
                "real_source": "m-a-p/SuperGPQA",
                "_real_kind": "choice",
            }
        return real_module._cached(
            cache_dir,
            "supergpqa_400_schema",
            lambda: real_module._hf_rows("m-a-p/SuperGPQA", 400, mapper),
        )

    # LiveBench reasoning stores the actual prompt in a one-element `turns` list.
    def load_livebench(cache_dir: str):
        def mapper(row, config, split):
            turns = row.get("turns")
            prompt = turns[0] if isinstance(turns, list) and turns else None
            answer = row.get("ground_truth")
            values = real_module._answers(answer)
            if not isinstance(prompt, str) or not prompt.strip() or not values:
                return None
            return {
                "id": f"REAL-LIVEBENCH-{row.get('question_id', abs(hash(prompt)))}",
                "prompt": prompt,
                "expected": values[0],
                "expected_answers": values,
                "real_source": "livebench/reasoning",
                "_real_kind": "freeform",
            }
        return real_module._cached(
            cache_dir,
            "livebench_reasoning_100_turns",
            lambda: real_module._hf_rows("livebench/reasoning", 100, mapper),
        )

    # BFCL's Hub repo is raw JSONL rather than Dataset-Server rows. Use the real
    # v3 simple questions + official possible-answer file. The flexible scorer is
    # installed below so optional/alternate accepted arguments stay valid.
    def load_bfcl(cache_dir: str):
        base = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/"

        def parse_lines(path):
            text = real_module._text_get(base + path)
            return [json.loads(line) for line in text.splitlines() if line.strip()]

        def build():
            questions = parse_lines("BFCL_v3_simple.json")
            answers = {
                str(row.get("id")): row
                for row in parse_lines("possible_answer/BFCL_v3_simple.json")
            }
            out = []
            for row in questions:
                rid = str(row.get("id") or "")
                answer_row = answers.get(rid)
                if not answer_row:
                    continue
                truth = answer_row.get("ground_truth")
                if not isinstance(truth, list) or len(truth) != 1 or not isinstance(truth[0], dict) or len(truth[0]) != 1:
                    continue
                name, arg_options = next(iter(truth[0].items()))
                if not isinstance(arg_options, dict):
                    continue
                prompt = _flatten_question(row.get("question"))
                functions = row.get("function")
                if not prompt or not functions:
                    continue
                normalized = {
                    str(key): (value if isinstance(value, list) else [value])
                    for key, value in arg_options.items()
                }
                out.append({
                    "id": f"REAL-BFCL-{rid}",
                    "prompt": prompt + "\n\nAvailable tools:\n" + json.dumps(functions, ensure_ascii=False),
                    "expected_tool": str(name),
                    "expected_arg_options": normalized,
                    "real_source": "gorilla-llm/Berkeley-Function-Calling-Leaderboard:BFCL_v3_simple",
                    "_real_kind": "bfcl_flexible",
                })
                if len(out) == 200:
                    return out
            raise RuntimeError(f"BFCL v3 simple produced only {len(out)}/200 usable rows")

        return real_module._cached(cache_dir, "bfcl_v3_simple_200", build)

    # LiveCodeBench hides many private cases in base64(zlib(pickle(JSON))). Decode
    # them and keep only stdin/stdout tests supported by this evaluator. Functional
    # cases are skipped rather than mis-scored as stdin programs.
    def normalize_io(raw):
        value = real_module._as_json(raw)
        if isinstance(value, str):
            try:
                value = json.loads(pickle.loads(zlib.decompress(base64.b64decode(value.encode("utf-8")))))
            except Exception:
                return []
        pairs = []
        if isinstance(value, dict):
            value = [value]
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
    real_module._load_livebench_reasoning = load_livebench
    real_module._load_bfcl = load_bfcl
    real_module._normalize_io_cases = normalize_io
    scoring_module._final_choice = _choice_aj


def install_runtime(real_module, runtime_module, phase4_module, cls) -> None:
    """Add only BFCL's flexible official-answer scoring to both eval stages."""
    original_eval = cls._evaluate_single_item
    original_task = phase4_module._task_user_prompt
    original_blind = phase4_module._candidate_passes_answer_blind
    original_hidden = phase4_module._hidden_reward_only_after_selection

    def prompt(item):
        return (
            f"{item['prompt']}\n"
            "Return ONLY one JSON object with keys `name` and `arguments` for the requested tool call."
        )

    def match(item, output):
        parsed = real_module._parse_json_call(output)
        if not parsed or parsed[0] != item.get("expected_tool"):
            return False
        args = parsed[1]
        expected = item.get("expected_arg_options") or {}
        if not isinstance(args, dict):
            return False
        for key in args:
            if key not in expected:
                return False
        for key, allowed in expected.items():
            allowed = allowed if isinstance(allowed, list) else [allowed]
            if key not in args:
                if "" in allowed or None in allowed:
                    continue
                return False
            if not any(args[key] == candidate for candidate in allowed if candidate != ""):
                return False
        return True

    def evaluate(self, split, item):
        if item.get("_real_kind") != "bfcl_flexible":
            return original_eval(self, split, item)
        user = prompt(item)
        formatted = runtime_module._chat(
            self.engine.tokenizer, user, system=phase4_module.SYSTEM_PROMPT
        )
        ceiling = getattr(self, "benchmark_max_tokens", None) or runtime_module._benchmark_ceiling(self)
        out = self._fast_generate(formatted, max_tokens=ceiling)
        self.last_raw_out = out
        runtime_module._append_raw_generation_log(self, formatted, user, out)
        return match(item, out)

    def task_prompt(split, item):
        if item.get("_real_kind") == "bfcl_flexible":
            return prompt(item)
        return original_task(split, item)

    def blind(self, split, item, candidate):
        if item.get("_real_kind") == "bfcl_flexible":
            return None
        return original_blind(self, split, item, candidate)

    def hidden(self, split, item, candidate):
        if item.get("_real_kind") == "bfcl_flexible":
            return match(item, candidate)
        return original_hidden(self, split, item, candidate)

    cls._evaluate_single_item = evaluate
    phase4_module._task_user_prompt = task_prompt
    phase4_module._candidate_passes_answer_blind = blind
    phase4_module._hidden_reward_only_after_selection = hidden
