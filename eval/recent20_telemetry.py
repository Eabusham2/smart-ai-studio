"""Non-invasive per-question telemetry for the master 4K benchmark.

This module records completed-question token usage and correctness without
changing prompts, generation, scoring, checkpoint semantics, or benchmark stages.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

TELEMETRY_PATH = Path("eval_results/recent20_questions.jsonl")


def _append_record(record):
    TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TELEMETRY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def install(cls):
    original_fast_generate = cls._fast_generate
    original_evaluate_one = cls._evaluate_single_item
    original_evaluate_all = cls._evaluate_all_splits

    def fast_generate_with_token_accounting(self, prompt, max_tokens=16384, stream=False):
        # Measure prompt tokens with the exact tokenizer used by the benchmark.
        try:
            self._recent_prompt_tokens = len(self.engine.tokenizer.encode(prompt))
        except Exception:
            self._recent_prompt_tokens = 0
        self._recent_output_tokens = 0

        output = original_fast_generate(self, prompt, max_tokens=max_tokens, stream=stream)

        # The fused runtime already tracks actual emitted token count. Fall back to
        # tokenizer.encode only if that runtime field is unavailable.
        try:
            generated = int(getattr(self, "live_generated_tokens", 0) or 0)
        except Exception:
            generated = 0
        if generated <= 0 and output:
            try:
                generated = len(self.engine.tokenizer.encode(output))
            except Exception:
                generated = 0
        self._recent_output_tokens = generated
        return output

    def evaluate_one_with_record(self, split_name, item):
        self._recent_prompt_tokens = 0
        self._recent_output_tokens = 0
        started_wall = time.time()
        started_perf = time.perf_counter()

        result = bool(original_evaluate_one(self, split_name, item))

        duration = max(0.0, time.perf_counter() - started_perf)
        prompt_tokens = int(getattr(self, "_recent_prompt_tokens", 0) or 0)
        output_tokens = int(getattr(self, "_recent_output_tokens", 0) or 0)
        record = {
            "timestamp": started_wall,
            "phase": str(getattr(self, "_recent_phase", "unknown")),
            "split": str(split_name),
            "item_id": str(item.get("id", "unknown")),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
            "correct": result,
            "tok_per_sec": float(getattr(self, "last_tok_per_sec", 0.0) or 0.0),
            "duration_seconds": duration,
        }
        _append_record(record)
        return result

    def evaluate_all_with_phase(self, splits, cache, phase, start, total):
        self._recent_phase = str(phase)
        return original_evaluate_all(self, splits, cache, phase, start, total)

    cls._fast_generate = fast_generate_with_token_accounting
    cls._evaluate_single_item = evaluate_one_with_record
    cls._evaluate_all_splits = evaluate_all_with_phase
