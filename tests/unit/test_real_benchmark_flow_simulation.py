"""Offline control-flow simulation for real benchmark adapters.

No model weights, network, Docker, benchmark answers, or official verifiers are
used. The goal is to prove prompt/scorer/wrapper routing across baseline, RSI and
Phase 4, including that flagship DeepSWE branches are generated with verification
disabled and verification is called only after answer-blind selection.
"""
from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from eval import deepswe_dataset_override as swe_override
from eval import deepswe_optional_flagship as deep
from eval import real_benchmark_runtime as real
from eval import real_choice_scoring
from eval import real_phase4_context


class _Result:
    def __init__(self, passed=True):
        self.passed = passed


class _Sandbox:
    def execute_python_code(self, code, test):
        return _Result("BAD" not in code and "BAD" not in test)


class _Tokenizer:
    def encode(self, text, *args, **kwargs):
        return str(text).split()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(f"{m['role']}::{m['content']}" for m in messages) + "\nassistant::"


class _KG:
    def __init__(self):
        self.rows = []

    def log_interaction(self, *args, **kwargs):
        self.rows.append((args, kwargs))


class _Engine:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.sandbox = _Sandbox()
        self.model = object()
        self.kg = _KG()


class _Eval:
    def __init__(self):
        self.engine = _Engine()
        self.outputs = {}
        self.prompts = []
        self._current_phase = "Phase 1: Baseline"

    def _fast_generate(self, prompt, max_tokens=32768, stream=False):
        self.prompts.append(prompt)
        for key, value in self.outputs.items():
            if key in prompt:
                return value
        return "OK"

    def _evaluate_single_item(self, split, item):
        raise AssertionError("fallback evaluator should not receive real items")


class RealFlowSimulation(unittest.TestCase):
    def _installed(self):
        runtime = types.SimpleNamespace()
        runtime._model_context_limit = lambda engine: 32768
        runtime._benchmark_ceiling = lambda self: 32768
        runtime.clean_output = lambda text: str(text).replace("```python", "").replace("```", "").strip()
        runtime._append_raw_generation_log = lambda *args, **kwargs: None
        runtime._chat = lambda tok, user, system: tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )

        scoring = types.SimpleNamespace(
            _boxed_values=lambda text: [],
            _last_nonempty_line=lambda text: [x for x in str(text).splitlines() if x.strip()][-1],
        )
        real_choice_scoring.install(scoring)

        phase4 = types.SimpleNamespace()
        phase4.SYSTEM_PROMPT = "BASE + UNIVERSAL ANTI-LOOP"
        phase4._task_user_prompt = lambda split, item: item.get("prompt", "")
        phase4._candidate_passes_answer_blind = lambda self, split, item, candidate: None
        phase4._hidden_reward_only_after_selection = lambda self, split, item, candidate: False
        phase4._append_pro_metadata = lambda self: None

        provider = type("Provider", (), {"load_all_4000_items": lambda self: {}})
        real.install(provider, runtime, phase4, _Eval)
        real_phase4_context.install(phase4, _Eval)
        return runtime, phase4

    def test_all_real_kinds_baseline_rsi_phase4(self):
        runtime, phase4 = self._installed()
        ev = _Eval()
        cases = [
            ("HumanEval-164", {"id":"HE","prompt":"HEMARK def f(): pass","test":"assert True","_real_kind":"humaneval"}, "```python\nreturn 1\n```", True),
            ("LiveCodeBench-Hard", {"id":"LCB","prompt":"LCBMARK solve stdin","test":"assert True","_real_kind":"livecodebench"}, "```python\nprint(1)\n```", True),
            ("GSM8K-500", {"id":"GSM","prompt":"GSMMARK math","expected":"42","_real_kind":"math"}, "\\boxed{42}", True),
            ("MMLU-Pro-1000", {"id":"MCQ","prompt":"MCQMARK\n(A) x\n(E) y","expected":"E","_real_kind":"choice"}, "E", True),
            ("BFCL-200", {"id":"BF","prompt":"BFMARK tools","expected_tool":"f","expected_args":{"x":1},"_real_kind":"bfcl"}, '{"name":"f","arguments":{"x":1}}', True),
            ("LiveBench-Reasoning-100", {"id":"LB","prompt":"LBMARK reason","expected":"done","_real_kind":"freeform"}, "done", True),
            ("CodeRepair", {"id":"CR","prompt":"CRMARK repair","buggy_code":"BAD","test":"assert True","_real_kind":"code_repair"}, "fixed = True", True),
        ]
        for split, item, output, expected in cases:
            ev.outputs = {item["prompt"].split()[0]: output}
            ev._current_phase = "Phase 1: Baseline"
            self.assertEqual(ev._evaluate_single_item(split, item), expected, split)
            self.assertIn("BASE + UNIVERSAL ANTI-LOOP", ev.prompts[-1], split)
            user = phase4._task_user_prompt(split, item)
            self.assertIn(item["prompt"], user, split)
            self.assertTrue(phase4._hidden_reward_only_after_selection(ev, split, item, output), split)
            ev._current_phase = "Phase 4: Post-Consolidation"
            self.assertEqual(ev._evaluate_single_item(split, item), expected, split)

    def test_swebench_verified_baseline_rsi_phase4_uses_verifier_hook(self):
        runtime, phase4 = self._installed()
        ev = _Eval()
        item = {
            "id":"SWE","instance_id":"repo__1","prompt":"SWEMARK ctx","prompt_27k":"SWEMARK ctx",
            "prompt_13k":"SWEMARK short","_real_kind":"swebench_verified",
        }
        ev.outputs = {"SWEMARK":"diff --git a/a b/a\n--- a/a\n+++ b/a\n"}
        calls = []
        with patch.object(swe_override, "_verify_official_swebench", side_effect=lambda i, c: calls.append((i,c)) or True):
            swe_override.install(real, runtime, phase4, _Eval)
            ev._current_phase = "Phase 1: Baseline"
            self.assertTrue(ev._evaluate_single_item("SWE-bench-Verified-50", item))
            self.assertEqual(len(calls), 1)
            self.assertTrue(phase4._hidden_reward_only_after_selection(ev, "SWE-bench-Verified-50", item, "patch"))
            self.assertEqual(len(calls), 2)
            ev._current_phase = "Phase 4: Post-Consolidation"
            self.assertTrue(ev._evaluate_single_item("SWE-bench-Verified-50", item))
            self.assertEqual(len(calls), 3)

    def test_flagship_deepswe_generation_is_blind_then_single_verify(self):
        events = []
        with tempfile.TemporaryDirectory() as td:
            tasks = []
            for name in ("task-a", "task-b"):
                p = Path(td) / name
                p.mkdir()
                (p / "instruction.md").write_text(f"Solve {name}")
                tasks.append(p)

            phase4 = types.SimpleNamespace()
            phase4.RSI_SESSION_ID = "rsi"
            phase4._run_rsi_self_improvement = lambda self, splits, cache: 0
            phase4._normalized_entropy = lambda self, prompt: 0.8
            phase4._pro_router = lambda self: types.SimpleNamespace(route=lambda entropy, has_test_cases: ("Pro-RLVR (N=16)", 16))
            phase4.get_ladder_temperatures = lambda n: [0.2 + i * 0.01 for i in range(n)]
            phase4._choose_without_ground_truth = lambda self, split, item, branches: (branches[0], 0, False, "blind consensus")

            class C:
                def __init__(self):
                    self.engine = _Engine()
                def run_full_suite(self):
                    cache = {}
                    self._evaluate_all_splits({}, cache, "Phase 1: Baseline", 0, 0)
                    phase4._run_rsi_self_improvement(self, {}, cache)
                    self._evaluate_all_splits({}, cache, "Phase 4: Post-Consolidation", 0, 0)
                def _evaluate_all_splits(self, splits, cache, phase, start, total):
                    return {}

            prompt = types.SimpleNamespace(GLOBAL_SYSTEM_SUFFIX="UNIVERSAL")
            runtime = types.SimpleNamespace(_model_context_limit=lambda engine: 1000000)

            class Bridge:
                def __init__(self, *args): pass
                def start(self): return "http://local/v1"
                def stop(self): pass

            fresh_state = {"benchmark_commit":deep.DEEPSWE_PIN,"baseline":{},"rsi":{},"final":{}}
            def fake_generate(pier, task, base, universal, stage, temperature, prior_patch=""):
                events.append(("generate", stage, task.name))
                return f"patch-{stage}-{task.name}"
            def fake_verify(pier, task, patch_text, stage):
                events.append(("verify", stage, task.name))
                return stage != "baseline"

            with patch.object(deep, "_ask_enabled", return_value=True), \
                 patch.object(deep, "_ensure_tools", return_value="pier"), \
                 patch.object(deep, "_ensure_official_repo", return_value=(Path(td), tasks)), \
                 patch.object(deep, "_load_state", return_value=fresh_state), \
                 patch.object(deep, "_save_state", return_value=None), \
                 patch.object(deep, "_LocalModelBridge", Bridge), \
                 patch.object(deep, "_generate_patch", side_effect=fake_generate), \
                 patch.object(deep, "_verify_selected_patch", side_effect=fake_verify):
                deep.install(runtime, phase4, prompt, C)
                obj = C()
                obj.run_full_suite()

            # 2 baseline generations + one verify each.
            self.assertEqual(sum(1 for e in events if e[0]=="generate" and e[1]=="baseline"), 2)
            self.assertEqual(sum(1 for e in events if e[0]=="verify" and e[1]=="baseline"), 2)
            # Baseline misses: RSI round 1 has four blind branches per task; selected patch verifies once.
            self.assertEqual(sum(1 for e in events if e[0]=="generate" and e[1].startswith("rsi-r1")), 8)
            self.assertEqual(sum(1 for e in events if e[0]=="verify" and e[1]=="rsi-r1-selected"), 2)
            # Phase-4 test-backed route forces existing N=16 Pro policy, verifier once after selection.
            self.assertEqual(sum(1 for e in events if e[0]=="generate" and e[1].startswith("phase4-b")), 32)
            self.assertEqual(sum(1 for e in events if e[0]=="verify" and e[1]=="phase4-selected"), 2)
            for idx, event in enumerate(events):
                if event[0] == "verify":
                    # A selected verification can only happen after at least one generation for that task/stage.
                    self.assertTrue(any(e[0]=="generate" and e[2]==event[2] for e in events[:idx]))

    def test_deepswe_context_policy(self):
        self.assertEqual(deep.DEEPSWE_CONTEXT_TOKENS, 226000)
        self.assertEqual(deep.DEEPSWE_MAX_OUTPUT_TOKENS, 8192)


if __name__ == "__main__":
    unittest.main()
