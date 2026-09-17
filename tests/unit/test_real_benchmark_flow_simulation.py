"""Offline control-flow simulation for the real benchmark adapters.

No model weights, network, Docker, hidden benchmark answers, or official verifiers
are used. It checks prompt/scorer/wrapper routing across baseline, RSI and Phase 4,
and proves flagship DeepSWE candidates are generated blind and verified only after
answer-blind selection.
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
from eval import scoring_hardening


class _Result:
    def __init__(self, passed=True):
        self.passed = passed


class _Sandbox:
    def execute_python_code(self, code, test):
        return _Result("SIM_FAIL" not in str(code) and "SIM_FAIL" not in str(test))


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


def _fresh_eval_class():
    class Eval:
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

    return Eval


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

        # Use the actual branch choice parser so SuperGPQA/MMLU-Pro A-J is covered.
        real_choice_scoring.install(scoring_hardening)

        phase4 = types.SimpleNamespace()
        phase4.SYSTEM_PROMPT = "BASE GEMINI + UNIVERSAL ANTI-LOOP"
        phase4._task_user_prompt = lambda split, item: item.get("prompt", "")
        phase4._candidate_passes_answer_blind = lambda self, split, item, candidate: None
        phase4._hidden_reward_only_after_selection = lambda self, split, item, candidate: False
        phase4._append_pro_metadata = lambda self: None

        provider = type("Provider", (), {"load_all_4000_items": lambda self: {}})
        Eval = _fresh_eval_class()
        real.install(provider, runtime, phase4, Eval)
        real_phase4_context.install(phase4, Eval)
        return runtime, phase4, Eval

    def test_all_real_kinds_baseline_rsi_phase4(self):
        runtime, phase4, Eval = self._installed()
        cases = [
            ("HumanEval-164", {"id":"HE","prompt":"HEMARK def f(): pass","test":"assert True","_real_kind":"humaneval"}, "```python\nreturn 1\n```"),
            ("LiveCodeBench-Hard", {"id":"LCB","prompt":"LCBMARK solve stdin","test":"assert True","_real_kind":"livecodebench"}, "```python\nprint(1)\n```"),
            ("GSM8K-500", {"id":"GSM","prompt":"GSMMARK math","expected":"42","_real_kind":"math"}, "\\boxed{42}"),
            ("MATH-500", {"id":"MATH","prompt":"MATHMARK problem","expected":"9","_real_kind":"math"}, "\\boxed{9}"),
            ("OlympiadBench-150", {"id":"OLY","prompt":"OLYMARK problem","expected":"17","_real_kind":"math"}, "\\boxed{17}"),
            ("SuperGPQA-400", {"id":"SG","prompt":"SGMARK\n(A) x\n(E) y","expected":"E","_real_kind":"choice"}, "E"),
            ("MMLU-Pro-1000", {"id":"MCQ","prompt":"MCQMARK\n(A) x\n(J) y","expected":"J","_real_kind":"choice"}, "J"),
            ("BFCL-200", {"id":"BF","prompt":"BFMARK tools","expected_tool":"f","expected_args":{"x":1},"_real_kind":"bfcl"}, '{"name":"f","arguments":{"x":1}}'),
            ("LogiQA-200", {"id":"LOG","prompt":"LOGMARK\n(A) x\n(B) y","expected":"B","_real_kind":"choice"}, "B"),
            ("LiveBench-Reasoning-100", {"id":"LB","prompt":"LBMARK reason","expected":"done","_real_kind":"freeform"}, "done"),
            ("CodeRepairShape", {"id":"CR","prompt":"CRMARK repair","buggy_code":"broken = True","test":"assert True","_real_kind":"code_repair"}, "fixed = True"),
        ]
        for split, item, output in cases:
            ev = Eval()
            ev.outputs = {item["prompt"].split()[0]: output}
            ev._current_phase = "Phase 1: Baseline"
            self.assertTrue(ev._evaluate_single_item(split, item), split)
            self.assertIn("BASE GEMINI + UNIVERSAL ANTI-LOOP", ev.prompts[-1], split)
            user = phase4._task_user_prompt(split, item)
            self.assertIn(item["prompt"], user, split)
            # RSI reward is checked only after a candidate already exists.
            self.assertTrue(phase4._hidden_reward_only_after_selection(ev, split, item, output), split)
            ev._current_phase = "Phase 4: Post-Consolidation"
            self.assertTrue(ev._evaluate_single_item(split, item), split)

    def test_swebench_verified_baseline_rsi_phase4_uses_verifier_hook(self):
        runtime, phase4, Eval = self._installed()
        ev = Eval()
        item = {
            "id":"SWE","instance_id":"repo__1","prompt":"SWEMARK ctx","prompt_27k":"SWEMARK ctx",
            "prompt_13k":"SWEMARK short","_real_kind":"swebench_verified",
        }
        ev.outputs = {"SWEMARK":"diff --git a/a b/a\n--- a/a\n+++ b/a\n"}
        calls = []
        with patch.object(swe_override, "_verify_official_swebench", side_effect=lambda i, c: calls.append((i,c)) or True):
            swe_override.install(real, runtime, phase4, Eval)
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

            def choose(self, split, item, branches):
                events.append(("select", split, "current"))
                return branches[0], 0, False, "answer-blind consensus"
            phase4._choose_without_ground_truth = choose

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

            with patch.object(deep, "DEEPSWE_TASK_COUNT", 2), \
                 patch.object(deep, "_ask_enabled", return_value=True), \
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

            self.assertEqual(sum(1 for e in events if e[0]=="generate" and e[1]=="baseline"), 2)
            self.assertEqual(sum(1 for e in events if e[0]=="verify" and e[1]=="baseline"), 2)
            # 4 branches per RSI round, then exactly one selected-patch verification.
            self.assertEqual(sum(1 for e in events if e[0]=="generate" and e[1].startswith("rsi-r1")), 8)
            self.assertEqual(sum(1 for e in events if e[0]=="verify" and e[1]=="rsi-r1-selected"), 2)
            # Existing test-backed Pro policy is N=16; still exactly one verifier call per task after selection.
            self.assertEqual(sum(1 for e in events if e[0]=="generate" and e[1].startswith("phase4-b")), 32)
            self.assertEqual(sum(1 for e in events if e[0]=="verify" and e[1]=="phase4-selected"), 2)
            for idx, event in enumerate(events):
                if event[0] == "verify" and ("rsi" in event[1] or "phase4" in event[1]):
                    self.assertTrue(any(e[0] == "select" for e in events[:idx]))

    def test_deepswe_context_policy(self):
        self.assertEqual(deep.DEEPSWE_CONTEXT_TOKENS, 226000)
        self.assertEqual(deep.DEEPSWE_MAX_OUTPUT_TOKENS, 8192)


if __name__ == "__main__":
    unittest.main()
