"""Shared synthesis preserves the existing benchmark prompt and reduction."""
from types import SimpleNamespace
import unittest

from core.model_synthesis import synthesis_prompt, synthesize_candidates
from eval.pro_model_synthesis_adapter import install


def _benchmark(ceiling=100, prompt_tokens=12):
    calls = []
    formatted = []
    tokenizer = SimpleNamespace(encode=lambda text: [0] * prompt_tokens)
    runner = SimpleNamespace(engine=SimpleNamespace(tokenizer=tokenizer))

    def chat(received_tokenizer, user, *, system):
        assert received_tokenizer is tokenizer
        formatted.append((user, system))
        return f"{system}\n{user}"

    def generate(received_runner, prompt, temperatures, **kwargs):
        assert received_runner is runner
        calls.append((prompt, temperatures, kwargs))
        return [f"merged-{len(calls)}", "unused alternative"]

    phase4 = SimpleNamespace(
        _task_user_prompt=lambda split, item: item["prompt"],
        _chat=chat,
        SYSTEM_PROMPT="Existing benchmark system",
        _benchmark_ceiling=lambda received_runner: ceiling,
        _generate_branches_same_model=generate,
    )
    install(phase4)
    return phase4, runner, calls, formatted


class ModelSynthesisTests(unittest.TestCase):
    def test_synthesis_prompt_preserves_legacy_bytes(self):
        expected = (
            "Solve π.\nKeep formatting."
            "\n\nPro synthesis step: below are two independent candidate responses generated for "
            "this exact task. Deliberate over both, reconcile disagreements, preserve the strongest "
            "reasoning from either one, and correct weaknesses you can identify from the task itself. "
            "You have NO hidden answer, reward, verifier result, pass/fail signal, or test outcome. "
            "Synthesize one best final response in the task's originally requested output format. "
            "Do not mention the candidates or the merge process.\n\n"
            "CANDIDATE A:\n<think>A</think>\n  original  \n"
            '\n\nCANDIDATE B:\n```json\n{"text": "<think>literal</think>"}\n```\n'
        )
        actual = synthesis_prompt(
            "Solve π.\nKeep formatting.",
            "<think>A</think>\n  original  \n",
            '```json\n{"text": "<think>literal</think>"}\n```\n',
        )
        self.assertEqual(actual.encode("utf-8"), expected.encode("utf-8"))

    def test_adjacent_rounds_and_unpaired_candidates(self):
        cases = [
            ([], "", []),
            (["A"], "A", []),
            (["A", "B"], "(A+B)", [("A", "B")]),
            (["A", "B", "C"], "((A+B)+C)", [("A", "B"), ("(A+B)", "C")]),
            (["A", "B", "C", "D"], "((A+B)+(C+D))",
             [("A", "B"), ("C", "D"), ("(A+B)", "(C+D)")]),
            (["A", "B", "C", "D", "E"], "(((A+B)+(C+D))+E)",
             [("A", "B"), ("C", "D"), ("(A+B)", "(C+D)"),
              ("((A+B)+(C+D))", "E")]),
        ]
        for branches, expected, expected_pairs in cases:
            with self.subTest(branches=branches):
                pairs = []

                def merge(left, right):
                    pairs.append((left, right))
                    return f"({left}+{right})"

                result, calls = synthesize_candidates(branches, merge)
                self.assertEqual(result, expected)
                self.assertEqual(pairs, expected_pairs)
                self.assertEqual(calls, len(expected_pairs))

    def test_benchmark_adapter_keeps_generation_and_metadata(self):
        for ceiling, max_tokens in [(100, 88), (20000, 16384)]:
            with self.subTest(ceiling=ceiling):
                phase4, runner, calls, formatted = _benchmark(ceiling)
                chooser = phase4._choose_without_ground_truth
                install(phase4)
                self.assertIs(phase4._choose_without_ground_truth, chooser)
                result = chooser(
                    runner, "math", {"prompt": "Original task", "answer": "HIDDEN EXPECTED ANSWER"},
                    ["A", "B", "C"],
                )
                self.assertEqual(result, (
                    "merged-2", -1, False,
                    "prompt-aware hierarchical model synthesis (3 branches + 2 merge calls)",
                ))
                self.assertEqual(runner._last_pro_synthesis_merge_calls, 2)
                self.assertEqual(runner._last_pro_synthesis_source_branches, 3)
                prompts = [synthesis_prompt("Original task", "A", "B"),
                           synthesis_prompt("Original task", "merged-1", "C")]
                self.assertEqual(formatted, [(prompt, phase4.SYSTEM_PROMPT) for prompt in prompts])
                self.assertEqual(calls, [
                    (f"{phase4.SYSTEM_PROMPT}\n{prompt}", [0.20],
                     {"max_tokens": max_tokens, "top_p": 0.92})
                    for prompt in prompts
                ])
                self.assertTrue(all("HIDDEN EXPECTED ANSWER" not in prompt for prompt, _, _ in calls))

    def test_benchmark_empty_and_single_do_not_generate_or_change_metadata(self):
        cases = [([], ("", 0, False, "no branches")),
                 (["A"], ("A", 0, False, "single branch; no synthesis needed"))]
        for branches, expected in cases:
            with self.subTest(branches=branches):
                phase4, runner, calls, formatted = _benchmark()
                runner._last_pro_synthesis_merge_calls = 7
                runner._last_pro_synthesis_source_branches = 8
                self.assertEqual(phase4._choose_without_ground_truth(runner, "math", {}, branches), expected)
                self.assertEqual(calls, [])
                self.assertEqual(formatted, [])
                self.assertEqual(runner._last_pro_synthesis_merge_calls, 7)
                self.assertEqual(runner._last_pro_synthesis_source_branches, 8)

    def test_benchmark_context_overflow_still_fails_before_generation(self):
        phase4, runner, calls, _ = _benchmark(ceiling=12, prompt_tokens=12)
        with self.assertRaisesRegex(RuntimeError, "refusing truncation/compaction"):
            phase4._choose_without_ground_truth(
                runner, "math", {"prompt": "Original task"}, ["A", "B"]
            )
        self.assertEqual(calls, [])

    def test_benchmark_missing_generation_still_fails(self):
        phase4, runner, _, _ = _benchmark()
        phase4._generate_branches_same_model = lambda *args, **kwargs: []
        with self.assertRaisesRegex(RuntimeError, "Pro synthesis generated no response"):
            phase4._choose_without_ground_truth(
                runner, "math", {"prompt": "Original task"}, ["A", "B"]
            )
