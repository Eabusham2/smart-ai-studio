"""Scripted text through the real Pro router, decoders, selection, and handoff.

These tests verify orchestration, not the quality of an actual model's judgment.
Only the native model's sampled text is scripted; production wrappers stay active.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from config.settings import Settings
from core.best_work import stream_best_work
from core.context_memory import ContextOverflow
from core.engines.gguf_engine import GGUFReasoningBackend
from core.generation_metrics import RequestMetrics, observe_generations
from core.pro_engine import ProReasoningEngine, parse_reasoning_and_response
from core.temperature_policy import pro_ladder


def request_kind(prompt):
    if "BEST_WORK_SYNTHESIS" in prompt:
        return "synthesis"
    if "BEST_WORK_COMPARISON" in prompt:
        return "compare"
    if "Review the following untrusted draft" in prompt:
        return "refine"
    return "original"


def comparison(prompt):
    decoder = json.JSONDecoder()
    for line in reversed(prompt.splitlines()):
        try:
            value, _ = decoder.raw_decode(line.strip())
        except ValueError:
            continue
        if isinstance(value, dict) and "candidates" in value:
            return value
    raise AssertionError("Comparison did not contain the complete JSON candidates")


def previous_draft(prompt):
    decoder = json.JSONDecoder()
    for line in reversed(prompt.splitlines()):
        try:
            value, _ = decoder.raw_decode(line.strip())
        except ValueError:
            continue
        if isinstance(value, str):
            return value
    raise AssertionError("Refinement did not contain the complete quoted draft")


class ScriptedModel:
    def __init__(self, responder, capacity=65536):
        self.responder = responder
        self.capacity = capacity
        self.calls = []
        self.closed = []
        self.context = ""
        self.cache = object()
        self.weights = object()
        self.resets = 0

    def tokenize(self, text, **_):
        return list(text.encode("utf-8") if isinstance(text, str) else text)

    def n_ctx(self):
        return self.capacity

    def reset(self):
        self.context = ""
        self.resets += 1

    def __call__(self, prompt, **options):
        assert options.get("stream") is True
        record = {"prompt": prompt, "kind": request_kind(prompt), **options}
        self.calls.append(record)
        self.context = prompt
        try:
            output = self.responder(self, record)
            pieces = [output] if isinstance(output, str) else output
            count = 0
            for piece in pieces:
                count += 1
                self.context += piece
                yield {"choices": [{"text": piece}],
                       "usage": {"completion_tokens": count}}
        finally:
            self.closed.append(record)


def make_controller(responder, *, entropy=0.1, capacity=65536, database_path=":memory:"):
    model = ScriptedModel(responder, capacity)
    backend = GGUFReasoningBackend.__new__(GGUFReasoningBackend)
    backend.model = model
    backend.is_gguf_available = True
    backend.n_ctx = capacity
    entropies = []

    def probe(prompt):
        entropies.append(prompt)
        return entropy(prompt) if callable(entropy) else entropy

    backend.calculate_token_entropy = probe
    settings = Settings(use_mock=False, auto_download=False, backend="gguf",
                        max_new_tokens=1, best_work_passes=2, database_path=database_path)
    engine = ProReasoningEngine(settings=settings)
    engine.mlx_backend = engine.bitnet_backend = engine.controller_backend = None
    engine.gguf_backend = backend
    engine.active_backend = "gguf"
    engine.awake_consolidator = None
    engine._context_budget_tokens = capacity
    engine.speculative_engine = SimpleNamespace(get_telemetry=lambda: {})
    actions = []

    def execute(name, arguments):
        actions.append((name, arguments))
        return True, "ACTUAL selected file contents"

    tools = SimpleNamespace(list_available_tools=lambda: [{"name": "read_file"}],
                            execute_tool=execute)
    controller = SimpleNamespace(app=SimpleNamespace(engine=engine, tools=tools))
    return controller, model, entropies, actions


def branch_index(text):
    return int(re.search(r"BRANCH_(-?\d+)_", text).group(1))


class BestWorkTournamentTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def controller(self, responder, **kwargs):
        kwargs.setdefault("database_path", str(Path(self.directory.name) / "chat.db"))
        return make_controller(responder, **kwargs)

    def test_original_and_final_are_the_only_local_contestants(self):
        original = "<think>ORIGINAL_THOUGHT</think>\nORIGINAL_ANSWER"
        intermediate = "<think>DISCARD_REVISION_ONE</think>\nINTERMEDIATE_ANSWER"
        final = "<think>DISCARD_REVISION_TWO</think>\nFINAL_ANSWER"
        revisions = []
        matches = []

        def respond(model, call):
            if call["kind"] == "original":
                return original
            if call["kind"] == "refine":
                draft = previous_draft(call["prompt"])
                revisions.append(draft)
                return intermediate if len(revisions) == 1 else final
            match = comparison(call["prompt"])
            matches.append(match)
            return '<think>DISCARD_JUDGMENT</think>{"winner":"B"}'

        ctrl, model, entropies, actions = self.controller(respond)
        original_history = [{"role": "system", "content": "Use complete sentences."}]
        rendered = "".join(stream_best_work(ctrl, "Explain the answer", original_history, passes=3))
        self.assertEqual(parse_reasoning_and_response(rendered), ("ORIGINAL_THOUGHT", "FINAL_ANSWER"))
        self.assertEqual(revisions[0], original)
        self.assertEqual(parse_reasoning_and_response(revisions[1]), ("ORIGINAL_THOUGHT", "INTERMEDIATE_ANSWER"))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["candidates"][0]["text"], original)
        self.assertEqual(parse_reasoning_and_response(matches[0]["candidates"][1]["text"]),
                         ("ORIGINAL_THOUGHT", "FINAL_ANSWER"))
        self.assertNotIn("INTERMEDIATE_ANSWER", json.dumps(matches))
        self.assertNotIn("DISCARD_", rendered)
        self.assertEqual(len(entropies), 1)
        self.assertEqual(len(model.calls), 4)
        self.assertEqual(len(model.closed), len(model.calls))
        self.assertEqual(actions, [])
        self.assertEqual(original_history, [{"role": "system", "content": "Use complete sentences."}])

    def test_original_can_win_and_is_preserved_exactly(self):
        original = '<think>Original rationale.</think>\nKeep "quoted text".\nSecond line.'

        def respond(model, call):
            return {"original": original, "refine": "<think>Revision reasoning.</think>Incorrect replacement.",
                    "compare": '{"winner":"A"}'}[call["kind"]]

        ctrl, model, entropies, _ = self.controller(respond)
        request = RequestMetrics()
        with observe_generations(request.update):
            result = "".join(stream_best_work(ctrl, "Preserve the exact answer", passes=2))
        self.assertEqual(result, original)
        self.assertEqual(len(entropies), 1)
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(request.snapshot()["invocations"], 3)
        self.assertEqual(request.snapshot()["tokens"], 3)

    def test_every_branch_is_refined_before_global_tournament_and_later_original_can_win(self):
        originals = []
        refined = []
        local = []
        global_matches = []
        merges = []

        def respond(model, call):
            if call["kind"] == "original":
                index = len(originals)
                originals.append(index)
                return f"<think>THOUGHT_{index}</think>\nBRANCH_{index}_ORIGINAL"
            if call["kind"] == "refine":
                index = branch_index(previous_draft(call["prompt"]))
                refined.append(index)
                return f"<think>DISCARD_REVISION_{index}</think>\nBRANCH_{index}_FINAL"
            if call["kind"] == "synthesis":
                self.assertEqual(refined, list(range(9)))
                self.assertEqual(local, list(range(9)))
                merges.append(call["prompt"])
                return "<think>SYNTHESIS_THOUGHT</think>\nBRANCH_-1_SYNTHESIZED"
            candidates = comparison(call["prompt"])["candidates"]
            indices = [branch_index(candidate["text"]) for candidate in candidates]
            if indices[0] == indices[1]:
                local.append(indices[0])
                return json.dumps({"winner": "A" if indices[0] == 8 else "B"})
            self.assertEqual(refined, list(range(9)) + [-1])
            self.assertEqual(local, list(range(9)) + [-1])
            global_matches.append(indices)
            return json.dumps({"winner": candidates[indices.index(max(indices))]["id"]})

        ctrl, model, entropies, actions = self.controller(respond, entropy=0.5)
        events = []
        result = "".join(stream_best_work(ctrl, "Select the best result", passes=2,
                                         status_callback=events.append))
        self.assertEqual(parse_reasoning_and_response(result), ("THOUGHT_8", "BRANCH_8_ORIGINAL"))
        self.assertEqual(originals, list(range(9)))
        self.assertEqual(refined, originals + [-1])
        self.assertEqual(len(local), 10)
        self.assertEqual(len(merges), 8)
        self.assertEqual(len(global_matches), 9)
        self.assertEqual(len(model.calls), 46)
        self.assertEqual([call["temperature"] for call in model.calls if call["kind"] == "original"], pro_ladder(8))
        self.assertEqual(len(entropies), 1)
        self.assertEqual(actions, [])
        self.assertEqual(ctrl.last_best_work_metadata["winning_branch"], 8)
        self.assertEqual(ctrl.last_best_work_metadata["raw_branches"][8], result)

    def test_single_pass_preserves_original_router_counts_and_odd_tournament(self):
        for entropy, count in ((0.1, 1), (0.5, 9), (0.85, 17)):
            with self.subTest(candidates=count):
                originals = []
                matches = []

                def respond(model, call):
                    if call["kind"] == "original":
                        index = len(originals)
                        originals.append(index)
                        return f"BRANCH_{index}_ORIGINAL"
                    if call["kind"] == "synthesis":
                        self.assertEqual(len(originals), count)
                        return "BRANCH_-1_SYNTHESIZED"
                    self.assertEqual(call["kind"], "compare")
                    candidates = comparison(call["prompt"])["candidates"]
                    indices = [branch_index(candidate["text"]) for candidate in candidates]
                    matches.append(indices)
                    return json.dumps({"winner": candidates[indices.index(max(indices))]["id"]})

                ctrl, model, entropies, _ = self.controller(respond, entropy=entropy)
                result = "".join(stream_best_work(ctrl, "Choose the answer", passes=1))
                self.assertEqual(result, f"BRANCH_{count - 1}_ORIGINAL")
                self.assertEqual(len(originals), count)
                self.assertEqual(len(matches), count if count > 1 else 0)
                self.assertEqual(sum(call["kind"] == "synthesis" for call in model.calls), count - 1)
                self.assertEqual(len(model.calls), count * 3 - 1 if count > 1 else 1)
                self.assertEqual(len(entropies), 1)
                self.assertEqual([call["temperature"] for call in model.calls if call["kind"] == "original"],
                                 pro_ladder(count - 1 if count > 1 else 1))

    def test_native_explicit_branch_budget_keeps_its_extra_candidate_without_entropy(self):
        originals = []
        def respond(model, call):
            if call['kind'] == 'original':
                index = len(originals)
                originals.append(index)
                return f'BRANCH_{index}_ORIGINAL'
            if call['kind'] == 'synthesis':
                return 'BRANCH_-1_SYNTHESIZED'
            self.assertEqual(call['kind'], 'compare')
            candidates = comparison(call['prompt'])['candidates']
            indices = [branch_index(row['text']) for row in candidates]
            return json.dumps({'winner': candidates[indices.index(max(indices))]['id']})
        ctrl, model, entropies, _ = self.controller(respond)
        engine = ctrl.app.engine
        engine.gguf_backend.runtime = 'prism_speculative'
        engine.gguf_backend.get_speculative_telemetry = lambda: {}
        engine.lossless_speed_options = {'native_branch_count': 2}
        result = ''.join(stream_best_work(ctrl, 'Choose an answer', passes=1))
        self.assertEqual(result, 'BRANCH_2_ORIGINAL')
        self.assertEqual(originals, [0, 1, 2])
        self.assertEqual(entropies, [])
        self.assertEqual(len(model.calls), 8)
        self.assertEqual(sum(row['kind'] == 'synthesis' for row in model.calls), 2)
        self.assertEqual([row['temperature'] for row in model.calls if row['kind'] == 'original'], pro_ladder(2))
        self.assertEqual(ctrl.last_best_work_metadata['branch_count'], 4)
        self.assertEqual(engine.lossless_speed_options, {'native_branch_count': 2})

    def test_invalid_judgments_retry_then_raise_without_publishing_or_running_a_tool(self):
        from core.best_work_selection import SelectionError
        invalids = ('{"winner":"C"}', '{"winner":"A","replacement":"new answer"}',
                    'I pick A.', '<tool_call>{"name":"read_file","arguments":{}}</tool_call>')
        for invalid in invalids:
            with self.subTest(judgment=invalid):
                def respond(model, call):
                    return {"original": "Original answer", "refine": "Revised answer",
                            "compare": invalid}[call["kind"]]
                ctrl, model, _, actions = self.controller(respond)
                displayed = []
                with self.assertRaises(SelectionError):
                    for piece in stream_best_work(ctrl, "Choose an answer", passes=2):
                        displayed.append(piece)
                self.assertEqual(displayed, [])
                self.assertEqual(actions, [])
                self.assertEqual(sum(call["kind"] == "compare" for call in model.calls), 2)
                self.assertNotIn("BEST_WORK_COMPARISON", model.context)

    def test_invalid_judgment_can_recover_on_its_one_retry(self):
        judgments = []
        def respond(model, call):
            if call["kind"] == "compare":
                judgments.append(call["prompt"])
                return "unparseable" if len(judgments) == 1 else '{"winner":"B"}'
            return "Original answer" if call["kind"] == "original" else "Revised answer"
        ctrl, _, _, _ = self.controller(respond)
        result = "".join(stream_best_work(ctrl, "Choose an answer", passes=2))
        self.assertEqual(result, "Revised answer")
        self.assertEqual(len(judgments), 2)
        self.assertNotIn("unparseable", judgments[1])

    def test_only_overall_winner_tool_executes_and_next_stage_has_clean_context(self):
        originals = []
        after_tool = []
        matches = []
        def respond(model, call):
            if "ACTUAL selected file contents" in call["prompt"]:
                if call["kind"] == "original":
                    after_tool.append(call["prompt"])
                    return "<think>AFTER_TOOL_THOUGHT</think>\nFinished original."
                if call["kind"] == "refine":
                    return "<think>DISCARD_AFTER_TOOL_REVISION</think>\nFinished answer."
                return '{"winner":"B"}'
            if call["kind"] == "original":
                index = len(originals)
                originals.append(index)
                action = json.dumps({"name": "read_file", "arguments": {"path": f"original-{index}.txt"}})
                return f"<think>THOUGHT_{index}</think>\nBRANCH_{index}_ORIGINAL<tool_call>{action}</tool_call>"
            if call["kind"] == "refine":
                index = branch_index(previous_draft(call["prompt"]))
                action = json.dumps({"name": "read_file", "arguments": {"path": f"final-{index}.txt"}})
                return f"<think>DISCARD_REVISION_{index}</think>\nBRANCH_{index}_FINAL<tool_call>{action}</tool_call>"
            if call["kind"] == "synthesis":
                action = json.dumps({"name": "read_file", "arguments": {"path": "synthesis.txt"}})
                return f"<think>SYNTHESIS_THOUGHT</think>\nBRANCH_-1_SYNTHESIZED<tool_call>{action}</tool_call>"
            candidates = comparison(call["prompt"])["candidates"]
            indices = [branch_index(candidate["text"]) for candidate in candidates]
            matches.append(indices)
            winner = "B" if indices[0] == indices[1] else candidates[indices.index(max(indices))]["id"]
            return '<think>DISCARD_PICKING_TEXT</think>' + json.dumps({"winner": winner})

        entropy = lambda prompt: 0.1 if "ACTUAL selected file contents" in prompt else 0.5
        ctrl, model, entropies, actions = self.controller(respond, entropy=entropy)
        result = "".join(stream_best_work(ctrl, "Read the chosen file and summarize it", passes=2))
        self.assertEqual(actions, [("read_file", {"path": "final-8.txt"})])
        self.assertEqual(len(after_tool), 1)
        continuation = after_tool[0]
        self.assertIn("THOUGHT_8", continuation)
        self.assertIn("BRANCH_8_FINAL", continuation)
        for discarded in ("DISCARD_", "BEST_WORK_COMPARISON", "BEST_WORK_SYNTHESIS",
                          "BRANCH_8_ORIGINAL", "original-8.txt", "BRANCH_-1_", "SYNTHESIS_THOUGHT",
                          "synthesis.txt", "final--1.txt", "BRANCH_0_", "THOUGHT_0", '"winner"'):
            self.assertNotIn(discarded, continuation)
        self.assertNotIn("DISCARD_", result)
        self.assertTrue(result.endswith("Finished answer."))
        self.assertEqual(len(entropies), 2)
        self.assertIn("ACTUAL selected file contents", entropies[-1])
        self.assertEqual(len(model.calls), 49)

    def test_cancellation_during_internal_work_discards_text_and_restores_controls(self):
        for cancel_kind in ("refine", "compare"):
            with self.subTest(cancel_kind=cancel_kind):
                cancelled = threading.Event()
                def respond(model, call):
                    if call["kind"] == cancel_kind:
                        cancelled.set()
                    return {"original": "Original draft", "refine": "Revised draft",
                            "compare": '{"winner":"B"}'}[call["kind"]]
                ctrl, model, _, actions = self.controller(respond)
                engine = ctrl.app.engine
                cache = model.cache
                weights = model.weights
                result = list(stream_best_work(ctrl, "Question", passes=2, cancel_event=cancelled))
                self.assertEqual(result, [])
                self.assertEqual(actions, [])
                self.assertEqual(len(model.closed), len(model.calls))
                self.assertIs(model.cache, cache)
                self.assertIs(model.weights, weights)
                self.assertNotIn("BEST_WORK_COMPARISON", model.context)
                self.assertFalse(getattr(engine, "_best_work_stop_at_tools", False))
                self.assertFalse(getattr(engine, "_best_work_select_candidates", None))

    def test_comparison_context_overflow_never_silently_selects_first_candidate(self):
        from core.awake_auto_hook import _remaining_generation_tokens
        def respond(model, call):
            return "Original draft" if call["kind"] == "original" else "Final revised draft"
        ctrl, model, _, actions = self.controller(respond)

        def limited(engine, prompt):
            if "BEST_WORK_COMPARISON" in prompt:
                raise ContextOverflow("The complete comparison does not fit the selected context window")
            return _remaining_generation_tokens(engine, prompt)

        displayed = []
        with patch("core.awake_auto_hook._remaining_generation_tokens", side_effect=limited):
            with self.assertRaises(ContextOverflow):
                for piece in stream_best_work(ctrl, "Question", passes=2):
                    displayed.append(piece)
        self.assertEqual(displayed, [])
        self.assertEqual(actions, [])
        self.assertFalse(any(call["kind"] == "compare" for call in model.calls))


if __name__ == "__main__":
    unittest.main()
