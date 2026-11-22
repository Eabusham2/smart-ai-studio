"""Reasoning stays disposable without modifying literal tool arguments."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from core.best_work import split_selected_action, stream_best_work
from core.best_work_selection import SelectionError, _complete_original_prefix, retain_original_thinking
from core.gui_thinking import _PAIRS
from tests.unit.test_best_work_tournament import comparison, make_controller, previous_draft


class BestWorkReasoningIsolationTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def controller(self, respond):
        return make_controller(respond, database_path=str(Path(self.directory.name) / "chat.db"))

    def implicit_template(self, model):
        def template(messages, **_kwargs):
            return "\n".join(str(turn["content"]) for turn in messages) + "\n<think>"
        model.apply_chat_template = template

    def test_explicit_reasoning_tool_examples_are_inert_in_all_supported_channels(self):
        example = '<tool_call>{"name":"read_file","arguments":{"path":"example-only"}}</tool_call>'
        for opening, closing in _PAIRS:
            with self.subTest(opening=opening):
                def respond(_model, call):
                    if call["kind"] == "original":
                        return "<think>Original rationale.</think>Original answer."
                    if call["kind"] == "refine":
                        return [opening, "Unexecuted example: ", example, closing, "Final revised answer."]
                    return '{"winner":"B"}'

                ctrl, model, _, actions = self.controller(respond)
                answer = "".join(stream_best_work(ctrl, "Explain the answer", passes=2))
                self.assertEqual(answer, "<think>Original rationale.</think>Final revised answer.")
                self.assertEqual(actions, [])
                self.assertEqual(len(model.calls), 3)
                self.assertNotIn("example-only", comparison(model.calls[-1]["prompt"])["candidates"][1]["text"])

    def test_literal_thinking_markup_in_chosen_tool_argument_is_unchanged(self):
        content = "Keep <think>literal text</think> and </analysis> exactly."
        invocation = '<tool_call>' + json.dumps({
            "name": "write_file", "arguments": {"path": "notes.txt", "content": content},
        }) + '</tool_call>'
        matched = []

        def respond(_model, call):
            if call["kind"] == "compare":
                matched.append(comparison(call["prompt"]))
                return '{"winner":"B"}'
            if "ACTUAL selected file contents" in call["prompt"]:
                return "<think>New original rationale.</think>Saved." if call["kind"] == "original" else "Saved."
            if call["kind"] == "original":
                return "<think>Original rationale.</think>Original answer."
            return "<analysis>Discard revision rationale.</analysis>" + invocation

        ctrl, _, _, actions = self.controller(respond)
        ctrl.app.tools.list_available_tools = lambda: [{"name": "write_file"}]
        result = "".join(stream_best_work(ctrl, "Write the exact content to notes.txt", passes=2))
        self.assertEqual(actions, [("write_file", {"path": "notes.txt", "content": content})])
        revised = matched[0]["candidates"][1]["text"]
        self.assertEqual(revised, "<think>Original rationale.</think>" + invocation)
        self.assertEqual(split_selected_action(revised)[1][2]["content"], content)
        self.assertNotIn("Discard revision rationale", result)

    def test_original_tool_argument_does_not_become_retained_original_thinking(self):
        literal = '<tool_call>{"name":"read_file","arguments":{"path":"<think>literal</think>"}}</tool_call>'
        result = retain_original_thinking("<think>Actual original.</think>" + literal,
                                          "<think>Discard revision.</think>Final answer.")
        self.assertEqual(result, "<think>Actual original.</think>Final answer.")

    def test_quoted_answer_markup_survives_revisions_and_selection_exactly(self):
        thought = "<think>Original reasoning.</think>"
        for opening, closing in _PAIRS:
            for kind in ("json", "double_quote", "single_quote"):
                with self.subTest(opening=opening, kind=kind):
                    def answer(label):
                        literal = opening + label + closing
                        if kind == "json":
                            return json.dumps({"html": 'Keep "' + literal + '" and ``` literally.'})
                        if kind == "double_quote":
                            return 'Keep the exact literal "' + literal + '".'
                        return "Don't change 'the user's " + literal + "'."

                    original = thought + answer("ORIGINAL_LITERAL")
                    intermediate = thought + answer("INTERMEDIATE_LITERAL")
                    final = thought + answer("FINAL_LITERAL")
                    drafts, compared = [], []

                    def respond(_model, call):
                        if call["kind"] == "original":
                            return original
                        if call["kind"] == "refine":
                            drafts.append(previous_draft(call["prompt"]))
                            label = "INTERMEDIATE_LITERAL" if len(drafts) == 1 else "FINAL_LITERAL"
                            return '<think>DISCARD_REVISION; literal "```".</think>' + answer(label)
                        compared.append(comparison(call["prompt"]))
                        return '<analysis>DISCARD_JUDGMENT</analysis>{"winner":"B"}'

                    ctrl, model, _, actions = self.controller(respond)
                    result = "".join(stream_best_work(ctrl, "Keep the literal text intact", passes=3))
                    self.assertEqual(result, final)
                    self.assertEqual(drafts, [original, intermediate])
                    self.assertEqual([row["text"] for row in compared[0]["candidates"]], [original, final])
                    self.assertEqual(actions, [])
                    self.assertEqual(model.context, "")

    def test_implicit_reasoning_ignores_quoted_closing_markers(self):
        for _opening, closing in _PAIRS:
            with self.subTest(closing=closing):
                sampled = 'The literal "' + closing + '" and "```" matter.</think>Answer.'
                canonical = _complete_original_prefix(sampled, starts_in_thought=True)
                self.assertEqual(canonical, "<think>" + sampled)
                self.assertEqual(retain_original_thinking(
                    canonical, "Don't keep revision reasoning.</think>Revised answer.",
                    starts_in_thought=True), canonical.removesuffix("Answer.") + "Revised answer.")
                with self.assertRaisesRegex(SelectionError, "without an answer"):
                    retain_original_thinking(canonical, 'Unclosed reasoning with "' + closing + '".',
                                             starts_in_thought=True)

    def test_quotes_inside_multiline_code_do_not_change_reasoning_boundaries(self):
        for fence in ("`", "```", "~~~"):
            with self.subTest(fence=fence):
                revision = '<think>Use ' + fence + 'a\n"b' + fence + ' carefully.</think>Answer "c".'
                self.assertEqual(retain_original_thinking("<think>Original.</think>Old answer.", revision),
                                 '<think>Original.</think>Answer "c".')

    def test_template_supplied_initial_marker_is_restored_without_rewriting_content(self):
        for opening, closing in _PAIRS:
            with self.subTest(opening=opening):
                sampled = 'Keep `</analysis>` literal. Original thought.' + closing + '\nExact answer.'
                canonical = _complete_original_prefix(sampled, starts_in_thought=True)
                self.assertEqual(canonical, opening + sampled)
                self.assertEqual(_complete_original_prefix(canonical, starts_in_thought=True), canonical)
                self.assertEqual(_complete_original_prefix(sampled), sampled)

    def test_implicit_original_winner_keeps_complete_thinking_after_tool_handoff(self):
        action = '<tool_call>' + json.dumps({
            'name': 'read_file', 'arguments': {'path': '<thinking>literal</thinking>'},
        }) + '</tool_call>'
        sampled = 'ORIGINAL_THOUGHT.</think>' + action
        continuations, compared = [], []

        def respond(_model, call):
            if call['kind'] == 'compare':
                compared.append(comparison(call['prompt']))
                return 'DISCARD_JUDGE.</think>{"winner":"A"}'
            if call['kind'] == 'refine':
                return 'DISCARD_REVISION.</think>Revised answer.'
            if 'ACTUAL selected file contents' in call['prompt']:
                continuations.append(call['prompt'])
                return 'AFTER_TOOL_THOUGHT.</think>Finished.'
            return sampled

        ctrl, model, _, actions = self.controller(respond)
        self.implicit_template(model)
        result = ''.join(stream_best_work(ctrl, 'Read the chosen file', passes=2))
        self.assertEqual(compared[0]['candidates'][0]['text'], '<think>' + sampled)
        self.assertEqual(actions, [('read_file', {'path': '<thinking>literal</thinking>'})])
        self.assertEqual(len(continuations), 1)
        self.assertIn('<think>' + sampled, continuations[0])
        self.assertNotIn('DISCARD_', continuations[0])
        self.assertNotIn('BEST_WORK_COMPARISON', continuations[0])
        self.assertIn('<think>AFTER_TOOL_THOUGHT.</think>Finished.', result)

    def test_unclosed_implicit_revision_thinking_is_rejected(self):
        def respond(_model, call):
            if call["kind"] == "original":
                return "Original rationale.</think>Original answer."
            self.assertEqual(call["kind"], "refine")
            return "Unclosed revision reasoning must never be an answer"

        ctrl, model, _, actions = self.controller(respond)
        self.implicit_template(model)
        with self.assertRaisesRegex(SelectionError, "without an answer"):
            list(stream_best_work(ctrl, "Explain the answer", passes=2))
        self.assertEqual(actions, [])
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.context, "")

    def test_thinking_only_original_tool_cannot_become_a_branch_winner(self):
        example = '<tool_call>{"name":"read_file","arguments":{"path":"thinking-only </think> literal"}}</tool_call>'

        def respond(_model, call):
            self.assertEqual(call["kind"], "original")
            return ["Unclosed original reasoning: ", example]

        ctrl, model, _, actions = self.controller(respond)
        self.implicit_template(model)
        with self.assertRaisesRegex(SelectionError, "original branch ended in thinking"):
            list(stream_best_work(ctrl, "Explain the answer", passes=2))
        self.assertEqual(actions, [])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.context, "")

    def test_json_inside_unclosed_implicit_judging_thinking_is_rejected(self):
        def respond(_model, call):
            return {"original": "Original rationale.</think>Original answer.",
                    "refine": "Revision rationale.</think>Revised answer.",
                    "compare": '{"winner":"B"}'}[call["kind"]]

        ctrl, model, _, actions = self.controller(respond)
        self.implicit_template(model)
        with self.assertRaisesRegex(SelectionError, "after two attempts"):
            list(stream_best_work(ctrl, "Explain the answer", passes=2))
        self.assertEqual(actions, [])
        self.assertEqual(sum(call["kind"] == "compare" for call in model.calls), 2)
        self.assertEqual(model.context, "")

    def test_duplicate_winner_keys_require_a_new_unambiguous_selection(self):
        comparisons = []

        def respond(_model, call):
            if call["kind"] == "compare":
                comparisons.append(call["prompt"])
                return '{"winner":"A","winner":"B"}' if len(comparisons) == 1 else '{"winner":"A"}'
            return "Original answer." if call["kind"] == "original" else "Revised answer."

        ctrl, _, _, actions = self.controller(respond)
        answer = "".join(stream_best_work(ctrl, "Explain the answer", passes=2))
        self.assertEqual(answer, "Original answer.")
        self.assertEqual(len(comparisons), 2)
        self.assertEqual(actions, [])


if __name__ == "__main__":
    unittest.main()
