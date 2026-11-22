"""Scripted model text through the real synthesis, refinement and tool handoff."""
from __future__ import annotations

import json
import re
import threading
from unittest.mock import patch

import pytest

from core.best_work import stream_best_work
from core.best_work_selection import SelectionError
from core.context_memory import ContextOverflow
from core.generation_metrics import RequestMetrics, observe_generations
from core.pro_engine import parse_reasoning_and_response
from core.temperature_policy import pro_ladder
from tests.unit.test_best_work_tournament import comparison, make_controller, previous_draft


def synthesis_inputs(prompt):
    decoder = json.JSONDecoder()
    return [decoder.raw_decode(prompt.split(f"CANDIDATE {side}:\n", 1)[1])[0]
            for side in ("A", "B")]


def identity(text):
    match = re.search(r"\b(NORMAL|MERGE)_(\d+)_(ORIGINAL|REVISION_\d+)", text)
    assert match is not None, text
    return match[1], int(match[2]), match[3]


class Responses:
    def __init__(self, winner="synthesis", tools=False):
        self.winner = winner
        self.tools = tools
        self.originals = []
        self.merges = []
        self.refinements = []
        self.matches = []
        self.order = []
        self.after_tool = []
        self.engine = None

    def text(self, kind, number, version):
        thought = f"{kind}_THOUGHT_{number}" if version == "ORIGINAL" else "DROP_REVISION_THOUGHT"
        output = f"<think>{thought}</think>\n{kind}_{number}_{version}"
        if self.tools:
            action = {"name": "read_file", "arguments": {"path": f"{kind}-{number}-{version}.txt"}}
            output += "<tool_call>" + json.dumps(action) + "</tool_call>"
        return output

    def __call__(self, model, call):
        prompt = call["prompt"]
        if "ACTUAL selected file contents" in prompt:
            if call["kind"] == "original":
                self.after_tool.append(prompt)
                return "<think>AFTER_TOOL_THOUGHT</think>Finished original."
            if call["kind"] == "refine":
                return "<think>DROP_AFTER_TOOL_THOUGHT</think>Finished answer."
            return '{"winner":"B"}'
        if "BEST_WORK_SYNTHESIS" in prompt:
            assert model.cache is None
            assert self.engine._best_work_inference_read_only is True
            assert self.engine.memory_auto_recall is False
            self.merges.append(synthesis_inputs(prompt))
            self.order.append("merge")
            return self.text("MERGE", len(self.merges), "ORIGINAL")
        if call["kind"] == "original":
            index = len(self.originals)
            output = self.text("NORMAL", index, "ORIGINAL")
            self.originals.append(output)
            self.order.append("original")
            return output
        if call["kind"] == "refine":
            draft = previous_draft(prompt)
            kind, number, _ = identity(draft)
            revision = 1 + sum(identity(old)[:2] == (kind, number) for old in self.refinements)
            self.refinements.append(draft)
            self.order.append("refine_synthesis" if kind == "MERGE" else "refine_normal")
            return self.text(kind, number, f"REVISION_{revision}")
        contenders = comparison(prompt)["candidates"]
        ids = [identity(row["text"]) for row in contenders]
        local = ids[0][:2] == ids[1][:2]
        self.matches.append([row["text"] for row in contenders])
        self.order.append(("local_synthesis" if ids[0][0] == "MERGE" else "local_normal")
                          if local else "tournament")
        if local:
            original = (ids[0][0] == "MERGE" and self.winner == "synthesis_original"
                        or ids[0][0] == "NORMAL" and ids[0][1] == len(self.originals) - 1)
            choice = "A" if original else "B"
        else:
            def rank(item):
                return (int((item[0] == "MERGE") == self.winner.startswith("synthesis")), item[1])
            choice = contenders[max(range(2), key=lambda index: rank(ids[index]))]["id"]
        return '<think>DROP_JUDGING_TEXT</think>' + json.dumps({"winner": choice})


def controller(tmp_path, responses, **kwargs):
    result = make_controller(responses, database_path=str(tmp_path / "chat.db"), **kwargs)
    responses.engine = result[0].app.engine
    return result


@pytest.mark.parametrize("winner,passes", [("normal", 2), ("synthesis", 3), ("synthesis_original", 2)])
def test_synthesis_competes_after_all_normal_best_work_and_keeps_exact_winner(tmp_path, winner, passes):
    responses = Responses(winner)
    ctrl, model, entropies, actions = controller(tmp_path, responses, entropy=0.5)
    original_cache, original_weights = model.cache, model.weights
    history = [{"role": "system", "content": "Preserve the complete request and literal <think> text."}]
    request = "Choose the best complete answer, retaining every constraint."
    metrics = RequestMetrics()
    with observe_generations(metrics.update):
        result = "".join(stream_best_work(ctrl, request, history, passes=passes))

    expected_kind, expected_index = ("NORMAL", 8) if winner == "normal" else ("MERGE", 8)
    expected_version = "ORIGINAL" if winner != "synthesis" else f"REVISION_{passes - 1}"
    assert parse_reasoning_and_response(result) == (
        f"{expected_kind}_THOUGHT_{expected_index}",
        f"{expected_kind}_{expected_index}_{expected_version}",
    )
    assert responses.order == (["original"] * 9 + ["refine_normal"] * (9 * (passes - 1))
                               + ["local_normal"] * 9 + ["merge"] * 8
                               + ["refine_synthesis"] * (passes - 1)
                               + ["local_synthesis"] + ["tournament"] * 9)
    merge_inputs = [text for pair in responses.merges for text in pair]
    for index in range(9):
        version = "ORIGINAL" if index == 8 else f"REVISION_{passes - 1}"
        assert any(identity(text) == ("NORMAL", index, version) for text in merge_inputs)
    assert all("DROP_" not in text for text in merge_inputs)
    synthesis_drafts = [text for text in responses.refinements if identity(text)[0] == "MERGE"]
    assert all("MERGE_THOUGHT_8" in text and "DROP_" not in text for text in synthesis_drafts)
    synthesis_match = next(pair for pair in responses.matches
                           if all(identity(text)[0] == "MERGE" for text in pair))
    assert [identity(text)[2] for text in synthesis_match] == ["ORIGINAL", f"REVISION_{passes - 1}"]
    assert all(request in call["prompt"] and history[0]["content"] in call["prompt"] for call in model.calls)

    meta = ctrl.last_best_work_metadata
    assert meta["branch_count"] == 10
    assert meta["winning_branch"] == (8 if winner == "normal" else 9)
    assert meta["winning_temp"] == (pro_ladder(8)[8] if winner == "normal" else 0.20)
    assert meta["temp_ladder"] == pro_ladder(8)
    assert meta["branches"][-1]["temp"] == 0.20
    assert meta["raw_branches"][meta["winning_branch"]] == result
    assert len(meta["best_work"]["branches"]) == 10
    assert meta["best_work"]["branches"][-1]["type"] == "synthesis"
    assert meta["best_work"]["branches"][-1]["passes_completed"] == passes
    assert len(meta["best_work"]["synthesis_summaries"]) == 8
    assert "DROP_" not in json.dumps(meta["best_work"])
    assert meta["pass_usage"] == {key: metrics.snapshot()[key] for key in ("invocations", "tokens")}
    assert metrics.snapshot()["invocations"] == len(model.calls)
    assert [call["temperature"] for call in model.calls if call["kind"] == "original"] == pro_ladder(8)
    assert all(call["temperature"] == 0.20 for call in model.calls if "BEST_WORK_SYNTHESIS" in call["prompt"])
    assert len(entropies) == 1 and actions == []
    assert model.context == "" and model.cache is original_cache and model.weights is original_weights
    assert len(model.closed) == len(model.calls)
    assert history == [{"role": "system", "content": "Preserve the complete request and literal <think> text."}]


@pytest.mark.parametrize("entropy,count", [(0.1, 1), (0.85, 17)])
def test_single_pass_keeps_router_budget_and_adds_only_one_synthesis_contender(tmp_path, entropy, count):
    responses = Responses()
    ctrl, model, entropies, _ = controller(tmp_path, responses, entropy=entropy)
    result = "".join(stream_best_work(ctrl, "Choose the answer", passes=1))
    assert len(responses.originals) == count
    assert len(responses.merges) == count - 1
    assert responses.refinements == []
    assert len(responses.matches) == (count if count > 1 else 0)
    if count > 1:
        assert ctrl.last_best_work_metadata["branch_count"] == count + 1
    else:
        assert not ctrl.last_best_work_metadata.get("best_work")
        assert len(model.calls) == 1  # Keep the existing N=1 live-stream path.
    assert identity(result) == (("MERGE", 16, "ORIGINAL") if count > 1 else ("NORMAL", 0, "ORIGINAL"))
    assert len(entropies) == 1
    assert [call["temperature"] for call in model.calls if call["kind"] == "original"] == pro_ladder(16 if count > 1 else 1)


@pytest.mark.parametrize("winner", ["normal", "synthesis"])
def test_only_the_overall_winner_tool_runs_and_continuation_has_no_merge_or_judge_context(tmp_path, winner):
    responses = Responses(winner, tools=True)
    entropy = lambda prompt: 0.1 if "ACTUAL selected file contents" in prompt else 0.5
    ctrl, model, entropies, actions = controller(tmp_path, responses, entropy=entropy)
    result = "".join(stream_best_work(ctrl, "Read the chosen file and summarize it", passes=2))
    kind, version = ("NORMAL", "ORIGINAL") if winner == "normal" else ("MERGE", "REVISION_1")
    assert actions == [("read_file", {"path": f"{kind}-8-{version}.txt"})]
    assert len(responses.after_tool) == 1
    continuation = responses.after_tool[0]
    assert f"{kind}_THOUGHT_8" in continuation and f"{kind}_8_{version}" in continuation
    for discarded in ("BEST_WORK_SYNTHESIS", "BEST_WORK_COMPARISON", "CANDIDATE A:",
                      "DROP_", "NORMAL_0_", "MERGE_1_", '"winner"'):
        assert discarded not in continuation
    assert ("NORMAL_8_" if winner == "synthesis" else "MERGE_8_") not in continuation
    assert result.endswith("Finished answer.")
    assert len(entropies) == 2 and "ACTUAL selected file contents" in entropies[-1]
    assert len(responses.merges) == 8
    assert len(model.closed) == len(model.calls)


def test_merge_cancellation_discards_all_candidates_and_restores_state(tmp_path):
    cancelled = threading.Event()
    responses = Responses(tools=True)

    def respond(model, call):
        output = responses(model, call)
        if "BEST_WORK_SYNTHESIS" in call["prompt"]:
            cancelled.set()
        return output

    ctrl, model, _, actions = make_controller(respond, entropy=0.5, database_path=str(tmp_path / "chat.db"))
    responses.engine = ctrl.app.engine
    cache, weights = model.cache, model.weights
    assert list(stream_best_work(ctrl, "Read the chosen file", passes=2, cancel_event=cancelled)) == []
    assert responses.order[-1] == "merge" and len(responses.merges) == 1
    assert actions == [] and model.context == ""
    assert model.cache is cache and model.weights is weights
    assert len(model.closed) == len(model.calls)
    assert not getattr(ctrl.app.engine, "_best_work_select_candidates", None)
    assert not getattr(ctrl.app.engine, "_best_work_inference_read_only", False)


def test_merge_overflow_reports_failure_without_truncating_or_dropping_the_extra_branch(tmp_path):
    from core.awake_auto_hook import _remaining_generation_tokens
    responses = Responses()
    ctrl, model, _, actions = controller(tmp_path, responses, entropy=0.5)
    blocked_prompts, displayed = [], []

    def limit(engine, prompt):
        if "BEST_WORK_SYNTHESIS" in prompt:
            blocked_prompts.append(prompt)
            raise ContextOverflow("Both complete merge candidates exceed the selected context")
        return _remaining_generation_tokens(engine, prompt)

    with patch("core.awake_auto_hook._remaining_generation_tokens", side_effect=limit):
        with pytest.raises(ContextOverflow, match="complete merge candidates"):
            for piece in stream_best_work(ctrl, "Preserve both complete candidates", passes=2):
                displayed.append(piece)
    assert len(blocked_prompts) == 1
    assert [identity(text) for text in synthesis_inputs(blocked_prompts[0])] == [
        ("NORMAL", 0, "REVISION_1"), ("NORMAL", 1, "REVISION_1")]
    assert responses.order[-1] == "local_normal"
    assert displayed == [] and actions == [] and model.context == ""
    assert len(model.closed) == len(model.calls)


def test_reasoning_only_merge_is_not_admitted_as_a_complete_candidate(tmp_path):
    responses = Responses()

    def respond(model, call):
        if "BEST_WORK_SYNTHESIS" in call["prompt"]:
            return "<think>Unfinished merge reasoning must not become a candidate."
        return responses(model, call)

    ctrl, model, _, actions = make_controller(respond, entropy=0.5, database_path=str(tmp_path / "chat.db"))
    responses.engine = ctrl.app.engine
    with pytest.raises(SelectionError):
        list(stream_best_work(ctrl, "Choose a complete answer", passes=1))
    assert actions == [] and model.context == ""
    assert not any(call["kind"] == "compare" for call in model.calls)
