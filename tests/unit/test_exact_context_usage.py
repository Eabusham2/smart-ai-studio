"""Actual prepared input IDs and native usage drive one invocation's Context."""
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from core.generation_metrics import (
    RequestMetrics, TokenMeter, observe_generations, observe_llama_tokens,
    observe_mlx_response, observe_server_event,
)
from core.gui_response_telemetry import active_backend_owner, context_label, context_limit, observed_prompt_tokens
from core.vlm_prompt_tokens import prepare_vlm_inputs
from tests.unit.test_prompt_reuse import runtime


def test_cached_suffix_usage_cannot_replace_complete_submitted_prompt():
    owner = SimpleNamespace(last_prompt_token_count=999)
    request = RequestMetrics()
    with observe_generations(request.update):
        meter = TokenMeter(owner, "mlx-lm", prompt_tokens=421)
        assert request.snapshot()["context"]["prompt_tokens"] == 421
        assert request.snapshot()["context"]["completion_tokens"] is None
        observe_mlx_response(meter, SimpleNamespace(prompt_tokens=1, generation_tokens=3, text="answer"))
        meter.finish()
    assert owner.last_prompt_token_count == 421
    context = request.snapshot()["context"]
    assert context["prompt_tokens"] == 421 and context["completion_tokens"] == 3


def test_llama_prompt_is_measured_at_actual_generate_boundary_with_special_ids():
    submitted = [10001, 10002, 48, 10003, 999, 10004]
    snapshots = []
    class Model:
        def generate(self, tokens):
            assert tokens is submitted
            yield 101
            yield 102
    model = Model()
    original = model.generate
    with observe_generations(snapshots.append):
        meter = TokenMeter()
        with observe_llama_tokens(model, meter):
            assert list(model.generate(submitted)) == [101, 102]
        # A runtime can report only uncached prefill tokens. These cannot
        # override the complete sequence already observed at its boundary.
        meter.observe(prompt_tokens=1)
        meter.finish()
    measured = [sample for sample in snapshots if sample["prompt_tokens"] is not None]
    assert measured[0]["prompt_tokens"] == 6 and measured[0]["tokens"] is None
    assert measured[-1]["prompt_tokens"] == 6 and measured[-1]["tokens"] == 2
    assert model.generate == original and "generate" not in model.__dict__


def test_context_is_current_invocation_not_total_pro_or_tool_work():
    metrics = RequestMetrics()
    with observe_generations(metrics.update):
        losing = TokenMeter(prompt_tokens=70)
        losing.observe(count=11)
        losing.finish()
        candidate = TokenMeter(prompt_tokens=70)
        candidate.observe(count=3)
        candidate.finish()
        # A real tool result and its template enlarge the next submitted input.
        after_tool = TokenMeter(prompt_tokens=108)
        assert metrics.snapshot()["context"]["prompt_tokens"] == 108
        assert metrics.snapshot()["context"]["completion_tokens"] is None
        after_tool.observe(count=2)
        after_tool.finish()
    measured = metrics.snapshot()
    assert measured["tokens"] == 16 and measured["invocations"] == 3
    assert measured["context"]["prompt_tokens"] == 108
    assert measured["context"]["completion_tokens"] == 2
    assert "Last invocation: 110" in context_label(108, 1000, completion_tokens=2, multiple_invocations=True)


def test_late_old_meter_finish_does_not_replace_newer_invocation():
    metrics = RequestMetrics()
    with observe_generations(metrics.update):
        old = TokenMeter(prompt_tokens=8)
        new = TokenMeter(prompt_tokens=19)
        new.observe(count=2)
        new.finish()
        old.finish()
    assert metrics.snapshot()["context"]["id"] == new.id
    assert metrics.snapshot()["context"]["prompt_tokens"] == 19


def test_server_without_live_prompt_usage_remains_unknown_until_final_record():
    meter = TokenMeter(SimpleNamespace(last_prompt_token_count=999))
    observe_server_event(meter, {"content": "Several words in one network chunk", "tokens": [31, 32]})
    assert meter.prompt_tokens is None and meter.tokens == 2
    assert context_label(None, 1000, completion_tokens=2, live=True) == (
        "📊 Live context: unknown / 1,000 · 2 generated"
    )
    observe_server_event(meter, {"usage": {"prompt_tokens": 28, "completion_tokens": 2}})
    assert meter.prompt_tokens == 28 and meter.tokens == 2
    observe_server_event(meter, {"timings": {"prompt_n": 29, "predicted_n": 3}})
    assert meter.prompt_tokens == 29 and meter.tokens == 3


def test_active_owner_and_stale_counts_are_never_borrowed_from_other_models():
    mlx = SimpleNamespace(is_mlx_available=True, last_prompt_token_count=99, context_budget_tokens=500)
    torch = SimpleNamespace(last_prompt_token_count=None, last_usage={"prompt_tokens": 17},
                            active_backend="torch", mlx_backend=mlx, _context_budget_tokens=1000)
    assert active_backend_owner(torch) is torch
    assert context_limit({}, torch, 32768) == 1000
    assert observed_prompt_tokens(torch) is None
    torch.active_backend = "mlx"
    assert active_backend_owner(torch) is mlx and observed_prompt_tokens(torch) == 99


def install_vlm_api(monkeypatch, *, accepts_inputs=True):
    generation = ModuleType("mlx_vlm.generate")
    prepared = []
    calls = []
    ids = np.array([[901, 902, 1, 2, 903, 904, 3, 4, 905, 906, 907]], dtype=np.int32)
    mask = np.ones_like(ids)
    def prepare(processor, **kwargs):
        prepared.append((processor, kwargs))
        return {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
    def stream(model, processor, *, input_ids=None, pixel_values=None, mask=None, **kwargs):
        calls.append((input_ids, mask, kwargs))
        assert input_ids is ids
        yield SimpleNamespace(text="a", prompt_tokens=1, generation_tokens=1, cached_tokens=10)
        yield SimpleNamespace(text="b", prompt_tokens=1, generation_tokens=2, cached_tokens=10)
    def unsupported_stream(model, processor, prompt, **kwargs):
        calls.append(kwargs)
        yield SimpleNamespace(text="answer", prompt_tokens=23, generation_tokens=2)
    generation.prepare_inputs = prepare
    generation.stream_generate = stream if accepts_inputs else unsupported_stream
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", generation)
    return generation, prepared, calls, ids, mask


@pytest.mark.parametrize("model_type,template,add_special", [
    ("bonsai", "template", True), ("gemma3", "template", False), ("gemma4", None, True),
])
def test_vlm_count_uses_same_final_template_processor_arrays_submitted_to_runtime(
        monkeypatch, model_type, template, add_special):
    _, prepared, _, ids, mask = install_vlm_api(monkeypatch)
    model = SimpleNamespace(config=SimpleNamespace(model_type=model_type, image_token_index=123))
    processor = SimpleNamespace(chat_template=template)
    formatted = "SYSTEM: fixed rules\nTOOLS: read_file(path)\nHISTORY: prior tool output\nASSISTANT:"
    inputs, count = prepare_vlm_inputs(model, processor, formatted)
    assert count == 11 and inputs["input_ids"] is ids and inputs["mask"] is mask
    assert "attention_mask" not in inputs and "token_type_ids" in inputs
    assert prepared == [(processor, {"images": [], "audio": None, "videos": None,
                                    "prompts": formatted, "image_token_index": 123,
                                    "resize_shape": None, "add_special_tokens": add_special})]


def test_vlm_runtime_with_prepare_helper_but_without_input_bypass_is_not_preprocessed(monkeypatch):
    _, prepared, _, _, _ = install_vlm_api(monkeypatch, accepts_inputs=False)
    inputs, count = prepare_vlm_inputs(SimpleNamespace(), object(), "final template")
    assert inputs == {} and count is None and prepared == []


def test_native_vlm_kwargs_bypass_contract_is_recognized(monkeypatch):
    generation, _, _, ids, _ = install_vlm_api(monkeypatch)
    def native_style(model, processor, prompt, **kwargs):
        if kwargs.get("input_ids", None) is not None:
            input_ids = kwargs.pop("input_ids")
            pixel_values = kwargs.pop("pixel_values", None)
            mask = kwargs.pop("mask", None)
        else:
            raise AssertionError("test request was preprocessed twice")
        yield input_ids
    generation.stream_generate = native_style
    inputs, count = prepare_vlm_inputs(SimpleNamespace(), object(), "final template")
    assert inputs["input_ids"] is ids and count == 11


def test_mlx_stream_and_pro_branches_count_native_prepared_ids(runtime, monkeypatch):
    backend = runtime.backend
    generation, prepared, calls, ids, mask = install_vlm_api(monkeypatch)
    backend._vlm_runtime = "mlx_vlm"
    backend.processor = SimpleNamespace()
    backend.context_budget_tokens = 16
    monkeypatch.setattr(backend, "_prepare_vlm_prompt", lambda prompt, images: "full template: " + prompt)
    monkeypatch.setattr(backend, "_vlm_reuse_kwargs", lambda *args, **kwargs: {})
    for branches in (False, True):
        metrics = RequestMetrics()
        with observe_generations(metrics.update):
            if branches:
                assert backend.generate_branches("question", branch_count=2) == ["ab", "ab"]
            else:
                assert "".join(backend.stream_generate_tokens("question")) == "ab"
        context = metrics.snapshot()["context"]
        assert context["prompt_tokens"] == 11 and context["completion_tokens"] == 2
        assert backend.last_prompt_token_count == 11
        assert backend.last_effective_max_tokens == 5
    # Preprocessing happens once per full request, not once for counting and
    # again for inference, nor once per Pro branch with identical input.
    assert len(prepared) == 2 and len(calls) == 3
    assert all(row[0] is ids and row[1] is mask for row in calls)
    assert all(row[2]["max_tokens"] == 5 for row in calls)


def test_mlx_prefix_reuse_does_not_shrink_live_context_to_suffix(runtime):
    backend = runtime.backend
    for _ in range(2):
        metrics = RequestMetrics()
        with observe_generations(metrics.update):
            assert len(list(backend.stream_generate_tokens("abcde"))) == 2
        assert metrics.snapshot()["context"]["prompt_tokens"] == 5
        assert metrics.snapshot()["context"]["completion_tokens"] == 2
    assert backend.last_prompt_reused_tokens == 4


@pytest.mark.parametrize("kind", ["image", "video", "audio"])
def test_media_review_replaces_old_text_context_with_unknown_then_new_measured_text(kind):
    from core.media_orchestrator import MediaController
    controller = MediaController.__new__(MediaController)
    controller._can_perceive = lambda value: value == kind
    metrics = RequestMetrics()
    calls = []
    native_result = {"perception_available": True, "description": "actual review fixture"}
    def review(**kwargs):
        calls.append(kwargs)
        context = metrics.snapshot()["context"]
        assert context["source"] == "media-review" and not context["finished"]
        assert context["prompt_tokens"] is None and context["completion_tokens"] is None
        assert "unknown" in context_label(context["prompt_tokens"], 2048, live=True)
        return native_result
    controller._controller_review_hook = lambda: review
    with observe_generations(metrics.update):
        prior = TokenMeter(source="previous-text", prompt_tokens=71)
        prior.observe(count=8)
        prior.finish()
        result = controller._review_with_controller({"kind": kind, "path": "fixture.media",
                                                    "prompt": "inspect exact artifact", "artifact_id": "artifact"})
        context = metrics.snapshot()["context"]
        assert context["source"] == "media-review" and context["finished"]
        assert context["prompt_tokens"] is None and context["completion_tokens"] is None
        continuation = TokenMeter(source="text-after-media", prompt_tokens=125)
        continuation.observe(count=3)
        continuation.finish()
    assert calls == [{"path": "fixture.media", "kind": kind, "prompt": "inspect exact artifact"}]
    assert result == {"status": "success", "artifact_id": "artifact", "review": native_result,
                      "weights_updated": False}
    assert metrics.snapshot()["context"]["prompt_tokens"] == 125
    assert metrics.snapshot()["context"]["completion_tokens"] == 3


def test_media_review_failure_finishes_unknown_invocation_without_restoring_old_usage():
    from core.media_orchestrator import MediaController
    controller = MediaController.__new__(MediaController)
    controller._can_perceive = lambda kind: True
    def fail(**kwargs):
        raise RuntimeError("media inference failed")
    controller._controller_review_hook = lambda: fail
    metrics = RequestMetrics()
    with observe_generations(metrics.update):
        prior = TokenMeter(prompt_tokens=99)
        prior.observe(count=9)
        prior.finish()
        with pytest.raises(RuntimeError, match="media inference failed"):
            controller._review_with_controller({"kind": "image", "path": "fixture.png"})
    context = metrics.snapshot()["context"]
    assert context["source"] == "media-review" and context["finished"]
    assert context["prompt_tokens"] is None and context["completion_tokens"] is None


def test_native_media_meter_remains_authoritative_when_outer_review_scope_finishes():
    from core.media_orchestrator import MediaController
    controller = MediaController.__new__(MediaController)
    controller._can_perceive = lambda kind: True
    metrics = RequestMetrics()
    def measured_review(**kwargs):
        native = TokenMeter(source="native-media", prompt_tokens=512)
        native.observe(count=9)
        native.finish()
        return {"perception_available": True}
    controller._controller_review_hook = lambda: measured_review
    with observe_generations(metrics.update):
        controller._review_with_controller({"kind": "image", "path": "fixture.png", "artifact_id": "artifact"})
    context = metrics.snapshot()["context"]
    assert context["source"] == "native-media" and context["finished"]
    assert context["prompt_tokens"] == 512 and context["completion_tokens"] == 9
