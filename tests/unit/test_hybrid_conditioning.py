"""Hybrid chooses a trainer against the actual tokens of the model's learning cue."""
import pytest

from core.learning_data import iter_probe_windows, tokenize_training_row
from core.legacy_torch_learning import LegacyCompletionTrainer, LegacyControllerTrainer, completion_rows_fit
from core.online_consolidator import HybridAwakeOnlineConsolidator
from tests.unit.test_native_context_learning import tiny_controller


@pytest.mark.parametrize("tiny_controller", [(96, "cpu")], indirect=True)
def test_untemplated_hybrid_learns_served_continuation_and_reloads(tiny_controller):
    backend = tiny_controller
    backend.unload_model()
    backend.learning_implementation = "hybrid"
    assert backend.load_model()
    tokenizer = backend.tokenizer
    assert not tokenizer.chat_template
    prompt, answer = "vault color", "slate"
    prefix, _ = LegacyControllerTrainer._format_training_example(tokenizer, prompt, answer)
    probe, = iter_probe_windows(tokenizer, prompt, answer)
    assert tuple(tokenizer(prefix)["input_ids"]) != probe.input_ids[:probe.loss_start]

    baseline = backend.score_learning_probe(prompt, answer)
    service = HybridAwakeOnlineConsolidator(backend, lambda_ewc=0)
    receipt = service.train_pairs([{"prompt": prompt, "completion": answer}], mode="rsi",
                                 steps=1, learning_rate=.001)
    metrics = receipt["training_metrics"]
    assert metrics["training_algorithm"] == "managed_causal_windows"
    assert metrics["training_windows"] == metrics["optimizer_steps"] == 1
    assert receipt["parameter_drift_l2"] > 0
    assert receipt["probe_nll_before"] == pytest.approx([baseline])
    learned = backend.score_learning_probe(prompt, answer)
    assert learned < baseline and receipt["probe_nll_after"] == pytest.approx([learned])
    manifest = backend.learning_state.current_manifest()
    assert manifest["generation"] == receipt["generation"] and manifest["parent"] is None
    backend.unload_model()
    assert backend.load_model() and backend.learning_state.current_manifest() == manifest
    assert backend.score_learning_probe(prompt, answer) == pytest.approx(learned)


@pytest.mark.parametrize("tiny_controller", [(96, "cpu")], indirect=True)
def test_hybrid_honors_tokenized_message_context_and_completion(tiny_controller):
    backend = tiny_controller
    backend.unload_model()
    backend.learning_implementation = "hybrid"
    assert backend.load_model()
    tokenizer = backend.tokenizer
    tokenizer.chat_template = "{% for m in messages %}{{ m.content }} {% endfor %}"
    row = {"prompt": "vault color", "completion": "slate"}
    conversation = [{"role": "user", "content": "vault color"}, {"role": "assistant", "content": "slate"}]
    assert completion_rows_fit(LegacyControllerTrainer, tokenizer, [dict(row, messages=conversation)], 96, None)

    with_history = dict(row, messages=[{"role": "system", "content": "red"}, *conversation])
    with_other_answer = dict(row, messages=[conversation[0], {"role": "assistant", "content": "red"}])
    for explicit in (with_history, with_other_answer):
        assert not completion_rows_fit(LegacyControllerTrainer, tokenizer, [explicit], 96, None)
    expected_inputs, _ = tokenize_training_row(tokenizer, with_history, supervise_prompt=False)
    assert expected_inputs == tokenizer.convert_tokens_to_ids(["red", "vault", "color", "slate"])
    metrics, drift = backend.train_mini_batch({}, [with_history], steps=1, validation_callback=lambda _: True)
    assert metrics["training_algorithm"] == "managed_causal_windows" and drift > 0
    assert metrics["supervised_tokens"] == metrics["training_windows"] == 1


@pytest.mark.parametrize("tiny_controller", [(96, "cpu")], indirect=True)
def test_hybrid_checks_tokenizer_added_bos_and_preserves_model_terminator(tiny_controller):
    from tokenizers.processors import TemplateProcessing

    tokenizer = tiny_controller.tokenizer
    tokenizer.chat_template = ("{% for m in messages %}{{ m.content }} {% endfor %}"
                               "{% if not add_generation_prompt %}{{ eos_token }}{% endif %}")
    rows = [{"prompt": "vault", "completion": "slate"}]
    assert completion_rows_fit(LegacyCompletionTrainer, tokenizer, rows, 8, None)
    tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A", special_tokens=[("[BOS]", tokenizer.convert_tokens_to_ids("[BOS]"))])
    assert not completion_rows_fit(LegacyCompletionTrainer, tokenizer, rows, 8, None)
