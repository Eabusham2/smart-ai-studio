"""VLM Auto memo reuses exact native text without retaining device arrays."""
import gc
import os
import sys
import types
import weakref

import numpy as np
import pytest

from core.vlm_prompt_tokens import PreparedTextInputMemo
from tests.unit.test_prompt_reuse import runtime


@pytest.fixture
def native_text(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    device = os.getenv("MLX_TEST_DEVICE", "cpu")
    if device == "metal":
        assert mx.metal.is_available(), "Metal lane must execute real Metal arrays"
        mx.set_default_device(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)
    generation = types.ModuleType("mlx_vlm.generate")
    calls, references = [], []

    class Tokenizer:
        chat_template = "template"
        added = {}

        def encode(self, text):
            return [90, *(ord(char) for char in text), 91]

        def get_added_vocab(self):
            return dict(self.added)

    class Processor:
        chat_template = "processor-template"

        def __init__(self):
            self.tokenizer = Tokenizer()
            self.padding = "right"

        def to_dict(self):
            return {"padding": self.padding}

    def prepare(processor, **kwargs):
        calls.append(kwargs)
        ids = mx.array([processor.tokenizer.encode(kwargs["prompts"])], dtype=mx.int32)
        result = {"input_ids": ids, "attention_mask": mx.ones_like(ids),
                  "token_type_ids": mx.zeros_like(ids)}
        references.extend(weakref.ref(value) for value in result.values())
        return result

    def stream(model, processor, *, input_ids=None, pixel_values=None, mask=None, **kwargs):
        # Same explicit bypass as native VLM, without loading model weights.
        yield types.SimpleNamespace(text="reply", prompt_tokens=input_ids.shape[-1], generation_tokens=1)

    generation.prepare_inputs = prepare
    generation.stream_generate = stream
    generation.generate_step = lambda *, prefill_step_size=2048: None
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", generation)
    model = types.SimpleNamespace(config={"model_type": "bonsai", "image_token_index": 100})
    return types.SimpleNamespace(mx=mx, model=model, processor=Processor(), calls=calls,
                                 refs=references, generation=generation)


def test_repeat_exact_native_text_reuses_cpu_bytes_and_returns_fresh_arrays(native_text):
    env = native_text
    memo = PreparedTextInputMemo()
    first, count = memo.prepare(env.model, env.processor, "SYSTEM rules; prior answer; question")
    expected = {key: None if value is None else np.asarray(value).copy() for key, value in first.items()}
    assert count == len("SYSTEM rules; prior answer; question") + 2
    assert memo.nbytes <= memo.max_bytes and memo.telemetry()["vlm_input_memo_hits"] == 0
    del first
    gc.collect()
    assert all(ref() is None for ref in env.refs), "CPU memo must not retain original device arrays"

    second, second_count = memo.prepare(env.model, env.processor, "SYSTEM rules; prior answer; question")
    assert len(env.calls) == 1 and memo.hits == 1 and second_count == count
    for key, value in expected.items():
        if value is None:
            assert second[key] is None
        else:
            np.testing.assert_array_equal(np.asarray(second[key]), value)
            assert np.asarray(second[key]).dtype == value.dtype
    second["input_ids"][0, 0] = 12
    third, _ = memo.prepare(env.model, env.processor, "SYSTEM rules; prior answer; question")
    np.testing.assert_array_equal(np.asarray(third["input_ids"]), expected["input_ids"])
    assert third["input_ids"] is not second["input_ids"]


@pytest.mark.parametrize("mutation", ["prompt", "model", "processor", "template", "tokenizer", "config", "runtime", "revision"])
def test_changed_input_or_identity_never_reuses_stale_preprocessing(native_text, mutation):
    env = native_text
    memo = PreparedTextInputMemo()
    memo.prepare(env.model, env.processor, "question")
    prompt = "question"
    if mutation == "prompt":
        prompt += " continued"
    elif mutation == "model":
        env.model = types.SimpleNamespace(config=dict(env.model.config))
    elif mutation == "processor":
        env.processor = type(env.processor)()
    elif mutation == "template":
        env.processor.chat_template = "changed-template"
    elif mutation == "tokenizer":
        env.processor.tokenizer.added = {"new-special": 100}
    elif mutation == "config":
        env.processor.padding = "left"
    elif mutation == "runtime":
        original = env.generation.prepare_inputs
        env.generation.prepare_inputs = lambda *args, **kwargs: original(*args, **kwargs)
    else:
        env.processor._smartai_revision = 1
    memo.prepare(env.model, env.processor, prompt)
    assert len(env.calls) == 2 and memo.hits == 0


def test_bound_disable_media_and_unknown_native_fields_keep_full_preprocessing(native_text):
    env = native_text
    memo = PreparedTextInputMemo(max_bytes=2048)
    for _ in range(2):
        _, count = memo.prepare(env.model, env.processor, "x" * 2000)
        assert count == 2002 and memo.nbytes == 0
    for _ in range(2):
        memo.prepare(env.model, env.processor, "small", enabled=False)
        assert memo.nbytes == 0
    assert len(env.calls) == 4 and not memo.hits

    original = env.generation.prepare_inputs
    for key, value in (("pixel_values", env.mx.ones((2, 2))), ("unknown_metadata", object())):
        def with_extra(*args, **kwargs):
            return {**original(*args, **kwargs), key: value}
        env.generation.prepare_inputs = with_extra
        for _ in range(2):
            inputs, _ = memo.prepare(env.model, env.processor, "small")
            assert inputs[key] is value and memo.nbytes == 0
    assert len(env.calls) == 8 and memo.hits == 0


def test_model_memos_are_independent_and_repeated_replacement_is_bounded(native_text):
    env = native_text
    left, right = PreparedTextInputMemo(max_bytes=8192), PreparedTextInputMemo(max_bytes=8192)
    for i in range(50):
        inputs, count = left.prepare(env.model, env.processor, f"request {i}")
        del inputs
        assert 0 < left.nbytes <= 8192 and count > 0
    inputs, _ = right.prepare(env.model, env.processor, "request 49")
    del inputs
    assert len(env.calls) == 51 and not right.hits
    gc.collect()
    assert all(ref() is None for ref in env.refs)
    left.clear()
    assert not left.nbytes and right.nbytes
    right.prepare(env.model, env.processor, "request 49")
    assert right.hits == 1


def test_backend_vlm_auto_memo_survives_deferred_tuning_and_invalidates_on_learning(runtime, monkeypatch):
    # Exercise the serving branch with the existing deterministic model fixture.
    from tests.unit.test_prompt_reuse import Array
    monkeypatch.setattr(Array, "__array__", lambda self, dtype=None: self.data.astype(dtype) if dtype else self.data,
                        raising=False)
    backend = runtime.backend
    backend._vlm_runtime = "jang_vlm"
    backend.processor = types.SimpleNamespace(tokenizer=backend.tokenizer, to_dict=lambda: {})
    backend.context_budget_tokens = 32768
    generation = types.ModuleType("mlx_vlm.generate")
    prepared, submitted = [], []

    def prepare(processor, **kwargs):
        prepared.append(kwargs["prompts"])
        return {"input_ids": Array([[1, 2, 3]]), "attention_mask": Array([[1, 1, 1]])}

    def stream(model, processor, *, input_ids=None, pixel_values=None, mask=None, **kwargs):
        submitted.append(input_ids)
        yield types.SimpleNamespace(text="reply", prompt_tokens=3, generation_tokens=1)

    generation.prepare_inputs = prepare
    generation.stream_generate = stream
    generation.generate_step = lambda *, prefill_step_size=2048: None
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", generation)
    import core.mlx_speed_tuning as tuning
    monkeypatch.setattr(tuning, "tuning_memory_ok", lambda *args: False)
    assert backend.configure_auto_speed()["status"] == "memory_pressure"
    assert list(backend.stream_generate_tokens("system+question")) == ["reply"]
    assert list(backend.stream_generate_tokens("system+question")) == ["reply"]
    assert prepared == ["system+question"] and submitted[0] is not submitted[1]
    assert backend.prompt_reuse_telemetry()["vlm_input_memo_hits"] == 1
    backend.invalidate_prompt_reuse("training or adapter mutation")
    assert backend.prompt_reuse_telemetry()["vlm_input_memo_bytes"] == 0
    assert list(backend.stream_generate_tokens("system+question")) == ["reply"]
    assert len(prepared) == 2



def test_qwen3vl_processor_and_real_tokenizer_reuse_exact_native_inputs(native_text, monkeypatch):
    # Both publisher loaders construct Qwen3VLProcessor. Build its text-facing
    # components offline with a tiny real fast tokenizer; no weights/downloads.
    from transformers import Qwen3VLProcessor, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers

    env = native_text
    raw = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2, "[EOS]": 3},
                                    unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", eos_token="[EOS]")
    processor = Qwen3VLProcessor.__new__(Qwen3VLProcessor)
    processor.tokenizer = tokenizer
    processor.chat_template = "template"
    processor.image_processor = processor.video_processor = None
    processor.image_token, processor.video_token = "<|image_pad|>", "<|video_pad|>"
    processor.image_token_id, processor.video_token_id = 98, 99

    class NoCopyDecoder:
        def __deepcopy__(self, memo):
            raise AssertionError("Text memo must not copy the generation decoder")

    processor.detokenizer = NoCopyDecoder()
    monkeypatch.setattr(processor, "to_dict", lambda: pytest.fail("Processor serialization copies runtime helpers"))
    prepares = []

    def prepare(current, **kwargs):
        prepares.append(True)
        # Native mlx-vlm initializes this on the first text-only call.
        if current.tokenizer.pad_token is None:
            current.tokenizer.pad_token = current.tokenizer.eos_token
        encoded = current(text=kwargs["prompts"], return_tensors="np", padding=True,
                          add_special_tokens=kwargs["add_special_tokens"])
        return {key: env.mx.array(value) for key, value in encoded.items()}

    env.generation.prepare_inputs = prepare
    memo = PreparedTextInputMemo()
    first, count = memo.prepare(env.model, processor, "hello world")
    second, repeated_count = memo.prepare(env.model, processor, "hello world")
    assert count == repeated_count == 2 and len(prepares) == 1
    assert memo.telemetry()["vlm_input_memo_used"] is True
    assert memo.telemetry()["vlm_input_memo_eligible"] is True
    for key in first:
        if first[key] is not None:
            np.testing.assert_array_equal(np.asarray(first[key]), np.asarray(second[key]))
            assert first[key].dtype == second[key].dtype
    tokenizer.padding_side = "left"
    memo.prepare(env.model, processor, "hello world")
    assert len(prepares) == 2 and not memo.telemetry()["vlm_input_memo_used"]


def test_uncacheable_processor_policy_remains_native_without_optimism(native_text):
    env = native_text
    env.processor.custom_configuration = object()
    memo = PreparedTextInputMemo()
    for _ in range(2):
        inputs, count = memo.prepare(env.model, env.processor, "question")
        assert count == 10 and inputs["input_ids"].shape == (1, 10)
    assert len(env.calls) == 2
    assert not memo.telemetry()["vlm_input_memo_eligible"]
    assert not memo.telemetry()["vlm_input_memo_used"]
    assert "not cacheable" in memo.telemetry()["vlm_input_memo_reason"]
