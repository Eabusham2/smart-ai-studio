"""Behavioral checks for atomic parameter/history state and model isolation."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def source_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[2] / "core" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


state = source_module("tested_learning_state", "learning_state.py")
data = source_module("tested_learning_data", "learning_data.py")


class Tokenizer:
    eos_token = "!"

    def encode(self, text, **kwargs):
        return list(text.encode())

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")


def commit(store, value=b"learned parameters", **metadata):
    transaction = store.begin(metadata=metadata)
    (transaction.directory / "adapter.safetensors").write_bytes(value)
    return transaction.commit("adapter.safetensors")


def test_complete_windows_cover_every_target_once():
    tokens = list(range(1203))
    windows = list(data.windows_from_tokens(tokens, max_tokens=64))
    targets = [value for window in windows for value in window.labels[1:]]
    assert targets == tokens[1:]
    assert max(len(window.input_ids) for window in windows) <= 64
    assert windows[-1].stop == len(tokens)


def test_long_prompt_cannot_hide_the_training_answer():
    row = {"prompt": "p" * 1200, "completion": "answer beyond the old 512 token cap"}
    windows = list(data.iter_training_windows(Tokenizer(), [row], max_tokens=64, supervise_prompt=False))
    targets = bytes(value for window in windows for value in window.labels[1:] if value != -100)
    assert targets == (row["completion"] + "</assistant>").encode()


def test_recall_probe_supervises_only_answer_and_no_role_delimiter():
    windows = list(data.iter_probe_windows(Tokenizer(), "question?", "answer" * 30, max_tokens=48))
    assert bytes(value for window in windows for value in window.labels[1:] if value != -100) == b"answer" * 30
    assert all(b"question?" in bytes(window.input_ids) for window in windows)


def test_probe_refuses_to_discard_a_question_that_cannot_fit():
    with pytest.raises(ValueError, match="full source-free"):
        list(data.iter_probe_windows(Tokenizer(), "q" * 14, "x", max_tokens=6))


def test_completion_only_refuses_a_divergent_native_template():
    class Divergent(Tokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return "different prefix" if add_generation_prompt else "user text plus answer"
    with pytest.raises(RuntimeError, match="completion boundary"):
        list(data.iter_training_windows(Divergent(), [{"prompt": "q", "completion": "a"}], supervise_prompt=False))


def test_each_row_must_supply_a_next_token_target():
    class CollapsedRow(Tokenizer):
        def encode(self, text, **kwargs):
            return [1] if "unsupported" in text else super().encode(text, **kwargs)

    rows = [{"prompt": "valid", "completion": "answer"},
            {"prompt": "unsupported", "completion": "answer"}]
    with pytest.raises(ValueError, match="row 1 has no supervised"):
        list(data.iter_training_windows(CollapsedRow(), rows))


def test_native_template_and_all_user_facts_enter_training():
    row = {"prompt": "user-only fact: violet", "completion": "understood"}
    windows = list(data.iter_training_windows(Tokenizer(), [row], max_tokens=9))
    reconstructed = bytes([windows[0].input_ids[0]] + [x for w in windows for x in w.input_ids[1:]])
    assert reconstructed == b"<user>user-only fact: violet</user><assistant>understood</assistant>"


def test_weights_and_retirement_publish_as_one_generation(tmp_path):
    store = state.LearningStateStore({"source": "bonsai2", "source_verified": True}, state_root=tmp_path)
    transaction = store.begin(update_id="update-1", metadata={"memory": {"retirement_head": "turn-4"}})
    (transaction.directory / "adapter.safetensors").write_bytes(b"candidate")
    assert store.current_manifest() is None
    manifest = transaction.commit("adapter.safetensors")
    assert Path(store.current_checkpoint()).read_bytes() == b"candidate"
    assert store.current_manifest()["metadata"]["memory"]["retirement_head"] == "turn-4"
    assert manifest["update_id"] == "update-1"


def test_aborted_stage_keeps_old_parameters_and_retirement(tmp_path):
    store = state.LearningStateStore({"source": "model"}, state_root=tmp_path)
    first = commit(store, memory={"retirement_head": "old"})
    with store.begin(metadata={"memory": {"retirement_head": "new"}}) as txn:
        (txn.directory / "adapter.safetensors").write_bytes(b"rejected")
    assert not txn.directory.exists()
    assert store.current_manifest()["generation"] == first["generation"]
    assert store.current_manifest()["metadata"]["memory"]["retirement_head"] == "old"


@pytest.mark.parametrize("published", [False, True])
def test_interrupt_at_publication_preserves_matching_generation(tmp_path, monkeypatch, published):
    store = state.LearningStateStore({"source": "model"}, state_root=tmp_path)
    previous = commit(store, b"previous")
    txn = store.begin(metadata={"memory": {"retirement_head": "new"}})
    (txn.directory / "adapter.safetensors").write_bytes(b"candidate")
    atomic = state._atomic_json

    def interrupt(path, value):
        if path.name == "current.json":
            if published:
                atomic(path, value)
            raise KeyboardInterrupt("publication boundary")
        atomic(path, value)

    monkeypatch.setattr(state, "_atomic_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        txn.commit("adapter.safetensors")
    txn.abort()
    assert txn.committed is published
    assert txn.directory.exists() is published
    assert store.current_manifest()["generation"] == (txn.generation if published else previous["generation"])
    assert Path(store.current_checkpoint()).read_bytes() == (b"candidate" if published else b"previous")


def test_raw_training_inherits_retirement_metadata(tmp_path):
    store = state.LearningStateStore({"source": "model"}, state_root=tmp_path)
    commit(store, memory={"retirement_head": "committed", "replay": []})
    second = commit(store, b"next weights", objective="plain training")
    assert second["metadata"]["memory"]["retirement_head"] == "committed"


def test_stale_plan_and_concurrent_commit_fail_closed(tmp_path):
    store = state.LearningStateStore({"source": "model"}, state_root=tmp_path)
    stale = store.begin(metadata={"expected_parent": None})
    (stale.directory / "adapter.safetensors").write_bytes(b"old candidate")
    commit(store)
    with pytest.raises(RuntimeError, match="older parameter"):
        store.begin(metadata={"expected_parent": None})
    with pytest.raises(RuntimeError, match="changed during"):
        stale.commit("adapter.safetensors")
    stale.abort()


def test_corrupt_checkpoint_is_never_restored(tmp_path):
    store = state.LearningStateStore({"source": "model"}, state_root=tmp_path)
    commit(store)
    checkpoint = Path(store.current_checkpoint())
    checkpoint.write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="integrity"):
        store.current_checkpoint()


def test_only_current_and_previous_committed_weights_remain(tmp_path):
    store = state.LearningStateStore({"source": "model"}, state_root=tmp_path)
    stages = [commit(store, str(i).encode()) for i in range(4)]
    generations = {path.name for path in (store.directory / "generations").iterdir()}
    assert generations == {stages[-1]["generation"], stages[-2]["generation"]}


def test_wrong_model_managed_file_cannot_bypass_as_legacy(tmp_path):
    from types import SimpleNamespace
    first = state.LearningStateStore({"source": "bonsai2"}, state_root=tmp_path)
    second = state.LearningStateStore({"source": "crack"}, state_root=tmp_path)
    commit(first)
    with pytest.raises(RuntimeError, match="different model"):
        state.validate_checkpoint(SimpleNamespace(learning_state=second), first.current_checkpoint(), allow_legacy=True)


def test_model_tokenizer_or_second_gguf_shard_changes_identity(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    first = model / "model-00001-of-00002.gguf"
    second = model / "model-00002-of-00002.gguf"
    first.write_bytes(b"shard1")
    second.write_bytes(b"shard2")
    identity = state.model_identity(first, state_root=tmp_path / "state", adapter_format="gguf")
    second.write_bytes(b"new shard2")
    changed = state.model_identity(first, state_root=tmp_path / "state", adapter_format="gguf")
    assert state.identity_digest(identity) != state.identity_digest(changed)
    assert len(identity["files"]) == 2


def test_directory_tokenizer_change_is_a_different_memory_bank(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"base")
    tokenizer = model / "tokenizer.json"
    tokenizer.write_text('{"vocab":1}')
    old = state.model_identity(model, state_root=tmp_path / "state")
    tokenizer.write_text('{"vocab":2}')
    new = state.model_identity(model, state_root=tmp_path / "state")
    assert old != new


@pytest.mark.parametrize("asset", ["hadamard.json", "merges.txt", "runtime/vision_artifact.py"])
def test_publisher_runtime_and_tokenizer_assets_are_identity_bound(tmp_path, asset):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"base")
    runtime = model / asset
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text("original")
    old = state.model_identity(model, state_root=tmp_path / "state")
    runtime.write_text("changed")
    new = state.model_identity(model, state_root=tmp_path / "state")
    assert old != new


def test_empty_checkpoint_directory_cannot_publish_unrelated_sibling(tmp_path):
    store = state.LearningStateStore({"source": "model"}, state_root=tmp_path)
    with store.begin() as txn:
        (txn.directory / "empty").mkdir()
        (txn.directory / "other.bin").write_bytes(b"not checkpoint")
        with pytest.raises(RuntimeError, match="authenticated"):
            txn.commit("empty")
    assert store.current_manifest() is None


def test_cancel_during_publication_keeps_prior_checkpoint(tmp_path):
    import threading
    store = state.LearningStateStore({"source": "model"}, state_root=tmp_path)
    previous = commit(store)
    cancel = threading.Event()
    cancel.set()
    with store.begin() as txn:
        (txn.directory / "adapter.safetensors").write_bytes(b"rejected")
        with pytest.raises(InterruptedError):
            txn.commit("adapter.safetensors", cancel_event=cancel)
    assert store.current_manifest()["generation"] == previous["generation"]


def test_checkpoint_cannot_escape_generation(tmp_path):
    store = state.LearningStateStore({"source": "model"}, state_root=tmp_path)
    with store.begin() as txn:
        with pytest.raises(ValueError, match="inside"):
            txn.commit("../different-model.safetensors")


def test_malformed_or_nonfinite_metadata_does_not_publish(tmp_path):
    store = state.LearningStateStore({"source": "model"}, state_root=tmp_path)
    with store.begin(metadata={"loss": float("nan")}) as txn:
        (txn.directory / "adapter.safetensors").write_bytes(b"parameters")
        with pytest.raises(ValueError):
            txn.commit("adapter.safetensors")
    assert store.current_manifest() is None
