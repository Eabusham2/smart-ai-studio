"""Behavioral tests for bounded Torch learning and native candidate transactions."""
from pathlib import Path
import math
import threading
from types import SimpleNamespace

import pytest

from core.controller_runtime import UniversalControllerBackend
from core.learning_state import LearningStateStore
from core.native_context_learning import score_server_probe, update_deployment


@pytest.fixture
def tiny_controller(tmp_path, monkeypatch, request):
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("accelerate")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    original_threads = torch.get_num_threads()
    rng_state = torch.random.get_rng_state()
    torch.set_num_threads(1)
    torch.manual_seed(29)
    base = tmp_path / "small-causal"
    if getattr(request, "param", None) == "split_queries":
        model = tf.LlamaForCausalLM(tf.LlamaConfig(vocab_size=16, max_position_embeddings=96, hidden_size=24,
                                  intermediate_size=48, num_hidden_layers=2, num_attention_heads=3,
                                  num_key_value_heads=1, bos_token_id=1, eos_token_id=2))
    else:
        model = tf.GPT2LMHeadModel(tf.GPT2Config(vocab_size=16, n_positions=getattr(request, "param", 16), n_embd=24,
                                                 n_layer=1, n_head=2, bos_token_id=1, eos_token_id=2))
    model.save_pretrained(base)
    vocabulary = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "color": 3, "slate": 4,
                  "vault": 5, "What": 6, "is": 7, "the": 8, "?": 9, "red": 10}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tf.PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]").save_pretrained(base)
    monkeypatch.setattr("config.paths.get_portable_data_dir", lambda: str(tmp_path / "state"))
    info = {"controller_runtime": "transformers_auto", "transformers_auto_class": "AutoModelForCausalLM"}
    backend = UniversalControllerBackend(str(base), info)
    try:
        assert backend.load_model()
        assert backend.training_ready()
        yield backend
    finally:
        backend.unload_model()
        torch.set_num_threads(original_threads)
        torch.random.set_rng_state(rng_state)


def test_real_peft_full_coverage_rollback_and_reload(tiny_controller, monkeypatch):
    import torch
    from core.learning_data import training_coverage

    backend = tiny_controller
    prompt, completion = "What is the vault color?", "slate"
    data = [{"prompt": ("vault color " * 280) + prompt, "completion": completion},
            {"prompt": prompt, "completion": completion}]
    expected = training_coverage(backend.tokenizer, data, max_tokens=16)
    base_weights = {k: p.detach().clone() for k, p in backend.model.named_parameters()}
    observed = []
    def accept(candidate):
        observed.append(candidate.score_learning_probe(prompt, completion))
        return {"accepted": math.isfinite(observed[-1])}

    metadata, drift = backend.train_mini_batch({}, data, learning_rate=0.006, steps=1,
                                              lambda_ewc=1.0, validation_callback=accept)
    assert metadata["training_windows"] == expected["windows"] > 32
    assert metadata["supervised_tokens"] == expected["supervised_tokens"] > 512
    assert metadata["ewc_lambda"] == 1.0 and drift > 0
    assert metadata["ewc_used"] and metadata["fisher_tensors"] > 0
    assert backend.learning_state.current_manifest()["metadata"]["metrics"] == metadata
    assert metadata["optimizer_resumed"] is False
    assert observed and backend.score_learning_probe(prompt, completion) == pytest.approx(observed[0])
    for k, p in backend.model.get_base_model().named_parameters():
        if "lora_" not in k:
            assert not p.requires_grad
            assert torch.equal(p, base_weights[k.replace(".base_layer.", ".")])

    generation = backend.learning_state.current_manifest()["generation"]
    state = {name: p.detach().clone() for name, p in backend.model.named_parameters() if p.requires_grad}
    with pytest.raises(ValueError, match="no positive diagonal Fisher"):
        backend.train_mini_batch({}, data[-1:], steps=1, lambda_ewc=1.,
                                 fisher_matrix={k: torch.zeros_like(v) for k, v in state.items()},
                                 validation_callback=accept)
    assert backend.learning_state.current_manifest()["generation"] == generation
    for action in ("reject", "cancel"):
        cancellation = threading.Event()
        def reject(candidate):
            assert any(not torch.equal(p, state[k]) for k, p in candidate.model.named_parameters() if k in state)
            if action == "cancel":
                cancellation.set()
                return True
            return {"accepted": False}
        with pytest.raises(RuntimeError, match="cancelled|validation"):
            backend.train_mini_batch({}, data[-1:], steps=1, learning_rate=0.008,
                                     cancel_event=cancellation, validation_callback=reject)
        assert backend.learning_state.current_manifest()["generation"] == generation
        assert all(torch.equal(p, state[k]) for k, p in backend.model.named_parameters() if k in state)
    backend.unload_model()
    assert backend.load_model()
    assert backend.score_learning_probe(prompt, completion) == pytest.approx(observed[0], abs=1e-6)
    from safetensors.torch import load_file
    previous = load_file(Path(backend.adapter_path).parent / "optimizer.safetensors")
    continued, _ = backend.train_mini_batch({}, data[-1:], steps=1, validation_callback=accept)
    assert continued["optimizer_resumed"]
    assert not continued["ewc_used"] and continued["fisher_tensors"] == 0
    resumed = load_file(Path(backend.adapter_path).parent / "optimizer.safetensors")
    for key in previous:
        if key.endswith("/step"):
            assert resumed[key].item() == previous[key].item() + 1
    from core.learning_state import LearningTransaction
    commit = LearningTransaction.commit
    def interrupt_after_publish(txn, *args, **kwargs):
        commit(txn, *args, **kwargs)
        raise KeyboardInterrupt()
    monkeypatch.setattr(LearningTransaction, "commit", interrupt_after_publish)
    with pytest.raises(KeyboardInterrupt):
        backend.train_mini_batch({}, data[-1:], steps=1, validation_callback=accept)
    assert backend._loaded_learning_checkpoint == backend.adapter_path == backend.learning_state.current_checkpoint()
    assert backend.last_learning_metrics == backend.learning_state.current_manifest()["metadata"]["metrics"]
    score = backend.score_learning_probe(prompt, completion)
    backend.unload_model()
    assert backend.load_model()
    assert backend.score_learning_probe(prompt, completion) == pytest.approx(score, abs=1e-6)


@pytest.mark.parametrize("failure", [None, "reload", "reject", "cancel", "converter", "post_commit_interrupt"])
def test_native_peft_and_served_file_publish_together(tmp_path, failure, monkeypatch):
    store = LearningStateStore({"base": "fixture"}, state_root=tmp_path)
    with store.begin() as first:
        (first.directory / "peft").mkdir()
        (first.directory / "peft" / "adapter_config.json").write_text('{"old":true}')
        (first.directory / "adapter.gguf").write_bytes(b"old")
        first.commit("adapter.gguf")
    original = store.current_checkpoint()
    cancellation = threading.Event()

    class Native:
        model = object()
        adapter_path = original
        adapters = {"old": True}
        learning_state = store
        _candidate_deployment = None

        def _verify_training_parent(self, trainer):
            return "verified-test-parent"

        def unload_model(self):
            self.model = None

        def load_model(self):
            if self._candidate_deployment and failure == "reload":
                return False
            self.adapter_path = self._candidate_deployment or store.current_checkpoint()
            self.model = Path(self.adapter_path).read_bytes()
            return True

    class Trainer:
        def train(self, data, *, output_directory, previous_peft, **kwargs):
            assert (previous_peft / "adapter_config.json").read_text() == '{"old":true}'
            assert backend.model is None
            peft = output_directory / "peft"
            peft.mkdir()
            (peft / "adapter_config.json").write_text('{"new":true}')
            if failure == "converter":
                raise RuntimeError("converter failed")
            artifact = output_directory / "adapter.gguf"
            artifact.write_bytes(b"new")
            return {"trainable_parameters_touched": 1}, 1.0, 1, str(artifact)

    def validate(candidate):
        assert candidate.model == b"new" and store.current_checkpoint() == original
        if failure == "cancel":
            cancellation.set()
        return failure != "reject"

    backend = Native()
    options = dict(learning_rate=0.1, steps=1, fisher_matrix=None, lambda_ewc=0.,
                   cancel_event=cancellation, validation_callback=validate, learning_metadata={})
    if failure == "post_commit_interrupt":
        from core.learning_state import LearningTransaction
        commit = LearningTransaction.commit
        def interrupt_after_publish(txn, *args, **kwargs):
            commit(txn, *args, **kwargs)
            raise KeyboardInterrupt()
        monkeypatch.setattr(LearningTransaction, "commit", interrupt_after_publish)
        with pytest.raises(KeyboardInterrupt):
            update_deployment(backend, Trainer(), [], **options)
        assert store.current_checkpoint() == backend.adapter_path
        assert Path(backend.adapter_path).read_bytes() == backend.model == b"new"
        assert backend.last_learning_metrics == store.current_manifest()["metadata"]["metrics"]
    elif failure is None:
        update_deployment(backend, Trainer(), [], **options)
        assert Path(store.current_checkpoint()).read_bytes() == backend.model == b"new"
    else:
        with pytest.raises(RuntimeError):
            update_deployment(backend, Trainer(), [], **options)
        assert store.current_checkpoint() == backend.adapter_path == original
        assert backend.model == b"old" and backend.adapters == {"old": True}
        assert len(list((store.directory / "generations").iterdir())) == 1


def test_teacher_forced_server_probe_requires_actual_token_ids():
    prefixes = []
    def request(path, payload):
        if path == "/tokenize":
            return {"tokens": [1, 2] if payload["content"].endswith("\n\n") else [1, 2, 3, 4]}
        assert payload["post_sampling_probs"] is False
        prefixes.append(list(payload["prompt"]))
        return {"completion_probabilities": [{"top_logprobs": [{"id": len(prefixes) + 2, "logprob": -0.5}]}]}
    assert score_server_probe(SimpleNamespace(n_ctx=8, _request=request), "Q", "AB") == 0.5
    assert prefixes == [[1, 2], [1, 2, 3]]
    def unavailable(path, payload):
        if path == "/tokenize":
            return {"tokens": [1] if payload["content"].endswith("\n\n") else [1, 1]}
        return {"completion_probabilities": [{"top_logprobs": [{"id": 2, "logprob": -0.3}]}]}
    with pytest.raises(RuntimeError, match="did not report"):
        score_server_probe(SimpleNamespace(n_ctx=8, _request=unavailable), "Q", "A")


def test_cpu_gguf_bridge_trains_real_parent_before_conversion(tiny_controller, tmp_path, monkeypatch):
    from safetensors.torch import load_file
    from core.gguf_lora_trainer import GGUFLoRATrainer

    output = tmp_path / "candidate"
    output.mkdir()
    trainer = GGUFLoRATrainer(model_path="selected.gguf", base_model_id=tiny_controller.model_path,
                              adapter_root=str(tmp_path / "bridge"), rank=2, alpha=2)
    import sys
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    def reject_conversion(peft_path, target):
        assert Path(peft_path).is_relative_to(output)
        weights = load_file(Path(peft_path) / "adapter_model.safetensors")
        assert any(t.abs().max().item() > 0 for name, t in weights.items() if "lora_B" in name)
        raise RuntimeError("converter rejected architecture")
    monkeypatch.setattr(trainer, "_convert_to_gguf", reject_conversion)
    with pytest.raises(RuntimeError, match="converter rejected"):
        trainer.train([{"prompt": "vault color", "completion": "slate"}], output_directory=output, steps=1)
    assert trainer.training_mode == "resident"
    assert not (output / "adapter.gguf").exists()


def test_ternary_rebuild_checks_tensor_layout_and_actual_token_vocabulary(tmp_path):
    from core.engines.bitnet_cpp_engine import BitNetCppReasoningBackend
    from tests.unit.gguf_fixtures import write_gguf
    selected, candidate = tmp_path / "base.gguf", tmp_path / "candidate.gguf"
    write_gguf(selected, tensor_type=36)
    backend = SimpleNamespace(_selected_model_file=str(selected))
    write_gguf(candidate, tensor_type=36)
    BitNetCppReasoningBackend._check_deployment_format(backend, str(candidate))
    write_gguf(candidate, tensor_type=37)
    with pytest.raises(RuntimeError, match="tensor_layout"):
        BitNetCppReasoningBackend._check_deployment_format(backend, str(candidate))
    write_gguf(candidate, tensor_type=36, extra={"tokenizer.ggml.tokens": [f"other{i}" for i in range(16)]})
    with pytest.raises(RuntimeError, match="tokenizer_layout"):
        BitNetCppReasoningBackend._check_deployment_format(backend, str(candidate))


def test_native_adapter_requires_exact_registered_path_and_unit_scale(tmp_path, monkeypatch):
    from core.engines.prism_speculative_engine import PrismSpeculativeReasoningBackend
    backend = PrismSpeculativeReasoningBackend("unused", "unused", adapter_root=tmp_path)
    backend.adapter_path = str(tmp_path / "candidate.gguf")
    for path, scale in ((backend.adapter_path, 0.), (str(tmp_path / "other.gguf"), 1.0)):
        monkeypatch.setattr(backend, "_request", lambda *a, **kw: {"adapters": [{"path": path, "scale": scale}]})
        with pytest.raises(ValueError, match="did not confirm"):
            backend._verify_served_adapter()
    monkeypatch.setattr(backend, "_request",
                        lambda *a, **kw: {"adapters": [{"path": backend.adapter_path, "scale": 1.0}]})
    assert backend._verify_served_adapter()


@pytest.mark.parametrize("tiny_controller", [96], indirect=True)
def test_train_pairs_measures_source_free_recall_on_restarted_cpu_model(tiny_controller):
    from core.online_consolidator import AwakeOnlineConsolidator
    backend = tiny_controller
    learner = AwakeOnlineConsolidator(backend)
    data = [{"prompt": "What is the vault color?", "completion": "the vault color is slate"}] * 8
    update = learner.train_pairs(data, evidence=["the vault color is slate"], learning_rate=0.004)
    assert update["probe_nll_after"][0] < update["probe_nll_before"][0]
    assert update["ewc_active"] and update["trainable_parameters_touched"] > 0
    persisted = backend.learning_state.current_manifest()["metadata"]["metrics"]
    assert persisted["ewc_used"] and persisted["fisher_tensors"] > 0
    backend.unload_model()
    assert backend.load_model()
    cue = "Complete the earlier statement: the vault color is"
    assert backend.score_learning_probe(cue, "slate") == pytest.approx(update["probe_nll_after"][0], abs=1e-6)


def test_converter_cancellation_reaps_the_owned_process(tmp_path):
    import sys
    import time
    import psutil
    from core.training_lineage import run_conversion
    pid_file = tmp_path / "converter.pid"
    event = threading.Event()
    errors = []

    def run():
        try:
            run_conversion([sys.executable, "-c",
                            "import os,sys,time; open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(30)",
                            str(pid_file)], cancel_event=event)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pid_file.exists()
    finally:
        event.set()
        worker.join(timeout=5)
    assert not worker.is_alive() and errors and "cancelled" in str(errors[0])
    assert not psutil.pid_exists(int(pid_file.read_text()))


def test_reward_rejection_restores_fresh_controller_and_commit_reloads(tiny_controller):
    import torch
    backend = tiny_controller
    initial = backend.model
    saved = {name: (p.detach().clone(), p.requires_grad) for name, p in initial.named_parameters()}
    prompt_ids = backend.tokenizer.encode("vault color", add_special_tokens=False)
    rows = [{"prompt": "vault color", "completion": word, "rollout_prompt_ids": prompt_ids,
             "rollout_completion_ids": backend.tokenizer.encode(word, add_special_tokens=False), "reward": reward}
            for word, reward in (("slate", 1.), ("red", 0.))]
    options = dict(objective="reward_policy", steps=1, learning_rate=.001, max_window_tokens=16)
    with pytest.raises(RuntimeError, match="validation"):
        backend.train_mini_batch({}, rows, validation_callback=lambda candidate: False, **options)
    assert backend.model is initial and backend.learning_state.current_manifest() is None
    assert all(name in saved and torch.equal(saved[name][0], p) and saved[name][1] == p.requires_grad
               for name, p in backend.model.named_parameters())
    metrics, drift = backend.train_mini_batch({}, rows, validation_callback=lambda candidate: True, **options)
    assert metrics["objective"] == "reward_policy" and drift > 0
    score = backend.score_learning_probe("vault color", "slate")
    backend.unload_model()
    assert backend.load_model() and backend.score_learning_probe("vault color", "slate") == pytest.approx(score)


def test_named_scope_does_not_restore_global_native_legacy(tmp_path, monkeypatch):
    from core.adaptation_capabilities import adaptation_capabilities
    from core.memory_scopes import LearningScope
    from core.native_context_learning import select_deployment
    legacy = tmp_path / "legacy.gguf"
    legacy.write_bytes(b"global")
    store = SimpleNamespace(scope=LearningScope("request", "this-request"), current_checkpoint=lambda: None)
    backend = SimpleNamespace(model=object(), training_base_model_id="base", legacy_adapter_path=str(legacy))
    monkeypatch.setattr("core.native_context_learning.bind_learning_state", lambda *a, **kw: store)
    assert select_deployment(backend, tmp_path, tmp_path, "gguf", legacy) is None
    assert backend._loaded_legacy_deployment is None
    store.scope = LearningScope("document", "sample", generation="1" * 32, pin="2" * 32)
    backend.learning_state = store
    result = adaptation_capabilities(backend)
    for key in ("causal_lm", "reward_policy", "query_only"):
        assert not result[key] and "read-only" in result["reasons"][key]

@pytest.mark.parametrize("tiny_controller", ["split_queries"], indirect=True)
def test_query_checkpoint_resumes_only_query_moments_and_rejects_cancel(tiny_controller):
    from safetensors.torch import load_file
    from core.adaptation_capabilities import adaptation_capabilities
    import torch

    backend = tiny_controller
    assert adaptation_capabilities(backend)["query_only"]
    rows = [{"prompt": "What is the vault color?", "completion": "slate"}]
    backend.train_mini_batch({}, rows, steps=1, validation_callback=lambda _: True)
    moments = load_file(Path(backend.adapter_path).parent / "optimizer.safetensors")
    weights = {k: v.detach().clone() for k, v in backend.model.named_parameters()}
    options = dict(objective="query_only", context_text="What is the vault color ? the vault color is slate",
                   max_context_tokens=32, max_window_tokens=4, steps=2, learning_rate=.002)
    metrics, _ = backend.train_mini_batch({}, rows, validation_callback=lambda _: True, **options)
    assert metrics == backend.learning_state.current_manifest()["metadata"]["metrics"]
    assert metrics["kv_prefills"] == 1 and metrics["optimizer_resumed"]
    for name, p in backend.model.named_parameters():
        if ".q_proj.lora_" not in name:
            assert torch.equal(p, weights[name])
    after = load_file(Path(backend.adapter_path).parent / "optimizer.safetensors")
    assert after.keys() == moments.keys()
    for key in after:
        if ".q_proj.lora_" not in key:
            assert torch.equal(after[key], moments[key])
        elif key.endswith("/step"):
            assert after[key].item() == moments[key].item() + 2
    expected = backend.score_learning_probe("vault color", "slate")
    checkpoint = backend.adapter_path
    event = threading.Event()
    def stop(_):
        event.set()
        return True
    with pytest.raises(RuntimeError, match="cancelled"):
        backend.train_mini_batch({}, rows, cancel_event=event, validation_callback=stop, **options)
    assert backend.score_learning_probe("vault color", "slate") == pytest.approx(expected)
    backend.unload_model()
    assert backend.load_model() and backend.adapter_path == checkpoint
    assert backend.score_learning_probe("vault color", "slate") == pytest.approx(expected)
