"""Persistent candidate deployment and teacher-forced scoring on native runtimes."""
from __future__ import annotations

import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from core.learning_state import bind_learning_state, model_identity, select_legacy_checkpoint
from core.torch_context_learning import check_cancelled, default_validation, validate_candidate


@contextmanager
def native_training_output(trainer, output_directory, destination, previous_peft, cancel_event, data, objective):
    """Give managed callers their stage and standalone callers the historical cumulative files."""
    if output_directory is not None:
        output = Path(output_directory)
        previous_dir, previous_cancel, previous_mode = trainer.peft_dir, trainer.cancel_event, trainer._legacy_training
        try:
            yield output, previous_peft, output / destination.name, data
        finally:
            trainer.peft_dir, trainer.cancel_event, trainer._legacy_training = previous_dir, previous_cancel, previous_mode
        return

    from core.learning_state import _process_commit_lock
    from core.training_lineage import standalone_training_parent

    if objective == "causal_lm":
        data = [row for row in data or () if str(row.get("prompt") or "").strip() and str(row.get("completion") or "").strip()]
        if not data:
            raise RuntimeError("The standalone trainer received no prompt/completion pairs")
    root = destination.parent
    root.mkdir(parents=True, exist_ok=True)
    peft = root / "peft"
    with _process_commit_lock(root), tempfile.TemporaryDirectory(prefix="native-training-", dir=root) as temporary:
        previous_peft_dir, previous_source = trainer.peft_dir, trainer.training_source
        previous_cancel, previous_mode = trainer.cancel_event, trainer._legacy_training
        output = Path(temporary)
        previous = previous_peft or (peft if (peft / "adapter_config.json").is_file() else None)
        try:
            check_cancelled(cancel_event)
            trainer.training_source = previous_source or standalone_training_parent(trainer.base_model_id)
            yield output, previous, destination, data
            check_cancelled(cancel_event)
            # Keep the cumulative PEFT factors, optimizer and converted deployment in sync.
            # Backups are renames on the same volume, including for full BitNet weights.
            files = [output / "peft", output / "optimizer.safetensors", output / destination.name]
            if not all(item.exists() for item in files):
                raise RuntimeError("The native trainer did not produce a complete standalone checkpoint")
            backups = []
            published = []
            try:
                for item in files:
                    target = root / item.name
                    backup = output / (item.name + ".previous")
                    if target.exists():
                        os.replace(target, backup)
                        backups.append((target, backup))
                    os.replace(item, target)
                    published.append((target, item))
            except BaseException:
                for target, item in reversed(published):
                    os.replace(target, item)
                for target, backup in reversed(backups):
                    os.replace(backup, target)
                raise
        finally:
            trainer.peft_dir = previous_peft_dir
            trainer.training_source = previous_source
            trainer.cancel_event, trainer._legacy_training = previous_cancel, previous_mode


def select_deployment(backend, base_path, root, kind, legacy_path):
    """Load only a committed generation for the selected base, or our live trial."""
    options = {"adapter_format": kind, "state_root": Path(root) / "learned_memory"}
    info = {"training_base_model_id": backend.training_base_model_id, "mmproj_path": getattr(backend, "mmproj_path", None)}
    trial = getattr(backend, "_candidate_deployment", None)
    if trial is not None:
        if model_identity(base_path, info, **options) != backend.learning_state.identity:
            raise RuntimeError("The selected base model changed during learning")
        if not Path(trial).is_file():
            raise RuntimeError("Candidate deployment is missing")
        return str(trial)
    store = bind_learning_state(backend, model_path=base_path, model_info=info, **options)
    current = store.current_checkpoint()
    backend._loaded_legacy_deployment = None
    legacy = select_legacy_checkpoint(backend, store, current, legacy_path,
                                      getattr(backend, "legacy_adapter_path", None), kind)
    if legacy:
        checker = getattr(backend, "_check_deployment_format", None)
        if checker is not None:
            checker(legacy)
        backend._loaded_legacy_deployment = legacy
        return legacy
    return current


def update_deployment(backend, trainer, data, *, learning_rate, steps, fisher_matrix,
                      lambda_ewc, cancel_event, validation_callback, learning_metadata, anchor_texts=None,
                      learning_mode="managed", objective='causal_lm', supervise_prompt=True, row_weights=None, max_window_tokens=256,
                      context_text=None, max_context_tokens=None,
                      reward_weighting="group_relative", entropic_kl_target=None, reference_kl_coef=0.0):
    """Validate a candidate using the *serving* backend before switching current.json."""
    if objective not in {"causal_lm", "reward_policy", "query_only"}:
        raise ValueError("Unknown native conversion objective")
    if backend.model is None:
        raise RuntimeError("Load the selected model before learning")
    if objective == "reward_policy":
        from core.reward_adaptation import reward_advantages
        reward_advantages(data, reward_weighting=reward_weighting, entropic_kl_target=entropic_kl_target)
    store = backend.learning_state
    txn = store.begin(metadata=learning_metadata, inherit_metadata=learning_mode == "managed")
    original_metadata = backend.adapters
    unloaded = False
    try:
        previous = store.current_checkpoint() or getattr(backend, "_loaded_legacy_deployment", None)
        if (previous or "") != (backend.adapter_path or ""):
            raise RuntimeError("The loaded model is behind its committed state. Reload before learning.")
        check_cancelled(cancel_event)
        # Merging and quantization can erase a BF16 update. Standalone
        # callers also validate behavior on the actual served deployment.
        if learning_mode == "managed" and validation_callback is None:
            validation_callback = default_validation(backend, data, objective=objective, reward_weighting=reward_weighting, entropic_kl_target=entropic_kl_target)
        trainer.allow_parent_download = learning_mode == "legacy"
        lineage = backend._verify_training_parent(trainer)
        check_cancelled(cancel_event)
        unloaded = True
        backend.unload_model()
        if learning_mode == "legacy":
            metrics, drift, touched, deployment = trainer.train(
                data, learning_mode="legacy", output_directory=txn.directory,
                previous_peft=(Path(previous).parent / "peft") if previous else None,
                learning_rate=learning_rate, steps=steps, max_length=max_window_tokens, cancel_event=cancel_event,
            )
        else:
            metrics, drift, touched, deployment = trainer.train(
                data, learning_mode="managed", output_directory=txn.directory,
                previous_peft=(Path(previous).parent / "peft") if previous else None,
                learning_rate=learning_rate, steps=steps, fisher_matrix=fisher_matrix,
                lambda_ewc=lambda_ewc, cancel_event=cancel_event, anchor_texts=anchor_texts,
                reward_weighting=reward_weighting, entropic_kl_target=entropic_kl_target, reference_kl_coef=reference_kl_coef,
                objective=objective, supervise_prompt=supervise_prompt, row_weights=row_weights, max_length=max_window_tokens,
                **({"context_text": context_text, "max_context_tokens": max_context_tokens} if objective == "query_only" else {}),
            )
        metrics["training_parent_sha256"] = lineage
        if objective == "query_only":
            metrics["kv_cache_runtime"] = "verified_transformers_parent"
        if not math.isfinite(drift) or drift <= 0 or touched <= 0:
            raise RuntimeError("Candidate trainer reported invalid parameter drift")
        check_cancelled(cancel_event)
        check_format = getattr(backend, "_check_deployment_format", None)
        if check_format is not None:
            check_format(deployment)
        backend._candidate_deployment = deployment
        if not backend.load_model():
            raise RuntimeError("Candidate deployment could not be loaded: " + str(getattr(backend, "last_error", "")))
        evidence = (validate_candidate(backend, validation_callback, cancel_event) if learning_mode == "managed"
                    else {"method": "deployment_reload"})
        check_cancelled(cancel_event)
        txn.commit(Path(deployment).name, metadata={"metrics": metrics, "validation": evidence, "drift_l2": drift},
                   cancel_event=cancel_event)
        return dict(metrics), drift
    except BaseException:
        if unloaded and not txn.committed:
            try:
                backend.unload_model()
            finally:
                backend._candidate_deployment = None
            if not backend.load_model():
                raise RuntimeError("Failed to restore the previous committed native deployment")
            if backend.adapter_path == (previous or ""):
                backend.adapters = original_metadata
            else:
                # A concurrent writer won the compare-and-swap. load_model()
                # restored that winner and its metrics must describe this model.
                winner = store.current_manifest(verify=False) or {}
                backend.adapters = winner.get("metadata", {}).get("metrics", {})
                backend.last_learning_metrics = dict(backend.adapters)
        raise
    finally:
        if txn.committed:
            backend.adapters = metrics
            backend.last_learning_metrics = dict(metrics)
        backend._candidate_deployment = None
        txn.abort()


def _probe_windows(tokenize, prompt, completion, n_ctx):
    from core.learning_data import iter_probe_windows

    def encode(text, **kwargs):
        ids = tokenize(text, True)
        if not isinstance(ids, (tuple, list)) or any(type(t) is not int or t < 0 for t in ids):
            raise RuntimeError("Native tokenizer returned invalid probe IDs")
        return ids

    # These GGUF adapters accept raw completion prompts. Use the same neutral
    # causal cue for the baseline and every candidate. Tokenize cue + answer
    # together so a boundary merge cannot invent an extra whitespace token.
    tokenizer = SimpleNamespace(encode=encode)
    for w in iter_probe_windows(tokenizer, prompt, completion, max_tokens=min(256, int(n_ctx))):
        yield list(w.input_ids[:w.loss_start]), w.input_ids[w.loss_start:]


def score_llama_probe(model, prompt, completion):
    from core.gguf_placement import last_token_logits

    def tokenize(text, bos):
        return model.tokenize(text.encode("utf-8"), add_bos=bos, special=True)

    total = count = 0
    try:
        for prefix, answer in _probe_windows(tokenize, prompt, completion, model.n_ctx()):
            model.reset()
            count += len(answer)
            model.eval(prefix)
            for i, target in enumerate(answer):
                logits = last_token_logits(model)
                peak = float(max(logits))
                # The target and normalizer come from the complete real vocabulary.
                if not math.isfinite(peak) or not 0 <= target < len(logits):
                    raise RuntimeError("Invalid target or logits in native learning probe")
                value = (peak - float(logits[target])
                         + math.log(math.fsum(math.exp(float(x) - peak) for x in logits)))
                if not math.isfinite(value):
                    raise RuntimeError("Non-finite native learning probe")
                total += value
                if i + 1 < len(answer):
                    model.eval([target])
    finally:
        model.reset()
    if not count:
        raise RuntimeError("The selected GGUF has no scoreable answer tokens")
    return total / count


def score_server_probe(backend, prompt, completion):
    def tokenize(text, bos):
        result = backend._request("/tokenize", {"content": text, "add_special": bos, "parse_special": True})
        tokens = result.get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError("The server does not expose exact tokenizer IDs")
        return tokens

    total = count = 0
    for prefix, answer in _probe_windows(tokenize, prompt, completion, backend.n_ctx):
        for target in answer:
            total -= _server_target_logprob(backend, tokenize, prefix, target)
            prefix.append(target)
            count += 1
    if not count:
        raise RuntimeError("The selected server has no scoreable answer tokens")
    return total / count


def _server_target_logprob(backend, tokenize, prefix, target):
    found = None
    expected_piece = None
    for count in (64, 2048, 32768):
        result = backend._request("/completion", {
            "prompt": prefix, "n_predict": 1, "stream": False, "cache_prompt": False,
            "n_probs": count, "post_sampling_probs": False, "temperature": 1.0,
            "top_k": 0, "top_p": 1.0, "min_p": 0.0, "typical_p": 1.0,
            "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "samplers": ["temperature"], "seed": 0,
        })
        rows = result.get("completion_probabilities", result.get("probs"))
        if result.get("truncated") or not isinstance(rows, list) or len(rows) != 1:
            raise RuntimeError("The selected server cannot verify complete target log-probabilities")
        row = rows[0]
        candidates = row.get("top_logprobs", row.get("probs", []))
        for token in candidates:
            token_id = token.get("id")
            if token_id is None:
                # Older bitnet.cpp servers identify vocabulary entries by
                # their decoded piece. Require a lossless, unambiguous ID.
                piece = token.get("tok_str")
                if not isinstance(piece, str) or not piece or "\ufffd" in piece:
                    continue
                if expected_piece is None:
                    expected_piece = backend._request("/detokenize", {"tokens": [target]}).get("content")
                if piece != expected_piece or tokenize(piece, False) != [target]:
                    continue
            elif type(token_id) is not int or token_id != target:
                continue
            value = token.get("logprob")
            if value is None:
                probability = token.get("prob")
                if type(probability) in (float, int) and 0 < probability <= 1:
                    value = math.log(probability)
            if type(value) in (float, int) and math.isfinite(value) and value <= 0:
                found = float(value)
                break
        if found is not None:
            break
    if found is None:
        raise RuntimeError("The selected server did not report the requested answer token's probability")
    return found
