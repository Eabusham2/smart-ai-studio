"""Disposable inference state for Best Work drafts and model comparisons.

The loaded model is shared. A fork owns only prompt/KV state, never another
copy of the weights or a learned-memory scope. The caller supplies the complete
conversation again when adopting the winner, so native runtimes may discard
their working KV rather than snapshotting an entire model context.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager


_MISSING = object()


def _backend(engine):
    from core.awake_auto_hook import _active_trainable_backend
    return _active_trainable_backend(engine)


@contextmanager
def _attributes(owner, changes):
    """Restore inherited methods as inherited methods, including on errors."""
    before = {name: vars(owner).get(name, _MISSING) for name in changes}
    try:
        for name, value in changes.items():
            setattr(owner, name, value)
        yield
    finally:
        for name, value in before.items():
            if value is _MISSING:
                vars(owner).pop(name, None)
            else:
                setattr(owner, name, value)


def _read_only(*_args, **_kwargs):
    raise RuntimeError("Best Work inference cannot update model weights or adapters")


@contextmanager
def inference_cycle(engine):
    """Freeze learning for one branch/refinement/selection cycle.

    Existing context consolidation runs BEFORE this guard. Its update lock
    also waits for an already-running background consolidation to finish.
    Acquire that lock on the caller, not MLX's owner thread: holding the MLX
    runtime RLock while dispatching a generation to its owner would deadlock.
    """
    with ExitStack() as stack:
        lifecycle = getattr(engine, "_model_lock", None)
        if lifecycle is not None:
            stack.enter_context(lifecycle)
        backend = _backend(engine)
        service = getattr(engine, "awake_consolidator", None)
        update = getattr(service, "_update_lock", None)
        if update is not None:
            stack.enter_context(update)
        if backend is not None:
            from core.native_speed import uses_mlx_thread_owner
            runtime = getattr(backend, "_runtime_model_lock", None)
            if runtime is not None and not uses_mlx_thread_owner(backend):
                stack.enter_context(runtime)
        stack.enter_context(_attributes(engine, {
            "_best_work_inference_read_only": True,
            "memory_auto_recall": False,
        }))
        if backend is not None:
            # Normal UI learning and consolidation wait on the locks above.
            # A direct accidental call from an internal inference callback
            # must fail instead of changing the shared model mid-tournament.
            blocked = {name: _read_only for name in
                       ("train_mini_batch", "inject_lora_adapters", "switch_learning_checkpoint")
                       if callable(getattr(backend, name, None))}
            stack.enter_context(_attributes(backend, blocked))
        yield


def _on_owner(backend, operation):
    owner = getattr(backend, "_mlx_owner", None)
    return owner.call(operation) if owner is not None else operation()


def _mlx_fork(backend):
    """Replace cache holders on the existing model's owner OS thread."""
    from core.prompt_reuse import ExactPrefixStore, TokenizerMemo
    from core.vlm_prompt_reuse import NativeVLMCacheReuse
    from core.vlm_prompt_tokens import PreparedTextInputMemo

    prefixes = backend._prompt_prefixes
    memo = backend._tokenizer_memo
    vlm = backend._vlm_prefixes
    inputs = backend._vlm_input_memo
    changes = {
        "_prompt_prefixes": ExactPrefixStore(
            max_entries=prefixes.max_entries, max_tokens=prefixes.max_tokens,
            max_bytes=prefixes.max_bytes, memory_admission=prefixes.memory_admission),
        "_tokenizer_memo": TokenizerMemo(memo.max_entries, memo.max_tokens, memo.max_bytes),
        "_vlm_prefixes": NativeVLMCacheReuse(vlm.max_bytes),
        "_vlm_input_memo": PreparedTextInputMemo(inputs.max_bytes),
        "kv_cache_manager": None,
        "_persistent_prompt_store": None,
        "_persistent_prompt_identity": None,
        "_prompt_reuse_identity": None,
        "_lossless_speedups": {**backend._lossless_speedups, "persistent_cache": False},
        "_prompt_disk_pristine": False,
    }
    return _attributes(backend, changes)


def _release_mlx_fork(backend):
    for name in ("_prompt_prefixes", "_tokenizer_memo", "_vlm_prefixes", "_vlm_input_memo"):
        getattr(backend, name).clear()
    arena = getattr(backend, "kv_cache_manager", None)
    if arena is not None:
        arena.release()
    store = getattr(backend, "_persistent_prompt_store", None)
    if store is not None:
        store.close()


@contextmanager
def inference_fork(engine):
    """Discard internal prompts/cache state, even after cancellation or errors.

    MLX uses fresh cache holders and restores the original holders afterwards.
    llama.cpp working KV is reset and its optional saved prompt cache disabled.
    HTTP adapters submit a complete independent prompt per request; native
    Prism already disables cache reuse, and BitNet observes the fork flag.
    Strata's complete messages remain its documented isolation boundary.
    """
    backend = _backend(engine)
    if backend is None:
        yield
        return
    with _attributes(backend, {"_best_work_inference_fork": True}):
        if all(hasattr(backend, name) for name in
               ("_prompt_prefixes", "_tokenizer_memo", "_vlm_prefixes", "_vlm_input_memo")):
            context = _on_owner(backend, lambda: _mlx_fork(backend))
            _on_owner(backend, context.__enter__)
            try:
                yield
            finally:
                try:
                    _on_owner(backend, lambda: _release_mlx_fork(backend))
                finally:
                    _on_owner(backend, lambda: context.__exit__(None, None, None))
            return
        model = getattr(backend, "model", None)
        reset = getattr(model, "reset", None)
        with ExitStack() as stack:
            if model is not None and hasattr(model, "cache"):
                stack.enter_context(_attributes(model, {"cache": None}))
            if callable(reset):
                reset()
            try:
                yield
            finally:
                if callable(reset):
                    reset()
