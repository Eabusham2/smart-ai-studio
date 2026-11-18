"""User-facing online learning and scoped parameter-memory lifecycle.

Scope transitions use an authenticated same-model adapter switch when the
backend supports it, otherwise the normal model loader. Tokenizer/source checks,
native deployment, MLX owner-thread execution and cache invalidation still apply.
The existing model lifecycle lock serializes scope changes with inference.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import functools
import inspect
from pathlib import Path
import threading
import time

from core.memory_scopes import LearningScope, as_scope


def active_learning_backend(engine):
    return getattr(getattr(engine, "awake_consolidator", None), "engine", None)


def _runtime_identity(engine):
    name = getattr(engine, "active_model_name", None)
    source = getattr(engine, "active_model_path", None)
    active = getattr(engine, "active_backend", None)
    backend = active_learning_backend(engine)
    controller = getattr(engine, "controller_backend", None)
    if getattr(engine, "device", None) == "cuda":
        if (active in ("torch", "cuda") and controller is None
                and getattr(engine, "model", None) is not None and getattr(engine, "tokenizer", None) is not None):
            return name, source, "direct_cuda", active
        if (active == "controller" and controller is backend and getattr(controller, "direct_cuda_loaded", False)
                and getattr(controller, "direct_torch_generation", False)
                and getattr(controller, "direct_backend_name", None) in ("torch", "cuda")):
            return name, source, "direct_cuda", controller.direct_backend_name
    return name, source, active, getattr(backend, "runtime", None)


def _implementation_binding(engine, mode, runtime, info):
    store = getattr(active_learning_backend(engine), "learning_state", None)
    if store is not None:
        return store.identity, getattr(store, "implementation_base_root", store.root)
    if runtime[2] == "direct_cuda" and mode == "legacy" and getattr(engine, "controller_backend", None) is None:
        # The original CUDA inference loader has no store. Authenticate the same
        # local weights and tokenizer the managed PEFT controller resolves.
        from core.controller_runtime import controller_learning_root
        from core.learning_state import cached_model_source, model_identity
        root = controller_learning_root()
        source = str(cached_model_source(runtime[1], info))
        identity = model_identity(source, info, adapter_format="peft", state_root=root)
        revision = getattr(getattr(engine.model, "config", None), "_commit_hash", None)
        if not identity["source_verified"] or (revision and identity["revision"] and revision != identity["revision"]):
            raise RuntimeError("Cannot verify the source of the resident direct CUDA model")
        return identity, root
    return None


def _live_checkpoint(backend):
    for key in ("_loaded_learning_checkpoint", "_last_managed_checkpoint", "_loaded_legacy_deployment", "adapter_path"):
        value = getattr(backend, key, None)
        if value:
            return Path(value).resolve()
    return None


def _verify_implementation(engine, mode, scope, runtime, binding, info):
    if not engine.is_model_loaded or _runtime_identity(engine) != runtime:
        raise RuntimeError("The reloaded model or runtime differs from the selected model")
    backend = active_learning_backend(engine)
    if getattr(backend, "learning_implementation", mode) != mode:
        raise RuntimeError("The reloaded backend uses a different learning implementation")
    if _implementation_binding(engine, mode, runtime, info) != binding:
        raise RuntimeError("The reloaded model identity or learned-state location changed")
    store = getattr(backend, "learning_state", None)
    if store is None:
        return
    base_root = binding[1]
    from core.learning_state import implementation_state_root
    expected_root = implementation_state_root(base_root, mode)
    if store.scope != scope or store.root != expected_root:
        raise RuntimeError("The reloaded model identity, memory scope or storage root differs from the selection")
    current = store.current_checkpoint()
    if current:
        checkpoint = Path(current).resolve()
        generation = getattr(backend, "_mlx_parent_generation", checkpoint.parent.name)
        if _live_checkpoint(backend) != checkpoint or generation != checkpoint.parent.name:
            raise RuntimeError("The reloaded backend did not activate the selected checkpoint")


class _PeakRSS:
    """Measure this process during the complete operation, including reloads."""

    def __enter__(self):
        self.peak = None
        self.stop = threading.Event()
        try:
            import psutil
            self.process = psutil.Process()
        except (ImportError, OSError):
            self.process = None
        self._sample()
        self.thread = threading.Thread(target=self._run, name="learning-memory-sample", daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        if self.process is not None:
            try:
                value = self.process.memory_info().rss
                self.peak = max(value, self.peak or 0)
            except (OSError, RuntimeError):
                pass

    def _run(self):
        while not self.stop.wait(0.05):
            self._sample()

    def __exit__(self, *_):
        self._sample()
        self.stop.set()
        self.thread.join(timeout=1)


def install_memory_runtime(cls):
    if getattr(cls, "_memory_runtime_installed", False):
        return
    original_init = cls.__init__
    original_load = cls.load_model
    original_solve = cls.solve
    original_stream = getattr(cls, "stream_solve", None)

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.learning_scope = LearningScope()
        self._memory_load_spec = None
        self._implementation_scopes = {}
        self._implementation_seeds = {}
        self._implementation_sessions = {}
        self._owned_request_stores = {}
        self._temporary_scope_depth = 0
        self.memory_auto_recall = True
        self._memory_routing_active = False
        self._last_memory_recall = None
        # Production already installs a reentrant speed/lifecycle lock. Small
        # embedders without that extension need the same lifecycle contract.
        if not hasattr(self, "_model_lock"):
            self._model_lock = threading.RLock()

    @functools.wraps(original_load)
    def load(self, model_name, model_path=None, backend=None, model_info=None):
        with self._model_lock:
            result = original_load(self, model_name, model_path=model_path, backend=backend,
                                   model_info=model_info)
            if isinstance(result, dict) and result.get("status") == "loaded":
                self._memory_load_spec = {
                    "model_name": model_name, "model_path": model_path, "backend": backend,
                    "model_info": deepcopy(model_info or {}),
                }
                result = {**result, "learning_scope": self.learning_scope.record}
            return result

    def prepare_call(self, method, args, kwargs):
        if getattr(getattr(self, "awake_consolidator", None), "learning_implementation", None) == "legacy":
            return args, kwargs, None
        current_scope = getattr(self, "learning_scope", LearningScope())
        if (not getattr(self, "memory_auto_recall", True) or not getattr(self, "is_model_loaded", False)
                or current_scope.pin or current_scope.kind == "request" or getattr(self, "_temporary_scope_depth", 0)):
            return args, kwargs, None
        backend = active_learning_backend(self)
        if (getattr(backend, "learning_state", None) is None
                or not callable(getattr(backend, "stream_generate_tokens", None))):
            return args, kwargs, None
        catalog = self.adapter_memory_bank()
        if not catalog.list_entries(limit=1):
            return args, kwargs, None
        bound = inspect.signature(method).bind_partial(self, *args, **kwargs)
        question = bound.arguments.get("prompt")
        if not isinstance(question, str):
            return args, kwargs, None
        extra = bound.arguments.get("kwargs", {})
        from core.memory_routing import prepare_recall_request
        history = bound.arguments.get("history", extra.get("history"))
        augmented, prepared_history, report = prepare_recall_request(self, question, history=history,
            cancel_event=bound.arguments.get("cancel_event", extra.get("cancel_event")))
        bound.arguments["prompt"] = augmented
        if prepared_history is not history:
            if "history" in bound.arguments:
                bound.arguments["history"] = prepared_history
            else:
                extra["history"] = prepared_history
        return bound.args[1:], bound.kwargs, report

    @functools.wraps(original_solve)
    def solve(self, *args, **kwargs):
        with self._model_lock:
            if getattr(self, "_memory_routing_active", False):
                return original_solve(self, *args, **kwargs)
            self._memory_routing_active = True
            try:
                args, kwargs, report = prepare_call(self, original_solve, args, kwargs)
                self._last_memory_recall = report
                result = original_solve(self, *args, **kwargs)
                if report is not None and isinstance(result, tuple) and len(result) == 2:
                    return result[0], {**(result[1] or {}), "memory_recall": report}
                return result
            finally:
                self._memory_routing_active = False

    @functools.wraps(original_stream)
    def stream_solve(self, *args, **kwargs):
        # Generator creation performs no inference. Hold the lock through
        # iteration and close, not merely while returning the generator object.
        with self._model_lock:
            if getattr(self, "_memory_routing_active", False):
                yield from original_stream(self, *args, **kwargs)
                return
            self._memory_routing_active = True
            try:
                args, kwargs, report = prepare_call(self, original_stream, args, kwargs)
                self._last_memory_recall = report
                yield from original_stream(self, *args, **kwargs)
            finally:
                self._memory_routing_active = False

    def set_implementation(self, implementation, *, cancel_event=None, preferred_scope=None):
        """Reload the selected model with its implementation-specific scope and learned weights."""
        from core.learning_implementation import get_learning_implementation
        from core.online_consolidator import make_online_consolidator
        selected = get_learning_implementation(implementation)

        def check_cancelled():
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Learning implementation change cancelled")

        def require_status(receipt, expected, message):
            if not isinstance(receipt, dict) or receipt.get("status") != expected:
                detail = receipt.get("error") if isinstance(receipt, dict) else None
                raise RuntimeError(f"{message}: {detail or receipt}")

        with self._model_lock:
            service = getattr(self, "awake_consolidator", None)
            previous = get_learning_implementation(getattr(self.settings, "learning_implementation", None))
            response = {"learning_implementation": selected, "previous_implementation": previous}
            if selected == previous and getattr(service, "learning_implementation", None) == selected:
                return {"status": "unchanged", **response}
            if (getattr(service, "is_consolidating", False) or self._temporary_scope_depth
                    or getattr(self, "speed_tuning_active", False)):
                raise RuntimeError("Finish the current model operation before changing the learning implementation")
            check_cancelled()
            loaded = bool(self.is_model_loaded and not getattr(self.settings, "use_mock", False))
            spec = deepcopy(self._memory_load_spec)
            if loaded and spec is None:
                raise RuntimeError("Reload this model before changing its learning implementation")
            backend = active_learning_backend(self)
            store = getattr(backend, "learning_state", None) if loaded else None
            runtime = _runtime_identity(self)
            info = (spec or {}).get("model_info", {})
            binding = _implementation_binding(self, previous, runtime, info) if loaded else None
            from core.learning_state import identity_digest
            model_key = (runtime, identity_digest(binding[0]) if binding else None) if loaded else None
            previous_scope = self.learning_scope
            if loaded:
                _verify_implementation(self, previous, previous_scope, runtime, binding, info)
            previous_checkpoint = _live_checkpoint(backend) if loaded else None
            remember_seed = bool(previous_checkpoint and store is not None and not store.current_checkpoint())
            preferred = as_scope(preferred_scope)
            if preferred.kind not in {"model", "session", "project"}:
                raise ValueError("A preferred memory scope must be model, session or project")
            selected_scope = (self._implementation_scopes.get((model_key, selected), preferred)
                              if selected != "legacy" else LearningScope())
            replacement = make_online_consolidator(
                None if loaded else backend, getattr(service, "db", None),
                max_context=getattr(service, "max_context", 8192),
                evict_ratio=getattr(service, "evict_ratio", .40), lambda_ewc=getattr(service, "lambda_ewc", 400.0),
                session_id=self._implementation_sessions.get((model_key, selected)), implementation=selected)
            replacement.watermark_tokens = getattr(service, "watermark_tokens", replacement.watermark_tokens)
            if loaded:
                require_status(self.unload_model(), "unloaded", "The previous runtime could not unload")
            global_seed = getattr(self, "lora_adapter_path", None)
            old_cancel = getattr(self, "_model_load_cancelled", None)
            attempted_load = False
            try:
                check_cancelled()
                self.settings.learning_implementation = selected
                self.awake_consolidator = replacement
                self.learning_scope = selected_scope
                result = {"status": "selected"}
                if loaded:
                    selected_spec = deepcopy(spec)
                    selected_info = selected_spec["model_info"]
                    selected_info.pop("legacy_adapter_path", None)
                    seed = self._implementation_seeds.get((model_key, selected, selected_scope))
                    if seed:
                        selected_info["legacy_adapter_path"] = seed
                    # The global explicit seed belongs to the source selection.
                    # A previously visited destination may have its own seed.
                    self.lora_adapter_path = None
                    if cancel_event is not None:
                        self._model_load_cancelled = lambda: cancel_event.is_set() or bool(old_cancel and old_cancel())
                    attempted_load = True
                    result = self.load_model(**selected_spec)
                    check_cancelled()
                    require_status(result, "loaded", "The selected implementation could not load")
                    _verify_implementation(self, selected, selected_scope, runtime, binding, selected_info)
            except BaseException as error:
                if attempted_load:
                    try:
                        require_status(self.unload_model(), "unloaded", "The candidate runtime could not unload")
                    except BaseException as shutdown:
                        raise RuntimeError(f"Implementation recovery could not stop the candidate: {shutdown}") from error
                self.settings.learning_implementation = previous
                self.awake_consolidator = service
                self.learning_scope = previous_scope
                if loaded:
                    self.lora_adapter_path = global_seed
                    self._model_load_cancelled = None
                    try:
                        restored = self.load_model(**spec)
                        require_status(restored, "loaded", "The previous model could not reload")
                        _verify_implementation(self, previous, previous_scope, runtime, binding, info)
                        if _live_checkpoint(active_learning_backend(self)) != previous_checkpoint:
                            raise RuntimeError("The previous checkpoint could not be restored")
                    except BaseException as restore_error:
                        raise RuntimeError(f"Implementation selection failed and recovery failed: {restore_error}") from error
                raise
            finally:
                if loaded:
                    self.lora_adapter_path = global_seed
                    self._model_load_cancelled = old_cancel
            if previous != "legacy":
                self._implementation_scopes[(model_key, previous)] = previous_scope
                self._implementation_sessions[(model_key, previous)] = getattr(service, "session_id", None)
            seed_key = model_key, previous, previous_scope
            if remember_seed:
                # Return to the caller's explicitly selected, uncommitted raw adapter.
                self._implementation_seeds[seed_key] = str(previous_checkpoint)
            else:
                self._implementation_seeds.pop(seed_key, None)
            self._last_memory_recall = None
            return {**result, **response}

    def set_scope(self, scope, *, seed_from_current=False, cancel_event=None, _temporary=False):
        """Select persistent state. On load failure restore the previous selection."""
        scope = as_scope(scope)
        if scope.kind == "request" and not _temporary:
            raise ValueError("Use use_learning_scope() to manage request memory and cleanup")
        with self._model_lock:
            if getattr(getattr(self, "awake_consolidator", None), "learning_implementation", None) == "legacy":
                if scope != getattr(self, "learning_scope", LearningScope()):
                    raise RuntimeError("Select managed persistent learning before changing memory scopes")
            if self._temporary_scope_depth and not _temporary:
                raise RuntimeError("Finish the temporary memory operation before changing scope")
            previous = self.learning_scope
            if previous == scope:
                return {"status": "unchanged", "learning_scope": scope.record}
            service = getattr(self, "awake_consolidator", None)
            if getattr(service, "is_consolidating", False):
                raise RuntimeError("Finish the current memory update before changing scope")
            loaded = bool(self.is_model_loaded)
            spec = deepcopy(self._memory_load_spec)
            if loaded and spec is None:
                raise RuntimeError("Reload this model before selecting a memory scope")
            old_store = getattr(active_learning_backend(self), "learning_state", None)
            if old_store is not None and loaded and old_store.scope != previous:
                raise RuntimeError("The loaded model does not match the selected memory scope")
            created = None
            if seed_from_current:
                if scope.kind != "request":
                    raise ValueError("Only a fresh request scope may copy the current learned state")
                if not loaded or old_store is None:
                    raise RuntimeError("Load a model with verified memory before adapting a request")
                if old_store is not None:
                    live = active_learning_backend(self)
                    manifest = old_store.current_manifest()
                    if manifest is None and (getattr(live, "adapter_path", None)
                            or getattr(live, "last_restored_adapter_path", None)
                            or getattr(live, "_loaded_legacy_deployment", None)):
                        raise RuntimeError("Commit and validate this legacy adapter before copying it to a request")
                    loaded_generation = getattr(live, "_mlx_parent_generation", None)
                    if loaded_generation and manifest and loaded_generation != manifest["generation"]:
                        raise RuntimeError("Reload the latest learned checkpoint before adapting a request")
                    loaded_path = (getattr(live, "_loaded_learning_checkpoint", None)
                                   or getattr(live, "adapter_path", None))
                    if isinstance(loaded_path, (str, Path)) and loaded_path and manifest:
                        expected = old_store.directory / "generations" / manifest["generation"] / manifest["checkpoint"]
                        if Path(loaded_path).resolve() != expected.resolve():
                            raise RuntimeError("Reload the latest learned checkpoint before adapting a request")
                    created = old_store.for_scope(scope, require_new=True)
                    try:
                        created.seed_from(old_store, cancel_event=cancel_event,
                                          expected_generation=(manifest or {}).get("generation"))
                    except BaseException:
                        # seed_from accepts only an empty transient namespace.
                        # Never remove a pre-existing request's committed data.
                        created.discard_request()
                        raise
                    self._owned_request_stores[scope.scope_id] = created
            elif scope.kind == "request" and scope.scope_id not in self._owned_request_stores:
                if not loaded or old_store is None:
                    raise RuntimeError("Load a model with verified memory before adapting a request")
                created = old_store.for_scope(scope, require_new=True)
                self._owned_request_stores[scope.scope_id] = created
            if not loaded:
                self.learning_scope = scope
                return {"status": "selected", "learning_scope": scope.record}
            self.learning_scope = scope
            try:
                switch = getattr(active_learning_backend(self), "switch_learning_checkpoint", None)
                if callable(switch) and switch(scope) is True:
                    result = {"status": "loaded", "learning_scope": scope.record,
                              "activation": "same_model_adapter_switch"}
                else:
                    result = self.load_model(**spec)
                if not isinstance(result, dict) or result.get("status") != "loaded":
                    raise RuntimeError((result or {}).get("error") or "The selected memory could not load")
                selected_store = getattr(active_learning_backend(self), "learning_state", None)
                if old_store is not None and (selected_store is None
                        or selected_store.scope != scope or selected_store.root != old_store.root
                        or selected_store.identity != old_store.identity):
                    raise RuntimeError("Model source, tokenizer or memory scope changed during reload")
                if scope.kind == "request" and selected_store is not None:
                    self._owned_request_stores[scope.scope_id] = selected_store
            except BaseException as error:
                self.learning_scope = previous
                restored = self.load_model(**spec)
                if not isinstance(restored, dict) or restored.get("status") != "loaded":
                    raise RuntimeError("Memory selection failed and the previous model could not reload: "
                                       + str((restored or {}).get("error") or restored)) from error
                restored_store = getattr(active_learning_backend(self), "learning_state", None)
                if old_store is not None and (restored_store is None or restored_store.scope != previous
                        or restored_store.identity != old_store.identity or restored_store.root != old_store.root):
                    raise RuntimeError("Memory selection failed and the previous model identity could not be restored") from error
                if created is not None:
                    created.discard_request()
                    self._owned_request_stores.pop(scope.scope_id, None)
                raise
            if previous.kind == "request":
                departed = self._owned_request_stores.get(previous.scope_id)
                if departed is not None:
                    departed.discard_request()
                    self._owned_request_stores.pop(previous.scope_id, None)
            return result

    @contextmanager
    def use_scope(self, scope=None, *, seed_from_current=True, cancel_event=None):
        """Temporarily adapt/use one scope, then restore the caller's loaded state."""
        selected = LearningScope.request() if scope is None else as_scope(scope)
        with self._model_lock:
            if self._temporary_scope_depth:
                raise RuntimeError("Nested temporary memory scopes are not supported")
            previous = self.learning_scope
            if selected == previous:
                yield active_learning_backend(self)
                return
            self.set_learning_scope(selected, seed_from_current=seed_from_current,
                                    cancel_event=cancel_event, _temporary=True)
            self._temporary_scope_depth += 1
            try:
                yield active_learning_backend(self)
            finally:
                # A failed native shutdown must keep its state on disk. Cleanup
                # follows a successful restore, never merely a requested unload.
                try:
                    self.set_learning_scope(previous, _temporary=True)
                finally:
                    self._temporary_scope_depth -= 1

    def bank(self):
        from core.adapter_memory_bank import AdapterMemoryBank
        store = getattr(active_learning_backend(self), "learning_state", None)
        if store is None or not self.is_model_loaded:
            raise RuntimeError("Load a model with verified learned-state support first")
        return AdapterMemoryBank(store)

    def solve_with_memory(self, prompt, *, memory_query=None, routing="lexical", top_k=3,
                          candidate_limit=8, memory_max_tokens=192, cache=True, **kwargs):
        """Route to the best matching pinned document adapter for this answer only."""
        with self._model_lock:
            if routing == "model":
                from core.memory_routing import prepare_recall_request
                previous = self._memory_routing_active
                self._memory_routing_active = True
                try:
                    augmented, history, report = prepare_recall_request(self, prompt, query=memory_query,
                        history=kwargs.get("history"), cancel_event=kwargs.get("cancel_event"),
                        top_k=top_k, candidate_limit=candidate_limit, max_tokens=memory_max_tokens, cache=cache)
                    if history is not kwargs.get("history"):
                        kwargs["history"] = history
                    response, meta = original_solve(self, augmented, **kwargs)
                    self._last_memory_recall = report
                    return response, {**(meta or {}), "memory_recall": report}
                finally:
                    self._memory_routing_active = previous
            if routing != "lexical":
                raise ValueError("Memory routing must be model or lexical")
            catalog = self.adapter_memory_bank()
            entries = catalog.select(memory_query or prompt, top_k=1)
            if not entries:
                previous = self._memory_routing_active
                self._memory_routing_active = True
                try:
                    response, meta = original_solve(self, prompt, **kwargs)
                    return response, {**(meta or {}), "memory_selection": None}
                finally:
                    self._memory_routing_active = previous
            entry = entries[0]
            with self.use_learning_scope(catalog.activation_scope(entry), seed_from_current=False):
                catalog.resolve(entry, active_learning_backend(self).learning_state)
                response, meta = self.solve(prompt, **kwargs)
            provenance = {key: value for key, value in entry.items() if key != "pin"}
            return response, {**(meta or {}), "memory_selection": provenance}

    def run_learning(self, request, *, cancel_event=None, progress=None):
        """Run online learning in the selected scope, or temporary practice on request."""
        from core.test_time_learning import OnlineLearning
        if not isinstance(request, dict):
            raise ValueError("An online learning request must be an object")
        options = dict(request)
        method = options.get("method", "ingest")
        requested_scope = options.pop("scope", None)
        temporary = options.pop("temporary", False)
        if type(temporary) is not bool:
            raise ValueError("temporary must be a boolean")
        if temporary and requested_scope is not None:
            raise ValueError("Choose temporary practice or a persistent scope, not both")
        if temporary and (method.startswith("bank_") or method == "capabilities"):
            raise ValueError("Temporary practice applies to learning tasks, not saved-memory management")
        if (requested_scope is not None and method.startswith("bank_")
                and as_scope(requested_scope) != self.learning_scope):
            raise ValueError("Select the saved-memory owner with set_learning_scope before managing its bank")
        started = time.perf_counter()
        with self._model_lock, _PeakRSS() as measured:
            if (method != "capabilities" and
                    getattr(getattr(self, "awake_consolidator", None), "learning_implementation", None) == "legacy"):
                raise RuntimeError("Select managed persistent learning in Learning & saved memory before running this task")
            if not self.is_model_loaded:
                raise RuntimeError("Load a model before starting online learning")
            if method == "capabilities":
                from core.adaptation_capabilities import adaptation_capabilities
                result = {"status": "completed", "capabilities": adaptation_capabilities(active_learning_backend(self))}
            elif method in {"bank_list", "bank_select", "bank_export", "bank_delete"}:
                catalog = self.adapter_memory_bank()
                if method == "bank_list":
                    result = {"status": "completed", "entries": catalog.list_entries(
                        limit=options.get("limit", 100), offset=options.get("offset", 0))}
                elif method == "bank_select":
                    result = {"status": "completed", "entries": catalog.select(
                        options.get("query", ""), top_k=options.get("top_k", 1))}
                elif method == "bank_export":
                    destination = catalog.export(options["key"], options["destination"])
                    result = {"status": "completed", "path": str(destination)}
                else:
                    result = {"status": "completed", "deleted": catalog.delete(options["key"])}
                if "entries" in result:
                    result["entries"] = [{k: v for k, v in entry.items() if k != "pin"}
                                         for entry in result["entries"]]
            elif method == "bank_answer":
                solve_options = {"cancel_event": cancel_event}
                if "force_branch_count" in options:
                    solve_options["force_branch_count"] = options["force_branch_count"]
                response, metadata = self.solve_with_memory(options["query"],
                    routing=options.get("routing", "model"), top_k=options.get("top_k", 3),
                    candidate_limit=options.get("candidate_limit", 8),
                    memory_max_tokens=options.get("max_tokens", 192), cache=options.get("cache", True),
                    **solve_options)
                result = {"status": "completed", "response": response, "metadata": metadata}
            elif method == "bank_ingest":
                catalog = self.adapter_memory_bank()
                key = options.pop("key", "")
                source = Path(str(options.get("source", ""))).expanduser().resolve()
                if not source.is_file():
                    raise ValueError("A document memory requires one local text or structured file")
                from core.learning_state import file_sha256
                source_hash = file_sha256(source, cancel_event=cancel_event)
                title = options.pop("title", "") or source.name
                keywords = options.pop("keywords", ())
                options["method"] = "ingest"
                options["source"] = str(source)
                with self.use_learning_scope(catalog.document_scope(key), seed_from_current=False):
                    result = OnlineLearning(self.awake_consolidator).run_request(
                        options, cancel_event=cancel_event, progress=progress)
                    store = active_learning_backend(self).learning_state
                    # Do not index an empty or cancelled/partial study as complete.
                    if (result.get("status") == "completed" and result.get("trained_chunks", 0) > 0
                            and not (cancel_event and cancel_event.is_set())):
                        manifest = store.current_manifest()
                        if not manifest:
                            raise RuntimeError("No validated document checkpoint was produced")
                        entry = catalog.register(key, store, title=title, source=str(source),
                                                 content=source, keywords=keywords,
                                                 expected_generation=manifest["generation"],
                                                 expected_content_sha256=source_hash, cancel_event=cancel_event)
                        result = {**result, "memory_entry": {k: v for k, v in entry.items() if k != "pin"}}
            else:
                selected = LearningScope.request() if temporary else (
                    as_scope(requested_scope) if requested_scope is not None else self.learning_scope)
                if selected == self.learning_scope:
                    result = OnlineLearning(self.awake_consolidator).run_request(
                        options, cancel_event=cancel_event, progress=progress)
                else:
                    with self.use_learning_scope(selected, seed_from_current=selected.kind == "request",
                                                 cancel_event=cancel_event):
                        result = OnlineLearning(self.awake_consolidator).run_request(
                            options, cancel_event=cancel_event, progress=progress)
                result = {**result, "scope": selected.record,
                          "state_persisted": selected.kind != "request"}
        return {**result, "total_wall_time_seconds": time.perf_counter() - started,
                "peak_process_rss_bytes": measured.peak}

    cls.__init__ = initialize
    cls.load_model = load
    cls.solve = solve
    if original_stream is not None:
        cls.stream_solve = stream_solve
    cls.set_learning_scope = set_scope
    cls.set_learning_implementation = set_implementation
    cls.use_learning_scope = use_scope
    cls.adapter_memory_bank = bank
    cls.solve_with_memory = solve_with_memory
    cls.run_online_learning = run_learning
    cls._memory_runtime_installed = True
