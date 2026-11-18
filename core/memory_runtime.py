"""User-facing online learning and scoped parameter-memory lifecycle.

Every scope transition uses the normal model loader, so tokenizer/source checks,
native deployment, MLX owner-thread execution and cache invalidation still apply.
The existing model lifecycle lock serializes scope changes with inference.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import functools
from pathlib import Path
import threading
import time

from core.memory_scopes import LearningScope, as_scope


def active_learning_backend(engine):
    return getattr(getattr(engine, "awake_consolidator", None), "engine", None)


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
        self._owned_request_stores = {}
        self._temporary_scope_depth = 0
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

    @functools.wraps(original_solve)
    def solve(self, *args, **kwargs):
        with self._model_lock:
            return original_solve(self, *args, **kwargs)

    @functools.wraps(original_stream)
    def stream_solve(self, *args, **kwargs):
        # Generator creation performs no inference. Hold the lock through
        # iteration and close, not merely while returning the generator object.
        with self._model_lock:
            yield from original_stream(self, *args, **kwargs)

    def set_scope(self, scope, *, seed_from_current=False, cancel_event=None, _temporary=False):
        """Select persistent state. On load failure restore the previous selection."""
        scope = as_scope(scope)
        if scope.kind == "request" and not _temporary:
            raise ValueError("Use use_learning_scope() to manage request memory and cleanup")
        with self._model_lock:
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

    def solve_with_memory(self, prompt, *, memory_query=None, **kwargs):
        """Route to the best matching pinned document adapter for this answer only."""
        with self._model_lock:
            catalog = self.adapter_memory_bank()
            entries = catalog.select(memory_query or prompt, top_k=1)
            if not entries:
                response, meta = self.solve(prompt, **kwargs)
                return response, {**(meta or {}), "memory_selection": None}
            entry = entries[0]
            with self.use_learning_scope(catalog.activation_scope(entry), seed_from_current=False):
                catalog.resolve(entry, active_learning_backend(self).learning_state)
                response, meta = self.solve(prompt, **kwargs)
            provenance = {key: value for key, value in entry.items() if key != "pin"}
            return response, {**(meta or {}), "memory_selection": provenance}

    def run_learning(self, request, *, cancel_event=None, progress=None):
        """Run a bounded online method; ephemeral task adaptation is the default."""
        from core.test_time_learning import OnlineLearning
        if not isinstance(request, dict):
            raise ValueError("An online learning request must be an object")
        options = dict(request)
        method = options.get("method", "ingest")
        requested_scope = options.pop("scope", None)
        started = time.perf_counter()
        with self._model_lock, _PeakRSS() as measured:
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
                response, metadata = self.solve_with_memory(options["query"], **solve_options)
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
                temporary = requested_scope is None and method in {
                    "few_shot", "reinforce", "query_adapt", "consolidate_candidates", "discover"}
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
    cls.use_learning_scope = use_scope
    cls.adapter_memory_bank = bank
    cls.solve_with_memory = solve_with_memory
    cls.run_online_learning = run_learning
    cls._memory_runtime_installed = True
