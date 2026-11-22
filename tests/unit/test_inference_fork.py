"""Read-only Best Work contexts use the real cache/lifecycle helpers."""
from __future__ import annotations

import json
import threading
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from core.inference_fork import inference_cycle, inference_fork
from core.prompt_reuse import ExactPrefixStore, TokenizerMemo
from core.vlm_prompt_reuse import NativeVLMCacheReuse
from core.vlm_prompt_tokens import PreparedTextInputMemo


class Model:
    def __init__(self):
        self.cache = {"canonical": "saved"}
        self.context = "canonical"
        self.resets = 0
        self.weights = [4.0]

    def reset(self):
        self.context = ""
        self.resets += 1

    def __deepcopy__(self, _memo):
        raise AssertionError("A fork must not duplicate loaded model weights")


class Backend:
    def __init__(self):
        self.model = Model()
        self.tokenizer = SimpleNamespace(encode=lambda text: list(text.encode()))
        self._runtime_model_lock = threading.RLock()

    def train_mini_batch(self):
        self.model.weights[0] += 1


def engine_for(backend, kind="gguf"):
    return SimpleNamespace(active_backend=kind, **{kind + "_backend": backend},
                           _model_lock=threading.RLock(), memory_auto_recall=True)


class MLXBackend(Backend):
    _mlx_thread_ownership_installed = True

    def __init__(self):
        super().__init__()
        self._prompt_prefixes = ExactPrefixStore()
        self._tokenizer_memo = TokenizerMemo()
        self._vlm_prefixes = NativeVLMCacheReuse()
        self._vlm_input_memo = PreparedTextInputMemo()
        self._lossless_speedups = {"prompt_reuse": True, "persistent_cache": True}
        self._prompt_reuse_identity = "original-identity"
        self._persistent_prompt_identity = "original-disk-identity"
        self._persistent_prompt_store = SimpleNamespace(close=lambda: self.fail_close())
        self._prompt_disk_pristine = True
        self.kv_cache_manager = None
        self.original_store_closed = False

    def fail_close(self):
        self.original_store_closed = True


class InferenceForkTests(unittest.TestCase):
    def legacy_preparation(self):
        from core.legacy_online_consolidator import LegacyAwakeOnlineConsolidator
        backend = Backend()
        backend.is_mlx_available = True
        backend.training_ready = lambda: True
        backend.count_tokens = lambda rows: 100
        def train(**kwargs):
            self.assertEqual(kwargs['data'], [{'prompt': 'retired question', 'completion': 'retired answer'}])
            backend.model.weights[0] += 1
            return {}, 1.0
        backend.train_mini_batch = train
        engine = engine_for(backend, 'mlx')
        service = LegacyAwakeOnlineConsolidator(backend, max_context=10)
        engine.awake_consolidator = service
        history = [{'role': 'user', 'content': 'retired question'},
                   {'role': 'assistant', 'content': 'retired answer'}]
        history += [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': 'recent'} for i in range(4)]
        return backend, engine, service, history

    def test_native_request_cannot_populate_canonical_cache(self):
        backend = Backend()
        engine = engine_for(backend)
        model = backend.model
        weights = model.weights
        original_cache = model.cache
        with inference_cycle(engine), inference_fork(engine):
            self.assertIsNone(model.cache)
            self.assertEqual(model.context, "")
            model.context = "PRIVATE JUDGING INSTRUCTIONS"
            self.assertIs(backend.model, model)
            self.assertIs(model.weights, weights)
        self.assertIs(model.cache, original_cache)
        self.assertEqual(original_cache, {"canonical": "saved"})
        self.assertEqual(model.context, "")
        self.assertEqual(model.resets, 2)
        self.assertNotIn("_best_work_inference_fork", vars(backend))

    def test_learning_is_read_only_and_methods_restore_after_failure(self):
        backend = Backend()
        engine = engine_for(backend)
        with self.assertRaisesRegex(ValueError, "cancelled"):
            with inference_cycle(engine):
                self.assertFalse(engine.memory_auto_recall)
                with self.assertRaisesRegex(RuntimeError, "cannot update"):
                    backend.train_mini_batch()
                self.assertEqual(backend.model.weights, [4.0])
                raise ValueError("cancelled")
        self.assertTrue(engine.memory_auto_recall)
        self.assertNotIn("train_mini_batch", vars(backend))
        self.assertNotIn("_best_work_inference_read_only", vars(engine))
        backend.train_mini_batch()
        self.assertEqual(backend.model.weights, [5.0])

    def test_nested_native_forks_keep_cache_disabled_until_outer_exit(self):
        backend = Backend()
        engine = engine_for(backend)
        original = backend.model.cache
        with inference_fork(engine):
            with inference_fork(engine):
                backend.model.context = "inner selection"
            self.assertTrue(backend._best_work_inference_fork)
            self.assertIsNone(backend.model.cache)
            self.assertEqual(backend.model.context, "")
        self.assertIs(backend.model.cache, original)
        self.assertEqual(backend.model.resets, 4)

    def test_mlx_fork_keeps_existing_cache_and_never_writes_it(self):
        backend = MLXBackend()
        engine = engine_for(backend, "mlx")
        model = backend.model
        originals = dict(vars(backend))
        backend._tokenizer_memo.encode(backend.tokenizer, "canonical prompt")
        with inference_cycle(engine), inference_fork(engine):
            self.assertIs(backend.model, model)
            self.assertIsNot(backend._prompt_prefixes, originals["_prompt_prefixes"])
            self.assertFalse(backend._lossless_speedups["persistent_cache"])
            self.assertIsNone(backend._persistent_prompt_store)
            backend._tokenizer_memo.encode(backend.tokenizer, "losing original")
            outer_memo = backend._tokenizer_memo
            with inference_fork(engine):
                backend._tokenizer_memo.encode(backend.tokenizer, "judge secrets")
                judge_memo = backend._tokenizer_memo
            self.assertFalse(judge_memo.entries)
            self.assertIs(backend._tokenizer_memo, outer_memo)
            self.assertIn("losing original", outer_memo.entries)
        for field in ("_tokenizer_memo", "_prompt_prefixes", "_vlm_prefixes", "_vlm_input_memo",
                      "_lossless_speedups", "_persistent_prompt_store"):
            self.assertIs(getattr(backend, field), originals[field])
        self.assertEqual(list(backend._tokenizer_memo.entries), ["canonical prompt"])
        self.assertFalse(outer_memo.entries)
        self.assertFalse(backend.original_store_closed)
        self.assertTrue(backend._prompt_disk_pristine)
        self.assertEqual(backend._prompt_reuse_identity, "original-identity")

    def test_mlx_cleanup_runs_on_owner_thread_including_error(self):
        from core.mlx_thread_owner import MLXThreadOwner
        backend = MLXBackend()
        backend._mlx_owner = MLXThreadOwner()
        engine = engine_for(backend, "mlx")
        original_memo = backend._tokenizer_memo
        releases = []
        try:
            with self.assertRaisesRegex(RuntimeError, "generation failure"):
                with inference_cycle(engine), inference_fork(engine):
                    backend.kv_cache_manager = SimpleNamespace(
                        release=lambda: releases.append(threading.get_ident()))
                    raise RuntimeError("generation failure")
            self.assertEqual(releases, [backend._mlx_owner._thread_id])
            self.assertIs(backend._tokenizer_memo, original_memo)
            self.assertIsNone(backend.kv_cache_manager)
        finally:
            backend._mlx_owner.shutdown()

    def test_legacy_background_update_waits_for_whole_cycle(self):
        from core.legacy_online_consolidator import LegacyAwakeOnlineConsolidator
        backend = Backend()
        engine = engine_for(backend)
        service = LegacyAwakeOnlineConsolidator(backend)
        engine.awake_consolidator = service
        started, finished = threading.Event(), threading.Event()

        def train(*_args):
            backend.train_mini_batch()
            finished.set()
            return True

        service._run_shadow_consolidation_locked = train

        def worker():
            started.set()
            service._run_shadow_consolidation([])

        with inference_cycle(engine):
            thread = threading.Thread(target=worker)
            thread.start()
            self.assertTrue(started.wait(1))
            self.assertFalse(finished.wait(0.05))
            self.assertEqual(backend.model.weights, [4.0])
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertTrue(finished.is_set())
        self.assertEqual(backend.model.weights, [5.0])

    def test_already_scheduled_legacy_preparation_finishes_before_inference(self):
        backend, engine, service, history = self.legacy_preparation()
        scheduled, allow_training, updated = threading.Event(), threading.Event(), threading.Event()
        guarding, entered, finished = threading.Event(), threading.Event(), threading.Event()
        errors = []
        update = service._run_shadow_consolidation
        guard = service.pause_updates_for_inference

        def scheduled_update(chunk, selected_backend):
            scheduled.set()
            allow_training.wait(2)
            try:
                return update(chunk, selected_backend)
            finally:
                updated.set()

        @contextmanager
        def observed_guard(cancel_event=None):
            guarding.set()
            with guard(cancel_event):
                yield

        service._run_shadow_consolidation = scheduled_update
        service.pause_updates_for_inference = observed_guard

        def infer():
            try:
                with inference_cycle(engine):
                    self.assertEqual(backend.model.weights, [5.0])
                    self.assertFalse(service.is_consolidating)
                    entered.set()
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        cleanup = {'before_mb': 0, 'after_mb': 0, 'released_mb': 0}
        with patch('core.legacy_online_consolidator.release_training_memory', return_value=cleanup):
            retained, pruned = service.check_and_prune(history)
            self.assertTrue(pruned)
            self.assertEqual(retained, history[2:])
            self.assertTrue(scheduled.wait(1))
            thread = threading.Thread(target=infer, daemon=True)
            thread.start()
            try:
                self.assertTrue(guarding.wait(1))
                self.assertFalse(entered.wait(0.05))
                self.assertEqual(backend.model.weights, [4.0])
            finally:
                allow_training.set()
                self.assertTrue(updated.wait(1))
                self.assertTrue(finished.wait(1))
                thread.join(1)
        self.assertEqual(errors, [])
        self.assertTrue(entered.is_set())
        self.assertEqual(service.consolidation_count, 1)

    def test_cancel_while_legacy_preparation_is_queued_or_running(self):
        for phase in ('queued', 'running'):
            with self.subTest(phase=phase):
                backend, engine, service, history = self.legacy_preparation()
                started, release, updated = threading.Event(), threading.Event(), threading.Event()
                guarding, finished, cancelled = threading.Event(), threading.Event(), threading.Event()
                errors, entered = [], []
                update = service._run_shadow_consolidation
                train = backend.train_mini_batch
                guard = service.pause_updates_for_inference

                def scheduled_update(chunk, selected_backend):
                    if phase == 'queued':
                        started.set()
                        release.wait(2)
                    try:
                        return update(chunk, selected_backend)
                    finally:
                        updated.set()

                def running_update(**kwargs):
                    started.set()
                    release.wait(2)
                    return train(**kwargs)

                @contextmanager
                def observed_guard(cancel_event=None):
                    guarding.set()
                    with guard(cancel_event):
                        yield

                service._run_shadow_consolidation = scheduled_update
                service.pause_updates_for_inference = observed_guard
                if phase == 'running':
                    backend.train_mini_batch = running_update

                def infer():
                    try:
                        with inference_cycle(engine, cancel_event=cancelled):
                            entered.append(True)
                    except BaseException as error:
                        errors.append(error)
                    finally:
                        finished.set()

                cleanup = {'before_mb': 0, 'after_mb': 0, 'released_mb': 0}
                with patch('core.legacy_online_consolidator.release_training_memory', return_value=cleanup):
                    _, pruned = service.check_and_prune(history)
                    self.assertTrue(pruned)
                    self.assertTrue(started.wait(1))
                    thread = threading.Thread(target=infer, daemon=True)
                    thread.start()
                    try:
                        self.assertTrue(guarding.wait(1))
                        cancelled.set()
                        self.assertTrue(finished.wait(1))
                        self.assertFalse(updated.is_set())
                    finally:
                        release.set()
                        self.assertTrue(updated.wait(1))
                        thread.join(1)
                self.assertEqual(entered, [])
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], InterruptedError)
                self.assertTrue(engine.memory_auto_recall)
                self.assertNotIn('_best_work_inference_read_only', vars(engine))
                self.assertEqual(backend.model.weights, [5.0])

    def test_nested_cycle_does_not_wait_for_update_queued_during_outer_cycle(self):
        backend, engine, service, history = self.legacy_preparation()
        update = service._run_shadow_consolidation
        scheduled, finished, cancelled = threading.Event(), threading.Event(), threading.Event()
        def queued_update(chunk, selected_backend):
            scheduled.set()
            try:
                return update(chunk, selected_backend)
            finally:
                finished.set()
        service._run_shadow_consolidation = queued_update
        timeout = threading.Timer(2, cancelled.set)
        cleanup = {'before_mb': 0, 'after_mb': 0, 'released_mb': 0}
        with patch('core.legacy_online_consolidator.release_training_memory', return_value=cleanup):
            try:
                timeout.start()
                with inference_cycle(engine, cancel_event=cancelled):
                    _, pruned = service.check_and_prune(history)
                    self.assertTrue(pruned)
                    self.assertTrue(scheduled.wait(1))
                    with inference_cycle(engine, cancel_event=cancelled):
                        self.assertEqual(backend.model.weights, [4.0])
                        self.assertFalse(finished.is_set())
            finally:
                timeout.cancel()
                self.assertTrue(finished.wait(1))
        self.assertEqual(backend.model.weights, [5.0])

    def test_awake_preparation_cannot_learn_internal_prompt(self):
        from core.awake_auto_hook import prepare_awake_request
        engine = engine_for(Backend())
        history = [{"role": "assistant", "content": "original"}]
        with inference_cycle(engine):
            prompt, retained = prepare_awake_request(engine, "private comparison", history)
        self.assertEqual(prompt, "private comparison")
        self.assertIs(retained, history)

    def test_bitnet_fork_disables_server_cache_reuse(self):
        from core.engines.bitnet_cpp_engine import BitNetCppReasoningBackend
        backend = BitNetCppReasoningBackend.__new__(BitNetCppReasoningBackend)
        backend.model = Model()
        backend._base_url = "http://localhost:12345"
        backend._request = lambda _path, payload, **_kwargs: captured.append(payload) or {"content": "A"}
        engine = engine_for(backend, "bitnet")
        captured = []
        with inference_fork(engine):
            self.assertEqual(backend.generate_branches("choose A or B"), ["A"])
        self.assertIs(captured[0]["cache_prompt"], False)
        backend.generate_branches("normal request")
        self.assertNotIn("cache_prompt", captured[1])

        class Response:
            def __iter__(self):
                return iter([b'data: {"content":"B","stop":true}\n'])

            def close(self):
                pass

        def request(req, **_kwargs):
            captured.append(json.loads(req.data))
            return Response()

        with inference_fork(engine), patch("urllib.request.urlopen", side_effect=request):
            self.assertEqual("".join(backend.stream_generate_tokens("judge")), "B")
        self.assertIs(captured[-1]["cache_prompt"], False)

    def test_prism_proxy_can_discard_worker_context(self):
        from core.engines.prism_gguf_engine import _PrismModelProxy
        requests = []
        model = _PrismModelProxy(SimpleNamespace(_call=lambda payload: requests.append(payload)))
        model.reset()
        self.assertEqual(requests, [{"op": "reset"}])


if __name__ == "__main__":
    unittest.main()
