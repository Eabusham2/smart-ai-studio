"""
Awake online synaptic consolidation.
Evicts old dialogue only when a real trainable active backend is present, then performs
serialized adapter updates, measures real drift, and persists the updated adapter
so learning survives process restarts. Normal watermark work may run in background;
a hard context-capacity request can use the same trainer synchronously so dialogue is
not removed before its parameter update has actually completed.
"""

import logging
import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from core.context_memory import ActiveChatHistory, ChatArchive
from core.learning_implementation import learning_mode_kwargs
from core.training_memory import backend_label, process_rss_mb
from core.training_memory import release_training_memory_on_owner as release_training_memory

logger = logging.getLogger(__name__)


class LegacyAwakeOnlineConsolidator:
    learning_implementation = "legacy"

    def __init__(
        self,
        mlx_engine: Any,
        memory_db: Optional[Any] = None,
        max_context: int = 8192,
        watermark: float = 0.80,
        evict_ratio: float = 0.40,
        lambda_ewc: float = 400.0,
    ):
        self.engine = mlx_engine
        self.db = memory_db
        self.max_context = max_context
        self.watermark_tokens = int(max_context * watermark)
        self.evict_ratio = evict_ratio
        self.lambda_ewc = lambda_ewc
        self.is_consolidating = False
        self.lock = threading.Lock()
        self._update_lock = threading.RLock()
        self.consolidation_count = 0
        self.total_param_shift = 0.0

    def _real_training_ready(self, backend=None) -> bool:
        backend = self.engine if backend is None else backend
        store = getattr(backend, "learning_state", None)
        if getattr(getattr(store, "scope", None), "pin", None):
            return False
        if not (
            backend is not None
            and getattr(backend, "model", None) is not None
            and getattr(backend, "tokenizer", None) is not None
            and callable(getattr(backend, "train_mini_batch", None))
        ):
            return False
        capability = getattr(backend, "training_ready", None)
        if callable(capability):
            try:
                return bool(capability())
            except Exception:
                return False
        return True

    @staticmethod
    def _conversation_training_pairs(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert real chat role/content turns into the prompt/completion schema the MLX trainer consumes."""
        pairs: List[Dict[str, str]] = []
        pending_user: List[str] = []
        for message in messages or []:
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            if role == "user":
                pending_user.append(content)
                continue
            if role == "assistant" and pending_user:
                prompt = "\n".join(pending_user).strip()
                if prompt:
                    pairs.append({"prompt": prompt, "completion": content})
                pending_user = []
        return pairs

    def check_and_prune(
        self,
        conversation_history: List[Dict[str, str]],
    ) -> Tuple[List[Dict[str, str]], bool]:
        """Prune only when real parameter consolidation can actually run."""
        # Persistent archives can retire turns only against a managed recall receipt.
        if isinstance(conversation_history, (ChatArchive, ActiveChatHistory)):
            return conversation_history, False
        backend = self.engine
        if not conversation_history or not self._real_training_ready(backend):
            return conversation_history, False

        with self.lock:
            if self.is_consolidating:
                return conversation_history, False

        token_count = (
            backend.count_tokens(conversation_history)
            if hasattr(backend, "count_tokens")
            else self._estimate_tokens(conversation_history)
        )
        if token_count < self.watermark_tokens:
            return conversation_history, False

        total_turns = len(conversation_history)
        if total_turns <= 2:
            return conversation_history, False

        evict_count = max(1, int(total_turns * self.evict_ratio))
        if evict_count % 2 != 0 and evict_count < total_turns - 2:
            evict_count += 1

        evicted_chunk = conversation_history[:evict_count]
        retained_history = conversation_history[evict_count:]

        # Keep the proven MLX behavior unchanged: its in-process adapter update can
        # safely run asynchronously. GGUF/native trainers may reject a device or
        # quantization at runtime, so never evict context until that update really succeeds.
        is_native_mlx = bool(getattr(backend, "is_mlx_available", False))
        if not is_native_mlx:
            success = self.consolidate_chunk_sync(evicted_chunk)
            return (retained_history, True) if success else (conversation_history, False)

        with self.lock:
            if self.is_consolidating:
                return conversation_history, False
            self.is_consolidating = True
            cycle = self.consolidation_count + 1

        worker = threading.Thread(
            target=self._run_shadow_consolidation,
            args=(evicted_chunk, backend),
            daemon=True,
            name=f"AwakeConsolidator-{cycle}",
        )
        try:
            worker.start()
        except Exception:
            with self.lock:
                self.is_consolidating = False
            raise

        return retained_history, True

    def consolidate_chunk_sync(self, chunk: List[Dict[str, str]], *, cancel_event=None) -> bool:
        """Synchronously consolidate a completed dialogue prefix before removing it.

        This is reserved for hard model-context pressure. It uses the exact same real
        LoRA/EWC trainer as background consolidation and returns True only after a
        measurable parameter update has completed. Callers must keep the original
        history if False is returned.
        """
        if not chunk or not self._real_training_ready():
            return False

        # Do not race a previous awake update against generation. Wait until the
        # existing serialized update completes, then claim the same consolidation lock.
        while True:
            if cancel_event is not None and cancel_event.is_set():
                return False
            with self.lock:
                if not self.is_consolidating:
                    self.is_consolidating = True
                    backend = self.engine
                    break
            time.sleep(0.05)

        return bool(self._run_shadow_consolidation(chunk, backend))

    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return max(1, total_chars // 4)

    def _run_shadow_consolidation(self, chunk: List[Dict[str, str]], backend=None) -> bool:
        # Best Work holds this same caller-side lock across its read-only
        # branch cycle. A queued update runs afterwards on the usual trainer.
        with self._update_lock:
            return self._run_shadow_consolidation_locked(chunk, backend)

    def _run_shadow_consolidation_locked(self, chunk: List[Dict[str, str]], backend=None) -> bool:
        """Run and persist a genuine parameter update; never fabricate drift."""
        backend = self.engine if backend is None else backend
        start_time = time.time()
        backend_name = backend_label(backend)
        ram_start_mb = process_rss_mb()
        logger.info(
            "[AwakeConsolidator] start backend=%s turns=%d RAM=%.0f MB",
            backend_name,
            len(chunk),
            ram_start_mb,
        )
        success = False

        try:
            if not self._real_training_ready(backend):
                raise RuntimeError("real trainable model/tokenizer unavailable for awake consolidation")

            training_pairs = self._conversation_training_pairs(chunk)
            if not training_pairs:
                raise RuntimeError("awake consolidation found no user/assistant training pairs")

            with self.lock:
                active_adapters = getattr(backend, "adapters", None)
                # train_mini_batch updates the live backend transactionally; this
                # argument is compatibility metadata, not a rollback snapshot.
                # A deep copy can duplicate every MLX LoRA tensor for no benefit.
                shadow_adapters = dict(active_adapters) if isinstance(active_adapters, dict) else {}

            updated_adapters, param_drift = backend.train_mini_batch(
                adapters=shadow_adapters,
                data=training_pairs,
                lambda_ewc=self.lambda_ewc,
                steps=3,
                save_path=getattr(backend, "adapter_path", None),
                **learning_mode_kwargs(backend.train_mini_batch, "legacy"),
            )

            if getattr(backend, "is_mlx_available", False) and isinstance(updated_adapters, dict) and updated_adapters:
                def materialize_adapters():
                    import mlx.core as mx
                    tensors = [value for value in updated_adapters.values() if isinstance(value, mx.array)]
                    if tensors:
                        mx.eval(*tensors)
                on_mlx = getattr(backend, "run_on_mlx_owner", None)
                # Materialize tensors on the exact thread where the trained
                # model created them, not on the background consolidator.
                if callable(on_mlx):
                    on_mlx(materialize_adapters)
                else:
                    try:
                        materialize_adapters()
                    except ImportError:
                        pass

            if not isinstance(param_drift, (int, float)) or not math.isfinite(param_drift) or param_drift <= 0:
                raise RuntimeError("awake consolidation produced no measurable parameter update")

            with self.lock:
                if hasattr(backend, "adapters"):
                    backend.adapters = updated_adapters
                self.consolidation_count += 1
                self.total_param_shift += float(param_drift)
                cycle = self.consolidation_count

            duration = time.time() - start_time
            if self.db and hasattr(self.db, "mark_traces_consolidated"):
                self.db.mark_traces_consolidated(chunk)
            success = True

            logger.info(
                "[AwakeConsolidator] complete backend=%s cycle=%d %.2fs drift=%.6f",
                backend_name,
                cycle,
                duration,
                float(param_drift),
            )

        except Exception as exc:
            logger.error(
                "[AwakeConsolidator] Real consolidation failed backend=%s: %s",
                backend_name,
                exc,
                exc_info=True,
            )
        finally:
            # Release transient gradients/optimizers/framework caches for every
            # backend without unloading the working inference model.
            updated_adapters = None
            shadow_adapters = None
            try:
                stats = release_training_memory(backend)
                logger.info(
                    "[AwakeConsolidator] cleanup backend=%s RAM %.0f -> %.0f MB (released %.0f MB)",
                    backend_name,
                    ram_start_mb,
                    stats["after_mb"],
                    max(0.0, ram_start_mb - stats["after_mb"]),
                )
            finally:
                with self.lock:
                    self.is_consolidating = False

        return success
