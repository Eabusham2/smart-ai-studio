"""
Awake online synaptic consolidation.
Evicts old dialogue only when a real trainable active backend is present, then performs
serialized adapter updates, measures real drift, and persists the updated adapter
so learning survives process restarts. Normal watermark work may run in background;
a hard context-capacity request can use the same trainer synchronously so dialogue is
not removed before its parameter update has actually completed.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from core.training_memory import backend_label, process_rss_mb, release_training_memory

logger = logging.getLogger(__name__)


class AwakeOnlineConsolidator:
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
        self.consolidation_count = 0
        self.total_param_shift = 0.0

    def _real_training_ready(self) -> bool:
        if not (
            self.engine is not None
            and getattr(self.engine, "model", None) is not None
            and getattr(self.engine, "tokenizer", None) is not None
            and callable(getattr(self.engine, "train_mini_batch", None))
        ):
            return False
        capability = getattr(self.engine, "training_ready", None)
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
        if not conversation_history or not self._real_training_ready():
            return conversation_history, False

        with self.lock:
            if self.is_consolidating:
                return conversation_history, False

        token_count = (
            self.engine.count_tokens(conversation_history)
            if hasattr(self.engine, "count_tokens")
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
        is_native_mlx = bool(getattr(self.engine, "is_mlx_available", False))
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
            args=(evicted_chunk,),
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

    def consolidate_chunk_sync(self, chunk: List[Dict[str, str]]) -> bool:
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
            with self.lock:
                busy = self.is_consolidating
            if not busy:
                break
            time.sleep(0.05)

        with self.lock:
            if self.is_consolidating:
                return False
            self.is_consolidating = True

        return bool(self._run_shadow_consolidation(chunk))

    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return max(1, total_chars // 4)

    def _run_shadow_consolidation(self, chunk: List[Dict[str, str]]) -> bool:
        """Run and persist a genuine parameter update; never fabricate drift."""
        start_time = time.time()
        backend_name = backend_label(self.engine)
        ram_start_mb = process_rss_mb()
        logger.info(
            "[AwakeConsolidator] start backend=%s turns=%d RAM=%.0f MB",
            backend_name,
            len(chunk),
            ram_start_mb,
        )
        success = False

        try:
            if not self._real_training_ready():
                raise RuntimeError("real trainable model/tokenizer unavailable for awake consolidation")

            training_pairs = self._conversation_training_pairs(chunk)
            if not training_pairs:
                raise RuntimeError("awake consolidation found no user/assistant training pairs")

            with self.lock:
                active_adapters = getattr(self.engine, "adapters", None)
                # train_mini_batch updates the live backend transactionally; this
                # argument is compatibility metadata, not a rollback snapshot.
                # A deep copy can duplicate every MLX LoRA tensor for no benefit.
                shadow_adapters = dict(active_adapters) if isinstance(active_adapters, dict) else {}

            updated_adapters, param_drift = self.engine.train_mini_batch(
                adapters=shadow_adapters,
                data=training_pairs,
                lambda_ewc=self.lambda_ewc,
                steps=3,
                save_path=getattr(self.engine, "adapter_path", None),
            )

            try:
                import mlx.core as mx
                if isinstance(updated_adapters, dict) and updated_adapters:
                    mx.eval(*list(updated_adapters.values()))
            except Exception:
                pass

            if not isinstance(param_drift, (int, float)) or float(param_drift) <= 0.0:
                raise RuntimeError("awake consolidation produced no measurable parameter update")

            with self.lock:
                if hasattr(self.engine, "adapters"):
                    self.engine.adapters = updated_adapters
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
            stats = release_training_memory(self.engine)
            logger.info(
                "[AwakeConsolidator] cleanup backend=%s RAM %.0f -> %.0f MB (released %.0f MB)",
                backend_name,
                ram_start_mb,
                stats["after_mb"],
                max(0.0, ram_start_mb - stats["after_mb"]),
            )
            with self.lock:
                self.is_consolidating = False

        return success
