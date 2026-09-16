"""
Awake online synaptic consolidation.
Evicts old dialogue only when a real trainable MLX model is present, then performs
serialized background LoRA/EWC updates, measures real drift, and persists the
updated adapter so learning survives process restarts.
"""

import copy
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

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
        return bool(
            self.engine is not None
            and getattr(self.engine, "model", None) is not None
            and getattr(self.engine, "tokenizer", None) is not None
            and getattr(self.engine, "is_mlx_available", False)
            and callable(getattr(self.engine, "train_mini_batch", None))
        )

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

    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return max(1, total_chars // 4)

    def _run_shadow_consolidation(self, chunk: List[Dict[str, str]]):
        """Run and persist a genuine parameter update; never fabricate drift."""
        start_time = time.time()
        logger.info("[AwakeConsolidator] Commencing consolidation on %d turns...", len(chunk))

        try:
            if not self._real_training_ready():
                raise RuntimeError("real MLX model/tokenizer unavailable for awake consolidation")

            training_pairs = self._conversation_training_pairs(chunk)
            if not training_pairs:
                raise RuntimeError("awake consolidation found no user/assistant training pairs")

            with self.lock:
                active_adapters = getattr(self.engine, "adapters", None)
                shadow_adapters = copy.deepcopy(active_adapters) if active_adapters is not None else {}

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
            logger.info(
                "[AwakeConsolidator] Cycle #%d complete in %.2fs | Param Drift ||ΔW||2: %.6f",
                cycle,
                duration,
                float(param_drift),
            )

            if self.db and hasattr(self.db, "mark_traces_consolidated"):
                self.db.mark_traces_consolidated(chunk)

        except Exception as exc:
            logger.error(
                "[AwakeConsolidator] Real background consolidation failed: %s",
                exc,
                exc_info=True,
            )
        finally:
            with self.lock:
                self.is_consolidating = False
