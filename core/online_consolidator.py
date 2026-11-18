"""Synchronous, validated updates of the active model's persistent adapters."""
from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
import time
from pathlib import Path
from uuid import uuid4

from core.context_memory import ChatArchive, RetirementLedger, completed_groups, oldest_completed_chunk, working_history
from core.training_memory import backend_label, process_rss_mb, release_training_memory
from memory.anchor_dataset import get_anchor_dataset, get_anchor_texts

logger = logging.getLogger(__name__)

# Maximum increases in mean negative log likelihood (nats per target token).
RETENTION_TOLERANCE = 0.10
REPLAY_LIMIT = 8


def source_probe(text):
    """A bounded verbatim continuation; the expected span is absent from the cue."""
    words = re.findall(r'\S+', text.strip())
    if len(words) < 2:
        return None
    # Prefer the value of an explicit association over a generic sentence suffix.
    split = next((i + 1 for i, w in enumerate(words[:-1]) if w.casefold() in
                  {"is", "equals", "means", "called", "named", "to", "="}), len(words) // 2)
    cue = " ".join(words[max(0, split - 16):split])[-256:]
    expected = " ".join(words[split:split + 12])[:192]
    if not expected or expected.casefold() in cue.casefold():
        return None
    return {"prompt": "Complete the earlier statement: " + cue, "completion": expected}


def _sample(rows, limit):
    if len(rows) <= limit:
        return rows
    return [rows[i * (len(rows) - 1) // (limit - 1)] for i in range(limit)]


def _probes(pairs, evidence):
    rows = []
    for text in evidence:
        probe = source_probe(text)
        if probe and probe not in rows:
            rows.append(probe)
    if not rows:
        # Single-word exchanges still supply an answer-blind question.
        for row in pairs:
            cue = row["prompt"][-256:]
            target = row["completion"][:192]
            if target.casefold() not in cue.casefold():
                rows.append({"prompt": cue, "completion": target})
    if not rows:
        raise ValueError("No independent continuation cue could be formed from this material")
    return _sample(rows, 4)


def _rehearsal(previous, new):
    # Retain the newest observations and a deterministic sample of earlier ones.
    rows = []
    for row in [*previous, *new]:
        if row not in rows:
            rows.append(row)
    if len(rows) <= REPLAY_LIMIT:
        return rows
    ranked = sorted(rows[:-2], key=lambda p: hashlib.sha256(
        (p["prompt"] + "\x00" + p["completion"]).encode()).digest())
    return ranked[:REPLAY_LIMIT - 2] + rows[-2:]


def _scores(backend, probes):
    values = [float(backend.score_learning_probe(p["prompt"], p["completion"])) for p in probes]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("The active model produced invalid continuation scores")
    return values


class AwakeOnlineConsolidator:
    def __init__(self, mlx_engine, memory_db=None, max_context=8192, watermark=0.80,
                 evict_ratio=0.40, lambda_ewc=400.0, *, session_id=None):
        self.engine = mlx_engine
        self.session_id = session_id or uuid4().hex
        self.db = memory_db
        self.max_context = max_context
        self.watermark_tokens = int(max_context * watermark)
        self.evict_ratio = evict_ratio
        self.lambda_ewc = lambda_ewc
        self.lock = threading.Lock()
        self._update_lock = threading.Lock()
        self.is_consolidating = False
        self.consolidation_count = 0
        self.total_param_shift = 0.0
        self.last_result = {}
        self.last_error = ""
        self._ledger = None

    def _retirement_ledger(self, store):
        if self._ledger is None or self._ledger.path.parent != store.directory:
            self._ledger = RetirementLedger(store.directory)
        return self._ledger

    def _real_training_ready(self):
        if not (self.engine is not None and getattr(self.engine, "model", None) is not None
                and getattr(self.engine, "tokenizer", None) is not None
                and callable(getattr(self.engine, "train_mini_batch", None))
                and callable(getattr(self.engine, "score_learning_probe", None))
                and getattr(self.engine, "learning_state", None) is not None):
            return False
        capability = getattr(self.engine, "training_ready", None)
        try:
            return capability is None or bool(capability())
        except Exception as error:
            self.last_error = str(error)
            return False

    @staticmethod
    def _conversation_training_pairs(messages):
        history = working_history(messages)
        return [{"prompt": "\n".join(history[i]["content"] for i in group[:-1]),
                 "completion": history[group[-1]]["content"]} for group in completed_groups(history)]

    @staticmethod
    def _memory(manifest):
        return (manifest or {}).get("metadata", {}).get("memory", {})

    def active_history(self, history):
        """Project archived turns against the checkpoint bound to this active backend."""
        store = getattr(self.engine, "learning_state", None)
        if store is None or not history:
            return working_history(history, self.session_id)
        manifest = store.current_manifest(verify=False)
        # Another process may have committed a newer adapter since this model
        # loaded. That checkpoint cannot authorize eviction from this live model.
        if manifest:
            loaded = getattr(self.engine, "_mlx_parent_generation", ...)
            if loaded is ...:
                path = getattr(self.engine, "adapter_path", None)
                expected = store.directory / "generations" / manifest["generation"] / manifest["checkpoint"]
                matches = bool(path) and Path(path).resolve() == expected.resolve()
            else:
                matches = loaded == manifest["generation"]
            if not matches:
                raise RuntimeError("Reload the latest learned checkpoint before retiring context")
        head = self._memory(manifest).get("retirement_head")
        if not head:
            return working_history(history, self.session_id)
        ledger = self._retirement_ledger(store)
        if isinstance(history, ChatArchive):
            return history.active_view(ledger, head)
        return ledger.active_turns(working_history(history, self.session_id), head)

    def check_and_prune(self, conversation_history):
        if not conversation_history or not self._real_training_ready():
            return conversation_history, False
        active = self.active_history(conversation_history)
        counter = getattr(self.engine, "count_tokens", None)
        count = counter(active) if counter else sum(len(m.get("content", "")) for m in active) // 4
        if count < self.watermark_tokens:
            return active, False
        evicted_chunk, retained_history = oldest_completed_chunk(active, self.evict_ratio)
        success = self.consolidate_chunk_sync(evicted_chunk)
        return (retained_history, True) if success else (active, False)

    def consolidate_chunk_sync(self, chunk, *, cancel_event=None, mode="awake"):
        if not chunk:
            return False
        try:
            self.train_pairs(self._conversation_training_pairs(chunk), turns=working_history(chunk, self.session_id),
                             cancel_event=cancel_event, mode=mode)
            mark = getattr(self.db, "mark_traces_consolidated", None)
            if callable(mark):
                try:
                    mark(chunk)
                except Exception as error:
                    logger.warning("The adapter committed, but the optional trace log could not be marked: %s", error)
            return True
        except Exception as error:
            self.last_error = str(error)
            logger.warning("%s consolidation retained the original dialogue: %s", mode, error)
            return False

    def _run_shadow_consolidation(self, chunk):
        """Compatibility with the conversation-teach evaluation entry point."""
        return self.consolidate_chunk_sync(chunk)

    def sleep(self, transcript, cancel_event=None):
        active = self.active_history(transcript)
        chunk, _ = oldest_completed_chunk(active, self.evict_ratio, keep_recent=False)
        if not chunk:
            return {"status": "idle", "memories_consolidated": 0}
        if not self.consolidate_chunk_sync(chunk, cancel_event=cancel_event, mode="sleep"):
            raise RuntimeError(self.last_error)
        return self.last_result.copy()

    def train_pairs(self, pairs, *, turns=(), evidence=(), mode="learn", cancel_event=None,
                    learning_rate=1e-4):
        """Serialize real updates. A committed manifest carries the exact retirement receipt."""
        with self._update_lock:
            with self.lock:
                self.is_consolidating = True
            backend = self.engine
            result = None
            try:
                if not self._real_training_ready():
                    raise RuntimeError("The active backend lacks a verified persistent trainer or probe scorer")
                result = self._train(backend, pairs, turns, evidence, mode, cancel_event, learning_rate)
                self.consolidation_count += 1
                self.total_param_shift += result["parameter_drift_l2"]
                self.last_result = result
                self.last_error = ""
                return result
            finally:
                try:
                    owner = getattr(backend, "run_on_mlx_owner", None)
                    if callable(owner) and getattr(backend, "_mlx_owner", None) is not None:
                        cleanup = owner(release_training_memory, backend)
                    else:
                        cleanup = release_training_memory(backend)
                    if result is not None:
                        result.update(ram_after_cleanup_mb=cleanup["after_mb"], ram_released_mb=cleanup["released_mb"])
                except Exception as error:
                    logger.warning("Training allocator cleanup failed: %s", error)
                finally:
                    with self.lock:
                        self.is_consolidating = False

    def _train(self, backend, pairs, turns, evidence, mode, cancel_event, learning_rate):
        def check_cancel():
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Learning cancelled before checkpoint commit")

        check_cancel()
        if not pairs:
            raise ValueError("No complete dialogue or learning pairs")
        store = backend.learning_state
        manifest = store.current_manifest(verify=False)
        memory = self._memory(manifest)
        parent_head = memory.get("retirement_head")
        old_replay = memory.get("replay", [])
        update_id = uuid4().hex
        ledger = None
        if turns:
            if len(self.active_history(turns)) != len(turns):
                raise RuntimeError("This update contains previously retired turns")
            ledger = self._retirement_ledger(store)
        evidence = list(evidence) or [
            turn["content"] for turn in turns if
            not (turn["role"] == "user" and turn["content"].strip().endswith("?"))
            and turn["content"].strip(" .!").casefold() not in {"yes", "ok", "okay", "got it", "understood"}
        ]
        evidence = evidence or [row["completion"] for row in pairs]
        probes = _probes(pairs, evidence)
        anchors = get_anchor_dataset()
        # Use different foundation items for rehearsal and for acceptance.
        guard = [anchors[2], *_sample(old_replay, 2)]
        baseline = _scores(backend, probes)
        baseline_guard = _scores(backend, guard)
        metrics = {}

        def validate(candidate):
            check_cancel()
            after = _scores(candidate, probes)
            after_guard = _scores(candidate, guard)
            metrics.update(probe_nll_before=baseline, probe_nll_after=after,
                           retention_nll_before=baseline_guard, retention_nll_after=after_guard)
            if sum(after) >= sum(baseline):
                raise RuntimeError("The candidate failed to improve the measured source continuations")
            if any(now > old + RETENTION_TOLERANCE for now, old in zip(after, baseline)):
                raise RuntimeError("The candidate regressed on a source continuation")
            if any(now > old + RETENTION_TOLERANCE for now, old in zip(after_guard, baseline_guard)):
                raise RuntimeError("The candidate regressed on a retention probe")
            check_cancel()
            return {"accepted": True, **metrics}

        new_memory = {"retirement_head": update_id if turns else parent_head,
                      "replay": _rehearsal(old_replay, probes)}
        metadata = {"update_id": update_id, "mode": mode, "memory": new_memory,
                    "expected_parent": (manifest or {}).get("generation")}
        # RSI trains the self-generated trace only. Retention scores are evaluated
        # independently after the trace is written and never fed into its generation.
        data = list(pairs)
        if mode != "rsi":
            data += [anchors[0], *_sample(old_replay, 2)]
        if ledger:
            ledger.stage(update_id, parent_head, [turn.turn_id for turn in turns])
        start = time.monotonic()
        ram = process_rss_mb()
        try:
            check_cancel()
            updated, drift = backend.train_mini_batch(
                adapters=getattr(backend, "adapters", {}) or {}, data=data,
                lambda_ewc=self.lambda_ewc, anchor_texts=get_anchor_texts()[:2],
                learning_rate=learning_rate, steps=3, cancel_event=cancel_event,
                validation_callback=validate, learning_metadata=metadata,
            )
            # Backends check finite drift and verify persisted native tensors before
            # publishing. Verify that this exact transaction became durable.
            committed = store.current_manifest()
            drift = float(drift)
            if (not metrics or not math.isfinite(drift) or drift <= 0 or not committed
                    or committed["update_id"] != update_id or self._memory(committed) != new_memory):
                raise RuntimeError("The trainer returned without a validated, durable parameter update")
        except BaseException:
            if ledger:
                current = store.current_manifest(verify=False)
                if (current or {}).get("generation") == (manifest or {}).get("generation"):
                    ledger.discard(update_id)
            raise

        backend_metrics = (committed.get("metadata", {}).get("metrics")
                           or getattr(backend, "last_learning_metrics", {}) or {})
        touched = int(backend_metrics.get("trainable_parameters_touched", 0))
        if not touched and isinstance(updated, dict):
            touched = int(updated.get("trainable_parameters_touched", 0))
        checkpoint = Path(store.directory) / "generations" / committed["generation"] / committed["checkpoint"]
        return {"status": "success", "mode": mode, "identity_digest": store.identity_digest, "memories_consolidated": len(pairs),
                "parameter_drift_l2": drift, "trainable_parameters_touched": touched,
                "trainable_parameters_m": touched / 1_000_000, "ram_before_mb": ram,
                "ewc_active": bool(backend_metrics.get("ewc_used") or backend_metrics.get("fisher_tensors", 0) > 0),
                "replay_examples": len(data) - len(pairs), "backend": backend_label(backend),
                "adapter_saved_to": str(checkpoint), "execution_time_seconds": time.monotonic() - start, **metrics}
