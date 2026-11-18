"""Synchronous, validated updates of the active model's persistent adapters."""
from __future__ import annotations

import hashlib
import logging
import math
import re
import sqlite3
import threading
import time
from itertools import chain, islice
from pathlib import Path
from uuid import uuid4

from core.context_memory import (ChatArchive, ContextOverflow, RetirementLedger, TranscriptTurn, completed_groups,
                                 iter_completed_exchanges, oldest_completed_chunk, working_history)
from core.learning_recall import (PromptTooLong, RecallBank, ReplayBank, SegmentedRecall, StreamCursorBank, check_cancel,
                                  generation_room, needs_segmented_recall, passes_recall, retirement_probes)
from core.legacy_online_consolidator import LegacyAwakeOnlineConsolidator
from core.learning_implementation import get_learning_implementation, learning_mode_kwargs
from core.training_memory import backend_label, process_rss_mb
from core.training_memory import release_training_memory_on_owner as release_training_memory
from memory.anchor_dataset import get_anchor_dataset, get_anchor_texts

logger = logging.getLogger(__name__)

# Maximum increases in mean negative log likelihood (nats per target token).
RETENTION_TOLERANCE = 0.10
REPLAY_LIMIT = 8


class LearningValidationError(RuntimeError):
    """The candidate was measured, rejected and rolled back by the trainer."""


def source_probe(text):
    """A bounded verbatim continuation; the expected span is absent from the cue."""
    markers = {"is", "equals", "means", "called", "named", "to", "="}
    split = None
    count = 0
    marker = False
    for word in re.finditer(r'\S+', text):
        if marker:
            split = count
            break
        count += 1
        marker = word.end() - word.start() <= 6 and word.group().casefold() in markers
    if split is None:
        if count < 2:
            return None
        split = count // 2
    # Bound even the temporary strings for a megabyte-long identifier.
    cue = expected = ""
    for i, word in enumerate(re.finditer(r'\S+', text)):
        if i < split - 16:
            continue
        if i >= split + 12:
            break
        if i < split:
            suffix = text[max(word.start(), word.end() - 256):word.end()]
            cue = (cue + " " + suffix).lstrip()[-256:]
        elif len(expected) < 192:
            prefix = text[word.start():min(word.end(), word.start() + 192)]
            expected = (expected + " " + prefix).lstrip()[:192]
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
    values = []
    sources = set()
    for row in probes:
        score = None
        if "rollout_completion_ids" in row:
            from core.policy_reference import sampled_action_nll
            score = sampled_action_nll(backend, row)
        if score is None:
            score = backend.score_learning_probe(row["prompt"], row["completion"])
            sources.add("serving_text_probe")
        else:
            sources.add("sampled_token_ids")
        values.append(float(score))
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("The active model produced invalid continuation scores")
    return values, sorted(sources)


class ManagedAwakeOnlineConsolidator:
    learning_implementation = "managed"

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
        self._update_lock = threading.RLock()
        self.is_consolidating = False
        self.consolidation_count = 0
        self.total_param_shift = 0.0
        self.last_result = {}
        self.last_error = ""
        self._attempt = threading.local()
        self._ledger = None

    def _retirement_ledger(self, store):
        if self._ledger is None or self._ledger.path.parent != store.directory:
            self._ledger = RetirementLedger(store.directory)
        return self._ledger

    def _real_training_ready(self):
        scope = getattr(getattr(self.engine, "learning_state", None), "scope", None)
        if getattr(scope, "pin", None):
            self.last_error = "The current document checkpoint is pinned for read-only inference"
            return False
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

    def _loaded_head(self, store):
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
        memory = self._memory(manifest)
        # Older receipts relied on continuation likelihood alone. Restore
        # their source turns until a new generation proves complete recall.
        return memory.get("retirement_head") if memory.get("recall_version") == 1 else None

    def active_history(self, history):
        """Project archived turns against the checkpoint bound to this active backend."""
        store = getattr(self.engine, "learning_state", None)
        head = self._loaded_head(store) if store is not None and history else None
        if not head:
            return history if isinstance(history, ChatArchive) else working_history(history, self.session_id)
        ledger = self._retirement_ledger(store)
        if isinstance(history, ChatArchive):
            return history.projected_history(ledger, head)
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

    def consolidate_chunk_sync(self, chunk, *, cancel_event=None, mode="awake", raise_operational_errors=False):
        self._attempt.candidate_rejected = False
        if not chunk:
            return False
        try:
            turns = working_history(chunk, self.session_id)
            if needs_segmented_recall(self.engine, turns):
                result = self._train_segments(turns, mode, cancel_event)
            else:
                result = self.train_pairs(self._conversation_training_pairs(turns), turns=turns,
                                          cancel_event=cancel_event, mode=mode)
            if result["turns_retired"] != len(chunk):
                self._attempt.candidate_rejected = True
                self.last_error = result["retirement_reason"]
                return False
            mark = getattr(self.db, "mark_traces_consolidated", None)
            if callable(mark):
                try:
                    mark(chunk)
                except Exception as error:
                    logger.warning("The adapter committed, but the optional trace log could not be marked: %s", error)
            return True
        except Exception as error:
            self._attempt.candidate_rejected = isinstance(error, (LearningValidationError, PromptTooLong))
            self.last_error = str(error)
            logger.warning("%s consolidation retained the original dialogue: %s", mode, error)
            if raise_operational_errors and not self._attempt.candidate_rejected:
                raise
            return False

    def consolidate_for_capacity(self, chunk, *, cancel_event=None):
        if not self._real_training_ready():
            self._attempt.candidate_rejected = False
            return False
        return self.consolidate_chunk_sync(chunk, cancel_event=cancel_event, raise_operational_errors=True)

    def _run_shadow_consolidation(self, chunk):
        """Compatibility with the conversation-teach evaluation entry point."""
        return self.consolidate_chunk_sync(chunk)

    def _train_segments(self, turns, mode, event):
        # Every interim checkpoint records an exact restart position. Only the
        # last candidate can publish a retirement receipt for the whole input.
        with self._update_lock:
            plan = SegmentedRecall(self.engine, turns, event)
            store = self.engine.learning_state
            memory = self._memory(store.current_manifest(verify=False))
            room = generation_room(self.engine, "", 10000000)
            job = hashlib.sha256(("chat-passages-v1/" + str(room) + "/" +
                                  "/".join(t.turn_id for t in turns)).encode()).hexdigest()
            cursor = StreamCursorBank(store.directory).lookup(memory.get("stream_head"), job)
            position = cursor.get("position", (0, 0))
            serial = cursor.get("next", 0)
            passages = plan.passages(*position)
            upcoming = next(passages, None)
            while upcoming:
                # Keep at most four addressed passages and one lookahead in
                # RAM. Large dialogue never becomes one giant optimizer batch.
                batch = [upcoming, *islice(passages, 3)]
                upcoming = next(passages, None)
                serial += 1
                rows = [r for r, _, _ in batch]
                # Start a fresh pass if a completed attempt cannot reconstruct
                # all earlier segments; an interruption resumes the current pass.
                next_state = {"next": serial, "position": batch[-1][2] if upcoming else [0, 0]}
                result = self.train_pairs(rows, turns=() if upcoming else turns, mode=mode,
                                          stream_cursor=(job, next_state), cancel_event=event,
                                          recall_plan=None if upcoming else plan,
                                          evidence=[r["completion"] for r in rows],
                                          training_options={"supervise_prompt": False})
                if not upcoming:
                    return result
            raise ValueError("No contiguous source passages to learn")

    def learn_current_document(self, source, cancel_event=None):
        """Verify explicit source material before substituting its address on the current request."""
        digest = hashlib.sha256()
        for offset in range(0, len(source), 8192):
            check_cancel(cancel_event)
            digest.update(source[offset:offset + 8192].encode())
        key = digest.hexdigest()
        turn = TranscriptTurn({"role": "document", "content": source}, key)
        if self.active_history([turn]):
            result = self._train_segments([turn], "current_document", cancel_event)
            verified = bool(result["turns_retired"])
        else:
            # A subsequent checkpoint may have sampled a different historical
            # receipt. Recheck this particular source before omitting it again.
            plan = SegmentedRecall(self.engine, [turn], cancel_event)
            verified = passes_recall(self.engine, plan.checks(), cancel_event)
        if not verified:
            raise RuntimeError("The complete document could not be recalled. Its source was retained.")
        return key

    def _current_receipt(self):
        store = getattr(self.engine, "learning_state", None)
        head = self._loaded_head(store) if store is not None else None
        return self._retirement_ledger(store) if head else None, head

    def _archive_pages(self, archive, page_size=16):
        return archive.iter_active_pages(*self._current_receipt(), page_size=page_size)

    def recover_after_rejected_chunk(self, history, chunk, active, count, budget, measure, cancel_event):
        """At the hard limit, try disjoint later exchanges once and keep the newest in text."""
        retired = 0
        if not getattr(self._attempt, "candidate_rejected", False):
            return active, count, retired
        last_failed_id = chunk[-1].turn_id
        if isinstance(history, ChatArchive):
            rows = history.active_turns_after(last_failed_id, self._current_receipt)
        else:
            rows = iter(history)
            for turn in rows:
                if getattr(turn, "turn_id", None) == last_failed_id:
                    break
        original_error = self.last_error
        previous = None
        for group in iter_completed_exchanges(rows):
            check_cancel(cancel_event)
            if previous:
                if self.consolidate_for_capacity(previous, cancel_event=cancel_event):
                    retired += len(previous)
                    active, count = measure()
                    if count < budget:
                        break
                elif not self._attempt.candidate_rejected:
                    break
            previous = group
        check_cancel(cancel_event)
        if count >= budget and not self.last_error:
            self.last_error = original_error
        return active, count, retired

    def pending_exchanges(self, history):
        if not isinstance(history, ChatArchive):
            return len(completed_groups(self.active_history(history)))
        pages = self._archive_pages(history)
        try:
            return sum(1 for _ in iter_completed_exchanges(chain.from_iterable(pages)))
        finally:
            pages.close()

    def _sleep_chunk(self, rows, request_size, keep_recent=False):
        target = min(32, max(2, int(request_size * self.evict_ratio)))
        selected = []
        recent = None
        for group in iter_completed_exchanges(rows):
            if not keep_recent:
                selected.extend(group)
            else:
                if recent:
                    selected.extend(recent)
                recent = group
            if len(selected) >= target:
                break
        return selected

    def prepare_archive(self, archive, packed_count, budget, cancel_event=None):
        """Page a cold/backlogged archive before any UI route copies it into the prompt.

        Stop collecting as soon as a page reaches the physical boundary. Retire
        from the oldest complete exchanges, then reopen at the published head.
        """
        trigger = max(1, int(budget * 0.8))
        target = max(1, int(budget * 0.6))
        retiring = False
        def preview(pages):
            active = []
            count = packed_count(active)
            try:
                for page in pages:
                    check_cancel(cancel_event)
                    active.extend(page)
                    count = packed_count(active)
                    if count >= budget:
                        return active, count, False
            finally:
                pages.close()
            return active, count, True

        while True:
            check_cancel(cancel_event)
            try:
                active, count, exhausted = preview(self._archive_pages(archive))
            except InterruptedError:
                raise
            except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
                # A different process may have published a new checkpoint.
                # This live model can still use the original visible transcript.
                self.last_error = str(exc)
                active, count, exhausted = preview(archive.iter_pages(page_size=16))
                if not exhausted or count >= budget:
                    if isinstance(exc, (OSError, sqlite3.Error)):
                        raise
                    raise ContextOverflow(f"Original transcript retained. Active context requires at least {count} "
                                          f"tokens with a budget of {budget}. {exc}") from exc
                return active
            if exhausted and (count <= target if retiring else count < trigger):
                return active
            retiring = True
            pages = self._archive_pages(archive)
            try:
                chunk = self._sleep_chunk(chain.from_iterable(pages), len(active), keep_recent=True)
            finally:
                pages.close()
            if not chunk and count >= budget:
                # Even the most recent completed exchange can be learned if it
                # would otherwise make the next request physically impossible.
                pages = self._archive_pages(archive)
                try:
                    chunk = self._sleep_chunk(chain.from_iterable(pages), len(active))
                finally:
                    pages.close()
            attempt = self.consolidate_for_capacity if count >= budget else self.consolidate_chunk_sync
            if chunk and attempt(chunk, cancel_event=cancel_event):
                continue
            check_cancel(cancel_event)
            if count >= budget:
                if chunk:
                    def measure():
                        active, count, _ = preview(self._archive_pages(archive))
                        return active, count
                    active, count, _ = self.recover_after_rejected_chunk(
                        archive, chunk, active, count, budget, measure, cancel_event)
                    if count < budget:
                        # The GUI next adds its guide and checks this working
                        # copy again. Only re-enter recovery if it actually
                        # fills the window. A fresh archive read has no marker.
                        if active:
                            active[0]._capacity_recovery = {"prepared": True}
                        return active
                detail = self.last_error if chunk else "No complete ordinary exchange can be retired."
                raise ContextOverflow(f"Active context requires at least {count} tokens with a budget of {budget}. "
                                   f"Original transcript retained. {detail}")
            return active

    @staticmethod
    def pending_verified_episodes(db):
        """The historical queue includes all verified rows, ordered by surprise."""
        return db.get_stats()["unconsolidated_verified"]

    def sleep_verified_episodes(self, db, *, settings, cancel_event=None, limit=50):
        """Replay shared verified episodes into the explicitly selected durable model scope."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("The verified episode batch size must be a positive integer")
        with self._update_lock, db.sleep_queue_lock():
            check_cancel(cancel_event)
            if not self._real_training_ready():
                raise RuntimeError(self.last_error or "Load a persistent trainable model before verified Sleep")
            store = self.engine.learning_state
            if store.scope.kind == "request":
                raise ValueError("Verified Sleep requires a durable memory scope")
            target = str(store.directory.resolve())
            token, rows = db.prepare_verified_sleep(target, limit)
            if not rows:
                return {"status": "idle", "memories_consolidated": 0}
            job = "verified-sleep/" + token
            bank = StreamCursorBank(store.directory)

            def committed_receipt():
                memory = self._memory(store.current_manifest())
                return bank.lookup(memory.get("stream_head"), job)

            receipt = committed_receipt()
            if receipt:
                result = {"status": "success", "recovered": True, "memories_consolidated": len(rows),
                          "adapter_saved_to": store.current_checkpoint(), "scope": store.scope.record,
                          "identity_digest": store.identity_digest}
            else:
                try:
                    strength = float(settings.ewc_lambda)
                    ratio = float(settings.episodic_replay_ratio)
                    if not 0 < ratio <= 1 or not math.isfinite(strength) or strength < 0:
                        raise ValueError("Invalid Sleep replay ratio or EWC strength")
                    # One independent foundation example is reserved for validation.
                    receipt = {"next": 1, "episode_digest": db.episode_digest(rows),
                               "anchors": len(get_anchor_dataset()) - 1, "ewc_lambda": strength}
                    result = self.train_pairs(
                        [{"prompt": r["prompt"], "completion": r["completion"]} for r in rows],
                        mode="verified_sleep", steps=1, learning_rate=settings.consolidation_lr,
                        training_options={"supervise_prompt": False, "episodic_ratio": ratio, "lambda_ewc": strength},
                        stream_cursor=(job, receipt), cancel_event=cancel_event)
                except BaseException:
                    # A backend may have committed before an error was reported. Keep its
                    # reservation for a later acknowledgement if the durable receipt exists.
                    if not committed_receipt():
                        db.release_verified_sleep(token)
                    raise
            db.finish_verified_sleep(token, result["adapter_saved_to"], receipt)
            return result

    def sleep(self, transcript, cancel_event=None):
        if isinstance(transcript, ChatArchive):
            store = self.engine.learning_state
            head = self._loaded_head(store)
            ledger = self._retirement_ledger(store) if head else None
            pages = transcript.iter_active_pages(ledger, head, page_size=16)
            try:
                chunk = self._sleep_chunk(chain.from_iterable(pages), request_size=len(transcript))
            finally:
                pages.close()  # Release the archive read transaction before training.
        else:
            active = self.active_history(transcript)
            chunk, _ = oldest_completed_chunk(active, self.evict_ratio, keep_recent=False)
        if not chunk:
            return {"status": "idle", "memories_consolidated": 0}
        if not self.consolidate_chunk_sync(chunk, cancel_event=cancel_event, mode="sleep"):
            if self.last_result.get("retirement_reason") == self.last_error and self.last_error:
                return {**self.last_result, "status": "retained"}
            raise RuntimeError(self.last_error)
        return self.last_result.copy()

    def train_pairs(self, pairs, *, turns=(), evidence=(), mode="learn", cancel_event=None,
                    learning_rate=1e-4, steps=3, training_options=None, stream_cursor=None,
                    validation_rows=None, policy_parent=..., recall_plan=None):
        """Serialize real updates. A committed manifest carries the exact retirement receipt."""
        with self._update_lock:
            with self.lock:
                self.is_consolidating = True
            backend = self.engine
            result = None
            try:
                if not self._real_training_ready():
                    raise RuntimeError(self.last_error or "The active backend lacks a verified persistent trainer or probe scorer")
                result = self._train(backend, pairs, turns, evidence, mode, cancel_event, learning_rate,
                                     steps, dict(training_options or {}), stream_cursor, validation_rows, policy_parent, recall_plan)
                self.consolidation_count += 1
                self.total_param_shift += result["parameter_drift_l2"]
                self.last_result = result
                self.last_error = ""
                return result
            finally:
                try:
                    cleanup = release_training_memory(backend)
                    if result is not None:
                        result.update(ram_after_cleanup_mb=cleanup["after_mb"], ram_released_mb=cleanup["released_mb"])
                except Exception as error:
                    logger.warning("Training allocator cleanup failed: %s", error)
                finally:
                    with self.lock:
                        self.is_consolidating = False

    def _train(self, backend, pairs, turns, evidence, mode, cancel_event, learning_rate,
               steps, options, stream_cursor, validation_rows, policy_parent, recall_plan):
        def check_cancel():
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Learning cancelled before checkpoint commit")

        check_cancel()
        if not pairs:
            raise ValueError("No complete dialogue or learning pairs")
        store = backend.learning_state
        manifest = store.current_manifest(verify=False)
        if policy_parent is not ... and (manifest or {}).get("generation") != policy_parent:
            raise RuntimeError("The active policy changed while sampling the rollout group")
        memory = self._memory(manifest)
        parent_head = memory.get("retirement_head") if memory.get("recall_version") == 1 else None
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
        objective = options.get("objective", "causal_lm")
        if objective == "reward_policy":
            # Only a *relative* gain on the rewarded actions is evidence of
            # learning this objective. Lower NLL on every rollout is insufficient.
            probes = pairs
            from core.reward_adaptation import reward_advantages, validation_contrast_weights, validation_gain_tolerance
            weighting = {}
            _, advantages = reward_advantages(
                pairs, reward_weighting=options.get("reward_weighting", "group_relative"),
                entropic_kl_target=options.get("entropic_kl_target", math.log(2)), diagnostics=weighting)
            contrast_weights = validation_contrast_weights(advantages)
        else:
            probes = list(validation_rows) if validation_rows is not None else _probes(pairs, evidence)
        if not probes:
            raise ValueError("Learning requires a measurable validation witness")
        anchors = get_anchor_dataset()
        # Use different foundation items for rehearsal and for acceptance.
        replay_bank = ReplayBank(store.directory)
        replay_head = memory.get("replay_head")
        historical = replay_bank.sample(replay_head, (manifest or {}).get("generation"))
        rehearsal = _sample(old_replay, 2)
        # Reserve at least one of the historical observations for a held-out
        # guard whenever two distinct facts exist. Never optimize a guard row.
        older = [r for r in historical if r not in rehearsal]
        if len(older) > 1:
            rehearsal.append(older[0])
        guard = [anchors[2], *_sample([r for r in historical + old_replay if r not in rehearsal], 3)]
        bank = RecallBank(store.directory)
        old_checks = bank.sample(parent_head, (manifest or {}).get("generation"))
        baseline, probe_sources = _scores(backend, probes)
        baseline_guard, _ = _scores(backend, guard)
        metrics = {}
        new_memory = {**memory, "retirement_head": parent_head, "recall_version": 1,
                      "replay": _rehearsal(old_replay, [] if objective == "reward_policy" else probes)}
        if stream_cursor:
            key, cursor = stream_cursor
            cursor_bank = StreamCursorBank(store.directory)
            stream_head = memory.get("stream_head")
            previous = cursor_bank.lookup(stream_head, key) or memory.get("streams", {}).get(key, {})
            if previous.get("next", 0) >= cursor["next"]:
                raise RuntimeError("This source chunk was already committed; resume from the current cursor")

        def validate(candidate):
            check_cancel()
            after, candidate_sources = _scores(candidate, probes)
            if candidate_sources != probe_sources:
                raise LearningValidationError("The candidate changed the validation scoring route")
            after_guard, _ = _scores(candidate, guard)
            metrics.update(probe_nll_before=baseline, probe_nll_after=after,
                           retention_nll_before=baseline_guard, retention_nll_after=after_guard)
            if objective == "reward_policy":
                deltas = [before - now for before, now in zip(baseline, after)]
                mean_delta = math.fsum(deltas) / len(deltas)
                # The leave-one-out entropic advantages need not sum to zero.
                # A shared improvement on every action cannot prove a shift
                # toward the verifier's preferred actions.
                gain = math.fsum(a * (d - mean_delta) for a, d in zip(contrast_weights, deltas)) / len(deltas)
                metrics["reward_logprob_contrast_gain"] = gain
                metrics["reward_logprob_weighted_gain"] = math.fsum(a * d for a, d in zip(advantages, deltas)) / len(deltas)
                metrics["reward_validation_sources"] = probe_sources
                metrics["reward_validation_measure"] = "centered change in mean serving log probability per token"
                metrics.update(weighting)
                metrics["reward_validation_tolerance"] = validation_gain_tolerance(baseline, after)
                if gain <= metrics["reward_validation_tolerance"]:
                    raise LearningValidationError("The candidate failed to improve relative likelihood of rewarded actions")
            elif sum(after) >= sum(baseline):
                raise LearningValidationError("The candidate failed to improve the measured source continuations")
            if objective != "reward_policy" and any(now > old + RETENTION_TOLERANCE for now, old in zip(after, baseline)):
                raise LearningValidationError("The candidate regressed on a source continuation")
            if any(now > old + RETENTION_TOLERANCE for now, old in zip(after_guard, baseline_guard)):
                raise LearningValidationError("The candidate regressed on a retention probe")
            if stream_cursor:
                head = stream_head
                for legacy_key, legacy_cursor in new_memory.pop("streams", {}).items():
                    head = cursor_bank.record(head, legacy_key, legacy_cursor)
                new_memory["stream_head"] = cursor_bank.record(head, key, cursor)
            if objective == "causal_lm" and mode != "rsi":
                replay_bank.stage(update_id, replay_head, _sample(probes, 4))
                new_memory["replay_head"] = update_id
            if old_checks and not passes_recall(candidate, old_checks, cancel_event):
                raise LearningValidationError("The candidate regressed on previously retired source-free recall")
            metrics.update(retention_recall_checks=len(old_checks), retirement_checks=0,
                           turns_retired=0, retirement_reason="")
            if turns:
                try:
                    metrics["retirement_checks"] = len(new_checks)
                    passed = bool(new_checks) and passes_recall(candidate, new_checks, cancel_event)
                except (OSError, sqlite3.Error, MemoryError):
                    raise
                except Exception as exc:
                    logger.info("Could not verify full exchange recall: %s", exc)
                    passed = False
                if passed:
                    ledger.stage(update_id, parent_head, [turn.turn_id for turn in turns])
                    new_memory["retirement_head"] = update_id
                    metrics["turns_retired"] = len(turns)
                else:
                    bank.discard(update_id)
                    metrics["retirement_reason"] = "Parameters updated; complete source-free recall was unproven."
            check_cancel()
            return {"accepted": True, **metrics}

        metadata = {"update_id": update_id, "mode": mode, "memory": new_memory,
                    "expected_parent": (manifest or {}).get("generation")}
        if objective == "reward_policy":
            from core.reward_adaptation import rollout_audit
            metadata["_sampled_rollout_group"] = rollout_audit(pairs, options)
        # RSI trains the self-generated trace only. Retention scores are evaluated
        # independently after the trace is written and never fed into its generation.
        data = list(pairs)
        strength = self.lambda_ewc
        fisher_anchors = get_anchor_texts()[:2]
        if mode == "verified_sleep":
            from consolidation.daemon import SleepConsolidationDaemon

            # Preserve historical surprise ordering and anchor interleaving. Leave
            # the common validation anchor out of both training and the Fisher fit.
            training_anchors = anchors[:2] + anchors[3:]
            texts = get_anchor_texts()
            fisher_anchors = texts[:2] + texts[3:]
            data = SleepConsolidationDaemon.interleave_replay_batches(
                data, training_anchors, options.pop("episodic_ratio"))
            strength = options.pop("lambda_ewc")
        elif mode != "rsi" and objective == "causal_lm":
            data += [anchors[0], *rehearsal]
        start = time.monotonic()
        ram = process_rss_mb()
        try:
            check_cancel()
            new_checks = []
            if turns:
                try:
                    checks = recall_plan.checks() if recall_plan else retirement_probes(backend, turns, cancel_event)
                    new_checks = bank.stage(update_id, parent_head, checks)
                except (OSError, sqlite3.Error, MemoryError):
                    raise
                except Exception as exc:
                    logger.info("Could not construct full exchange recall probes: %s", exc)
            updated, drift = backend.train_mini_batch(
                adapters=getattr(backend, "adapters", {}) or {}, data=data,
                lambda_ewc=strength, anchor_texts=fisher_anchors,
                learning_rate=learning_rate, steps=steps, cancel_event=cancel_event,
                validation_callback=validate, learning_metadata=metadata, **options,
                **learning_mode_kwargs(backend.train_mini_batch, "managed"),
            )
            # Backends check finite drift and verify persisted native tensors before
            # publishing. Verify that this exact transaction became durable.
            committed = store.current_manifest()
            drift = float(drift)
            if (not metrics or not math.isfinite(drift) or drift <= 0 or not committed
                    or committed["update_id"] != update_id or self._memory(committed) != new_memory):
                raise RuntimeError("The trainer returned without a validated, durable parameter update")
        except BaseException:
            current = store.current_manifest(verify=False)
            if (current or {}).get("generation") == (manifest or {}).get("generation"):
                replay_bank.discard(update_id)
                if ledger:
                    ledger.discard(update_id)
                    bank.discard(update_id)
            raise

        backend_metrics = (committed.get("metadata", {}).get("metrics")
                           or getattr(backend, "last_learning_metrics", {}) or {})
        touched = int(backend_metrics.get("trainable_parameters_touched", 0))
        if not touched and isinstance(updated, dict):
            touched = int(updated.get("trainable_parameters_touched", 0))
        checkpoint = Path(store.directory) / "generations" / committed["generation"] / committed["checkpoint"]
        return {"status": "success", "mode": mode, "generation": committed["generation"],
                "objective": objective, "identity_digest": store.identity_digest, "scope": store.scope.record, "memories_consolidated": len(pairs),
                "parameter_drift_l2": drift, "trainable_parameters_touched": touched,
                "trainable_parameters_m": touched / 1_000_000, "ram_before_mb": ram,
                "ewc_active": bool(backend_metrics.get("ewc_used") or backend_metrics.get("fisher_tensors", 0) > 0),
                "replay_examples": len(data) - len(pairs), "backend": backend_label(backend),
                "training_metrics": {k: backend_metrics[k] for k in (
                    "training_windows", "trained_windows", "supervised_tokens", "optimizer_resumed",
                    "updated_projection", "kv_prefills", "cached_context_tokens", "group_size",
                    "adapter_target") if k in backend_metrics},
                "adapter_saved_to": str(checkpoint), "execution_time_seconds": time.monotonic() - start, **metrics}


# The original public constructor retains the original learning behavior.
AwakeOnlineConsolidator = LegacyAwakeOnlineConsolidator


def make_online_consolidator(mlx_engine, memory_db=None, max_context=8192, watermark=0.80,
                             evict_ratio=0.40, lambda_ewc=400.0, *, session_id=None, implementation=None):
    """Construct the selected chat consolidation service for this model."""
    args = mlx_engine, memory_db, max_context, watermark, evict_ratio, lambda_ewc
    if get_learning_implementation(implementation) == "managed":
        return ManagedAwakeOnlineConsolidator(*args, session_id=session_id)
    return LegacyAwakeOnlineConsolidator(*args)
