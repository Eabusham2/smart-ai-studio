"""Activate awake learning while preserving the historical Pro chat path.

The rich GUI streams through ProReasoningEngine.stream_solve(). The existing entropy
router remains authoritative: easy N=1 prompts keep real-time streaming at the tuned
T=0.65 single-pass temperature, while N>1 prompts delegate to the historical solve()
Pro path with the calibrated convex ladder.

Chat has one total Context budget (prompt/history + generated tokens), not a separate
output-token allowance. At the Gemini-designed 80% context watermark, old completed
dialogue pairs are synchronously consolidated into the real trainable adapter and are
removed only after that real parameter update succeeds. Generation then owns every
token remaining in Context until natural EOS.
"""
from __future__ import annotations

import re
import sqlite3
from typing import Any, Dict, List, Optional

from core.context_memory import (ActiveChatHistory, ChatArchive, ContextOverflow, completed_groups,
                                 oldest_completed_chunk, working_history)
from core.platform import get_auto_context_window_size
from core.temperature_policy import CHAT_N1_TEMPERATURE


def _active_trainable_backend(engine):
    """Resolve the loaded backend by capability, not by one hard-coded format."""
    active = str(getattr(engine, "active_backend", "") or "").lower()
    named = {
        "controller": getattr(engine, "controller_backend", None),
        "gguf": getattr(engine, "gguf_backend", None),
        "bitnet": getattr(engine, "bitnet_backend", None),
        "mlx": getattr(engine, "mlx_backend", None),
    }
    backend = named.get(active)
    if backend is not None and getattr(backend, "model", None) is not None:
        return backend

    consolidator = getattr(engine, "awake_consolidator", None)
    backend = getattr(consolidator, "engine", None) if consolidator is not None else None
    if backend is not None and getattr(backend, "model", None) is not None:
        return backend

    for backend in (
        getattr(engine, "controller_backend", None),
        getattr(engine, "gguf_backend", None),
        getattr(engine, "bitnet_backend", None),
        getattr(engine, "mlx_backend", None),
    ):
        if backend is not None and getattr(backend, "model", None) is not None:
            return backend
    # Direct PyTorch has no separate backend object. Its actual tokenizer and
    # model config must still define the shared prompt+output context boundary.
    if getattr(engine, "model", None) is not None and getattr(engine, "tokenizer", None) is not None:
        return engine
    return None


def _model_context_limit(engine) -> Optional[int]:
    backend = _active_trainable_backend(engine)
    model = getattr(backend, "model", None)
    tokenizer = getattr(backend, "tokenizer", None)
    values = []

    # Native backends may expose context either on the backend or tokenizer/model
    # facade. Treat it uniformly so the single GUI Context control is backend-neutral.
    if backend is not None:
        for candidate in (
            getattr(backend, "n_ctx", None),
            getattr(model, "n_ctx", None),
        ):
            try:
                value = int(candidate() if callable(candidate) else candidate)
            except Exception:
                continue
            if 1 <= value <= 10_000_000:
                values.append(value)

    config = getattr(model, "config", None)
    for obj in (getattr(model, "args", None), getattr(config, "text_config", None), config, tokenizer):
        if obj is None:
            continue
        for name in (
            "max_position_embeddings",
            "n_positions",
            "max_seq_len",
            "max_sequence_length",
            "context_length",
            "model_max_length",
        ):
            try:
                value = int(getattr(obj, name))
            except Exception:
                continue
            if 1 <= value <= 10_000_000:
                values.append(value)
    return min(values) if values else None


def _selected_context_budget(engine) -> int:
    """Resolve the session's one prompt+output Context budget."""
    requested = int(getattr(engine, "_context_budget_tokens", 0) or 0)
    if requested <= 0:
        requested = int(get_auto_context_window_size())
        engine._context_budget_tokens = requested

    physical = _model_context_limit(engine)
    effective = min(requested, int(physical)) if physical else requested
    effective = max(1, int(effective))
    engine._effective_context_budget_tokens = effective

    backend = getattr(engine, "mlx_backend", None)
    if backend is not None:
        # Preserve the user's requested value; MLX's context-budget layer clamps it
        # against the physical model window when calculating generation room.
        backend.context_budget_tokens = max(1024, int(requested))
    return effective


def _formatted_token_count(engine, formatted_prompt: str, *, budget: Optional[int] = None) -> int:
    backend = _active_trainable_backend(engine)
    tokenizer = getattr(backend, "tokenizer", None) if backend is not None else None

    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            return max(0, len(encode(formatted_prompt)))
        except MemoryError:
            raise
        except Exception:
            pass

    tokenize = getattr(tokenizer, "tokenize", None)
    if callable(tokenize):
        try:
            return max(0, len(tokenize(formatted_prompt.encode("utf-8"))))
        except MemoryError:
            raise
        except Exception:
            pass

    counter = getattr(backend, "count_tokens", None) if backend is not None else None
    if callable(counter):
        try:
            return max(
                0,
                int(counter([{"role": "user", "content": formatted_prompt}])),
            )
        except MemoryError:
            raise
        except Exception:
            pass

    # Conservative final estimate only when a real tokenizer facade is unavailable.
    estimate = max(1, len(str(formatted_prompt)) // 4)
    if budget is not None and estimate >= budget:
        # Preserve the refusal when only an estimate is available, but never
        # advertise it as a tokenizer-confirmed overflow to a recovery caller.
        raise RuntimeError(f"Packed prompt is estimated at {estimate:,} tokens with active Context "
                           f"{budget:,}; the exact token count is unavailable. Refusing truncation.")
    return estimate


def _remaining_generation_tokens(engine, formatted_prompt: str) -> int:
    """One Context budget = packed prompt/history + generated output for every backend."""
    backend = _active_trainable_backend(engine)
    if getattr(backend, "runtime", None) == "strata":
        # The HTTP adapter's count_tokens is a conservative UTF-8 estimate, not
        # the model tokenizer. Subtracting it can reject valid long conversations
        # or artificially shorten replies. Strata applies its real chat template,
        # counts native tokens, rejects overlong prompts and clamps max_tokens to
        # the exact remaining window. Send that window as the output upper bound.
        return _strata_generation_context(engine, backend)
    if getattr(backend, "runtime", None) == "prism_speculative":
        # Count inside the new native request after its cancellation state is
        # reset. The adapter subtracts exact prompt tokens from this total cap.
        context_budget = _selected_context_budget(engine)
        backend.context_budget_tokens = context_budget
        return context_budget
    context_budget = _selected_context_budget(engine)
    prompt_tokens = _formatted_token_count(engine, formatted_prompt, budget=context_budget)
    remaining = int(context_budget) - int(prompt_tokens)
    if remaining <= 0:
        raise ContextOverflow(
            f"Packed prompt requires {prompt_tokens:,} tokens but active Context is "
            f"{context_budget:,}; refusing truncation."
        )
    return int(remaining)


def _strata_generation_context(engine, backend) -> int:
    """Use the loaded server's context; changes require a deliberate reload."""
    context = backend.n_ctx()
    if type(context) is not int or not 1024 <= context <= 10_000_000:
        raise RuntimeError("Strata has no valid loaded context window.")
    requested = int(getattr(engine, "_context_budget_tokens", 0) or 0)
    if requested <= 0:
        # Direct API callers need the same loaded default throughout Pro's
        # second pass; the native RAM heuristic must not replace it mid-query.
        requested = context
        engine._context_budget_tokens = context
    if requested != context:
        raise RuntimeError(
            f"Strata is loaded with Context {context:,}, but {requested:,} is selected. "
            "Unload the model, apply the desired Context, then load it again."
        )
    engine._effective_context_budget_tokens = context
    return context


def _oldest_completed_chunk(history: List[Dict[str, str]], ratio: float) -> tuple[list, list]:
    """Return complete old exchanges, retaining system/tool turns and recent dialogue."""
    return oldest_completed_chunk(working_history(history), ratio)


def _explicit_document(prompt):
    """Exactly one top-level source block, with the user's instructions outside it."""
    tags = [tag for tag in ("document", "context") if f"<{tag}>" in prompt or f"</{tag}>" in prompt]
    if len(tags) == 1:
        tag = tags[0]
        opening, closing = f"<{tag}>", f"</{tag}>"
        if prompt.count(opening) == prompt.count(closing) == 1:
            before, _, tail = prompt.partition(opening)
            source, end, after = tail.partition(closing)
            # A tag shown inside a Markdown code sample is part of the user's
            # literal instruction, even when the sample exceeds the window.
            fence = None
            for match in re.finditer(r"(?m)^ {0,3}(`{3,}|~{3,})", before):
                mark = match[1]
                if fence is None:
                    fence = mark
                elif mark.startswith(fence):
                    fence = None
            inline = before.rsplit("\n", 1)[-1].count("`") % 2
            if end and source.strip() and (before.strip() or after.strip()) and not fence and not inline:
                return before, opening, source, closing, after
    return None


def prepare_awake_request(engine, prompt, history, cancel_event=None):
    """Prepare the bounded text before a caller copies or routes the visible archive."""
    if isinstance(history, ActiveChatHistory):
        history = history.archive
    service = getattr(engine, "awake_consolidator", None)
    backend = _active_trainable_backend(engine)
    if (not prompt or service is None or getattr(service, "engine", None) is not backend
            or backend is None or getattr(backend, "runtime", None) == "strata"):
        return prompt, history
    if getattr(backend, "runtime", None) == "prism_speculative" and not service._real_training_ready():
        return prompt, history
    budget = _selected_context_budget(engine)
    backend._learning_context_budget_tokens = budget
    def count(text, rows):
        return _formatted_token_count(engine, engine._format_prompt_with_history(text, rows), budget=budget)

    archive = isinstance(history, ChatArchive)
    if archive and not callable(getattr(service, "prepare_archive", None)):
        return prompt, history
    possible_source = ("<document>" in prompt or "<context>" in prompt)
    oversized = possible_source and count(prompt, []) >= budget
    if archive and not oversized:
        try:
            return prompt, service.prepare_archive(history, lambda rows: count(prompt, rows), budget, cancel_event)
        except ContextOverflow:
            if not possible_source:
                raise
            oversized = True
    elif possible_source and not oversized and history:
        # Check protected messages before adapting the live prompt. The normal
        # awake path still handles evictable historical exchanges first.
        identified = working_history(history, getattr(service, "session_id", ""))
        eligible = {i for group in completed_groups(identified) for i in group}
        protected = [turn for i, turn in enumerate(identified) if i not in eligible]
        oversized = count(prompt, protected) >= budget

    original = prompt
    duplicate = None
    if oversized and callable(getattr(service, "learn_current_document", None)) and service._real_training_ready():
        document = _explicit_document(prompt)
        if document:
            before, opening, source, closing, after = document
            def instruction(key):
                return before + opening + f"Learned document {key}" + closing + after
            if count(instruction("0" * 64), []) < budget:
                key = service.learn_current_document(source, cancel_event)
                prompt = instruction(key)
                if history:
                    last = history[-1]
                    if last.get("role") == "user" and last.get("content") == original and set(last) == {"role", "content"}:
                        duplicate = getattr(last, "turn_id", None)
                        if not archive:
                            history = history[:-1]
    if archive:
        def packed(rows):
            messages = [t for t in rows if t.turn_id != duplicate] if duplicate else rows
            return count(prompt, messages)
        history = service.prepare_archive(history, packed, budget, cancel_event)
        if duplicate:
            history = [t for t in history if t.turn_id != duplicate]
    return prompt, history


def install_awake_auto_learning(cls) -> None:
    if getattr(cls, "_awake_auto_learning_installed", False):
        return

    original_stream_solve = cls.stream_solve
    original_solve = cls.solve

    def _apply_awake_learning(self, history, prompt: Optional[str] = None, cancel_event=None):
        if not prompt:
            return history

        backend = _active_trainable_backend(self)
        if getattr(backend, "runtime", None) == "strata":
            context_budget = _strata_generation_context(self, backend)
            self._last_context_consolidation_note = (
                f"Strata inference only · Context {context_budget:,} includes prompt/history + output. "
                "The server counts exact tokens; history is kept without weight consolidation."
            )
            return history

        native_trainable = False
        is_native = getattr(backend, "runtime", None) == "prism_speculative"
        if is_native:
            capability = getattr(backend, "training_ready", None)
            try:
                native_trainable = callable(capability) and bool(capability())
            except Exception:
                native_trainable = False
        if is_native and not native_trainable:
            context_budget = _selected_context_budget(self)
            self._last_context_consolidation_note = (
                f"Native inference only · Context {context_budget:,} includes prompt/history + output. "
                "Complete history is retained; target weights and adapters are immutable."
            )
            return history

        context_budget = _selected_context_budget(self)
        consolidator = getattr(self, "awake_consolidator", None)
        tokenizer = getattr(backend, "tokenizer", None)
        if consolidator is None or backend is None or tokenizer is None:
            return history

        # Resolve the authoritative checkpoint before measuring even a short request.
        # The visible transcript can arrive as a fresh copy on each GUI/tool round.
        original_history = history
        active_history = getattr(consolidator, "active_history", None)
        if callable(active_history):
            if getattr(consolidator, "engine", backend) is not backend:
                self._last_context_consolidation_note = "The active model changed; keeping the transcript."
                return original_history
            try:
                history = active_history(history)
            except (ValueError, RuntimeError, OSError, sqlite3.Error) as error:
                self._last_context_consolidation_note = str(error)
                return original_history

        # Gemini's rolling-memory design: start learning old turns before the hard
        # boundary, at 80% of total Context. After triggering, reduce the active
        # textual working set toward 60% so the answer has substantial free room.
        trigger_tokens = max(1, int(context_budget * 0.80))
        target_tokens = max(1, int(context_budget * 0.60))
        consolidator.max_context = int(context_budget)
        consolidator.watermark_tokens = int(trigger_tokens)

        def packed_token_count(active_history) -> int:
            formatted = self._format_prompt_with_history(prompt, active_history)

            # Preserve the native MLX path exactly when its tokenizer exposes encode().
            encode = getattr(tokenizer, "encode", None)
            if callable(encode):
                return len(encode(formatted))

            # llama.cpp tokenizer facade uses tokenize(bytes), not encode(str).
            tokenize = getattr(tokenizer, "tokenize", None)
            if callable(tokenize):
                return len(tokenize(formatted.encode("utf-8")))

            # Final backend-neutral fallback uses the backend's own token counter.
            count_tokens = getattr(backend, "count_tokens", None)
            if callable(count_tokens):
                return int(count_tokens([{"role": "user", "content": formatted}]))

            raise RuntimeError("Active trainable backend exposes no token counter")

        try:
            current_tokens = packed_token_count(history or [])
        except Exception:
            return history

        if (isinstance(original_history, list) and not isinstance(original_history, ChatArchive)
                and len(original_history) != len(history)):
            original_history[:] = history
        # Copies used by the GUI and Best Work share a single-use marker.
        # A later request reusing either working list gets normal watermarks.
        recovered = False
        for turn in history or ():
            marker = getattr(turn, "_capacity_recovery", None)
            if marker is not None:
                recovered = marker.pop("prepared", False) or recovered
                del turn._capacity_recovery
        if current_tokens < trigger_tokens or (recovered and current_tokens < context_budget):
            self._last_context_consolidation_note = (
                f"Context {context_budget:,}: prompt/history {current_tokens:,}; "
                f"generation may use the remaining {max(0, context_budget-current_tokens):,} tokens until EOS."
            )
            return history

        if not history:
            self._last_context_consolidation_note = (
                f"Context {context_budget:,}: current prompt alone uses {current_tokens:,}; "
                "there is no older completed dialogue to consolidate."
            )
            if current_tokens >= context_budget:
                raise ContextOverflow(f"Packed prompt requires {current_tokens} tokens with an active Context of "
                                      f"{context_budget}; refusing truncation.")
            return history

        retained = list(history)
        consolidated_turns = 0
        original_tokens = current_tokens
        while current_tokens > target_tokens:
            chunk, candidate_retained = _oldest_completed_chunk(
                retained,
                getattr(consolidator, "evict_ratio", 0.40),
            )
            if not chunk and current_tokens >= context_budget:
                chunk, candidate_retained = oldest_completed_chunk(
                    working_history(retained), getattr(consolidator, "evict_ratio", 0.40), keep_recent=False)
            if not chunk:
                break
            strict_attempt = getattr(consolidator, "consolidate_for_capacity", None) if current_tokens >= context_budget else None
            try:
                if callable(strict_attempt):
                    learned = strict_attempt(chunk, cancel_event=cancel_event)
                elif cancel_event is None:
                    learned = bool(consolidator.consolidate_chunk_sync(chunk))
                else:
                    learned = bool(consolidator.consolidate_chunk_sync(chunk, cancel_event=cancel_event))
            except Exception:
                if cancel_event is not None and cancel_event.is_set():
                    return retained
                if callable(strict_attempt):
                    raise
                learned = False
            if not learned:
                # Never remove dialogue that was not successfully consolidated.
                # Recovery may learn later exchanges while keeping this source.
                recover = getattr(consolidator, "recover_after_rejected_chunk", None)
                if current_tokens >= context_budget and callable(recover):
                    def measure():
                        projected = consolidator.active_history(retained)
                        return projected, packed_token_count(projected)
                    try:
                        retained, current_tokens, additional = recover(
                            retained, chunk, retained, current_tokens, context_budget, measure, cancel_event)
                    except InterruptedError:
                        if cancel_event is not None and cancel_event.is_set():
                            return retained
                        raise
                    consolidated_turns += additional
                break
            consolidated_turns += len(chunk)
            retained = candidate_retained
            try:
                current_tokens = packed_token_count(retained)
            except Exception:
                break

        if cancel_event is not None and cancel_event.is_set():
            return retained
        if current_tokens >= context_budget:
            detail = str(getattr(consolidator, "last_error", "") or "No eligible completed exchange remains.")
            raise ContextOverflow(f"Packed prompt requires {current_tokens} tokens with an active Context of "
                                  f"{context_budget}; refusing truncation. Original transcript retained. {detail}")
        if consolidated_turns:
            # Direct chat() callers receive the reduced working list. GUI archives
            # retain all messages for display and export, including after restart.
            if isinstance(original_history, list) and not isinstance(original_history, ChatArchive):
                original_history[:] = retained
            history = retained
            self._last_context_consolidation_note = (
                f"Context {context_budget:,}: consolidated {consolidated_turns} old dialogue turns into "
                f"trainable weights ({original_tokens:,} → {current_tokens:,} active prompt tokens); "
                f"generation may use all {max(0, context_budget-current_tokens):,} remaining tokens until EOS."
            )
            return history

        detail = getattr(consolidator, "last_error", "")
        self._last_context_consolidation_note = (
            f"Context {context_budget:,}: active prompt uses {original_tokens:,} tokens; no completed old dialogue "
            "pair could be safely consolidated, so the physical Context boundary remains authoritative."
            + (f" {detail}" if detail else "")
        )
        return history

    def stream_solve_with_awake_learning(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.75,
        top_p: float = 0.92,
        cancel_event: Optional[Any] = None,
        progress_callback=None,
        decision_callback=None,
    ):
        from core.lossless_speed_control import cancel_before_request
        if cancel_before_request(self, cancel_event):
            self._last_stream_pro_meta = None
            return
        try:
            prompt, history = prepare_awake_request(self, prompt, history, cancel_event)
        except InterruptedError:
            if cancel_event is None or not cancel_event.is_set():
                raise
            self._last_stream_pro_meta = None
            return
        history = _apply_awake_learning(self, history, prompt, cancel_event)
        self._last_stream_pro_meta = None
        if cancel_event is not None and cancel_event.is_set():
            return

        # An entropy probe can itself run a model forward. Check the complete
        # active request first, even on runtimes with no awake trainer.
        formatted = self._format_prompt_with_history(prompt, history)
        remaining = _remaining_generation_tokens(self, formatted)

        # Preserve the historical chat decision point: entropy chooses N=1/8/16.
        native = getattr(self, "gguf_backend", None)
        if bool(getattr(self, "_gui_force_single_stream", False)):
            entropy = None
            branch_count = 1
            mode = "Instant GUI stream (Pro unchecked)"
        elif getattr(native, "runtime", None) == "prism_speculative":
            # Native completion does not expose full-vocabulary entropy. Use
            # the explicit GUI/API branch budget for both Off and draft modes;
            # never substitute a prompt heuristic for the model router.
            branch_count = getattr(self, "lossless_speed_options", {}).get("native_branch_count", 1)
            if type(branch_count) is not int or not 1 <= branch_count <= 64:
                raise ValueError("Select a native branch count from 1 through 64 in Speedups.")
            entropy = None
            mode = f"Native (N={branch_count})"
        else:
            routing_prompt = formatted if getattr(self, "_best_work_route_with_history", False) else prompt
            entropy = float(self.calculate_token_entropy(routing_prompt))
            mode, branch_count = self.router.route(entropy, has_test_cases=False)

        if decision_callback is not None:
            decision_callback({"kind": "route", "branch_count": int(branch_count),
                               "total": int(branch_count) + (1 if int(branch_count) > 1 else 0)})

        if int(branch_count) <= 1:
            previous_max = getattr(self.settings, "max_new_tokens", None)
            self.settings.max_new_tokens = int(remaining)
            try:
                yield from original_stream_solve(
                    self,
                    prompt,
                    history=history,
                    temperature=CHAT_N1_TEMPERATURE,
                    top_p=top_p,
                    cancel_event=cancel_event,
                )
            finally:
                self.settings.max_new_tokens = previous_max
            return

        # N>1 delegates to the original Pro engine: same entropy router, branch
        # generator, calibrated convex temperature ladder, verifier/consensus and metadata.
        self._awake_stream_history_prepared = True
        previous_routing = getattr(self, "_best_work_routing_decision", None)
        if getattr(self, "_best_work_raw_winner", False):
            self._best_work_routing_decision = (entropy, mode, int(branch_count))
        try:
            response, metadata = self.solve(
                prompt,
                history=history,
                cancel_event=cancel_event,
                force_branch_count=int(branch_count),
                temperature=None,
                **({"progress_callback": progress_callback} if progress_callback is not None else {}),
            )
        finally:
            self._awake_stream_history_prepared = False
            self._best_work_routing_decision = previous_routing

        metadata = dict(metadata or {})
        metadata["entropy"] = entropy
        metadata["mode"] = mode
        note = str(getattr(self, "_last_context_consolidation_note", "") or "")
        if note:
            metadata["context_consolidation"] = note
        self._last_stream_pro_meta = metadata

        # Pro must finish its search before one final answer exists. Preserve the
        # newer GUI renderer by yielding that completed answer in small chunks.
        text = str(response or "")
        if getattr(self, "_best_work_raw_winner", False):
            from core.best_work import _selected_text
            text = _selected_text(response, metadata)
        if decision_callback is not None:
            decision_callback({"kind": "selected", "total": metadata.get("branch_count", branch_count),
                               "winning_branch": metadata.get("winning_branch", 0)})
        step = 64
        for idx in range(0, len(text), step):
            if cancel_event and cancel_event.is_set():
                break
            yield text[idx : idx + step]

    def solve_with_awake_learning(
        self,
        prompt: str,
        test_cases: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        cancel_event: Optional[Any] = None,
        force_branch_count: Optional[int] = None,
        temperature: Optional[float] = None,
        progress_callback=None,
    ):
        if not getattr(self, "_awake_stream_history_prepared", False):
            try:
                prompt, history = prepare_awake_request(self, prompt, history, cancel_event)
            except InterruptedError:
                if cancel_event is None or not cancel_event.is_set():
                    raise
                return "", {"cancelled": True, "verified": False, "branch_count": 0, "raw_branches": []}
            history = _apply_awake_learning(self, history, prompt, cancel_event)
        else:
            _selected_context_budget(self)

        if cancel_event is not None and cancel_event.is_set():
            return "", {"cancelled": True, "verified": False, "branch_count": 0, "raw_branches": []}
        formatted = self._format_prompt_with_history(prompt, history)
        remaining = _remaining_generation_tokens(self, formatted)
        previous_max = getattr(self.settings, "max_new_tokens", None)
        self.settings.max_new_tokens = int(remaining)
        try:
            return original_solve(
                self,
                prompt,
                test_cases=test_cases,
                history=history,
                cancel_event=cancel_event,
                force_branch_count=force_branch_count,
                temperature=temperature,
                **({"progress_callback": progress_callback} if progress_callback is not None else {}),
            )
        finally:
            self.settings.max_new_tokens = previous_max

    cls.stream_solve = stream_solve_with_awake_learning
    cls.solve = solve_with_awake_learning
    cls._awake_auto_learning_installed = True
