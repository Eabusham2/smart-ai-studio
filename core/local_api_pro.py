"""API-only bridge to the existing Best Work model tournament.

No desktop solve(), system prompts, tool registry, memory projection or learning
is entered here. Candidate generation uses the API's loaded decoder, template,
context ceiling, sampling and request-local speedup controls.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import math


def _check(cancel):
    if cancel.is_set():
        raise InterruptedError("API Pro cancelled")


def run_pro(adapter, backend, request, prompt, prompt_count, maximum, context,
            options, config, explicit_temperature, cancel):
    from core.best_work_selection import select_best_work_candidates, SelectionError
    from core.context_memory import ContextOverflow
    from core.generation_metrics import RequestMetrics, observe_generations, valid_count
    from core.gui_thinking import ReasoningParser, _OPEN
    from core.inference_fork import inference_cycle, inference_fork
    from core.local_api import APIError, _OutputParser, _json
    from core.local_api_engine import FormattedPrompt
    from core.temperature_policy import pro_ladder

    if not valid_count(context) or not valid_count(prompt_count):
        raise APIError(400, "Pro requires a verified loaded context and tokenizer.", code="unsupported_option")
    passes = config["best_work_passes"]
    count = config["branch_count"]
    if type(passes) is not int or not 1 <= passes <= 999 or type(count) is not int or not 1 <= count <= 64:
        raise APIError(400, "Invalid Pro branch count or pass count.")
    engine = adapter.engine
    total = RequestMetrics()
    records = {}
    entropy = None
    mode = "fixed"
    prepare = getattr(backend, "prepare_request", None)

    def prepare_decoder():
        _check(cancel)
        if callable(prepare):
            prepare()

    def candidate(req, packed, cap, temp):
        prepare_decoder()
        packed._smartai_api_max_tokens = cap
        # Use the API parser rather than the desktop's JSON-only tool parser.
        # Normalize native XML tool syntax to the same inert JSON representation
        # accepted by the existing selector. Argument values are never rewritten.
        parser = _OutputParser(replace(req, include_reasoning=True), first_tool_only=True)
        initial = packed.rstrip().casefold().endswith(tuple(_OPEN))
        parser.reasoning = ReasoningParser(initial)
        parser.prefix = "<think>" if initial else ""
        pieces, thought, byte_count = [], False, 0

        def consume(events):
            nonlocal thought
            for kind, value in events:
                if kind == "thought":
                    if not thought:
                        pieces.append("<think>")
                        thought = True
                    pieces.append(value)
                else:
                    if thought:
                        pieces.append("</think>")
                        thought = False
                    if kind == "tool":
                        pieces.append("<tool_call>" + _json({"name": value["name"],
                                                              "arguments": value["arguments"]}) + "</tool_call>")
                    elif kind == "text":
                        pieces.append(value)

        measured = RequestMetrics()
        stream = adapter._decode_stream(backend, {**options, "prompt": packed,
                                                "max_tokens": cap, "temperature": temp}, cancel)
        with observe_generations(measured.update):
            try:
                for piece in stream:
                    _check(cancel)
                    if not isinstance(piece, str):
                        raise APIError(500, "Model emitted a non-text Pro chunk.", "server_error")
                    byte_count += len(piece.encode("utf-8"))
                    if byte_count > 16 * 1024 * 1024:
                        raise APIError(413, "Pro candidate exceeded the server output limit.")
                    consume(parser.feed(piece))
                    if parser.stopped:
                        break
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
        _check(cancel)
        consume(parser.feed("", final=True))
        if thought:
            pieces.append("</think>")
        text = "".join(pieces)
        if not text.strip():
            raise SelectionError("Pro generation returned an empty candidate.")
        if (req.tool_choice == "required" or isinstance(req.tool_choice, dict)) and not parser.calls:
            raise SelectionError("Pro candidate did not return the requested client tool.")
        usage = measured.snapshot()
        tokens = usage["tokens"]
        record = {"text": text, "tokens": tokens, "tool_call": bool(parser.calls),
                  "stop_sequence": parser.stop_sequence,
                  "finish_reason": "stop" if parser.stopped else adapter._finish_reason(backend, tokens, cap),
                  "temperature": temp}
        records.setdefault(text, record)
        return text

    def internal_request(instruction, history):
        comparing = instruction.startswith("BEST_WORK_COMPARISON\n")
        rows = [dict(turn) for turn in history or ()]
        rows.append({"role": "user", "content": instruction})
        return replace(request, messages=rows, tools=[] if comparing else request.tools,
                       tool_choice="none" if comparing else request.tool_choice,
                       stop=[] if comparing else request.stop)

    # The desktop selector takes a narrow engine-shaped interface. This facade
    # has no desktop solver, tools, learner or weight mutation methods. Its
    # token counter delegates to the actual API tokenizer, not an estimate.
    class Selector:
        def __init__(self):
            self.settings = SimpleNamespace(max_new_tokens=maximum)
            self._context_budget_tokens = context
            self.model = SimpleNamespace()
            self.tokenizer = SimpleNamespace()
            self._prepared = None

        def _format_prompt_with_history(self, instruction, history):
            prepare_decoder()
            req = internal_request(instruction, history)
            packed = adapter._format_prompt(backend, req)
            self._prepared = (instruction, req, packed)
            return packed

        def count_tokens(self, messages):
            value = adapter._prompt_tokens(backend, messages[0]["content"], request.speedups)
            if not valid_count(value):
                raise ContextOverflow("Pro cannot verify the internal native token count.")
            return value

        def generate_parallel_branches(self, instruction, *, branch_count, history, temperatures, cancel_event):
            _check(cancel)
            if branch_count != 1:
                raise SelectionError("An internal Pro pass must produce exactly one candidate.")
            if self._prepared is None or self._prepared[0] != instruction:
                self._format_prompt_with_history(instruction, history)
            _, req, packed = self._prepared
            self._prepared = None
            # The selector's desktop wrapper measures full context first. Recheck
            # the actual processor count and enforce the client's per-call cap.
            used = self.count_tokens([{"content": packed}])
            room = context - used
            if room <= 0:
                raise ContextOverflow("Pro internal prompt exceeds the requested context.")
            return [candidate(req, packed, min(maximum, room), temperatures[0])]

    # The API lease already owns the actual lifecycle lock. The guard below
    # reuses the existing training/cache isolation without reacquiring a
    # potentially non-reentrant lock in an embedded adapter.
    guard = SimpleNamespace(awake_consolidator=getattr(engine, "awake_consolidator", None),
                            controller_backend=backend, active_backend="controller")
    try:
        with inference_cycle(guard, cancel), inference_fork(engine), observe_generations(total.update):
            _check(cancel)
            if config["routing"] == "auto":
                if getattr(backend, "runtime", None) in {"prism_speculative", "tensorfold_mlx"}:
                    mode = "native_configured"
                else:
                    measure = getattr(backend, "calculate_token_entropy", None)
                    router = getattr(engine, "router", None)
                    if not callable(measure) or not callable(getattr(router, "route", None)):
                        raise APIError(400, "This runtime has no entropy router. Choose fixed Pro routing.", code="unsupported_option")
                    entropy = float(measure(prompt))
                    if not math.isfinite(entropy):
                        raise APIError(500, "The model returned invalid routing entropy.", "server_error")
                    mode, count = router.route(entropy, has_test_cases=False)
            if type(count) is not int or not 1 <= count <= 64:
                raise APIError(400, "Pro router returned an invalid branch count.")
            # Preserve the established extra fixed branch for N>1. A caller
            # who explicitly sets temperature controls each original rollout.
            temps = pro_ladder(count) if count > 1 else [options["temperature"]]
            if explicit_temperature is not None:
                temps = [explicit_temperature] * len(temps)
            originals = [candidate(request, prompt, maximum, temp) for temp in temps]
            _check(cancel)
            question = next((m["content"] for m in reversed(request.messages)
                             if m.get("role") == "user" and m.get("content")), "")
            if not question:
                question = "Respond to the supplied conversation and client tool results."
            if len(originals) == 1 and passes == 1:
                winners, index, report = originals, 0, {"passes_requested": 1, "passes_completed": 1}
            else:
                winners, index, report = select_best_work_candidates(
                    Selector(), prompt=question, history=request.messages,
                    candidates=originals, temperatures=temps, passes=passes,
                    cancel_event=cancel, generation_usage=total.snapshot())
            _check(cancel)
            answer = winners[index]
            if not isinstance(answer, str) or not answer.strip():
                raise SelectionError("Pro selection returned no answer.")
            selected = records.get(answer)
            # Best Work can retain the original thinking and adopt a revised
            # answer. Such a composite may not correspond to one measured
            # invocation. Check its representation fits the output allowance;
            # do not invent sampled-token telemetry for it.
            if selected is None:
                prepare_decoder()
                used = adapter._prompt_tokens(backend, FormattedPrompt(answer), request.speedups)
                if not valid_count(used) or used > maximum:
                    raise ContextOverflow("The selected Pro composite exceeds the requested output limit.")
                final_parser = _OutputParser(replace(request, include_reasoning=True), first_tool_only=True)
                final_parser.feed(answer, final=True)
                if (request.tool_choice == "required" or isinstance(request.tool_choice, dict)) and not final_parser.calls:
                    raise SelectionError("Selected Pro response did not contain the required tool.")
                selected = {"tokens": None, "tool_call": bool(final_parser.calls),
                            "stop_sequence": final_parser.stop_sequence,
                            "finish_reason": "length" if any(r["finish_reason"] == "length" for r in records.values()) else "stop",
                            "temperature": None}
            stats = total.snapshot()
            # Standard usage describes the selected response, not all hidden
            # work. Separate capability telemetry reports measured total work.
            public = {"status": "completed", "routing": config["routing"], "mode": mode,
                      "entropy": entropy, "routed_branches": count, "original_candidates": len(originals),
                      "winning_branch": index, "best_work_passes": passes,
                      "passes_completed": report.get("passes_completed"),
                      "invocations": stats["invocations"], "generated_tokens": stats["tokens"],
                      "selected_completion_tokens": selected["tokens"],
                      "selected_temperature": selected["temperature"], "original_temperatures": temps}
            text = answer + (selected["stop_sequence"] or "")
            return {"text": text, "answer": answer, "completion_tokens": selected["tokens"],
                    "finish_reason": selected["finish_reason"], "tool_call": selected["tool_call"],
                    "stop_sequence": selected["stop_sequence"], "report": public}
    except ContextOverflow as exc:
        raise APIError(400, str(exc), code="context_length_exceeded") from exc
    except SelectionError as exc:
        raise APIError(502, str(exc), "model_error", "pro_selection_failed") from exc
