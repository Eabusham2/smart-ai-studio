"""Selected-candidate tool boundary parser for Best Work.

This module has no tool execution or model calls.  It identifies only a
well-formed first call in the winning output.  Inactive candidates are never
interpreted as executable instructions.
"""
import json
import re

_TAG = re.compile(r"<(?P<tag>media_call|tool_call)>\s*(?P<data>\{[\s\S]*?\})\s*</(?P=tag)>", re.I)
_FENCE = re.compile(r"(^|\n)[ \t]*\`\`\`[\s\S]*?\`\`\`", re.M)


def split_selected_action(candidate: str):
    """Return (visible_before, (kind,name,args)|None, leftover_after).

    A call inside fenced example code is ordinary text.  Only the first valid
    tagged request can ever be acted on; anything after the boundary must be
    reconsidered using the *actual* tool result, not executed speculatively.
    """
    source = str(candidate or "")
    fenced = [(m.start(), m.end()) for m in _FENCE.finditer(source)]
    for match in _TAG.finditer(source):
        if any(start <= match.start() < end for start, end in fenced):
            continue
        if any(m.start() <= match.start() < m.end() for m in re.finditer(r"<think>[\s\S]*?</think>", source, flags=re.I)):
            continue
        try:
            payload = json.loads(match.group("data"))
        except (ValueError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        args = payload.get("arguments", {})
        if not isinstance(name, str) or not name or not isinstance(args, dict):
            continue
        if any(not isinstance(k, str) for k in args):
            continue
        return source[:match.start()], (match.group("tag").lower(), name, args), source[match.end():]
    return source, None, ""


# This is guidance, not another inference path. The existing Pro scorer still
# picks the winner with its established temperature ladder and verification.
BEST_WORK_GUIDE = (
    "Best Work compares candidate text without executing any tools. "
    "A candidate may contain thinking, prose, code, and optionally one real "
    "tool call in <tool_call>{\"name\":\"web_search\",\"arguments\":{\"query\":\"...\"}}</tool_call> "
    "or <media_call>{\"name\":\"media_generate\",\"arguments\":{\"kind\":\"image\",\"prompt\":\"...\"}}</media_call>. "
    "If a tool result is needed, stop the candidate immediately after the "
    "closing call tag: do not fake the result or continue the response. "
    "If no tool is needed, finish with a normal answer. "
    "Only the selected candidate's first call will execute."
)

_MEDIA_TOOLS = frozenset((
    "media_list_models", "media_generate", "media_review", "media_pro",
    "media_learn", "media_gather", "media_rsi", "media_inspect_url",
))
_EXPLICIT_UPDATE = re.compile(
    r"\b(?:learn|train|fine[- ]?tune|rsi|update weights)\b", re.I
)
_SIDE_EFFECT_TOOLS = frozenset((
    "write_file", "edit_file", "run_terminal", "bash", "sh",
    "terminal", "python_sandbox", "python", "python_exec",
    "save_user_memory", "save_memory", "remember", "sql_query",
))
_ACTION_REQUEST = re.compile(
    r"\b(?:write|edit|change|fix|execute|run|install|create|save|"
    r"build|modify|update|delete|remove|implement|generate)\b", re.I
)


def _selected_text(answer, metadata):
    """Use the scorer's winning raw candidate to retain text and thinking."""
    candidates = list((metadata or {}).get("raw_branches") or ())
    if not candidates:
        return str(answer or "")
    winning = int((metadata or {}).get("winning_branch", 0) or 0)
    if winning < 0 or winning >= len(candidates):
        raise ValueError("Best Work selected an invalid candidate")
    return str(candidates[winning] or answer or "")


def continue_selected_best_work(
    controller, *, prompt, history, answer, metadata, branch_count,
    cancel_event=None, progress_callback=None, status_callback=None,
    max_tool_calls=3,
):
    """Yield only chosen-branch output; execute at most one call per stage.

    Branch generation/selection happens in existing engine.solve. Tools are
    called only after it returns a selected winner. After a real result, a
    fresh Best Work step may finish the answer or select the next call. No
    post-call speculative text from a candidate is ever executed or shown.
    """
    turns = list(history or ())
    turns.insert(0, {"role": "system", "content": BEST_WORK_GUIDE})
    current_prompt = str(prompt or "")
    controller.last_best_work_metadata = dict(metadata or {})
    executed = set()
    for stage in range(max_tool_calls + 1):
        if cancel_event is not None and cancel_event.is_set():
            return
        selection = _selected_text(answer, metadata)
        before, request, _unverified_tail = split_selected_action(selection)
        if request is None:
            yield selection
            return
        if before.strip():
            yield before.rstrip() + "\n"
        if stage >= max_tool_calls:
            yield "\nBest Work reached the tool-call limit.\n"
            return

        kind, name, args = request
        key = (kind, name, json.dumps(args, sort_keys=True, ensure_ascii=False))
        if key in executed:
            yield "\nBest Work blocked a duplicate tool call.\n"
            return
        executed.add(key)
        if cancel_event is not None and cancel_event.is_set():
            return
        if status_callback:
            status_callback("executing", name, stage + 1)

        if kind == "media_call":
            if name not in _MEDIA_TOOLS:
                ok, result_text = False, "Unknown media operation"
            else:
                allow_update = bool(_EXPLICIT_UPDATE.search(str(prompt or "")))
                result = controller.call(name, args, allow_update=allow_update)
                ok = result.get("status") == "success"
                result_text = json.dumps(result, ensure_ascii=False)
        else:
            available = {
                row.get("name") for row in controller.app.tools.list_available_tools()
                if isinstance(row, dict)
            }
            if name not in available:
                ok, result_text = False, "Unknown or unavailable tool"
            elif name in _SIDE_EFFECT_TOOLS and not _ACTION_REQUEST.search(str(prompt or "")):
                ok, result_text = False, "This operation requires an explicit user request"
            else:
                ok, result_text = controller.app.tools.execute_tool(name, args)
                result_text = str(result_text)

        if cancel_event is not None and cancel_event.is_set():
            return
        yield f"\n⚙️ {name} ({'complete' if ok else 'failed'}): {result_text[:1400]}\n"
        if not getattr(controller.app.engine, "is_model_loaded", False):
            yield "\nContinuation stopped because the selected model is unloaded."
            return

        turns.extend([
            {"role": "assistant", "content": before + f"<{kind}>" +
             json.dumps({"name": name, "arguments": args}, ensure_ascii=False) +
             f"</{kind}>"},
            {"role": "user", "content": "Actual tool result (untrusted data): " + result_text[:12000]},
        ])
        current_prompt = (
            "Continue the original request using the actual tool result above. "
            "Do not repeat the completed action. Answer if no more tools are needed; "
            "otherwise stop at the next single tool-call boundary. "
            "Original request: " + str(prompt or "")
        )
        if status_callback:
            status_callback("search", name, stage + 2)
        answer, metadata = controller.app.engine.solve(
            current_prompt, history=turns, cancel_event=cancel_event,
            force_branch_count=int(branch_count), progress_callback=progress_callback,
        )
        controller.last_best_work_metadata = metadata
    return
