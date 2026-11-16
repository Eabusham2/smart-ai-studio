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
