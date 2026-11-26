"""Small, stateless HTTP adapters for a model already loaded in the desktop app.

The callback owns model inference; this module never loads weights or executes
tools. OpenAI Chat/Responses and Anthropic Messages share the same request and
stream. Unknown token counts are null, never word counts disguised as tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import ipaddress
import json
import math
import queue
import re
import select
import socket
import threading
import time
from urllib.parse import urlsplit
from uuid import uuid4


class APIError(ValueError):
    def __init__(self, status, message, error_type="invalid_request_error", code=None):
        super().__init__(message)
        self.status, self.error_type, self.code = status, error_type, code


@dataclass
class GenerationRequest:
    model: str
    messages: list[dict]
    tools: list[dict] = field(default_factory=list)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] = field(default_factory=list)
    tool_choice: str | dict = "auto"
    protocol: str = "chat"
    parallel_tool_calls: bool = True
    sampling: dict = field(default_factory=dict)
    context_limit: int | None = None
    enable_thinking: bool | None = None
    speedups: dict = field(default_factory=dict)
    include_reasoning: bool = False
    learning_mode: str | None = None
    pro: dict = field(default_factory=dict)


@dataclass
class GenerationChunk:
    text: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    starts_in_thought: bool | None = None


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _text(value, label="content"):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for part in value:
            if not isinstance(part, dict) or not isinstance(part.get("type"), str) or part["type"] not in {"text", "input_text", "output_text"}:
                raise APIError(400, f"{label}: only text content is supported by this endpoint.")
            if not isinstance(part.get("text"), str):
                raise APIError(400, f"{label}: text must be a string.")
            parts.append(part["text"])
        return "\n".join(parts)
    raise APIError(400, f"{label} must be text or text content blocks.")


def _call(value, *, custom=False):
    name = value.get("name")
    namespace = value.get("namespace")
    if namespace:
        if not isinstance(namespace, str):
            raise APIError(400, "Tool namespaces must be strings.")
        name = str(namespace).rstrip(".") + "." + str(name or "")
    if not isinstance(name, str) or not name:
        raise APIError(400, "Tool calls need a name.")
    args = {"input": value.get("input", "")} if custom else value.get("arguments", {})
    if custom and not isinstance(args["input"], str):
        raise APIError(400, "Custom tool input must be a string.")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, TypeError):
            raise APIError(400, "Tool arguments must be a JSON object.") from None
    if not isinstance(args, dict):
        raise APIError(400, "Tool arguments must be a JSON object.")
    call_id = value.get("call_id") or value.get("id")
    if not isinstance(call_id, str) or not call_id:
        raise APIError(400, "Tool calls need their original call ID.")
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": _json(args)}}


def _messages(body, protocol):
    messages = []
    if protocol == "responses":
        if body.get("previous_response_id") or body.get("conversation"):
            raise APIError(400, "This server is stateless. Send full input history; previous_response_id/conversation is unsupported.")
        if body.get("instructions"):
            messages.append({"role": "system", "content": _text(body["instructions"], "instructions")})
        source = body.get("input", [])
        if isinstance(source, str):
            source = [{"role": "user", "content": source}]
    else:
        source = body.get("messages")
        if protocol == "anthropic" and body.get("system"):
            messages.append({"role": "system", "content": _text(body["system"], "system")})
    if not isinstance(source, list) or len(source) > 4096:
        raise APIError(400, "messages/input must be a list of at most 4096 items.")
    for item in source:
        if not isinstance(item, dict):
            raise APIError(400, "Every input item must be an object.")
        kind = item.get("type", "message")
        if not isinstance(kind, str):
            raise APIError(400, "Input item types must be strings.")
        if protocol == "responses" and kind in {"function_call", "custom_tool_call"}:
            messages.append({"role": "assistant", "content": "", "tool_calls": [_call(item, custom=kind == "custom_tool_call")]})
            continue
        if protocol == "responses" and kind in {"function_call_output", "custom_tool_call_output"}:
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise APIError(400, "Tool outputs need call_id.")
            messages.append({"role": "tool", "tool_call_id": call_id, "content": _text(item.get("output"), "tool output")})
            continue
        if protocol == "responses" and kind == "reasoning":
            # A caller can replay Responses output items. Reasoning is not a
            # user instruction or a tool result and is not inserted as one.
            continue
        if kind != "message":
            raise APIError(400, f"Unsupported input item type: {kind}.")
        role = item.get("role")
        if role == "developer":
            role = "system"
        if not isinstance(role, str) or role not in {"system", "user", "assistant", "tool"}:
            raise APIError(400, "Unsupported or missing message role.")
        content = item.get("content")
        if protocol == "anthropic" and isinstance(content, list):
            text_parts, calls = [], []
            def flush():
                if text_parts or calls:
                    row = {"role": role, "content": "\n".join(text_parts)}
                    if calls:
                        row["tool_calls"] = list(calls)
                    messages.append(row)
                    text_parts.clear()
                    calls.clear()
            for block in content:
                if not isinstance(block, dict):
                    raise APIError(400, "Content blocks must be objects.")
                block_kind = block.get("type")
                if not isinstance(block_kind, str):
                    raise APIError(400, "Content block types must be strings.")
                if block_kind == "text":
                    text_parts.append(_text(block.get("text")))
                elif block_kind == "tool_use" and role == "assistant":
                    calls.append(_call({"id": block.get("id"), "name": block.get("name"), "arguments": block.get("input")}))
                elif block_kind == "tool_result" and role == "user":
                    flush()
                    call_id = block.get("tool_use_id")
                    if not isinstance(call_id, str) or not call_id:
                        raise APIError(400, "Tool results need tool_use_id.")
                    text = _text(block.get("content"), "tool result")
                    if block.get("is_error"):
                        text = "Tool error: " + text
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
                elif block_kind in {"thinking", "redacted_thinking"} and role == "assistant":
                    continue
                else:
                    raise APIError(400, f"Unsupported content block: {block_kind}. Only text and client tool calls/results are supported.")
            flush()
            continue
        row = {"role": role, "content": _text(content)}
        if item.get("reasoning_content") is not None:
            if role != "assistant" or not isinstance(item["reasoning_content"], str):
                raise APIError(400, "reasoning_content must be assistant text.")
            row["reasoning_content"] = item["reasoning_content"]
        if item.get("tool_calls"):
            if role != "assistant" or not isinstance(item["tool_calls"], list):
                raise APIError(400, "tool_calls must be an assistant message list.")
            row["tool_calls"] = []
            for tool in item["tool_calls"]:
                if not isinstance(tool, dict) or tool.get("type", "function") != "function" or not isinstance(tool.get("function"), dict):
                    raise APIError(400, "Only function tool calls are supported.")
                row["tool_calls"].append(_call({**tool["function"], "id": tool.get("id")}))
        if role == "tool":
            if not isinstance(item.get("tool_call_id"), str) or not item["tool_call_id"]:
                raise APIError(400, "Tool messages need tool_call_id.")
            row["tool_call_id"] = item["tool_call_id"]
        messages.append(row)
    if not messages or all(row["role"] == "system" for row in messages):
        raise APIError(400, "Provide at least one conversation message.")
    return messages


def _tools(source, protocol, namespace=""):
    if source is None:
        return []
    if not isinstance(source, list) or len(source) > 256:
        raise APIError(400, "tools must be a list of at most 256 definitions.")
    result = []
    for tool in source:
        if not isinstance(tool, dict):
            raise APIError(400, "Tool definitions must be objects.")
        kind = tool.get("type", "function" if protocol != "anthropic" else "custom")
        if not isinstance(kind, str):
            raise APIError(400, "Tool types must be strings.")
        if tool.get("description") is not None and not isinstance(tool["description"], str):
            raise APIError(400, "Tool descriptions must be strings.")
        if protocol == "responses" and kind == "namespace":
            if not isinstance(tool.get("name"), str) or not tool["name"] or namespace:
                raise APIError(400, "Invalid tool namespace.")
            result.extend(_tools(tool.get("tools"), protocol, tool["name"]))
            continue
        if protocol == "chat":
            if kind != "function" or not isinstance(tool.get("function"), dict):
                raise APIError(400, "Only client function tools are supported.")
            function = dict(tool["function"])
        elif protocol == "anthropic":
            if kind not in {"custom", "function"}:
                raise APIError(400, "Hosted Anthropic tools are unsupported; provide client tools with input_schema.")
            function = {"name": tool.get("name"), "description": tool.get("description", ""), "parameters": tool.get("input_schema", {"type": "object", "properties": {}})}
        elif kind == "function":
            function = {key: tool[key] for key in ("name", "description", "parameters", "strict") if key in tool}
        elif kind == "custom":
            function = {"name": tool.get("name"), "description": tool.get("description", "") + "\nSupply the tool's raw input in the input string.", "parameters": {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}}
        else:
            raise APIError(400, f"Unsupported hosted tool type: {kind}. This server supports client function and custom tools.")
        name = function.get("name")
        if not isinstance(name, str) or not name or len(name) > 256:
            raise APIError(400, "Each tool needs a nonempty name of at most 256 characters.")
        function.setdefault("parameters", {"type": "object", "properties": {}})
        if not isinstance(function["parameters"], dict):
            raise APIError(400, "Tool parameters/input_schema must be a JSON schema object.")
        normalized = {"type": "function", "function": function}
        if protocol == "responses" and kind == "custom":
            normalized["original_type"] = "custom"
        if namespace:
            normalized["namespace"], normalized["original_name"] = namespace, name
            function["name"] = namespace + "." + name
        result.append(normalized)
    names = [tool["function"]["name"] for tool in result]
    if len(result) > 256 or len(set(names)) != len(names):
        raise APIError(400, "Tool names must be unique and no more than 256 tools may be supplied.")
    return result


def _optional_execution_options(body, protocol):
    """Validate the deliberately bounded, optional execution-only API surface."""
    settings = body.get("smartai")
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise APIError(400, "smartai must be an object.")
    unknown = set(settings) - {"context_limit", "enable_thinking", "speedups", "include_reasoning", "learning_mode", "pro"}
    if unknown:
        raise APIError(400, "Unknown smartai option: " + ", ".join(sorted(unknown)))
    pro = settings.get("pro")
    if pro is None:
        pro = {}
    if not isinstance(pro, dict) or set(pro) - {"enabled", "routing", "branch_count", "best_work_passes"}:
        raise APIError(400, "smartai.pro accepts enabled, routing, branch_count and best_work_passes.")
    pro = dict(pro)
    if "enabled" in pro and type(pro["enabled"]) is not bool:
        raise APIError(400, "smartai.pro.enabled must be a boolean.")
    if "routing" in pro and pro["routing"] not in ("auto", "fixed"):
        raise APIError(400, "smartai.pro.routing must be auto or fixed.")
    for key, maximum in (("branch_count", 64), ("best_work_passes", 999)):
        if key in pro and (type(pro[key]) is not int or not 1 <= pro[key] <= maximum):
            raise APIError(400, f"smartai.pro.{key} must be an integer from 1 to {maximum}.")
    if "branch_count" in pro:
        if pro.get("routing") == "auto":
            raise APIError(400, "Choose fixed routing when overriding branch_count.")
        pro.setdefault("routing", "fixed")
    learning = settings.get("learning_mode")
    if learning is not None and (not isinstance(learning, str) or learning not in {"legacy", "managed", "hybrid", "off"}):
        raise APIError(400, "smartai.learning_mode must be legacy, managed, hybrid, or off.")
    context = settings.get("context_limit")
    if context is not None and (type(context) is not int or not 1 <= context <= 10_000_000):
        raise APIError(400, "smartai.context_limit must be a positive integer up to 10000000.")
    thinking = settings.get("enable_thinking")
    if body.get("enable_thinking") is not None:
        if type(body["enable_thinking"]) is not bool:
            raise APIError(400, "enable_thinking must be a boolean.")
        if thinking is not None and body["enable_thinking"] != thinking:
            raise APIError(400, "Conflicting enable_thinking options.")
        thinking = body["enable_thinking"]
    template_kwargs = body.get("chat_template_kwargs")
    if template_kwargs is not None:
        if not isinstance(template_kwargs, dict) or set(template_kwargs) - {"enable_thinking"}:
            raise APIError(400, "Only chat_template_kwargs.enable_thinking is supported.")
        if "enable_thinking" in template_kwargs:
            if type(template_kwargs["enable_thinking"]) is not bool:
                raise APIError(400, "enable_thinking must be a boolean.")
            if thinking is not None and template_kwargs["enable_thinking"] != thinking:
                raise APIError(400, "Conflicting enable_thinking options.")
            thinking = template_kwargs["enable_thinking"]
    # A local reasoning template has an on/off switch, not vendor effort tiers.
    # Explicit local controls take precedence over a harness's effort default.
    reasoning = body.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, dict):
        raise APIError(400, "reasoning must be an object.")
    effort = body.get("reasoning_effort")
    if effort is None and isinstance(reasoning, dict):
        effort = reasoning.get("effort")
    if effort is not None:
        if not isinstance(effort, str) or effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise APIError(400, "Unsupported reasoning effort.")
        if thinking is None:
            thinking = effort != "none"
    vendor_thinking = body.get("thinking")
    if protocol == "anthropic" and vendor_thinking is not None:
        if not isinstance(vendor_thinking, dict) or vendor_thinking.get("type") != "disabled":
            raise APIError(400, "Signed Anthropic thinking/budget controls are unsupported; use smartai.enable_thinking.")
        if thinking is None:
            thinking = False
    if thinking is not None and type(thinking) is not bool:
        raise APIError(400, "enable_thinking must be a boolean.")
    include = settings.get("include_reasoning", False)
    if type(include) is not bool or (include and protocol != "chat"):
        raise APIError(400, "Plain reasoning_content is optional on Chat Completions only.")
    speedups = settings.get("speedups")
    if speedups is None:
        speedups = {}
    if not isinstance(speedups, dict):
        raise APIError(400, "smartai.speedups must be an object.")
    speedups = dict(speedups)
    bool_keys = {"enabled", "prompt_reuse", "tokenizer_memo", "metal_warmup", "persistent_cache"}
    unknown = set(speedups) - bool_keys - {"speculative_mode", "max_draft_tokens"}
    if unknown:
        raise APIError(400, "Unknown or non-request speedup: " + ", ".join(sorted(unknown)))
    for key in bool_keys & set(speedups):
        if type(speedups[key]) is not bool:
            raise APIError(400, key + " must be a boolean.")
    if speedups.get("persistent_cache"):
        raise APIError(400, "API prompt persistence is disabled; the client owns memory.")
    speedups.pop("persistent_cache", None)
    mode = speedups.get("speculative_mode")
    if mode is not None and (not isinstance(mode, str) or mode not in {"auto", "none", "mtp"}):
        raise APIError(400, "speculative_mode must be auto, none, or mtp.")
    if mode is None:
        speedups.pop("speculative_mode", None)
    draft = speedups.get("max_draft_tokens")
    if draft is not None and (type(draft) is not int or not 1 <= draft <= 64):
        raise APIError(400, "max_draft_tokens must be an integer from 1 to 64.")
    if draft is None:
        speedups.pop("max_draft_tokens", None)
    if speedups.get("enabled") is False and (
            any(speedups.get(key) is True for key in bool_keys - {"enabled", "persistent_cache"})
            or mode in {"auto", "mtp"} or draft is not None):
        raise APIError(400, "Speedups cannot be enabled individually when enabled is false.")
    if mode == "none" and draft is not None:
        raise APIError(400, "Draft tokens require active speculative decoding.")
    sampling = {}
    for key, low, high, integer in (
        ("top_k", 0, 200000, True), ("min_p", 0, 1, False),
        ("typical_p", 0, 1, False), ("repeat_penalty", 0.01, 10, False),
        ("repeat_last_n", 0, 1000000, True), ("seed", 0, 4294967295, True),
        ("frequency_penalty", -2, 2, False), ("presence_penalty", -2, 2, False),
    ):
        value = body.get(key)
        if key == "repeat_penalty" and body.get("repetition_penalty") is not None:
            alias = body["repetition_penalty"]
            if value is not None and value != alias:
                raise APIError(400, "Conflicting repetition penalty options.")
            value = alias
        if value is None:
            continue
        if type(value) not in ((int,) if integer else (int, float)) or not low <= value <= high:
            raise APIError(400, f"{key} must be {'an integer' if integer else 'a number'} between {low} and {high}.")
        sampling[key] = value
    return {"context_limit": context, "enable_thinking": thinking, "speedups": speedups,
            "include_reasoning": include, "sampling": sampling, "learning_mode": learning, "pro": pro}


def normalize_request(body, protocol, model):
    if not isinstance(body, dict):
        raise APIError(400, "The JSON body must be an object.")
    if type(body.get("n", 1)) is not int or body.get("n", 1) != 1 or body.get("logprobs") or body.get("top_logprobs"):
        raise APIError(400, "Multiple choices and log probabilities are not supported.")
    if body.get("background") or body.get("store"):
        raise APIError(400, "This server is stateless; background/store must be false.")
    if body.get("stream") is not None and type(body["stream"]) is not bool:
        raise APIError(400, "stream must be a boolean.")
    if body.get("parallel_tool_calls") is not None and type(body["parallel_tool_calls"]) is not bool:
        raise APIError(400, "parallel_tool_calls must be a boolean.")
    for key in ("text", "stream_options"):
        if body.get(key) is not None and not isinstance(body[key], dict):
            raise APIError(400, f"{key} must be an object.")
    response_format = body.get("response_format") or (body.get("text") or {}).get("format") or {"type": "text"}
    if not isinstance(response_format, dict) or response_format.get("type", "text") != "text":
        raise APIError(400, "Constrained JSON/schema output is not supported by this adapter.")
    if body.get("logit_bias") or body.get("audio") or body.get("modalities") not in (None, ["text"]):
        raise APIError(400, "Only text output without logit_bias is supported.")
    tools = _tools(body.get("tools"), protocol)
    choice = body.get("tool_choice", "auto")
    if isinstance(choice, dict):
        kind = choice.get("type")
        if not isinstance(kind, str):
            raise APIError(400, "tool_choice type must be a string.")
        if protocol == "anthropic" and kind in {"auto", "none", "any"}:
            choice = "required" if kind == "any" else kind
        elif kind in {"function", "tool", "custom"}:
            if protocol == "chat" and not isinstance(choice.get("function"), dict):
                raise APIError(400, "A function tool_choice must include a function object.")
            name = (choice.get("function") or {}).get("name") if protocol == "chat" else choice.get("name")
            choice = {"type": "function", "function": {"name": name}}
        else:
            raise APIError(400, "Unsupported tool_choice.")
    elif not isinstance(choice, str) or choice not in {"auto", "none", "required"}:
        raise APIError(400, "Unsupported tool_choice.")
    if isinstance(choice, dict) and (not isinstance(choice["function"]["name"], str)
                                    or choice["function"]["name"] not in {t["function"]["name"] for t in tools}):
        raise APIError(400, "tool_choice names a tool that was not supplied.")
    if choice == "required" and not tools:
        raise APIError(400, "tool_choice=required needs tool definitions.")
    options = {}
    for key, lower, upper in (("temperature", 0, 2), ("top_p", 0, 1)):
        value = body.get(key)
        if value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
            raise APIError(400, f"{key} must be a finite number between {lower} and {upper}.")
        options[key] = float(value)
    maximum = body.get("max_output_tokens" if protocol == "responses" else "max_tokens")
    if maximum is None and protocol == "chat":
        maximum = body.get("max_completion_tokens")
    if maximum is not None and (type(maximum) is not int or maximum < 1):
        raise APIError(400, "Maximum output tokens must be a positive integer.")
    stops = body.get("stop_sequences" if protocol == "anthropic" else "stop") or []
    if isinstance(stops, str):
        stops = [stops]
    if not isinstance(stops, list) or len(stops) > 16 or any(not isinstance(s, str) or not s or len(s) > 1024 for s in stops):
        raise APIError(400, "stop must contain at most 16 nonempty strings (up to 1024 characters each).")
    return GenerationRequest(model=model, messages=_messages(body, protocol), tools=tools,
                             max_tokens=maximum, stop=stops, tool_choice=choice, protocol=protocol,
                             parallel_tool_calls=body.get("parallel_tool_calls", True) is not False,
                             **options, **_optional_execution_options(body, protocol))


class _OutputParser:
    """Stream prose, suppress explicit thoughts, and recognize complete tool JSON.

    Existing literal-code/thought rules are shared with desktop tool handoff.
    Only supplied tools can become calls. No local registry is consulted.
    """
    def __init__(self, request, *, first_tool_only=False):
        from core.gui_thinking import ReasoningParser
        self.first_tool_only = first_tool_only
        self.reasoning = ReasoningParser()
        self.request = request
        self.buffer, self.prefix = "", ""
        self.tools = {t["function"]["name"]: t for t in request.tools}
        self.stopped, self.stop_sequence = False, None
        self.calls = 0

    def _display_events(self, events):
        return [(kind, value) for kind, value in events
                if kind == "text" or (kind == "thought" and self.request.include_reasoning)]

    def _parse_call(self, source, final):
        if not source.startswith("<function="):
            try:
                return json.JSONDecoder().raw_decode(source)
            except ValueError:
                if final:
                    raise APIError(502, "The model returned an incomplete or invalid tool call.", "model_error") from None
                return None
        # Qwen 3.5 / Bonsai 2.5's published template uses these XML-like
        # delimiters. Do not feed parameter values through the thought parser
        # or XML entity rewriting: values can contain source code verbatim.
        opening = re.match(r"<function=([^<>]+)>\s*", source)
        if not opening:
            if final:
                raise APIError(502, "The model returned an invalid function tag.", "model_error")
            return None
        name, position = opening.group(1).strip(), opening.end()
        definition = self.tools.get(name, {}).get("function", {}).get("parameters", {})
        properties = definition.get("properties", {})
        arguments = {}
        while True:
            whitespace = re.match(r"\s*", source[position:])
            position += whitespace.end()
            if source.startswith("</function>", position):
                return {"name": name, "arguments": arguments}, position + len("</function>")
            parameter = re.match(r"<parameter=([^<>]+)>", source[position:])
            if parameter is None:
                if final:
                    raise APIError(502, "The model returned an incomplete or invalid parameter tag.", "model_error")
                return None
            key = parameter.group(1).strip()
            start = position + parameter.end()
            end = source.find("</parameter>", start)
            if end < 0:
                if final:
                    raise APIError(502, "The model did not close its parameter value.", "model_error")
                return None
            if not key or key in arguments:
                raise APIError(502, "The model returned an empty or duplicate parameter.", "model_error")
            value = source[start:end]
            if value.startswith("\r\n"):
                value = value[2:]
            elif value.startswith("\n"):
                value = value[1:]
            if value.endswith("\r\n"):
                value = value[:-2]
            elif value.endswith("\n"):
                value = value[:-1]
            schema = properties.get(key, {}) if isinstance(properties, dict) else {}
            parameter_type = schema.get("type") if isinstance(schema, dict) else None
            if parameter_type != "string":
                try:
                    value = json.loads(value)
                except ValueError:
                    if parameter_type in ("object", "array", "integer", "number", "boolean", "null"):
                        raise APIError(502, "The model returned invalid JSON for a typed tool parameter.", "model_error") from None
            arguments[key] = value
            position = end + len("</parameter>")

    def feed(self, text, final=False):
        # Parse tool JSON before reasoning markup: code arguments can contain
        # literal <think>, backticks, or </tool_call> without being rewritten.
        self.buffer += text
        events = self._drain(final)
        if final:
            events.extend(self._display_events(self.reasoning.feed("", final=True)))
        return events

    def _drain(self, final):
        from core.best_work import _protected_prefix
        events = []
        def emit(text):
            if text:
                self.prefix += text
                events.extend(self._display_events(self.reasoning.feed(text)))
        while self.buffer and not self.stopped:
            low = self.buffer.casefold()
            marker = low.find("<tool_call>") if self.tools and self.request.tool_choice != "none" else -1
            stop_positions = [(self.buffer.find(s), s) for s in self.request.stop if s in self.buffer]
            first_stop = min(stop_positions, default=None)
            if first_stop and (marker < 0 or first_stop[0] < marker):
                emit(self.buffer[:first_stop[0]])
                self.stop_sequence, self.stopped, self.buffer = first_stop[1], True, ""
                break
            if marker >= 0 and _protected_prefix(self.prefix + self.buffer[:marker]):
                emit(self.buffer[:marker + len("<tool_call>")])
                self.buffer = self.buffer[marker + len("<tool_call>"):]
                continue
            if marker < 0:
                # Hold only delimiter/stop prefixes; ordinary text is immediate.
                needles = self.request.stop + (["<tool_call>"] if self.tools and self.request.tool_choice != "none" else [])
                hold = max((n for needle in needles for n in range(1, min(len(needle), len(self.buffer)) + 1)
                            if self.buffer[-n:].casefold() == needle[:n].casefold()), default=0)
                if final:
                    hold = 0
                emit(self.buffer[:-hold] if hold else self.buffer)
                self.buffer = self.buffer[-hold:] if hold else ""
                break
            emit(self.buffer[:marker])
            self.buffer = self.buffer[marker:]
            payload_text = self.buffer[len("<tool_call>"):].lstrip()
            parsed = self._parse_call(payload_text, final)
            if parsed is None:
                break
            payload, end = parsed
            close = re.match(r"\s*</tool_call>", payload_text[end:], re.I)
            if not close:
                if final:
                    raise APIError(502, "The model did not close its tool call.", "model_error")
                break
            if not isinstance(payload, dict) or payload.get("name") not in self.tools:
                raise APIError(502, "The model requested a tool not supplied by this client.", "model_error")
            args = payload.get("arguments", payload.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    raise APIError(502, "The model returned invalid tool arguments.", "model_error") from None
            if not isinstance(args, dict):
                raise APIError(502, "The model returned non-object tool arguments.", "model_error")
            tool = self.tools[payload["name"]]
            if tool.get("original_type") == "custom" and not isinstance(args.get("input"), str):
                raise APIError(502, "The custom tool needs a string input.", "model_error")
            if isinstance(self.request.tool_choice, dict) and payload["name"] != self.request.tool_choice["function"]["name"]:
                raise APIError(502, "The model did not select the requested tool.", "model_error")
            self.calls += 1
            if self.calls > 1 and not self.request.parallel_tool_calls:
                raise APIError(502, "The model emitted parallel tools when they were disabled.", "model_error")
            events.append(("tool", {"id": "call_" + uuid4().hex, "name": payload["name"], "arguments": args, "definition": tool}))
            self.buffer = payload_text[end + close.end():]
            if self.first_tool_only:
                # Pro candidates stop after the first complete client call.
                # Ordinary API streaming retains parallel-tool support.
                self.stopped = True
                self.buffer = ""
        return events


class _Reply:
    def __init__(self, request, emit=None):
        self.request, self.emit = request, emit
        self.protocol = request.protocol
        self.id = ("chatcmpl-" if self.protocol == "chat" else "resp_" if self.protocol == "responses" else "msg_") + uuid4().hex
        self.created = int(time.time())
        self.output, self.text, self.calls = [], "", []
        self.reasoning_text = ""
        self.prompt_tokens = self.completion_tokens = None
        self.finish_reason, self.stop_sequence = "stop", None
        self.sequence, self.text_index, self.text_id = 0, None, None

    def event(self, name, **values):
        if self.emit is None:
            return
        values = {"type": name, **values}
        if self.protocol == "responses":
            values["sequence_number"] = self.sequence
            self.sequence += 1
        self.emit(name, values)

    def chat_event(self, delta, finish=None, usage=False):
        value = {"id": self.id, "object": "chat.completion.chunk", "created": self.created,
                 "model": self.request.model, "choices": [] if usage else [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if usage:
            value["usage"] = self.usage()
        if self.emit:
            self.emit(None, value)

    def begin(self):
        if self.protocol == "chat":
            self.chat_event({"role": "assistant", "content": ""})
        elif self.protocol == "responses":
            self.event("response.created", response=self.result("in_progress"))
            self.event("response.in_progress", response=self.result("in_progress"))
        else:
            initial = self.result("in_progress")
            initial["usage"] = {"input_tokens": None, "output_tokens": 0}
            self.event("message_start", message=initial)

    def add_reasoning(self, text):
        if self.protocol == "chat" and self.request.include_reasoning and text:
            self.reasoning_text += text
            self.chat_event({"reasoning_content": text})

    def add_text(self, text):
        if not text:
            return
        self.text += text
        if self.protocol == "chat":
            self.chat_event({"content": text})
            return
        if self.text_index is None:
            self.text_index = len(self.output)
            if self.protocol == "responses":
                self.text_id = "msg_" + uuid4().hex
                item = {"id": self.text_id, "type": "message", "role": "assistant", "status": "in_progress", "content": []}
                self.event("response.output_item.added", output_index=self.text_index, item=item)
                part = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
                self.event("response.content_part.added", item_id=self.text_id, output_index=self.text_index, content_index=0, part=part)
                item = {**item, "content": [part]}
            else:
                item = {"type": "text", "text": ""}
                self.event("content_block_start", index=self.text_index, content_block=item)
            self.output.append(item)
        if self.protocol == "responses":
            self.output[self.text_index]["content"][0]["text"] += text
            self.event("response.output_text.delta", item_id=self.text_id, output_index=self.text_index, content_index=0, delta=text, logprobs=[])
        else:
            self.output[self.text_index]["text"] += text
            self.event("content_block_delta", index=self.text_index, delta={"type": "text_delta", "text": text})

    def close_text(self):
        if self.text_index is None:
            return
        index, item = self.text_index, self.output[self.text_index]
        if self.protocol == "responses":
            part = item["content"][0]
            self.event("response.output_text.done", item_id=self.text_id, output_index=index, content_index=0, text=part["text"], logprobs=[])
            self.event("response.content_part.done", item_id=self.text_id, output_index=index, content_index=0, part=part)
            item["status"] = "completed"
            self.event("response.output_item.done", output_index=index, item=item)
        elif self.protocol == "anthropic":
            self.event("content_block_stop", index=index)
        self.text_index, self.text_id = None, None

    def add_tool(self, call):
        self.close_text()
        index = len(self.calls)
        self.calls.append(call)
        arguments = _json(call["arguments"])
        if self.protocol == "chat":
            self.chat_event({"tool_calls": [{"index": index, "id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": ""}}]})
            self.chat_event({"tool_calls": [{"index": index, "function": {"arguments": arguments}}]})
            return
        index = len(self.output)
        if self.protocol == "anthropic":
            item = {"type": "tool_use", "id": call["id"], "name": call["name"], "input": call["arguments"]}
            self.event("content_block_start", index=index, content_block={**item, "input": {}})
            self.event("content_block_delta", index=index, delta={"type": "input_json_delta", "partial_json": arguments})
            self.event("content_block_stop", index=index)
        else:
            tool = call["definition"]
            custom = tool.get("original_type") == "custom"
            item_id = ("ctc_" if custom else "fc_") + uuid4().hex
            field, event = ("input", "custom_tool_call_input") if custom else ("arguments", "function_call_arguments")
            value = call["arguments"]["input"] if custom else arguments
            item = {"id": item_id, "type": "custom_tool_call" if custom else "function_call", "call_id": call["id"],
                    "name": tool.get("original_name", call["name"]), field: value, "status": "completed"}
            if tool.get("namespace"):
                item["namespace"] = tool["namespace"]
            self.event("response.output_item.added", output_index=index, item={**item, field: "", "status": "in_progress"})
            self.event("response." + event + ".delta", item_id=item_id, output_index=index, delta=value)
            self.event("response." + event + ".done", item_id=item_id, output_index=index, **{field: value})
            self.event("response.output_item.done", output_index=index, item=item)
        self.output.append(item)

    def usage(self):
        if self.protocol == "anthropic":
            return {"input_tokens": self.prompt_tokens, "output_tokens": self.completion_tokens}
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        if self.protocol == "chat":
            return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                    "total_tokens": self.prompt_tokens + self.completion_tokens}
        return {"input_tokens": self.prompt_tokens, "output_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens}

    def result(self, status="completed"):
        finish = "tool_calls" if self.calls else self.finish_reason
        if self.protocol == "chat":
            message = {"role": "assistant", "content": self.text or (None if self.calls else "")}
            if self.request.include_reasoning:
                message["reasoning_content"] = self.reasoning_text
            if self.calls:
                message["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": _json(c["arguments"])}} for c in self.calls]
            return {"id": self.id, "object": "chat.completion", "created": self.created, "model": self.request.model,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": self.usage()}
        if self.protocol == "anthropic":
            stop = "tool_use" if self.calls else "max_tokens" if finish == "length" else "stop_sequence" if self.stop_sequence else "end_turn"
            return {"id": self.id, "type": "message", "role": "assistant", "model": self.request.model,
                    "content": self.output, "stop_reason": None if status == "in_progress" else stop,
                    "stop_sequence": self.stop_sequence, "usage": self.usage()}
        if status == "completed" and finish == "length" and not self.calls:
            status = "incomplete"
        choice = self.request.tool_choice
        if isinstance(choice, dict):
            name = choice["function"]["name"]
            tool = next(t for t in self.request.tools if t["function"]["name"] == name)
            choice = {"type": tool.get("original_type", "function"), "name": tool.get("original_name", name)}
            if tool.get("namespace"):
                choice["namespace"] = tool["namespace"]
        return {"id": self.id, "object": "response", "created_at": self.created, "status": status,
                "completed_at": int(time.time()) if status in {"completed", "incomplete"} else None,
                "model": self.request.model, "output": self.output, "error": None,
                "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
                "usage": self.usage(), "parallel_tool_calls": self.request.parallel_tool_calls,
                "previous_response_id": None, "store": False, "tools": [], "tool_choice": choice,
                "temperature": self.request.temperature, "top_p": self.request.top_p, "max_output_tokens": self.request.max_tokens,
                "text": {"format": {"type": "text"}}, "truncation": "disabled", "metadata": {}}

    def finish(self, include_usage=False):
        self.close_text()
        if self.request.tool_choice == "required" or isinstance(self.request.tool_choice, dict):
            if not self.calls:
                raise APIError(502, "The model finished without the required tool call.", "model_error")
        if self.protocol == "chat":
            self.chat_event({}, "tool_calls" if self.calls else self.finish_reason)
            if include_usage:
                self.chat_event({}, usage=True)
            if self.emit:
                self.emit(None, "[DONE]")
        elif self.protocol == "responses":
            result = self.result()
            self.event("response.incomplete" if result["status"] == "incomplete" else "response.completed", response=result)
        else:
            result = self.result()
            self.event("message_delta", delta={"stop_reason": result["stop_reason"], "stop_sequence": result["stop_sequence"]}, usage=self.usage())
            self.event("message_stop")


class _Job:
    def __init__(self):
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.queue = queue.Queue(maxsize=32)
        self.thread = None


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    request_queue_size = 16

    def __init__(self, address, handler, owner):
        self.owner = owner
        self._connections = threading.BoundedSemaphore(32)
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        if not self._connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connections.release()


class LocalModelAPIServer:
    """Serve one loaded model, with at most one active inference callback.

    ``generate(request, cancel_event)`` yields text or ``GenerationChunk``.
    ``model_info()`` returns a model dict with ``id`` (and optional ``aliases``),
    or None if unloaded. ``busy()`` optionally reports desktop work. ``stop``
    returns False until a cancelling callback has actually finished; callers
    must not unload/reuse its model before it returns True.
    """
    def __init__(self, generate, model_info, *, host="127.0.0.1", port=1234, api_key="",
                 busy=None, count_tokens=None, max_body_bytes=8 * 1024 * 1024, max_output_bytes=16 * 1024 * 1024):
        host = str(host).strip()
        if host == "localhost":
            host = "127.0.0.1"
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            raise ValueError("Host must be localhost or an IP address.") from None
        if not address.is_loopback and not api_key:
            raise ValueError("An API key is required when listening beyond localhost.")
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("Port must be between 0 and 65535.")
        self.generate, self.model_info, self.busy, self.count_tokens = generate, model_info, busy, count_tokens
        self.host, self.port, self.api_key = host, port, str(api_key)
        self.max_body_bytes, self.max_output_bytes = max_body_bytes, max_output_bytes
        self._http, self._thread = None, None
        self._jobs, self._lock = set(), threading.RLock()
        self._stopping = threading.Event()

    @property
    def is_running(self):
        return self._http is not None and not self._stopping.is_set()

    @property
    def active_requests(self):
        with self._lock:
            return len(self._jobs)

    @property
    def base_url(self):
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}/v1"

    def start(self):
        with self._lock:
            if self.is_running:
                return self.base_url
            if self._jobs:
                raise RuntimeError("The previous request is still stopping.")
            self._stopping.clear()
            self._http = _HTTPServer((self.host, self.port), _Handler, self)
            self.port = self._http.server_address[1]
            self._thread = threading.Thread(target=self._http.serve_forever, kwargs={"poll_interval": 0.1}, name="smartai-api", daemon=True)
            self._thread.start()
        return self.base_url

    def stop(self, timeout=2.0):
        self._stopping.set()
        with self._lock:
            jobs = list(self._jobs)
            http, self._http = self._http, None
        for job in jobs:
            job.cancel.set()
        if http is not None:
            http.shutdown()
            http.server_close()
        deadline = time.monotonic() + max(0, timeout)
        for job in jobs:
            job.done.wait(max(0, deadline - time.monotonic()))
        return self.active_requests == 0

    def _begin(self, request):
        with self._lock:
            if self._stopping.is_set():
                raise APIError(503, "The API server is stopping.", "server_error")
            if self._jobs or (callable(self.busy) and self.busy()):
                raise APIError(409, "The model is busy. Retry when the current request or desktop operation finishes.", "server_error", "model_busy")
            if not self.model_info():
                raise APIError(503, "Load a text model in Smart AI Studio first.", "server_error", "model_not_loaded")
            job = _Job()
            self._jobs.add(job)
        def put(value):
            while not job.cancel.is_set():
                try:
                    job.queue.put(value, timeout=0.1)
                    return
                except queue.Full:
                    pass
        def run():
            iterator = None
            try:
                iterator = iter(self.generate(request, job.cancel))
                for item in iterator:
                    if job.cancel.is_set():
                        break
                    put(("chunk", item))
            except Exception as exc:
                put(("error", exc))
            finally:
                try:
                    close = getattr(iterator, "close", None)
                    if callable(close):
                        close()
                finally:
                    # Removal is after generator.close: the model must remain
                    # leased while an owner thread cleans up its lazy cache.
                    with self._lock:
                        self._jobs.discard(job)
                    job.done.set()
        job.thread = threading.Thread(target=run, name="smartai-api-inference", daemon=True)
        job.thread.start()
        return job

    def _count(self, request):
        # Tokenizers and lazy MLX wrappers can share model state. Counting
        # participates in the same lease as generation, including shutdown.
        with self._lock:
            if self._stopping.is_set() or self._jobs or (callable(self.busy) and self.busy()):
                raise APIError(409, "The model is busy.", "server_error", "model_busy")
            job = _Job()
            self._jobs.add(job)
        try:
            return self.count_tokens(request)
        finally:
            with self._lock:
                self._jobs.discard(job)
            job.done.set()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "SmartAI/1"

    def log_message(self, *_):
        # Prompts, tool outputs and keys never enter HTTP logs.
        pass

    def setup(self):
        super().setup()
        self.connection.settimeout(30)
        self._stream_started = False

    @property
    def owner(self):
        return self.server.owner

    def _headers(self, status, content_type, length=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Content-Type-Options", "nosniff")
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.end_headers()
        self.close_connection = True

    def _send_json(self, status, value):
        data = _json(value).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(data))
        self.wfile.write(data)

    def _auth(self):
        if self.headers.get("Origin"):
            # This desktop service is for local/native harnesses. It does not
            # grant arbitrary websites cross-origin model or tool access.
            raise APIError(403, "Browser cross-origin requests are not enabled.", "permission_error")
        host = self.headers.get("Host", "")
        try:
            hostname = urlsplit("//" + host).hostname
            address = ipaddress.ip_address(self.owner.host)
            if address.is_loopback and hostname not in {"localhost", "127.0.0.1", "::1", self.owner.host}:
                raise APIError(403, "Unexpected Host header.", "permission_error")
        except APIError:
            raise
        except ValueError:
            raise APIError(400, "Invalid Host header.") from None
        if self.owner.api_key:
            auth = self.headers.get("Authorization", "")
            bearer = auth[7:] if auth.lower().startswith("bearer ") else ""
            supplied = bearer or self.headers.get("x-api-key", "")
            if not hmac.compare_digest(supplied.encode(), self.owner.api_key.encode()):
                raise APIError(401, "Invalid or missing API key.", "authentication_error")

    def _error(self, error, protocol="chat"):
        if not isinstance(error, APIError):
            error = APIError(500, "Model generation failed: " + str(error), "server_error")
        details = {"type": error.error_type, "message": str(error)}
        if error.code:
            details["code"] = error.code
        body = {"error": details}
        if protocol == "anthropic":
            body["type"] = "error"
        if self._stream_started:
            if protocol == "responses":
                reply = getattr(self, "_reply", None)
                if reply is not None:
                    reply.event("error", code=error.code or error.error_type, message=str(error), param=None)
                else:
                    self._sse("error", {"type": "error", "code": error.code or error.error_type, "message": str(error), "param": None, "sequence_number": 0})
            else:
                self._sse("error" if protocol == "anthropic" else None, body)
        else:
            self._send_json(error.status, body)

    def do_GET(self):
        try:
            self._auth()
            path = urlsplit(self.path).path.rstrip("/")
            if path in {"/health", "/healthz"}:
                self._send_json(200, {"status": "ok", "model_loaded": bool(self.owner.model_info()), "active_requests": self.owner.active_requests})
            elif path == "/v1/smartai/capabilities":
                info = self.owner.model_info()
                if not info:
                    raise APIError(503, "Load a text model in Smart AI Studio first.", "server_error", "model_not_loaded")
                self._send_json(200, {"model": info.get("id", "smart-ai-studio"),
                                      "name": info.get("name"), "backend": info.get("backend"),
                                      "capabilities": info.get("capabilities", {})})
            elif path == "/v1/models":
                info = self.owner.model_info()
                data = [] if not info else [{"id": info.get("id", "smart-ai-studio"), "object": "model", "created": 0,
                                             "owned_by": "local", "display_name": info.get("name", info.get("id", "smart-ai-studio")), "type": "model"}]
                self._send_json(200, {"object": "list", "data": data, "has_more": False, "first_id": data[0]["id"] if data else None, "last_id": data[-1]["id"] if data else None})
            else:
                raise APIError(404, "Unknown endpoint.", "not_found_error")
        except (BrokenPipeError, ConnectionError, socket.timeout):
            pass
        except Exception as exc:
            self._error(exc)

    def _body(self):
        if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Encoding") not in (None, "identity"):
            raise APIError(400, "Use an uncompressed JSON body with Content-Length.")
        values = self.headers.get_all("Content-Length", [])
        if len(values) != 1:
            raise APIError(411, "Provide one Content-Length header.")
        try:
            length = int(values[0])
        except ValueError:
            raise APIError(400, "Invalid Content-Length.") from None
        if length < 1 or length > self.owner.max_body_bytes:
            raise APIError(413, "The request body exceeds the configured limit or is empty.")
        if self.headers.get_content_type() != "application/json":
            raise APIError(415, "Use Content-Type: application/json.")
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise APIError(400, "Incomplete request body.")
        try:
            return json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON number")))
        except (UnicodeError, ValueError):
            raise APIError(400, "The request body is not valid UTF-8 JSON.") from None

    def _sse(self, event, data):
        value = data if isinstance(data, str) else _json(data)
        prefix = f"event: {event}\n" if event else ""
        self.wfile.write((prefix + "data: " + value + "\n\n").encode("utf-8"))
        self.wfile.flush()

    def _disconnected(self):
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            return True

    def do_POST(self):
        path = urlsplit(self.path).path.rstrip("/")
        protocol = {"/v1/chat/completions": "chat", "/v1/responses": "responses",
                    "/v1/messages": "anthropic", "/v1/messages/count_tokens": "anthropic"}.get(path, "chat")
        job = None
        try:
            self._auth()
            if path not in {"/v1/chat/completions", "/v1/responses", "/v1/messages", "/v1/messages/count_tokens"}:
                raise APIError(404, "Unknown endpoint.", "not_found_error")
            body = self._body()
            info = self.owner.model_info()
            if not info:
                raise APIError(503, "Load a text model in Smart AI Studio first.", "server_error", "model_not_loaded")
            if not isinstance(body, dict):
                raise APIError(400, "The JSON body must be an object.")
            model = body.get("model") or info.get("id", "smart-ai-studio")
            if not isinstance(model, str) or model not in {info.get("id", "smart-ai-studio"), *info.get("aliases", [])}:
                raise APIError(404, "That model is not loaded. Use a model ID from /v1/models.", "not_found_error", "model_not_found")
            request = normalize_request(body, protocol, model)
            if path.endswith("/count_tokens"):
                if not callable(self.owner.count_tokens):
                    raise APIError(501, "Exact token counting is unavailable for the loaded backend.", "api_error")
                count = self.owner._count(request)
                if type(count) is not int or count < 0:
                    raise APIError(501, "Exact token counting is unavailable for the loaded backend.", "api_error")
                self._send_json(200, {"input_tokens": count})
                return
            job = self.owner._begin(request)
            stream = body.get("stream", False)
            if stream:
                self._headers(200, "text/event-stream; charset=utf-8")
                self._stream_started = True
            reply = _Reply(request, self._sse if stream else None)
            self._reply = reply
            reply.begin()
            parser, total_bytes, ping_at = _OutputParser(request), 0, time.monotonic()
            def consume(events):
                for kind, value in events:
                    if kind == "tool":
                        reply.add_tool(value)
                    elif kind == "thought":
                        reply.add_reasoning(value)
                    else:
                        reply.add_text(value)
            while True:
                if self._disconnected():
                    job.cancel.set()
                    return
                if self.owner._stopping.is_set():
                    raise APIError(503, "The API server was stopped.", "server_error", "cancelled")
                try:
                    kind, value = job.queue.get(timeout=0.1)
                except queue.Empty:
                    if job.done.is_set():
                        break
                    if stream and time.monotonic() - ping_at > 10:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        ping_at = time.monotonic()
                    continue
                if kind == "error":
                    raise value
                chunk = GenerationChunk(value) if isinstance(value, str) else value
                if not isinstance(chunk, GenerationChunk) or not isinstance(chunk.text, str):
                    raise APIError(500, "The model adapter returned an invalid stream chunk.", "server_error")
                if chunk.starts_in_thought is not None and total_bytes == 0:
                    from core.gui_thinking import ReasoningParser
                    parser.reasoning = ReasoningParser(starts_in_thought=chunk.starts_in_thought)
                    parser.prefix = "<think>" if chunk.starts_in_thought else ""
                for field in ("prompt_tokens", "completion_tokens"):
                    count = getattr(chunk, field)
                    if type(count) is int and count >= 0:
                        setattr(reply, field, count)
                if chunk.finish_reason in {"stop", "length"}:
                    reply.finish_reason = chunk.finish_reason
                total_bytes += len(chunk.text.encode("utf-8"))
                if total_bytes > self.owner.max_output_bytes:
                    raise APIError(413, "Generated output exceeded the configured server limit.", "server_error")
                consume(parser.feed(chunk.text))
                if parser.stopped:
                    job.cancel.set()
                    break
            consume(parser.feed("", final=True))
            reply.stop_sequence = parser.stop_sequence
            if parser.stopped:
                reply.finish_reason = "stop"
            reply.finish(include_usage=bool((body.get("stream_options") or {}).get("include_usage")))
            if not stream:
                self._send_json(200, reply.result())
        except (BrokenPipeError, ConnectionError, socket.timeout):
            pass
        except Exception as exc:
            try:
                self._error(exc, protocol)
            except (OSError, ValueError):
                pass
        finally:
            if job is not None:
                job.cancel.set()
