"""Incremental reasoning markup and stable, reopenable Tk thinking controls.

Only explicit model delimiters (or a chat-template reasoning prefix) start a
thinking section. Ordinary prose is never guessed to be private reasoning.
"""
from __future__ import annotations

import re


_PAIRS = (
    ("<think>", "</think>"), ("<thinking>", "</thinking>"),
    ("<reasoning>", "</reasoning>"), ("<analysis>", "</analysis>"),
    ("<｜begin▁of▁thought｜>", "<｜end▁of▁thought｜>"),
    ("<|channel|>analysis<|message|>", "<|channel|>final<|message|>"),
)
_OPEN = {a.casefold() for a, _ in _PAIRS}
_CLOSE = {b.casefold() for _, b in _PAIRS}
_TOKENS = tuple(sorted(_OPEN | _CLOSE | {"```", "~~~", "`"}, key=len, reverse=True))
_DELIMITER = re.compile("|".join(map(re.escape, _TOKENS)), re.IGNORECASE)


class ReasoningParser:
    """Yield text/open/thought/close events without leaking split delimiters."""

    def __init__(self, starts_in_thought=False):
        self.thinking = bool(starts_in_thought)
        self._initial_open = self.thinking
        self._buffer = ""
        self._fence = None
        self._inline = False

    def feed(self, text, *, final=False):
        events = []
        if self._initial_open:
            events.append(("open", ""))
            self._initial_open = False
        self._buffer += str(text)

        def emit(value):
            if value:
                kind = "thought" if self.thinking else "text"
                if events and events[-1][0] == kind:
                    events[-1] = (kind, events[-1][1] + value)
                else:
                    events.append((kind, value))

        while self._buffer:
            lowered = self._buffer.casefold()
            match = _DELIMITER.search(self._buffer)
            if match is None:
                # Retain only a possible delimiter suffix, not a growing answer.
                pending = 0
                for length in range(1, min(len(lowered), max(map(len, _TOKENS))) + 1):
                    if any(token.startswith(lowered[-length:]) for token in _TOKENS):
                        pending = length
                if final and not (pending >= 3 and self._buffer[-pending:].startswith("<")):
                    pending = 0
                emit(self._buffer[:-pending] if pending else self._buffer)
                self._buffer = self._buffer[-pending:] if pending and not final else ""
                break
            position, token = match.start(), match.group().casefold()
            # A backtick at the end could still become a fenced-code delimiter.
            tail = self._buffer[position:].casefold()
            if not final and any(candidate.startswith(tail) and len(candidate) > len(tail)
                                 for candidate in _TOKENS):
                emit(self._buffer[:position])
                self._buffer = self._buffer[position:]
                break
            emit(self._buffer[:position])
            raw = match.group()
            self._buffer = self._buffer[match.end():]
            if token in {"```", "~~~"}:
                if self._fence == token:
                    self._fence = None
                elif self._fence is None and not self._inline:
                    self._fence = token
                emit(raw)
            elif token == "`":
                if self._fence is None:
                    self._inline = not self._inline
                emit(raw)
            elif self._fence is not None or self._inline:
                emit(raw)
            elif token in _OPEN:
                if not self.thinking:
                    self.thinking = True
                    events.append(("open", ""))
            elif token in _CLOSE:
                if self.thinking:
                    self.thinking = False
                    events.append(("close", ""))
                else:
                    events.append(("close_unopened", ""))
        if final and self.thinking:
            self.thinking = False
            events.append(("close", ""))
        return events


def reasoning_segments(text, *, starts_in_thought=False):
    """Return interleaved (is_thought, text) segments for replay and storage."""
    parser = ReasoningParser(starts_in_thought)
    events = parser.feed(text, final=True)
    controls = [kind for kind, _ in events if kind in {"open", "close_unopened"}]
    # Some tokenizers include the opening tag in the prompt, not the output.
    # The parser's code-literal protection also applies to this inference.
    if not starts_in_thought and controls and controls[0] == "close_unopened":
        events = ReasoningParser(True).feed(text, final=True)
    segments = []
    for kind, value in events:
        if kind == "open":
            segments.append((True, ""))
            continue
        if kind not in {"text", "thought"}:
            continue
        is_thought = kind == "thought"
        if segments and segments[-1][0] == is_thought:
            segments[-1] = (is_thought, segments[-1][1] + value)
        else:
            segments.append((is_thought, value))
    return segments


def starts_in_thinking(engine, info=None):
    """Inspect the selected template's assistant prefix, never its model name."""
    info = info or {}
    if "starts_in_thinking" in info:
        return bool(info["starts_in_thinking"])
    active_backend = getattr(engine, "active_backend", None)
    name = {"mlx": "mlx_backend", "controller": "controller_backend",
            "gguf": "gguf_backend", "bitnet": "bitnet_backend"}.get(active_backend)
    owner = getattr(engine, name, None) if name else (engine if active_backend in {"torch", "cuda"} else None)
    loader = getattr(owner, "_vlm_loader", None) or owner
    prepare_vlm = getattr(loader, "_mlx_prompt", None)
    if callable(prepare_vlm):
        try:
            # Bonsai 2's MLX/VLM path owns its processor template separately
            # from the tokenizer. Ask that exact text-only formatting path.
            prefix = prepare_vlm("", 0)
            if isinstance(prefix, str) and prefix.rstrip().casefold().endswith(tuple(_OPEN)):
                return True
        except Exception:
            import logging
            logging.getLogger(__name__).debug("VLM reasoning prefix unavailable", exc_info=True)
    tokenizer = getattr(owner, "tokenizer", None)
    template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(template):
        return False
    try:
        prompt = template([{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True)
        return str(prompt).rstrip().casefold().endswith(tuple(_OPEN))
    except Exception:
        import logging
        logging.getLogger(__name__).debug("Tokenizer reasoning prefix unavailable", exc_info=True)
        return False


def _label(app, thought_id):
    meta = app.thinking_meta[thought_id]
    expanded = app.thinking_expanded[thought_id]
    arrow, action = ("▼", "Collapse") if expanded else ("▶", "Expand")
    title = "Thinking…" if meta.get("live") else f"Thought for {meta.get('duration_s', 0.0):.1f}s"
    tokens = meta.get("tokens")
    hint = f" ({tokens} tokens)" if type(tokens) is int and tokens >= 0 else ""
    return f"  {arrow} 💭 {title}{hint}{meta.get('speed_str', '')} [Click to {action}]  "


def update_thought(app, thought_id):
    """Refresh a pill without deleting its reasoning body or its click binding."""
    meta = app.thinking_meta[thought_id]
    stream = meta.get("stream", app.chat_stream)
    tag = f"tag_btn_{thought_id}"
    ranges = stream.tag_ranges(tag)
    if ranges:
        start = stream.index(ranges[0])
        stream.delete(ranges[0], ranges[1])
        stream.insert(start, _label(app, thought_id), ("think_dropdown_btn", tag))
    stream.tag_configure(f"tag_body_{thought_id}", elide=not app.thinking_expanded[thought_id])


def add_thought(app, content="", *, duration_s=0.0, live=False, thought_id=None, expanded=None):
    stream = app.chat_stream
    if thought_id is None:
        app._think_counter += 1
        thought_id = f"think_{app._think_counter}"
    app.thinking_cache[thought_id] = content
    app.thinking_expanded[thought_id] = bool(live) if expanded is None else bool(expanded)
    app.thinking_meta[thought_id] = {"duration_s": float(duration_s), "tokens": None,
                                   "speed_str": "", "live": bool(live), "stream": stream}
    tag, body = f"tag_btn_{thought_id}", f"tag_body_{thought_id}"
    stream.insert("end", _label(app, thought_id), ("think_dropdown_btn", tag))
    stream.insert("end", "\n")
    stream.insert("end", content + "\n", ("think_body", body))
    if live:
        mark = f"thought_end_{thought_id}"
        stream.mark_set(mark, "end-2c")
        stream.mark_gravity(mark, "right")
    stream.insert("end", "\n")
    stream.tag_configure(body, elide=not app.thinking_expanded[thought_id])
    stream.tag_bind(tag, "<Button-1>", lambda event, tid=thought_id: app._on_toggle_thinking_dropdown(tid))
    return thought_id


class ThinkingStream:
    """Tk-thread renderer; new thoughts expand, close once, then obey the user."""

    def __init__(self, app, *, starts_in_thought=False):
        self.app = app
        self.parser = ReasoningParser(starts_in_thought)
        self.ids = []
        self.active_id = None

    def feed(self, text, elapsed_s=0.0, *, final=False):
        app, stream = self.app, self.app.chat_stream
        for kind, value in self.parser.feed(text, final=final):
            if kind == "open":
                self.active_id = add_thought(app, live=True)
                self.ids.append(self.active_id)
            elif kind == "thought":
                tid = self.active_id
                app.thinking_cache[tid] += value
                stream.insert(f"thought_end_{tid}", value, ("think_body", f"tag_body_{tid}"))
            elif kind == "close" and self.active_id is not None:
                tid = self.active_id
                app.thinking_meta[tid].update(live=False, duration_s=float(elapsed_s))
                app.thinking_expanded[tid] = False
                update_thought(app, tid)
                stream.mark_unset(f"thought_end_{tid}")
                self.active_id = None
            elif kind == "text":
                stream.insert("end", value, "ai_msg")


def transcript_payload(segments):
    """Keep display reasoning outside the answer passed back to the model."""
    content, thoughts = "", []
    for is_thought, value in segments:
        if is_thought:
            thoughts.append({"offset": len(content), "text": value})
        else:
            content += value
    result = {"role": "assistant", "content": content}
    if thoughts:
        result["display_thoughts"] = thoughts
    return result


def transcript_segments(turn):
    text = str(turn.get("content", ""))
    thoughts = turn.get("display_thoughts")
    if not thoughts:
        return reasoning_segments(text)
    result, cursor = [], 0
    for thought in thoughts:
        offset = max(cursor, min(len(text), int(thought["offset"])))
        result.extend(((False, text[cursor:offset]), (True, str(thought["text"]))))
        cursor = offset
    result.append((False, text[cursor:]))
    return result


def render_segments(app, segments, *, duration_s=0.0, existing_ids=(), open_canvas=False):
    """Final/replayed markup shares the same stable thinking-button behavior."""
    available = list(existing_ids)
    stream = app.chat_stream
    for is_thought, value in segments:
        if is_thought:
            # A tokenizer may omit its opening delimiter; the final parser can
            # recognize that earlier implicit thought only when a close arrives.
            # Match existing pills by content, not an offset into a changed list.
            tid = next((candidate for candidate in available if app.thinking_cache.get(candidate) == value), None)
            if tid is not None:
                available.remove(tid)
            expanded = app.thinking_expanded.get(tid, False) if tid else False
            duration = app.thinking_meta.get(tid, {}).get("duration_s", duration_s) if tid else duration_s
            add_thought(app, value, duration_s=duration, thought_id=tid, expanded=expanded)
            continue
        for index, part in enumerate(value.split("```")):
            if index % 2 == 0:
                app._render_styled_markdown(part)
                continue
            lines = part.split("\n", 1)
            lang = lines[0].strip() if lines else "python"
            content = lines[1] if len(lines) > 1 else part
            app._code_counter += 1
            code_id = f"code_{app._code_counter}"
            app.code_snippets[code_id] = content.strip()
            copy_tag, canvas_tag = f"tag_cp_{code_id}", f"tag_cv_{code_id}"
            stream.insert("end", f"  📄 {lang or 'python'}  ", "code_hdr")
            stream.insert("end", " [ 📋 Copy ] ", ("code_action_copy", copy_tag))
            stream.insert("end", " [ 🎨 Open in Canvas ] \n", ("code_action_canvas", canvas_tag))
            stream.insert("end", content.rstrip(), "code_block")
            stream.insert("end", "\n\n")
            stream.tag_bind(copy_tag, "<Button-1>", lambda event, cid=code_id: app._on_copy_code_snippet(cid))
            stream.tag_bind(canvas_tag, "<Button-1>", lambda event, cid=code_id: app._on_open_snippet_in_canvas(cid))
            if open_canvas and len(content.splitlines()) > 4 and hasattr(app, "txt_canvas"):
                app._open_in_canvas(content.strip())
    for tid in available:
        for cache in (app.thinking_cache, app.thinking_meta, app.thinking_expanded):
            cache.pop(tid, None)
        stream.tag_delete(f"tag_btn_{tid}", f"tag_body_{tid}")
        stream.mark_unset(f"thought_end_{tid}")


def clear_display_state(app):
    """Release per-view widgets/tags when changing or clearing a transcript."""
    stream = app.chat_stream
    for child in stream.winfo_children():
        child.destroy()
    for name in ("thinking_cache", "thinking_expanded", "thinking_meta", "code_snippets", "_generation_visuals"):
        getattr(app, name, {}).clear()
    for tag in stream.tag_names():
        if tag.startswith(("tag_btn_", "tag_body_", "tag_cp_", "tag_cv_", "think_body_")):
            stream.tag_delete(tag)
    for mark in stream.mark_names():
        if mark.startswith(("thought_end_", "stream_ai_")):
            stream.mark_unset(mark)
