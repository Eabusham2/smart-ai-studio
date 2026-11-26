"""Incremental parser, real Tk thought controls, scroll ownership and replay."""
from types import SimpleNamespace

import pytest

from core.gui_thinking import (
    ReasoningParser, ThinkingStream, clear_display_state, completed_display_segments, reasoning_segments,
    render_segments, starts_in_thinking, transcript_payload, transcript_segments,
)
from tests.unit.test_best_work_chat_ui import work_app


@pytest.mark.parametrize("opening,closing", [
    ("<think>", "</think>"), ("<THINK>", "</THINK>"),
    ("<thinking>", "</thinking>"), ("<reasoning>", "</reasoning>"),
    ("<analysis>", "</analysis>"),
    ("<｜begin▁of▁thought｜>", "<｜end▁of▁thought｜>"),
    ("<|channel|>analysis<|message|>", "<|channel|>final<|message|>"),
])
def test_every_split_boundary_preserves_body_and_never_emits_control_tags(opening, closing):
    raw = f"İstanbul {opening}first thought{closing}answer"
    for boundary in range(1, len(raw)):
        parser = ReasoningParser()
        events = parser.feed(raw[:boundary]) + parser.feed(raw[boundary:], final=True)
        assert "".join(value for kind, value in events if kind == "text") == "İstanbul answer"
        assert "".join(value for kind, value in events if kind == "thought") == "first thought"
        assert [kind for kind, _ in events if kind in {"open", "close"}] == ["open", "close"]


def test_character_at_a_time_code_literals_stay_literal_and_plain_prose_is_not_thought():
    raw = 'Use `<think>` literally.\n```xml\n<think>example</think>\n```\nThen <think>real</think>answer'
    parser = ReasoningParser()
    events = [event for character in raw for event in parser.feed(character)] + parser.feed("", final=True)
    assert "".join(value for kind, value in events if kind == "thought") == "real"
    assert "".join(value for kind, value in events if kind == "text") == raw.replace("<think>real</think>", "")
    assert reasoning_segments("a normal answer") == [(False, "a normal answer")]
    assert reasoning_segments("`</think>` is a closing tag") == [(False, "`</think>` is a closing tag")]


def test_prompt_owned_opening_and_unterminated_reasoning_are_retained():
    assert reasoning_segments("thought</think>answer") == [(True, "thought"), (False, "answer")]
    assert reasoning_segments("<think>unfinished reasoning") == [(True, "unfinished reasoning")]
    parser = ReasoningParser(True)
    assert parser.feed("") == [("open", "")]
    assert parser.feed("reason", final=True) == [("thought", "reason"), ("close", "")]
    assert reasoning_segments("<think>a</think><think>b</think>c") == [(True, "a"), (True, "b"), (False, "c")]
    assert reasoning_segments("<think>reason</thi") == [(True, "reason")]
    assert reasoning_segments("ordinary answer <") == [(False, "ordinary answer <")]


@pytest.mark.parametrize("raw", ["Hello! This is the answer.", "Use `<think>` literally.", "Code:\n```xml\n<think>example</think>\n```"])
def test_completed_reply_without_reasoning_markers_is_not_hidden_by_template_hint(raw):
    segments = completed_display_segments(raw, starts_in_thought=True)
    assert segments == [(False, raw)]
    assert transcript_payload(segments) == {"role": "assistant", "content": raw}
    # Candidate/tool parsing and interrupted reasoning retain their strict semantics.
    assert reasoning_segments(raw, starts_in_thought=True) == [(True, raw)]


def test_completed_display_preserves_real_thoughts_but_omits_empty_ones():
    assert completed_display_segments("reason</think>answer", starts_in_thought=True) == [(True, "reason"), (False, "answer")]
    assert completed_display_segments("<think>reason</think>answer") == [(True, "reason"), (False, "answer")]
    assert completed_display_segments("\n</think>answer", starts_in_thought=True) == [(False, "answer")]
    assert completed_display_segments("<think>\n</think>answer") == [(False, "answer")]
    assert completed_display_segments("<think>unfinished reasoning") == [(True, "unfinished reasoning")]


def test_completed_unmarked_reply_is_visible_and_provisional_thought_is_removed(ui):
    app, stream, _scroll = ui
    renderer = ThinkingStream(app, starts_in_thought=True)
    renderer.feed("")
    renderer.feed("Hello! ", .1)
    renderer.feed("This is the answer.", .2)
    renderer.feed("", .3, final=True)
    # Same final replacement used by the real chat request.
    stream.delete("1.0", "end")
    render_segments(app, completed_display_segments("Hello! This is the answer.", starts_in_thought=True),
                    existing_ids=renderer.ids)
    assert visible(stream) == "Hello! This is the answer."
    assert not app.thinking_cache and not app.thinking_meta and not app.thinking_expanded


def test_reasoning_prefix_comes_from_selected_native_and_vlm_templates():
    tokenizer = SimpleNamespace(apply_chat_template=lambda *_a, **_k: "assistant\n<think>\n")
    engine = SimpleNamespace(active_backend="mlx", mlx_backend=SimpleNamespace(tokenizer=tokenizer))
    assert starts_in_thinking(engine)
    tokenizer.apply_chat_template = lambda *_a, **_k: "assistant\n"
    assert not starts_in_thinking(engine)
    engine.mlx_backend._vlm_loader = SimpleNamespace(_mlx_prompt=lambda text, images: "assistant<think>\n")
    assert starts_in_thinking(engine)
    assert not starts_in_thinking(engine, {"starts_in_thinking": False})
    tokenizer.apply_chat_template = lambda *_a, **_k: "assistant\n<think>\n"
    engine.mlx_backend._vlm_loader._mlx_prompt = lambda text, images: "assistant<think>\n</think>\n"
    assert not starts_in_thinking(engine)  # actual VLM prefix overrides tokenizer default
    # Another loaded/previous backend must not influence the selected model.
    engine.active_backend = "gguf"
    engine.gguf_backend = SimpleNamespace()
    engine.tokenizer = SimpleNamespace(apply_chat_template=lambda *_a, **_k: "<think>")
    assert not starts_in_thinking(engine)


@pytest.mark.parametrize("crack", [False, True])
def test_bonsai_vlm_chat_is_formatted_once_with_reasoning_enabled(crack):
    from config.bonsai_models import bonsai_model_info
    from core.controller_runtime import UniversalControllerBackend
    from core.full_context_hardening import install

    calls = []
    class Engine:
        pass
    install(Engine)
    engine = Engine()
    info = bonsai_model_info(crack=crack)
    loader = UniversalControllerBackend("unused", info)
    loader.runtime = "mlx_repo_vlm"
    def render(messages, **kwargs):
        calls.append((messages, kwargs))
        return "packed history\nassistant\n<think>\n" if kwargs.get("enable_thinking") else "packed history\nassistant\n<think>\n</think>\n"
    def double_wrap(*args, **kwargs):
        pytest.fail("A fully templated desktop chat was wrapped a second time")
    loader.tokenizer = SimpleNamespace(apply_chat_template=render)
    loader._apply_chat_template = double_wrap
    engine.active_backend = "mlx"
    engine.mlx_backend = SimpleNamespace(tokenizer=loader.tokenizer, _vlm_loader=loader, model_info=info)

    history = [{"role": "system", "content": "Follow instructions."}, {"role": "user", "content": "Earlier question"}, {"role": "assistant", "content": "Earlier reply"}]
    packed = engine._format_prompt_with_history("Current question", history)
    assert packed.endswith("<think>\n")
    assert calls[-1] == (history + [{"role": "user", "content": "Current question"}],
                         {"tokenize": False, "add_generation_prompt": True, "enable_thinking": True})
    assert loader._mlx_prompt(packed, 0) is packed
    assert starts_in_thinking(engine)

    raw = "Reasoning is visible.</think>Final answer is visible."
    parser = ReasoningParser(starts_in_thinking(engine))
    events = [event for part in ("Reasoning is ", "visible.</thi", "nk>Final answer is visible.")
              for event in parser.feed(part)]
    assert ("thought", "Reasoning is ") in events
    assert ("close", "") in events
    assert ("text", "Final answer is visible.") in events
    assert completed_display_segments(raw, starts_in_thought=True) == [
        (True, "Reasoning is visible."), (False, "Final answer is visible.")]


def test_fenced_example_with_unbalanced_inline_backtick_remains_visible():
    raw = '```text\n`example\n<think>literal</think>\n```\n<think>real</think>answer'
    parser = ReasoningParser()
    events = [event for char in raw for event in parser.feed(char)] + parser.feed("", final=True)
    assert "".join(value for kind, value in events if kind == "thought") == "real"
    assert "<think>literal</think>" in "".join(value for kind, value in events if kind == "text")


@pytest.fixture
def ui():
    import tkinter as tk
    from app_gui import SmartAIChatbotApp
    root = tk.Tk()
    root.geometry("600x240")
    stream = tk.Text(root, wrap="word")
    stream.pack(side="left", fill="both", expand=True)
    scrollbar = tk.Scrollbar(root, command=stream.yview)
    scrollbar.pack(side="right", fill="y")
    stream.configure(yscrollcommand=scrollbar.set)
    app = SimpleNamespace(root=root, chat_stream=stream, active_tab_id="bonsai", thinking_cache={},
                          thinking_meta={}, thinking_expanded={}, code_snippets={}, _generation_visuals={},
                          _think_counter=0, _code_counter=0)
    app._on_toggle_thinking_dropdown = lambda tid: SmartAIChatbotApp._on_toggle_thinking_dropdown(app, tid)
    app._render_styled_markdown = lambda text: stream.insert("end", text)
    root.update()
    yield app, stream, scrollbar
    root.destroy()


def visible(stream):
    return str(stream.tk.call(stream._w, "get", "-displaychars", "1.0", "end-1c"))


def test_live_thought_opens_can_collapse_during_decode_then_closes_once_and_reopens(ui):
    app, stream, _scroll = ui
    renderer = ThinkingStream(app)
    renderer.feed("<thi")
    assert visible(stream) == ""
    renderer.feed("nk>first thought", .1)
    tid = renderer.ids[0]
    assert app.thinking_expanded[tid]
    assert "first thought" in visible(stream)
    assert "<think>" not in visible(stream)
    app._on_toggle_thinking_dropdown(tid)
    assert "first thought" not in visible(stream)
    stream.configure(state="normal")
    renderer.feed(" and more", .2)
    assert "and more" not in visible(stream)
    app._on_toggle_thinking_dropdown(tid)
    assert "first thought and more" in visible(stream)
    stream.configure(state="normal")
    renderer.feed("</thi", .3)
    assert app.thinking_expanded[tid]
    renderer.feed("nk>answer", .4)
    assert not app.thinking_expanded[tid]
    assert "first thought" not in visible(stream)
    assert "answer" in visible(stream)
    app._on_toggle_thinking_dropdown(tid)
    stream.configure(state="normal")
    renderer.feed(" continues", .5)
    renderer.feed("", .6, final=True)
    assert app.thinking_expanded[tid]  # never repeatedly closes an ended block
    assert "first thought and more" in visible(stream)
    # Final Markdown replacement preserves the same permanent button/state.
    stream.delete("1.0", "end")
    render_segments(app, [(True, "first thought and more"), (False, "answer continues")], existing_ids=renderer.ids)
    assert app.thinking_expanded[tid]
    assert "first thought and more" in visible(stream)
    app._on_toggle_thinking_dropdown(tid)
    assert "first thought" not in visible(stream)
    app._on_toggle_thinking_dropdown(tid)
    assert "first thought and more" in visible(stream)


def test_known_reasoning_template_starts_expanded_before_first_token_and_multiple_blocks_survive(ui):
    app, stream, _scroll = ui
    renderer = ThinkingStream(app, starts_in_thought=True)
    renderer.feed("")
    first = renderer.ids[0]
    assert app.thinking_expanded[first]
    assert "Thinking…" in visible(stream)
    renderer.feed("step one</think>first answer<analysis>step two</analysis>final answer", 1.0)
    assert len(renderer.ids) == 2
    assert all(not app.thinking_expanded[tid] for tid in renderer.ids)
    for tid in renderer.ids:
        app._on_toggle_thinking_dropdown(tid)
    assert "step one" in visible(stream) and "step two" in visible(stream)
    assert "first answer" in visible(stream) and "final answer" in visible(stream)


def test_final_implicit_thought_does_not_transfer_another_blocks_expansion(ui):
    app, stream, _scroll = ui
    raw = "implicit first</think>first answer<think>explicit second</think>last answer"
    renderer = ThinkingStream(app)
    renderer.feed(raw, 1.0)
    second = renderer.ids[0]
    app._on_toggle_thinking_dropdown(second)
    stream.configure(state="normal")
    stream.delete("1.0", "end")
    render_segments(app, reasoning_segments(raw), existing_ids=renderer.ids)
    first = next(tid for tid, value in app.thinking_cache.items() if value == "implicit first")
    assert not app.thinking_expanded[first]
    assert app.thinking_expanded[second]
    assert "explicit second" in visible(stream)
    assert "implicit first" not in visible(stream)


def test_transcript_reasoning_replays_in_its_model_and_display_caches_are_released(ui, tmp_path):
    from core.context_memory import ChatArchive
    app, stream, _scroll = ui
    normal = ChatArchive(str(tmp_path / "transcripts.sqlite"), "hybrid:normal")
    crack = ChatArchive(str(tmp_path / "transcripts.sqlite"), "hybrid:crack")
    segments = [(True, "normal model thought"), (False, "normal answer")]
    normal.append(transcript_payload(segments))
    crack.append({"role": "assistant", "content": "crack answer"})
    assert normal[0]["content"] == "normal answer"
    assert transcript_segments(normal[0]) == [(False, ""), *segments]
    render_segments(app, transcript_segments(normal[0]))
    tid = next(iter(app.thinking_cache))
    app._on_toggle_thinking_dropdown(tid)
    assert "normal model thought" in visible(stream)
    clear_display_state(app)
    stream.configure(state="normal")
    stream.delete("1.0", "end")
    render_segments(app, transcript_segments(crack[0]))
    assert visible(stream) == "crack answer"
    assert not app.thinking_cache and not app.thinking_meta
    assert not [tag for tag in stream.tag_names() if tag.startswith("tag_body_")]


def test_manual_scroll_preempts_pending_tail_callbacks_and_new_chunks_do_not_jump(ui):
    from core.gui_chat_scroll import follow_chat_tail, install_chat_scroll
    app, stream, scrollbar = ui
    install_chat_scroll(app, scrollbar)
    stream.insert("end", "".join(f"Line {number}\n" for number in range(250)))
    app.root.update()
    follow_chat_tail(app, force=True)
    app.root.update()
    assert stream.yview()[1] > .995
    follow_chat_tail(app)  # leaves a queued after_idle callback
    app.root.tk.call(scrollbar.cget("command"), "moveto", .2)
    app.root.update()
    assert not app._chat_follow_tail and stream.yview()[1] < .5
    anchor = stream.index("@0,0")
    for _ in range(10):
        stream.insert("end", "new generated text\n")
        follow_chat_tail(app)
        app.root.update()
        assert stream.index("@0,0") == anchor
    app.root.tk.call(scrollbar.cget("command"), "moveto", 1.0)
    app.root.update()
    assert app._chat_follow_tail
    stream.insert("end", "last generated line\n")
    follow_chat_tail(app)
    app.root.update()
    assert stream.yview()[1] > .995


def test_mouse_wheel_scroll_disables_follow_until_reader_returns_to_bottom(ui):
    from core.gui_chat_scroll import follow_chat_tail, install_chat_scroll
    app, stream, scrollbar = ui
    install_chat_scroll(app, scrollbar)
    stream.insert("end", "line\n" * 300)
    follow_chat_tail(app, force=True)
    app.root.update()
    stream.event_generate("<MouseWheel>", delta=360)
    app.root.update()
    assert not app._chat_follow_tail
    first = stream.index("@0,0")
    stream.insert("end", "line\n" * 20)
    follow_chat_tail(app)
    app.root.update()
    assert stream.index("@0,0") == first


@pytest.mark.parametrize("location", ["history", "plain", "markdown", "code", "reopened_thought", "repeated"])
def test_completion_reformat_preserves_the_readers_visible_content(work_app, location):
    """Finishing an answer must not relocate a reader inside the replaced text."""
    app, root = work_app, work_app.root
    root.deiconify()
    root.geometry("1100x900")
    root.update()
    stream = app.chat_stream
    stream.configure(state="normal")
    stream.delete("1.0", "end")
    stream.insert("end", "".join(f"Earlier paragraph {number:03d}: saved conversation.\n" for number in range(100)))
    stream.configure(state="disabled")
    before = []
    reopened = []
    body = "".join(f"Paragraph {number:03d}: reading this unchanged sentence.\n" for number in range(250))
    if location == "markdown":
        body = "# Heading\n" + body.replace("Paragraph ", "**Paragraph ").replace(": reading", "**: reading")
    elif location == "code":
        body = "```python\n" + body + "```\n"
    elif location == "reopened_thought":
        body = "<think>" + body + "</think>Answer after thinking.\n"
    elif location == "repeated":
        body = "Repeated paragraph with identical content.\n" * 250

    def generate(*_args, **_kwargs):
        yield body
        # Run the actual scheduled stream painter, then emulate a reader using
        # the scrollbar while generation is still active.
        root.update()
        if location == "reopened_thought":
            tid = next(iter(app.thinking_cache))
            app._on_toggle_thinking_dropdown(tid)
            reopened.append(tid)
        target = "Earlier paragraph 050" if location == "history" else "Paragraph 130"
        if location == "repeated":
            target = "Repeated paragraph with identical content."
            first = stream.search(target, "1.0", "end")
            position = stream.index(f"{first}+130lines")
        else:
            position = stream.search(target, "1.0", "end")
        assert position
        app._chat_follow_tail = False
        app._chat_scroll_epoch += 1
        stream.yview(position)
        root.update()
        before.append(stream.get("@0,0", "@0,0 lineend").replace("**", ""))
        assert target in before[0]
        yield "Final continuation."

    app.multimodal = SimpleNamespace(busy=False, handle_command=lambda _msg: None, stream_solve=generate)
    app._set_generating_state(True)
    app._process_message_thread("Example prompt", "Example prompt")
    root.update()
    assert len(before) == 1
    assert not app._chat_follow_tail
    assert stream.get("@0,0", "@0,0 lineend") == before[0]
    if reopened:
        assert app.thinking_expanded[reopened[0]]
        assert "Paragraph 130" in visible(stream)
    if location == "repeated":
        first_line = int(stream.search(before[0], "1.0", "end").split(".")[0])
        assert int(stream.index("@0,0").split(".")[0]) - first_line == 130


def test_automatic_transcript_paging_on_completion_does_not_move_a_scrolled_reader():
    from unittest.mock import Mock
    from app_gui import SmartAIChatbotApp
    app = SimpleNamespace(settings=SimpleNamespace(learning_implementation="hybrid"),
        is_generating=False, active_tab_id="normal", chat_history={"normal": [{}] * 101},
        _chat_follow_tail=False, _transcript_browsing=False, chat_stream=Mock(), _show_transcript_page=Mock())
    app.chat_stream.count.return_value = (300_000,)
    SmartAIChatbotApp._limit_rendered_transcript(app)
    app._show_transcript_page.assert_not_called()


def test_actual_gui_generation_pipeline_uses_incremental_renderer_and_keeps_reopenable_history(tmp_path):
    import tkinter as tk
    from app_gui import SmartAIChatbotApp
    from config.settings import get_settings
    root = tk.Tk()
    try:
        app = SmartAIChatbotApp(root, settings=get_settings(backend="mock", database_path=str(tmp_path / "state.db")))
        app.multimodal = SimpleNamespace(busy=False, handle_command=lambda _msg: None,
            stream_solve=lambda *_args, **_kwargs: iter(["<thi", "nk>first", " thought</thi", "nk>", "Final answer."]))
        app._set_generating_state(True)
        app._process_message_thread("Example prompt", "Example prompt")
        root.update()
        assert "<think>" not in visible(app.chat_stream) and "</thi" not in visible(app.chat_stream)
        assert "Final answer." in visible(app.chat_stream)
        tid = next(tid for tid, thought in app.thinking_cache.items() if thought == "first thought")
        assert not app.thinking_expanded[tid]
        app._on_toggle_thinking_dropdown(tid)
        assert "first thought" in visible(app.chat_stream)
        app._on_toggle_thinking_dropdown(tid)
        assert "first thought" not in visible(app.chat_stream)
        turn = app.chat_history[app.active_tab_id][-1]
        assert turn["content"] == "Final answer."
        assert turn["display_thoughts"] == [{"offset": 0, "text": "first thought"}]
        app._show_transcript_page()
        restored = next(tid for tid, thought in app.thinking_cache.items() if thought == "first thought")
        app._on_toggle_thinking_dropdown(restored)
        assert "first thought" in visible(app.chat_stream)
    finally:
        root.destroy()


def test_failed_generation_finishes_reasoning_and_keeps_its_reopenable_toggle(tmp_path):
    import tkinter as tk
    from app_gui import SmartAIChatbotApp
    from config.settings import get_settings
    root = tk.Tk()
    try:
        app = SmartAIChatbotApp(root, settings=get_settings(backend="mock", database_path=str(tmp_path / "state.db")))
        def failing_stream(*_args, **_kwargs):
            yield "<thi"
            yield "nk>reasoning before failure"
            raise RuntimeError("fixture decoder failure")
        app.multimodal = SimpleNamespace(busy=False, handle_command=lambda _msg: None,
                                        stream_solve=failing_stream)
        app._set_generating_state(True)
        app._process_message_thread("Example prompt", "Example prompt")
        root.update()
        assert not app.is_generating
        assert "fixture decoder failure" in visible(app.chat_stream)
        tid = next(tid for tid, text in app.thinking_cache.items() if text == "reasoning before failure")
        assert not app.thinking_meta[tid]["live"]
        assert not app.thinking_expanded[tid]
        assert "Thinking…" not in visible(app.chat_stream)
        app._on_toggle_thinking_dropdown(tid)
        assert "reasoning before failure" in visible(app.chat_stream)
        app._on_toggle_thinking_dropdown(tid)
        assert "reasoning before failure" not in visible(app.chat_stream)
    finally:
        root.destroy()
