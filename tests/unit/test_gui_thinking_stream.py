"""Incremental parser, real Tk thought controls, scroll ownership and replay."""
from types import SimpleNamespace

import pytest

from core.gui_thinking import (
    ReasoningParser, ThinkingStream, clear_display_state, reasoning_segments,
    render_segments, starts_in_thinking, transcript_payload, transcript_segments,
)


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


def test_reasoning_prefix_comes_from_selected_native_and_vlm_templates():
    tokenizer = SimpleNamespace(apply_chat_template=lambda *_a, **_k: "assistant\n<think>\n")
    engine = SimpleNamespace(active_backend="mlx", mlx_backend=SimpleNamespace(tokenizer=tokenizer))
    assert starts_in_thinking(engine)
    tokenizer.apply_chat_template = lambda *_a, **_k: "assistant\n"
    assert not starts_in_thinking(engine)
    engine.mlx_backend._vlm_loader = SimpleNamespace(_mlx_prompt=lambda text, images: "assistant<think>\n")
    assert starts_in_thinking(engine)
    assert not starts_in_thinking(engine, {"starts_in_thinking": False})
    # Another loaded/previous backend must not influence the selected model.
    engine.active_backend = "gguf"
    engine.gguf_backend = SimpleNamespace()
    engine.tokenizer = SimpleNamespace(apply_chat_template=lambda *_a, **_k: "<think>")
    assert not starts_in_thinking(engine)


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
