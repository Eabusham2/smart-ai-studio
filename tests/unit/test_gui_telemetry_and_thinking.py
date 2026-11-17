"""No-model response metrics, live thought folding and Tk layout checks."""
import pytest

from core.gui_response_telemetry import (
    context_label, context_limit, observed_prompt_tokens, response_footer,
)


def test_response_footer_uses_measured_counts_and_rates():
    sample = {"tokens": 57, "decode_tps": 22.75, "invocations": 1, "all_counts_available": True}
    assert response_footer(sample) == "⚡ Generated tokens: 57 · 22.8 tok/s decode"
    assert "across Pro branches" in response_footer(sample, pro=True)
    partial = {"tokens": None, "decode_tps": 900, "all_counts_available": False}
    assert response_footer(partial, pro=True) == (
        "⚡ Generated tokens: unavailable · TPS unavailable · across Pro branches"
    )
    assert "unavailable" in response_footer({})


def test_context_never_fabricates_word_based_token_counts():
    assert context_label(None, 32768) == "📊 Context: measuring / 32,768"
    assert context_label(100, 32768) == "📊 Prompt: 100 / 32,768 tokens (0.3%)"
    assert context_label(100, 32768, completion_tokens=51) == (
        "📊 Last request: 151 / 32,768 tokens (0.5%)"
    )
    assert context_label(100, None) == "📊 Context: limit unavailable"


def test_context_model_and_backend_actual_budget_only():
    from types import SimpleNamespace
    mlx = SimpleNamespace(is_mlx_available=True, last_prompt_token_count=241,
                          last_model_context_limit=65536, context_budget_tokens=32768)
    engine = SimpleNamespace(active_backend="mlx", mlx_backend=mlx)
    assert context_limit({"max_context": 262144}, engine, 65536) == 32768
    # A loaded runtime's configured 65K cap must not be reduced by a stale
    # 32K startup UI guess.
    mlx.context_budget_tokens = 65536
    assert context_limit({"max_context": 262144}, engine, 32768) == 65536
    mlx.context_budget_tokens = 32768
    assert observed_prompt_tokens(engine) == 241
    mlx.last_prompt_token_count = None
    assert observed_prompt_tokens(engine) is None


def test_finished_thought_collapses_before_answer_and_click_restores():
    import tkinter as tk
    from config.settings import get_settings
    from app_gui import SmartAIChatbotApp
    root = tk.Tk()
    root.withdraw()
    app = SmartAIChatbotApp(root, settings=get_settings(backend="mock"))
    stream = app.chat_stream
    stream.configure(state="normal")
    stream.mark_set("unit_stream", "end-1c")
    stream.mark_gravity("unit_stream", "left")
    stream.insert("end", "<think>first part of reasoning</thi")
    ids = []
    app._collapse_completed_stream_thinking("unit_stream", ids, 0.4)
    assert ids == []
    stream.insert("end", "nk>The final response continues")
    app._collapse_completed_stream_thinking("unit_stream", ids, 0.5)
    assert len(ids) == 1
    assert app.thinking_expanded[ids[0]] is False
    assert "first part of reasoning" not in stream.get("unit_stream", "end-1c")
    assert "The final response continues" in stream.get("unit_stream", "end-1c")
    app._on_toggle_thinking_dropdown(ids[0])
    assert "first part of reasoning" in stream.get("unit_stream", "end-1c")
    app._on_toggle_thinking_dropdown(ids[0])
    assert "first part of reasoning" not in stream.get("unit_stream", "end-1c")
    root.destroy()


def test_top_bar_never_clips_and_aligns_consistently_during_resize():
    import tkinter as tk
    from config.settings import get_settings
    from app_gui import SmartAIChatbotApp
    root = tk.Tk()
    app = SmartAIChatbotApp(root, settings=get_settings(backend="mock"))
    root.update()
    assert root.winfo_width() <= root.winfo_screenwidth()
    for width in (1280, 960, 800, 1536, 760):
        root.geometry(f"{width}x800")
        root.update()
        for frame in app.hud_bar.pack_slaves():
            assert frame.winfo_width() <= app.hud_bar.winfo_width()
            controls = [child for child in frame.winfo_children() if child.winfo_manager() == "place"]
            assert controls
            for widget in controls:
                assert widget.winfo_x() >= 0
                assert widget.winfo_width() >= 10
                assert widget.winfo_x() + widget.winfo_width() <= frame.winfo_width() + 2
            # Every row starts at the same x, rather than justified spacing.
            leftmost = {}
            for widget in controls:
                y = widget.winfo_y()
                leftmost[y] = min(leftmost.get(y, 10**6), widget.winfo_x())
            assert set(leftmost.values()) == {3}
        assert app.chk_pro_branches.winfo_manager() == "place"
        bar = app.speed_tuning_bar
        for widget in (app.chk_auto_speed, app.btn_speed_tune, app.btn_speed_cancel,
                       app.btn_lossless_speedups):
            assert widget.winfo_x() + widget.winfo_width() <= bar.winfo_width()

    # Model names, memory badges, and load-status labels can grow without a
    # window resize. Their live requested widths must trigger reflow too.
    import time
    root.geometry("1050x800")
    root.update()
    app.model_var.set("Bonsai 2 Ternary EXTREMELY LONG IMPORTED MODEL LABEL")
    app.lbl_model_status.configure(
        text="Loaded Bonsai 2 — quantized runtime initializing extra context and media modules"
    )
    time.sleep(0.22)
    root.update()
    for widget in (app.btn_model_menu, app.lbl_model_status, app.chk_pro_branches):
        assert widget.winfo_width() >= widget.winfo_reqwidth() - 3
        assert widget.winfo_x() + widget.winfo_width() <= widget.master.winfo_width() + 2
    root.destroy()
