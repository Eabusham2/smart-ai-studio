from pathlib import Path

import eval.live_generation_stream as live


class DummyEval:
    _current_phase = "Phase 1: Baseline"
    _current_split = "LiveCodeBench-Hard"
    _current_item_id = "LCB_Hard_42"
    last_prompt_tokens = 123
    last_output_tokens = 7
    last_tok_per_sec = 5.5
    last_generation_seconds = 1.25


def test_live_file_contains_exact_full_prompt_and_raw_thinking(tmp_path, monkeypatch):
    log = tmp_path / "live_generation.log"
    monkeypatch.setattr(live, "LIVE_GENERATION_LOG", str(log))
    prompt = (
        "<|im_start|>system\nGemini-tested system prompt<|im_end|>\n"
        "<|im_start|>user\nFULL USER PROMPT HERE<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    live._write_live_header(DummyEval(), prompt)
    live._append_live_text("<think>2+2=4</think>\\boxed{4}")

    text = log.read_text(encoding="utf-8")
    assert prompt in text
    assert "<think>2+2=4</think>" in text
    assert "FULL FORMATTED MODEL PROMPT" in text
    assert "RAW MODEL OUTPUT — LIVE" in text
    assert "GITHUB" not in text.upper()


def test_final_snapshot_keeps_exact_output_and_measured_metrics(tmp_path, monkeypatch):
    log = tmp_path / "live_generation.log"
    monkeypatch.setattr(live, "LIVE_GENERATION_LOG", str(log))
    prompt = "SYSTEM + USER + ASSISTANT PROMPT"
    raw = "<think>x=3</think>final answer"

    live._write_final_snapshot(DummyEval(), prompt, raw)
    live._append_live_result(True)
    text = log.read_text(encoding="utf-8")

    assert prompt in text
    assert raw in text
    assert "Speed: 5.500 t/s" in text
    assert "Prompt tokens: 123" in text
    assert "Output tokens: 7" in text
    assert "Total tokens: 130" in text
    assert "RESULT: PASS" in text


def test_launcher_installs_live_stream_before_phase4_wrapper_and_routes_logger():
    src = (Path(__file__).resolve().parents[2] / "master_4000_eval_suite.py").read_text(encoding="utf-8")
    base = src.index("master_runtime.install(Master4000EvaluationEngine)")
    live_base = src.index("install_baseline_stream(master_runtime, Master4000EvaluationEngine)")
    phase4_logger = src.index("phase4_pro_rsi._append_raw_generation_log = master_runtime._append_raw_generation_log")
    phase4 = src.index("phase4_pro_rsi.install(Master4000EvaluationEngine)")
    live_phase4 = src.index("install_phase4_stream(phase4_pro_rsi)")
    assert base < live_base < phase4_logger < phase4 < live_phase4
