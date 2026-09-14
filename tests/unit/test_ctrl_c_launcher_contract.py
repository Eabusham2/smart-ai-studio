"""Regression coverage for clean Ctrl+C benchmark shutdown."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_master_launcher_catches_keyboard_interrupt_without_traceback():
    source = (ROOT / "master_4000_eval_suite.py").read_text(encoding="utf-8")
    assert "except KeyboardInterrupt:" in source
    assert "Ctrl+C: clean shutdown." in source
    assert "raise SystemExit(130)" in source


def test_runtime_still_saves_checkpoint_before_propagating_ctrl_c():
    source = (ROOT / "eval" / "master_4000_runtime.py").read_text(encoding="utf-8")
    assert "except KeyboardInterrupt:" in source
    assert "self.checkpoint_mgr.save_checkpoint(cache, phase, start)" in source
    assert "Ctrl+C: checkpoint saved." in source
    assert "raise" in source
