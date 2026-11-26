"""Real media bytes, playback, saving and transcript rendering contracts."""
from __future__ import annotations

import json
import time
import tkinter as tk
import wave
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest

from core.gui_media import MediaPreview, media_artifacts, read_audio_samples, save_media_copy, waveform_points


@pytest.fixture
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("A display is required; the release gate runs this test under Xvfb")
    root.geometry("1100x850")
    yield root
    root.destroy()


def make_image(tmp_path):
    path = tmp_path / "original.png"
    values = np.zeros((768, 1024, 3), dtype=np.uint8)
    values[:, :, 0] = np.arange(1024, dtype=np.uint16) % 256
    values[:, :, 1] = np.arange(768, dtype=np.uint16)[:, None] % 256
    Image.fromarray(values).save(path)
    return path


def test_save_as_copies_original_pixels_and_keeps_source(tmp_path):
    source = make_image(tmp_path)
    target = tmp_path / "saved.png"
    save_media_copy(source, target)
    assert source.read_bytes() == target.read_bytes()
    with Image.open(target) as image:
        assert image.size == (1024, 768)
    save_media_copy(source, source)
    assert source.read_bytes() == target.read_bytes()


def test_audio_waveform_uses_real_samples_including_opposite_phase_stereo(tmp_path):
    path = tmp_path / "audio.wav"
    samples = np.array([[0, 0], [16384, -16384], [0, 0], [32767, -32767]], dtype="<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(samples.tobytes())
    actual, rate = read_audio_samples(path)
    assert rate == 8000
    assert np.allclose(actual, samples / 32768)
    points = waveform_points(actual, 12, 100)
    assert points[0][1] == points[0][3] == 50
    assert points[1][3] - points[1][1] > 39
    assert points[3][3] - points[3][1] == 80


def test_media_tool_candidates_render_only_actual_success_records():
    record = {"status": "success", "kind": "video", "path": "/real/video.mp4"}
    result = {"status": "success", "candidates": [
        {"generation": record}, {"generation": dict(record)},
        {"generation": {"status": "error", "kind": "image", "path": "/not/generated.png"}},
        {"review": {"kind": "image", "path": "/not/an/output.png"}},
    ]}
    assert media_artifacts(result) == [record]


def test_large_media_preview_expands_and_saves_original(tk_root, tmp_path, monkeypatch):
    source = make_image(tmp_path)
    viewer = MediaPreview(tk_root, source, "image", {}, bounds=(880, 620))
    viewer.pack()
    tk_root.update()
    assert viewer._photo.width() >= 800 and viewer._photo.height() == 620
    assert viewer._source_image.size == (1024, 768)
    expanded = viewer.expand()
    tk_root.update()
    child = next(item for item in expanded.winfo_children() if isinstance(item, MediaPreview))
    assert child._source_image.size == (1024, 768)
    destination = tmp_path / "saved.png"
    monkeypatch.setattr("core.gui_media.filedialog.asksaveasfilename", lambda **_: str(destination))
    viewer.save_as()
    assert destination.read_bytes() == source.read_bytes()
    expanded.destroy()
    viewer.destroy()


def test_generated_video_plays_seeks_and_closes_decoder(tk_root, tmp_path):
    import imageio.v2 as imageio
    path = tmp_path / "actual.mp4"
    frames = [np.full((64, 96, 3), value, dtype=np.uint8) for value in (15, 55, 95, 135, 175, 215)]
    imageio.mimwrite(str(path), frames, fps=10, codec="libx264", macro_block_size=1)
    viewer = MediaPreview(tk_root, path, "video", {}, bounds=(880, 620))
    viewer.pack()
    tk_root.update()
    first = np.asarray(viewer._source_image).mean()
    viewer.toggle_playback()
    deadline = time.monotonic() + 1.5
    while viewer._frame_index < 2 and time.monotonic() < deadline:
        tk_root.update()
        time.sleep(0.01)
    assert viewer._frame_index >= 2
    assert np.asarray(viewer._source_image).mean() > first + 50
    viewer.toggle_playback()
    viewer._seek("0.4")
    assert viewer._frame_index == 4
    viewer.destroy()
    assert viewer._reader is None and viewer._job is None


def test_media_transcript_uses_real_preview_without_tool_dialogue(tk_root, tmp_path):
    from app_gui import SmartAIChatbotApp
    app = SmartAIChatbotApp.__new__(SmartAIChatbotApp)
    app.root = tk_root
    app.C = {"bg_card": "#172033", "bg_chat": "#080c14", "text_muted": "#94a3b8",
             "accent_green": "#22c55e", "accent_red": "#f87171", "accent_cyan": "#38bdf8"}
    app.chat_stream = tk.Text(tk_root, width=120, height=45)
    app.chat_stream.pack(fill="both", expand=True)
    app._generation_visuals = {}
    app.active_tab_id = "media_test"
    app.chat_history = {app.active_tab_id: []}
    app._scroll_chat_to_bottom = lambda: None
    app._append_ai_message = Mock()
    app._append_rendered_ai_turn = Mock()
    tk_root.update()
    source = make_image(tmp_path)
    app._create_generation_visual("running", "image", "Image")
    state = app._generation_visuals["running"]
    assert state["active"] and state["preview"] is None
    app._update_generation_visual("running", 0.1, "Loading model")
    assert state["preview"] is None
    app._append_tool_call("media_generate", "prompt", json.dumps({
        "status": "success", "kind": "image", "path": str(source), "visual_id": "running",
    }))
    tk_root.update()
    assert not state["active"] and isinstance(state["preview"], MediaPreview)
    assert "Tool Execution" not in app.chat_stream.get("1.0", "end")
    assert len(app.chat_history[app.active_tab_id]) == 1
    assert app.chat_history[app.active_tab_id][0]["media"]["path"] == str(source)
    source.unlink()
    app._append_media_result("image", str(source), record=False)
    assert len(app.chat_history[app.active_tab_id]) == 1
    app._append_ai_message.assert_not_called()
    app._append_rendered_ai_turn.assert_called_once()
    app._clear_media_previews()
    assert not app._generation_visuals
    assert not state["frame"].winfo_exists()
