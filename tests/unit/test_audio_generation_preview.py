"""Real waveform output and pinned audio loading, without downloading weights."""
import json
import sys
import types

import numpy as np

from core.audio_engine import AudioGenerationEngine


def test_completed_waveform_preview_comes_from_saved_audio(tmp_path):
    engine = AudioGenerationEngine()
    waveform = np.sin(np.arange(8000, dtype=np.float32) / 13.0) * 0.75
    engine.backend = "transformers_text_to_audio"
    engine.model = lambda prompt: {"audio": waveform[None, :], "sampling_rate": 8000}
    previews = []
    output = tmp_path / "generated.wav"
    result = engine.generate("a tone", str(output), preview_callback=previews.append)
    assert result["status"] == "success" and output.stat().st_size > 100
    assert len(previews) == 1
    preview = previews[0]
    indices = np.linspace(0, len(waveform) - 1, 2048, dtype=int)
    np.testing.assert_allclose(preview["samples"], waveform[indices])
    assert preview["kind"] == "audio" and preview["sample_rate"] == 8000
    assert preview["duration_seconds"] == 1.0
    assert preview["label"] == "Completed audio waveform"


def test_mlx_audio_previews_only_real_returned_chunks(tmp_path):
    engine = AudioGenerationEngine()
    chunks = [np.full((400, 2), 0.25, dtype=np.float32), np.full((800, 2), -0.5, dtype=np.float32)]
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        for chunk in chunks:
            yield types.SimpleNamespace(audio=chunk, sample_rate=8000)
    engine.backend = "mlx_audio"
    engine.model_info = {"audio_variant": "small-music"}
    engine.model = types.SimpleNamespace(generate=generate, sample_rate=8000)
    previews = []
    result = engine.generate("music", str(tmp_path / "music.wav"), duration_seconds=0.15,
                             steps=8, cfg_scale=1.0, preview_callback=previews.append)
    assert result["status"] == "success"
    assert calls[0] == {"text": "music", "duration": 0.15, "steps": 8, "cfg_scale": 1.0}
    assert len(previews) == 2
    assert set(previews[0]["samples"]) == {0.25}
    assert set(previews[1]["samples"]) == {-0.5}
    import wave
    with wave.open(result["path"], "rb") as saved:
        assert saved.getnframes() == 1200 and saved.getnchannels() == 2
        assert saved.getframerate() == 8000


def test_invalid_audio_does_not_produce_a_success_or_fake_preview(tmp_path):
    engine = AudioGenerationEngine()
    engine.backend = "transformers_text_to_audio"
    engine.model = lambda prompt: {"audio": np.array([float("nan")]), "sampling_rate": 8000}
    previews = []
    result = engine.generate("broken", str(tmp_path / "invalid.wav"), preview_callback=previews.append)
    assert result["status"] == "error" and "non-finite" in result["error"]
    assert previews == [] and not (tmp_path / "invalid.wav").exists()


def test_unsupported_audio_guard_prevents_download_and_keeps_current_model(monkeypatch):
    engine = AudioGenerationEngine()
    current = engine.model = object()
    monkeypatch.setattr("core.audio_engine.snapshot_download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("download")))
    result = engine.load_model({"repo_id": "owner/audio", "download_unavailable_reason": "Missing publisher runtime"})
    assert result == {"status": "error", "error": "Missing publisher runtime"}
    assert engine.model is current


def test_native_audio_uses_pinned_weights_and_their_local_text_conditioner(tmp_path, monkeypatch):
    root = tmp_path / "snapshot"
    conditioner = root / "t5gemma-b-b-ul2"
    conditioner.mkdir(parents=True)
    (root / "model.safetensors").write_bytes(b"test checkpoint")
    (conditioner / "model.safetensors").write_bytes(b"test conditioner")
    config = {"sample_rate": 44100, "model": {"conditioning": {"configs": [
        {"type": "t5gemma", "config": {"repo_id": "stabilityai/audio", "subfolder": "t5gemma-b-b-ul2"}},
    ]}}}
    (root / "model_config.json").write_text(json.dumps(config))
    transfers, loads = [], []
    def download(repo, **kwargs):
        transfers.append((repo, kwargs))
        return str(root)
    def load_diffusion(model_config, checkpoint, **kwargs):
        loads.append((model_config, checkpoint, kwargs))
        return types.SimpleNamespace()
    class Model:
        def __init__(self, model, model_config, device, half):
            self.model_config = model_config
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("The unpinned named-variant loader must not run")
    monkeypatch.setitem(sys.modules, "stable_audio_3", types.SimpleNamespace(StableAudioModel=Model))
    monkeypatch.setitem(sys.modules, "stable_audio_3.loading_utils", types.SimpleNamespace(load_diffusion_cond=load_diffusion))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))))
    monkeypatch.setattr("core.audio_engine.snapshot_download", download)
    engine = AudioGenerationEngine()
    monkeypatch.setattr(engine, "_saved_lora_files", lambda repo: [])
    required = ["model_config.json", "model.safetensors", "t5gemma-b-b-ul2/model.safetensors"]
    result = engine.load_model({"repo_id": "stabilityai/audio", "audio_backend": "stable_audio_3",
                                "audio_variant": "small-music", "repo_revision": "b" * 40,
                                "download_required_files": required, "download_allow_patterns": required})
    assert result["status"] == "loaded"
    assert transfers == [("stabilityai/audio", {"revision": "b" * 40, "allow_patterns": required})]
    loaded_config, checkpoint, options = loads[0]
    assert checkpoint == str(root / "model.safetensors")
    assert loaded_config["model"]["conditioning"]["configs"][0]["config"] == {
        "repo_id": str(root.resolve()), "subfolder": "t5gemma-b-b-ul2"}
    assert options == {"device": "cpu", "model_half": False}
    assert json.loads((root / "model_config.json").read_text()) == config
