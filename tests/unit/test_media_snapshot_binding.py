"""Media generation remains bound to the snapshot accepted during admission."""
from pathlib import Path
import sys
import types

from core.media_engine import MediaGenerationEngine


REVISION = "a" * 40


def admitted_info(snapshot: Path, **updates):
    info = {
        "repo_id": "owner/media-model",
        "source_repo_id": "owner/media-model",
        "source_origin": "hf_fetch",
        "source_kind": "hf_fetch",
        "source_revision": REVISION,
        "repo_revision": REVISION,
        "model_path": str(snapshot),
        "download_required_files": ["model_index.json", "weights/model.safetensors"],
        "model_type": "image",
    }
    info.update(updates)
    return info


def write_snapshot(snapshot: Path):
    (snapshot / "weights").mkdir(parents=True)
    (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
    (snapshot / "weights" / "model.safetensors").write_bytes(b"weights")


def test_lazy_load_binds_admitted_path_and_revision(tmp_path):
    write_snapshot(tmp_path)
    engine = MediaGenerationEngine()

    result = engine.load_model(admitted_info(tmp_path))

    assert result["status"] == "loaded"
    assert result["model_path"] == str(tmp_path.resolve())
    assert result["revision"] == REVISION
    assert engine.active_model_path == str(tmp_path.resolve())


def test_incomplete_admitted_snapshot_fails_closed(tmp_path):
    write_snapshot(tmp_path)
    (tmp_path / "weights" / "model.safetensors").unlink()
    engine = MediaGenerationEngine()

    result = engine.load_model(admitted_info(tmp_path))

    assert result["status"] == "error"
    assert "snapshot is incomplete" in result["error"]
    assert engine.active_model_path == ""


def test_invalid_manifest_paths_fail_before_artifact_lookup(tmp_path):
    write_snapshot(tmp_path)
    for name in ("../outside.safetensors", "/absolute.safetensors", "weights\\model.safetensors"):
        info = admitted_info(tmp_path, download_required_files=[name])

        result = MediaGenerationEngine().load_model(info)

        assert result["status"] == "error"
        assert "invalid artifact path" in result["error"]


def test_admitted_identity_never_falls_back_to_repository(tmp_path, monkeypatch):
    calls = []
    hub = types.SimpleNamespace(snapshot_download=lambda **kwargs: calls.append(kwargs))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    info = admitted_info(tmp_path, model_path="")

    result = MediaGenerationEngine().load_model(info)

    assert result["status"] == "error"
    assert "admitted media snapshot is missing" in result["error"]
    assert calls == []


def test_diffusers_receives_only_local_admitted_snapshot(tmp_path, monkeypatch):
    write_snapshot(tmp_path)
    calls = []

    class Pipeline:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls.append((source, kwargs))
            return cls()

        def to(self, device):
            self.device = device

    torch = types.SimpleNamespace(float16="float16", float32="float32")
    diffusers = types.SimpleNamespace(DiffusionPipeline=Pipeline)
    learning = types.SimpleNamespace(apply_saved_media_adapter=lambda pipe, repo_id: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "core.media_learning", learning)
    engine = MediaGenerationEngine()
    monkeypatch.setattr(engine, "_torch_device", lambda: "cpu")
    info = admitted_info(tmp_path)

    model_path, revision = engine._resolve_snapshot(info)
    engine._load_pipeline(info, model_path, revision)

    assert calls == [(str(tmp_path.resolve()), {
        "torch_dtype": "float32", "low_cpu_mem_usage": True, "local_files_only": True,
    })]
    assert engine.active_revision == REVISION


def test_mflux_receives_local_snapshot_and_reports_revision(tmp_path, monkeypatch):
    write_snapshot(tmp_path)
    output = tmp_path / "result.png"
    commands = []
    run_options = []

    def run(command, **kwargs):
        commands.append(command)
        run_options.append(kwargs)
        output.write_bytes(b"png")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    learning = types.SimpleNamespace(get_saved_media_adapter=lambda repo_id: None)
    monkeypatch.setitem(sys.modules, "core.media_learning", learning)
    monkeypatch.setattr("core.media_engine.shutil.which", lambda name: "/usr/bin/mflux")
    monkeypatch.setattr("core.media_engine.subprocess.run", run)
    info = admitted_info(tmp_path, image_backend="mflux")

    result = MediaGenerationEngine().generate_image(info, "a lighthouse", str(output))

    assert commands[0][commands[0].index("--model") + 1] == str(tmp_path.resolve())
    assert run_options[0]["env"]["HF_HUB_OFFLINE"] == "1"
    assert run_options[0]["env"]["DIFFUSERS_OFFLINE"] == "1"
    assert result["status"] == "success" and result["revision"] == REVISION


def test_legacy_cached_preset_stays_offline_and_revision_bound(tmp_path, monkeypatch):
    snapshot = tmp_path / ("b" * 40)
    snapshot.mkdir()
    calls = []
    hub = types.SimpleNamespace(snapshot_download=lambda **kwargs: calls.append(kwargs) or str(snapshot))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    engine = MediaGenerationEngine()

    result = engine.load_model({"repo_id": "owner/legacy", "model_type": "video"})

    assert calls == [{"repo_id": "owner/legacy", "local_files_only": True}]
    assert result["status"] == "loaded"
    assert result["model_path"] == str(snapshot.resolve())
    assert result["revision"] == "b" * 40
