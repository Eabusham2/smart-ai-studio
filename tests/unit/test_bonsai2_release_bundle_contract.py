from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_bonsai2_release_bundle_contains_both_mlx_repositories():
    bundler = _src("build_bonsai2_bundled_zip.py")
    assert '"prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"' in bundler
    assert '"dealignai/Bonsai-2-27B-CRACK-Ternary-JANG"' in bundler
    assert "preloaded_models/" in bundler
    assert "DEFAULT_PART_BYTES = 1_900_000_000" in bundler
    assert '"gh", "release", "upload"' in bundler


def test_app_prefers_packaged_bonsai2_snapshots_when_present():
    app = _src("app_gui.py")
    paths = _src("config/paths.py")
    cache = _src("core/hf_downloader.py")

    assert "get_bundled_model_path" in app
    assert (
        'get_bundled_model_path("prism-ml/Ternary-Bonsai-2-27B-mlx-2bit") '
        "if is_apple_silicon else None"
    ) in app
    assert (
        'get_bundled_model_path("dealignai/Bonsai-2-27B-CRACK-Ternary-JANG") '
        "if is_apple_silicon else None"
    ) in app
    assert '"preloaded_models"' in paths
    assert "if get_bundled_model_path(repo_id):" in cache


def test_release_workflow_streams_oversized_macos_bundle():
    workflow = _src(".github/workflows/ci-build-release.yml")
    build = _src("build_app.py")

    assert 'SMARTAI_MODEL_BUNDLE_RELEASE: "1"' in workflow
    assert "python build_bonsai2_bundled_zip.py" in workflow
    assert '--archive-name SmartAI-macOS-arm64.zip' in workflow
    assert '--upload-tag "${{ steps.release_info.outputs.tag_name }}"' in workflow
    assert "timeout-minutes: 180" in workflow
    assert 'bundle_release = os.getenv("SMARTAI_MODEL_BUNDLE_RELEASE", "0") == "1"' in build

    # The giant logical zip is streamed to release parts instead of being uploaded
    # as one impossible >2 GiB GitHub release asset.
    release_tail = workflow.split("- name: Publish or Update GitHub Release", 1)[1]
    assert "dist/SmartAI-macOS-arm64.zip" not in release_tail
