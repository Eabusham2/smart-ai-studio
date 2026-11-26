import io
import json
import os
from pathlib import Path
from unittest.mock import patch

from build_bonsai2_bundled_zip import _hf_hub_url, _repo_files


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
    assert "REMOTE_RANGE_BYTES" in bundler
    assert 'headers["Range"]' in bundler
    assert '"Content-Range"' in bundler
    assert "while expected_size <= 0 or written < expected_size:" in bundler
    assert "_validate_range_response(response, range_start, expected_size)" in bundler


def test_app_selects_bonsai25_and_retains_legacy_bundle_lookup_support():
    from config.bonsai_models import BONSAI_MLX_REPO, CRACK_MLX_REPO, bonsai_model_info
    app = _src("app_gui.py")
    paths = _src("config/paths.py")
    cache = _src("core/hf_downloader.py")

    assert "get_bundled_model_path" in app
    assert "bonsai_model_info(is_apple_silicon=is_apple_silicon)" in app
    assert "bonsai_model_info(crack=True, is_apple_silicon=is_apple_silicon)" in app
    assert bonsai_model_info()["repo_id"] == BONSAI_MLX_REPO
    assert bonsai_model_info(crack=True)["repo_id"] == CRACK_MLX_REPO
    assert '"preloaded_models"' in paths
    assert "if get_bundled_model_path(repo_id):" in cache

    eval_runtime = _src("run_studio_complete.py")
    assert "from config.paths import get_bundled_model_path" in eval_runtime
    assert f'"{BONSAI_MLX_REPO}"' in eval_runtime
    assert "get_bundled_model_path(" in eval_runtime


def test_release_workflow_streams_oversized_macos_bundle():
    workflow = _src(".github/workflows/ci-build-release.yml")
    build = _src("build_app.py")

    assert 'SMARTAI_MODEL_BUNDLE_RELEASE: "1"' in workflow
    assert "python build_bonsai2_bundled_zip.py" in workflow
    assert '--archive-name SmartAI-macOS-arm64.zip' in workflow
    assert 'RELEASE_TAG: ${{ steps.release_info.outputs.tag_name }}' in workflow
    assert '--upload-tag "$RELEASE_TAG"' in workflow
    assert "--no-models" in workflow
    assert "HF_TOKEN:" not in workflow
    assert "timeout-minutes: 45" in workflow
    assert 'bundle_release = os.getenv("SMARTAI_MODEL_BUNDLE_RELEASE", "0") == "1"' in build

    # The giant logical zip is streamed to release parts instead of being uploaded
    # as one impossible >2 GiB GitHub release asset.
    release_tail = workflow.split("- name: Publish or Update GitHub Release", 1)[1]
    assert "dist/SmartAI-macOS-arm64.zip" not in release_tail


def test_release_bundler_avoids_third_party_install_with_tokens():
    workflow = _src(".github/workflows/ci-build-release.yml")
    bundler = _src("build_bonsai2_bundled_zip.py")
    release_job = workflow.split("  build-and-release:", 1)[1]

    assert "pip install" not in release_job
    assert "huggingface_hub" not in bundler


def test_release_hf_metadata_preserves_authenticated_snapshot_downloads():
    response = io.BytesIO(
        json.dumps(
            {
                "sha": "a" * 40,
                "siblings": [
                    {"rfilename": "config.json", "size": 42},
                    {"rfilename": "weights/model weights.safetensors", "size": 1024},
                ],
            }
        ).encode("utf-8")
    )
    with (
        patch.dict(os.environ, {"HF_TOKEN": "test-token"}),
        patch(
            "build_bonsai2_bundled_zip.urllib.request.urlopen",
            return_value=response,
        ) as urlopen,
    ):
        revision, files = _repo_files("owner/model")

    request = urlopen.call_args.args[0]
    assert request.full_url == "https://huggingface.co/api/models/owner/model?blobs=true"
    assert request.get_header("Authorization") == "Bearer test-token"
    assert revision == "a" * 40
    assert files == [
        {"name": "config.json", "size": 42},
        {"name": "weights/model weights.safetensors", "size": 1024},
    ]
    assert _hf_hub_url(
        "owner/model",
        "weights/model weights.safetensors",
        "refs/pr/1",
    ) == (
        "https://huggingface.co/owner/model/resolve/"
        "refs%2Fpr%2F1/weights/model%20weights.safetensors"
    )
