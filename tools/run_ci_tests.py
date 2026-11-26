"""Run a small, explicit offline release gate, or the full suite on request."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
REGRESSION = (
    "test_gui_media_playback.py",
    "test_desktop_app.py",
    "test_best_work_tournament.py",
    "test_best_work_synthesis.py",
    "test_best_work_boundary.py",
    "test_best_work_reasoning_isolation.py",
    "test_model_admission.py",
    "test_gui_model_import.py",
    "test_hf_downloader_cache.py",
    "test_media_generation.py",
    "test_bonsai_image_runtime.py",
    "test_media_orchestrator.py",
    "test_speed_memory_and_media.py",
    "test_audio_generation_preview.py",
    "test_stable_audio_runtime.py",
    "test_gui_media.py",
    "test_native_speculation.py",
    "test_native_exactness_patch.py",
    "test_native_ci_reuse.py",
    "test_bundled_native.py",
    "test_gui_lossless_speedups.py",
    "test_lossless_speed_control.py",
    "test_prompt_reuse.py",
    "test_app_updates.py",
    "test_app_update_ui.py",
    "test_app_update_packaging.py",
    "test_app_eval_integration_contract.py",
    "test_update_installer.py",
    "test_portable_runtime.py",
    "test_portable_download.py",
    "test_release_provenance.py",
    "test_learning_release_gate.py",
    "test_macos_full_installer.py",
    "test_macos_dependency_bundle.py",
    "test_ci_selection.py",
)
NATIVE = (
    "test_mlx_packed_terminal.py",
    "test_mlx_publisher_runtime_compat.py",
    "test_mlx_thread_owner.py",
    "test_mlx_runtime_lock.py",
    "test_vlm_input_memo.py",
)
NATIVE_EXTENDED = (
    "test_mlx_learning_implementations.py",
    "test_query_reward_adaptation.py",
    "test_native_context_learning.py",
    "test_mlx_context_memory.py",
    "test_mlx_adapter_persistence.py",
    "test_mlx_speed_tuning.py",
    "test_memory_routing.py",
    "test_online_task_extensions.py",
    "test_verifier_large_program.py",
    "test_consolidation_memory_lifecycle.py",
    "test_mlx_memory_guard.py",
    "test_exact_context_usage.py",
)


def selected_tests(suite, full=False):
    if suite == "regression" and full:
        return ["tests/unit", "tests/live"]
    names = REGRESSION if suite == "regression" else NATIVE + (NATIVE_EXTENDED if full else ())
    paths = ["tests/unit/" + name for name in names]
    if suite == "regression":
        # Keep the API/model feature gate current when its focused tests are
        # split into additional files, without collecting unrelated slow suites.
        for pattern in ("test_local_api*.py", "test_bonsai25*.py"):
            paths.extend(str(path.relative_to(ROOT)) for path in sorted((ROOT / "tests/unit").glob(pattern)))
    return list(dict.fromkeys(paths))


def verify_desktop_report(report):
    cases = list(ET.parse(report).iter("testcase"))
    names = ("test_chatbot_ui_initialization", "test_dual_model_tab_switching",
             "test_large_media_preview_expands_and_saves_original",
             "test_generated_video_plays_seeks_and_closes_decoder",
             "test_media_transcript_uses_real_preview_without_tool_dialogue")
    for name in names:
        matched = [case for case in cases if case.get("name") == name]
        if not matched or any(case.find("skipped") is not None for case in matched):
            raise RuntimeError("Desktop smoke did not execute: " + name)


def verify_source_syntax():
    paths = subprocess.check_output(["git", "ls-files", "-z", "--", "*.py"], cwd=ROOT).decode().split("\0")
    for name in filter(None, paths):
        compile((ROOT / name).read_bytes(), name, "exec")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("regression", "native"))
    parser.add_argument("--report", required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    full = args.full or os.environ.get("SMARTAI_CI_FULL_TESTS") == "1"
    verify_source_syntax()
    paths = selected_tests(args.suite, full)
    env = dict(os.environ, SMARTAI_TEST_OFFLINE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", RUN_REAL_WEIGHTS_EVAL="0")
    print(f"Running {'full' if full else 'focused'} {args.suite} gate ({len(paths)} paths)", flush=True)
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=short",
                             "--junitxml=" + args.report, *paths], cwd=ROOT, env=env)
    if result.returncode:
        return result.returncode
    if args.suite == "regression":
        verify_desktop_report(ROOT / args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
