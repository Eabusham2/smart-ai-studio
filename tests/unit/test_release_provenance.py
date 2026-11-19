"""Exercise the workflow's final provenance gate with a local fake GitHub API."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest


WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/ci-build-release.yml"
STEP_NAME = "Verify Release Assets and Update Rolling Tag"
SHA = "a" * 40


def final_script():
    source = WORKFLOW.read_text(encoding="utf-8")
    step = source.split(f"- name: {STEP_NAME}\n", 1)[1]
    assert "if: success()" in step.split("script: |", 1)[0]
    return textwrap.dedent(step.split("script: |\n", 1)[1])


def run_release_gate(tmp_path, *, tag="latest", fault=None):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the GitHub Actions JavaScript gate")
    dist = tmp_path / "dist"
    dist.mkdir()
    manifest = {"archive_name": "SmartAI-macOS-arm64.zip", "parts": [
        {"name": "SmartAI-macOS-arm64.zip.part000", "size": 9},
        {"name": "SmartAI-macOS-arm64.zip.part001", "size": 4},
    ]}
    (dist / "SmartAI-macOS-arm64.bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    # The extra Apple Silicon NoAdmin kit and independently verified SHA256 are
    # mandatory, alongside the *existing* three-platform/Setup/native packages.
    kit_name = "SmartAI-macOS-arm64-NoAdmin-Portable-Kit.zip"
    kit_payload = b"verified no-admin portable kit"
    (dist / kit_name).write_bytes(kit_payload)
    (dist / (kit_name + ".sha256")).write_text(
        f"{hashlib.sha256(kit_payload).hexdigest()}  {kit_name}\n", encoding="ascii")
    installer_name = "Install-SmartAI-macOS-Full.command"
    installer = (f'#!/bin/bash\nCOMMIT="{SHA}"\n'
                 f'KIT_SHA256="{hashlib.sha256(kit_payload).hexdigest()}"\n').encode()
    (dist / installer_name).write_bytes(installer)
    (dist / (installer_name + '.sha256')).write_text(
        f'{hashlib.sha256(installer).hexdigest()}  {installer_name}\n', encoding='ascii')
    for name in ("Reassemble-SmartAI-macOS-arm64.command", "SmartAI-Windows.zip",
                 "SmartAI-Linux-x86_64.tar.gz", "SmartAI-Unix-POSIX.tar.gz",
                 "SmartAI-native-cpu-Linux-X64.zip", "SmartAI-native-metal-macOS-ARM64.zip"):
        (dist / name).write_bytes(b"built package")
    for target in ("Linux-x86_64", "Windows-x86_64", "macOS-arm64"):
        files=[]
        for suffix in ("portable.zip", "Setup.zip"):
            path=dist/f"SmartAI-{target}-{suffix}"
            path.write_bytes(b"native-tested-package")
            files.append({"name":path.name,"size":path.stat().st_size})
        (dist/f"SmartAI-{target}.receipt.json").write_text(json.dumps({
            "commit":SHA,"gui_smoke":"passed","setup_smoke":"passed","dmg_smoke":"passed","files":files}),encoding="utf-8")
    assets = [*manifest["parts"], *({"name": path.name, "size": path.stat().st_size} for path in dist.iterdir())]
    for index, asset in enumerate(assets):
        asset.update(id=index + 1, state="uploaded")
    assets += [
        {"id": 100, "name": "SmartAI-macOS-arm64.zip", "size": 2_260_000, "state": "uploaded"},
        {"id": 101, "name": "SmartAI-macOS-arm64.zip.backup", "size": 123, "state": "uploaded"},
    ]
    if fault == "missing_part":
        assets = [item for item in assets if item["name"] != manifest["parts"][1]["name"]]
    elif fault == "missing_package":
        assets = [item for item in assets if item["name"] != "SmartAI-Windows-x86_64-portable.zip"]
    elif fault == "missing_native":
        assets = [item for item in assets if item["name"] != "SmartAI-native-cpu-Linux-X64.zip"]
    elif fault == "missing_noadmin_kit":
        assets = [item for item in assets if item["name"] != "SmartAI-macOS-arm64-NoAdmin-Portable-Kit.zip"]
    elif fault == "missing_noadmin_receipt":
        assets = [item for item in assets if item["name"] != "SmartAI-macOS-arm64-NoAdmin-Portable-Kit.zip.sha256"]
    elif fault == "bad_noadmin_receipt":
        (dist / "SmartAI-macOS-arm64-NoAdmin-Portable-Kit.zip.sha256").write_text(
            "0" * 64 + "  SmartAI-macOS-arm64-NoAdmin-Portable-Kit.zip\n", encoding="ascii")
    elif fault == "missing_full_installer":
        assets = [item for item in assets if item["name"] != installer_name]
    elif fault == "missing_full_receipt":
        assets = [item for item in assets if item["name"] != installer_name + '.sha256']
    elif fault in {"bad_full_receipt", "wrong_full_commit", "wrong_full_kit"}:
        if fault == "bad_full_receipt":
            (dist / (installer_name + '.sha256')).write_text('0' * 64 + f'  {installer_name}\n')
        else:
            bad = installer.replace(SHA.encode(), b'b' * 40) if fault == "wrong_full_commit" else installer.replace(
                hashlib.sha256(kit_payload).hexdigest().encode(), b'0' * 64)
            (dist / installer_name).write_bytes(bad)
            (dist / (installer_name + '.sha256')).write_text(
                f'{hashlib.sha256(bad).hexdigest()}  {installer_name}\n')
    elif fault == "wrong_size":
        assets[0]["size"] += 1
    elif fault == "upload_in_progress":
        assets[0]["state"] = "starter"
    elif fault == "duplicate_part":
        manifest["parts"].append(dict(manifest["parts"][0]))
        path = dist / "SmartAI-macOS-arm64.bundle.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        next(item for item in assets if item["name"] == path.name)["size"] = path.stat().st_size

    harness = """
const script = SCRIPT;
const assets = ASSETS;
const fault = FAULT;
const context = {repo: {owner: 'test-owner', repo: 'test-repo'}, sha: SHA};
const calls = [];
const github = {
  paginate: async (method, args) => { calls.push(['assets', args]); return assets; },
  rest: {
    repos: {
      getReleaseByTag: async args => { calls.push(['release', args]); return {data: {id: 42}}; },
      listReleaseAssets: function () {},
      deleteReleaseAsset: async args => {
        calls.push(['delete', args]);
        if (fault === 'delete_failed') throw new Error('Cannot delete obsolete asset');
      },
    },
    git: {
      updateRef: async args => { calls.push(['update', args]); },
      getRef: async args => {
        calls.push(['getRef', args]);
        return {data: {object: {type: 'commit', sha: fault === 'tag_mismatch' ? 'old' : context.sha}}};
      },
    },
  },
};
const core = {notice: message => calls.push(['notice', message])};
const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
new AsyncFunction('github', 'context', 'core', 'require', script)(github, context, core, require)
  .then(() => process.stdout.write(JSON.stringify({calls, error: null})))
  .catch(error => process.stdout.write(JSON.stringify({calls, error: error.message})));
"""
    # Replace exact, unique harness markers only. Broad `.replace("SHA", ..)`
    # previously corrupted the release gate's literal "SHA256" error text.
    harness = (harness.replace("SCRIPT", json.dumps(final_script()))
               .replace("ASSETS", json.dumps(assets)).replace("FAULT", json.dumps(fault))
               .replace("const context = {repo: {owner: 'test-owner', repo: 'test-repo'}, sha: SHA};",
                        "const context = {repo: {owner: 'test-owner', repo: 'test-repo'}, sha: " + json.dumps(SHA) + "};"))
    result = subprocess.run(
        [node, "-e", harness], cwd=tmp_path,
        # The first Node process on a hosted Windows runner can spend more than
        # ten seconds starting. Keep a bounded process timeout without turning
        # that startup latency into a false failure of the release assertions.
        env={**os.environ, "RELEASE_TAG": tag}, capture_output=True, text=True, timeout=30, check=True,
    )
    parsed = json.loads(result.stdout)
    parsed['assets'] = assets
    return parsed


def test_successful_release_removes_only_obsolete_packages_and_moves_exact_rolling_commit(tmp_path):
    result = run_release_gate(tmp_path)
    assert result["error"] is None
    calls = result["calls"]
    deleted = {args["asset_id"] for method, args in calls if method == "delete"}
    assert {asset['name'] for asset in result['assets'] if asset['id'] in deleted} == {
        'SmartAI-macOS-arm64.zip', 'SmartAI-Windows.zip', 'SmartAI-Unix-POSIX.tar.gz',
        'SmartAI-Linux-x86_64.tar.gz',
    }
    assert 101 not in deleted  # An unrelated named backup must be preserved.
    assert [args for method, args in calls if method == "update"] == [{
        "owner": "test-owner", "repo": "test-repo", "ref": "tags/latest", "sha": SHA, "force": True,
    }]
    order = [method for method, _args in calls]
    assert order.index("assets") < order.index("delete") < order.index("update") < order.index("getRef")


@pytest.mark.parametrize("fault", ["missing_part", "missing_package", "missing_native", "missing_noadmin_kit", "missing_noadmin_receipt", "bad_noadmin_receipt", "missing_full_installer", "missing_full_receipt", "bad_full_receipt", "wrong_full_commit", "wrong_full_kit", "wrong_size", "upload_in_progress", "duplicate_part"])
def test_incomplete_replacement_neither_deletes_legacy_download_nor_moves_tag(tmp_path, fault):
    result = run_release_gate(tmp_path, fault=fault)
    assert result["error"]
    assert not any(method in {"delete", "update", "getRef"} for method, _ in result["calls"])


def test_failed_legacy_cleanup_does_not_move_rolling_tag(tmp_path):
    result = run_release_gate(tmp_path, fault="delete_failed")
    assert "Cannot delete" in result["error"]
    assert not any(method == "update" for method, _ in result["calls"])


def test_immutable_version_tag_is_never_rewritten(tmp_path):
    result = run_release_gate(tmp_path, tag="v1.2.3")
    assert result["error"] is None
    assert not any(method in {"update", "getRef"} for method, _ in result["calls"])


def test_tag_readback_must_confirm_the_exact_packaged_sha(tmp_path):
    result = run_release_gate(tmp_path, fault="tag_mismatch")
    assert "does not point to the packaged commit" in result["error"]
    assert not any(method == "notice" for method, _ in result["calls"])


def test_provenance_gate_runs_after_all_uploads_and_missing_package_is_fatal():
    source = WORKFLOW.read_text(encoding="utf-8")
    assert source.index("- name: Stream Both Bonsai") < source.index("- name: Build no-admin Apple Silicon portable kit") < source.index("- name: Publish or Update GitHub Release")
    assert all(target in source for target in ("Linux-x86_64", "Windows-x86_64", "macOS-arm64"))
    assert "SmartAI-native-" in source
    assert "SmartAI-macOS-arm64-NoAdmin-Portable-Kit.zip" in source
    assert source.index("- name: Build standalone full Mac installer") < source.index("- name: Publish or Update GitHub Release")
    assert 'Install-SmartAI-macOS-Full.command' in source
    assert source.index("- name: Publish or Update GitHub Release") < source.index(f"- name: {STEP_NAME}")
    assert 'python tools/verify_release_packages.py dist --commit "$GITHUB_SHA"' in source
    assert ",check=True)" in source



def test_release_does_not_delete_previous_downloads_before_verification():
    source = WORKFLOW.read_text(encoding='utf-8')
    preparation = source.split('- name: Prepare GitHub Release', 1)[1].split('- name: Download native-OS', 1)[0]
    assert 'delete-asset' not in preparation
