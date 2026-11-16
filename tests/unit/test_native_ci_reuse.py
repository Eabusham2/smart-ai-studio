"""Execute native artifact-reuse gates with fake GitHub data and inert binaries."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/ci-build-release.yml"
HEAD = "a" * 40
OLD = "b" * 40
BASE = "c" * 40


def workflow_step(name):
    step = WORKFLOW.read_text(encoding="utf-8").split(f"      - name: {name}\n", 1)[1]
    return step.split("\n      - name:", 1)[0].split("\n  test:", 1)[0]


def preflight_script():
    return textwrap.dedent(workflow_step("Detect reusable native compile artifacts").split("script: |\n", 1)[1])


def run_preflight(tmp_path, fault=None):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the actual GitHub Actions preflight")
    current_workflow = WORKFLOW.read_text(encoding="utf-8")
    prior_workflow = current_workflow.replace(
        "    if: needs.preflight.outputs.reuse_green != 'true' && needs.preflight.outputs.native_reuse_run_id == ''",
        "    if: needs.preflight.outputs.reuse_green != 'true'",
    )
    if fault == "command_changed":
        prior_workflow = prior_workflow.replace("--jobs 2", "--jobs 4")
    if fault == "runner_changed":
        prior_workflow = prior_workflow.replace("- os: macos-latest", "- os: macos-14")
    if fault == "job_environment_changed":
        prior_workflow = prior_workflow.replace('          HF_HUB_OFFLINE: "1"', '          HF_HUB_OFFLINE: "0"')
    if fault == "global_environment_changed":
        prior_workflow = "env:\n  CC: custom-compiler\n" + prior_workflow
    if fault == "current_defaults_added":
        current_workflow = "defaults:\n  run:\n    shell: zsh\n" + current_workflow
    sources = {}
    for ref, workflow in ((HEAD, current_workflow), (OLD, prior_workflow)):
        sources[ref] = {".github/workflows/ci-build-release.yml": workflow}
        for name in ("core/native_exactness.py", "tools/prepare_speculative_runtime.py"):
            sources[ref][name] = (ROOT / name).read_text(encoding="utf-8")
    if fault == "patcher_changed":
        sources[OLD]["core/native_exactness.py"] += "\n# different implementation\n"
    if fault == "helper_changed":
        sources[OLD]["tools/prepare_speculative_runtime.py"] += "\n# different compiler configuration\n"
    files = {
        ref: {name: {"type": "file", "encoding": "base64", "sha": hashlib.sha1(text.encode()).hexdigest(),
                     "text": text} for name, text in values.items()}
        for ref, values in sources.items()
    }
    artifacts = [
        {"name": "SmartAI-native-cpu-Linux-X64", "expired": False, "expires_at": "2999-01-01T00:00:00Z", "size_in_bytes": 100},
        {"name": "SmartAI-native-metal-macOS-ARM64", "expired": False, "expires_at": "2999-01-01T00:00:00Z", "size_in_bytes": 200},
    ]
    if fault == "expired_artifact":
        artifacts[0]["expired"] = True
    if fault == "invalid_expiry":
        artifacts[0]["expires_at"] = "not-a-date"
    if fault == "expiry_in_past":
        artifacts[0]["expires_at"] = "2000-01-01T00:00:00Z"
    if fault == "missing_artifact":
        artifacts.pop()
    if fault == "duplicate_artifact":
        artifacts.append(dict(artifacts[0]))
    jobs = [{"name": f"Compile Native Verifier ({backend}, no models)", "conclusion": "success"}
            for backend in ("cpu", "metal")]
    if fault == "failed_native_job":
        jobs[0]["conclusion"] = "failure"
    payload = {"script": preflight_script(), "files": files, "fault": fault, "artifacts": artifacts, "jobs": jobs,
               "head": HEAD, "old": OLD, "base": BASE}
    (tmp_path / "fixture.json").write_text(json.dumps(payload), encoding="utf-8")
    # A file avoids platform command-line limits for the full workflow fixtures.
    harness = r"""
const p = require('./fixture.json');
const outputs = {}, calls = [];
const context = {repo: {owner: 'owner', repo: 'repo'}, sha: p.head, runId: 42, workflow: 'CI',
  payload: {pull_request: {number: 16, head: {ref: 'feature', repo: {full_name: 'owner/repo'}}, base: {sha: p.base}}}};
const run = {id: 41, name: 'CI', path: '.github/workflows/ci-build-release.yml', head_sha: p.old, head_branch: 'feature', head_repository: {full_name: 'owner/repo'},
  pull_requests: [{number: 16, base: {sha: p.base}}]};
if (p.fault === 'other_repository') run.head_repository.full_name = 'fork/repo';
if (p.fault === 'other_workflow') run.path = '.github/workflows/other.yml';
if (p.fault === 'other_pr') run.pull_requests[0].number = 17;
if (p.fault === 'base_changed') run.pull_requests[0].base.sha = 'old-base';
const jobsMethod = () => {}, artifactsMethod = () => {};
const github = {
  paginate: async (method, args) => {
    calls.push(['paginate', args]);
    if (method === jobsMethod) return p.jobs;
    if (method === artifactsMethod) return p.artifacts;
    throw Error('Unexpected pagination');
  },
  rest: {
    repos: {
      getContent: async args => {
        calls.push(['source', args]);
        if (p.fault === 'missing_source') throw Error('Missing source blob');
        const file = p.files[args.ref][args.path];
        return {data: {...file, content: Buffer.from(file.text).toString('base64')}};
      },
      compareCommits: async args => {
        calls.push(['ancestry', args]);
        return {data: {status: p.fault === 'diverged_base' ? 'diverged' : 'ahead'}};
      },
    },
    actions: {
      listWorkflowRunsForRepo: async args => { calls.push(['runs', args]); return {data: {workflow_runs: [run]}}; },
      listJobsForWorkflowRun: jobsMethod,
      listWorkflowRunArtifacts: artifactsMethod,
    },
  },
};
const core = {setOutput: (name, value) => outputs[name] = value, notice: message => calls.push(['notice', message])};
const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
new AsyncFunction('github', 'context', 'core', p.script)(github, context, core)
  .then(() => process.stdout.write(JSON.stringify({outputs, calls})))
  .catch(error => {process.stderr.write(error.stack); process.exitCode = 1;});
"""
    script = tmp_path / "gate.js"
    script.write_text(harness, encoding="utf-8")
    result = subprocess.run([node, str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=30, check=True)
    return json.loads(result.stdout)


def test_identical_native_inputs_reuse_both_artifacts_despite_changed_job_gate(tmp_path):
    result = run_preflight(tmp_path)
    assert result["outputs"]["native_reuse_run_id"] == "41"
    assert json.loads(result["outputs"]["native_reuse_matrix"]) == {"include": [
        {"backend": "cpu", "name": "SmartAI-native-cpu-Linux-X64"},
        {"backend": "metal", "name": "SmartAI-native-metal-macOS-ARM64"},
    ]}
    assert any(method == "ancestry" and args["base"] == BASE and args["head"] == OLD
               for method, args in result["calls"])


@pytest.mark.parametrize("fault", [
    "patcher_changed", "helper_changed", "command_changed", "runner_changed", "job_environment_changed",
    "global_environment_changed", "current_defaults_added", "expired_artifact", "expiry_in_past", "invalid_expiry",
    "missing_artifact", "duplicate_artifact", "failed_native_job", "other_repository", "other_pr",
    "base_changed", "diverged_base", "missing_source", "other_workflow",
])
def test_unverifiable_native_inputs_fall_back_to_both_compilations(tmp_path, fault):
    result = run_preflight(tmp_path, fault)
    assert result["outputs"]["native_reuse_run_id"] == ""
    assert json.loads(result["outputs"]["native_reuse_matrix"]) == {"include": []}


def receipt_script():
    source = workflow_step("Verify reused binary and receipt against current source")
    return textwrap.dedent(source.split("python - <<'PY'\n", 1)[1].rsplit("          PY", 1)[0])


def run_receipt_check(tmp_path, fault=None, backend="cpu"):
    (tmp_path / "core").mkdir()
    (tmp_path / "docs").mkdir()
    patcher = tmp_path / "core/native_exactness.py"
    shutil.copyfile(ROOT / "core/native_exactness.py", patcher)
    (tmp_path / "docs/ternary-speedups.md").write_text("Current README\n", encoding="utf-8")
    runtime = tmp_path / "native-release"
    runtime.mkdir()
    binary = runtime / "llama-server"
    binary.write_bytes(b"inert binary fixture: never executed")
    (runtime / "Prism-llama-LICENSE").write_bytes(b"license fixture")
    (runtime / "README.md").write_bytes(b"old README")
    # Parse the constants exactly as production does; no neural runtime import.
    import ast
    constants = {node.targets[0].id: ast.literal_eval(node.value)
                 for node in ast.parse(patcher.read_text(encoding="utf-8")).body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in {"CONTRACT", "SOURCE_ARCHIVE_SHA256"}}
    receipt = {"schema_version": 1, "backend": backend, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
               "patcher_sha256": hashlib.sha256(patcher.read_bytes()).hexdigest(),
               "source_archive_sha256": constants["SOURCE_ARCHIVE_SHA256"], "exactness_contract": constants["CONTRACT"]}
    if fault in receipt:
        receipt[fault] = "mismatch"
    sidecar = runtime / "llama-server.smartai.json"
    sidecar.write_text(json.dumps(receipt), encoding="utf-8")
    original_receipt = sidecar.read_bytes()
    original_binary = binary.read_bytes()
    if fault == "tampered_binary":
        binary.write_bytes(b"changed bytes")
    if fault == "unexpected_file":
        (runtime / "unexpected").write_text("extra", encoding="utf-8")
    result = subprocess.run([sys.executable, "-c", receipt_script()], cwd=tmp_path,
                            env={**os.environ, "NATIVE_BACKEND": backend}, capture_output=True, text=True, timeout=10)
    return result, runtime, original_binary, original_receipt


@pytest.mark.parametrize("backend", ["cpu", "metal"])
def test_reused_native_bytes_stay_identical_and_readme_refreshes(tmp_path, backend):
    result, runtime, binary, receipt = run_receipt_check(tmp_path, backend=backend)
    assert result.returncode == 0, result.stderr
    assert (runtime / "llama-server").read_bytes() == binary
    assert (runtime / "llama-server.smartai.json").read_bytes() == receipt
    assert (runtime / "README.md").read_text(encoding="utf-8") == "Current README\n"


@pytest.mark.parametrize("fault", ["schema_version", "backend", "binary_sha256", "patcher_sha256",
                                    "source_archive_sha256", "exactness_contract", "tampered_binary", "unexpected_file"])
def test_bad_reused_receipt_or_payload_is_rejected_before_readme_refresh(tmp_path, fault):
    result, runtime, _binary, _receipt = run_receipt_check(tmp_path, fault)
    assert result.returncode != 0
    assert (runtime / "README.md").read_bytes() == b"old README"


def test_all_python_lanes_still_run_and_release_requires_verified_native_job():
    source = WORKFLOW.read_text(encoding="utf-8")
    python_job = source.split("\n  test:\n", 1)[1].split("\n  native-build:\n", 1)[0]
    assert "native_reuse" not in python_job
    assert "os: [ubuntu-latest, macos-latest, windows-latest]" in python_job
    assert "python-version: ['3.10', '3.11']" in python_job
    assert "python -m pytest -v" in python_job
    assert "needs: [preflight, test, native-build, reuse-native]" in source
    assert "(needs.native-build.result == 'success' || needs.reuse-native.result == 'success')" in source
    assert "name: ${{ matrix.name }}" in source
    assert "run-id: ${{ needs.preflight.outputs.native_reuse_run_id }}" in source
