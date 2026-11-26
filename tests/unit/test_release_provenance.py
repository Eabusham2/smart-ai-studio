"""Run the actual publication gate against a local, stateful fake GitHub API."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import textwrap
import zipfile

import pytest

from tools.verify_release_packages import file_record, verify
from tools.build_macos_full_installer import INSTALLER, INSTALLER_ZIP, write_installer_zip

WORKFLOW = Path(__file__).resolve().parents[2] / '.github/workflows/ci-build-release.yml'
STEP_NAME = 'Verify Release Assets and Update Rolling Tag'
SHA = 'a' * 40
OBSOLETE = {
    'Install-SmartAI-macOS-Full.command',
    'Install-SmartAI-macOS-Full.command.sha256',
    'Reassemble-SmartAI-Linux-x86_64-portable.zip.ps1',
    'Reassemble-SmartAI-Linux-x86_64-portable.zip.sh',
    'Reassemble-SmartAI-Linux-x86_64.tar.gz.ps1',
    'Reassemble-SmartAI-Linux-x86_64.tar.gz.sh',
    'Reassemble-SmartAI-macOS-arm64.command',
    'SmartAI-Linux-x86_64-portable.zip.sha256',
    'SmartAI-Linux-x86_64.tar.gz.part000',
    'SmartAI-Linux-x86_64.tar.gz.part001',
    'SmartAI-Linux-x86_64.tar.gz.parts.json',
    'SmartAI-Linux-x86_64.tar.gz.sha256',
    'SmartAI-macOS-arm64-NoAdmin-Portable-Kit.zip',
    'SmartAI-macOS-arm64-NoAdmin-Portable-Kit.zip.sha256',
    'SmartAI-macOS-arm64.bundle.json',
    'SmartAI-macOS-arm64.dmg',
    'SmartAI-macOS-arm64.zip.part000',
    'SmartAI-native-cpu-Linux-X64.zip',
    'SmartAI-native-metal-macOS-ARM64.zip',
    'SmartAI-macOS-arm64.zip',
    'SmartAI-Windows.zip',
    'SmartAI-Unix-POSIX.tar.gz',
    'SmartAI-Linux-x86_64.tar.gz',
}


def final_script():
    step = WORKFLOW.read_text(encoding='utf-8').split(f'- name: {STEP_NAME}\n', 1)[1]
    assert 'if: success()' in step.split('script: |', 1)[0]
    return textwrap.dedent(step.split('script: |\n', 1)[1])


def package_fixture(root, *, split_linux=True):
    root.mkdir()
    for target in ('Linux-x86_64', 'Windows-x86_64', 'macOS-arm64'):
        files = []
        for suffix in ('portable.zip', 'Setup.zip'):
            path = root / f'SmartAI-{target}-{suffix}'
            path.write_bytes(b'verified package fixture')
            if target.startswith('Linux') and suffix == 'portable.zip' and split_linux:
                payload = path.read_bytes()
                parts = []
                for index, chunk in enumerate((payload[:10], payload[10:])):
                    part = path.with_name(path.name + f'.part{index:03d}')
                    part.write_bytes(chunk)
                    parts.append(file_record(part))
                manifest = path.with_name(path.name + '.parts.json')
                manifest.write_text(json.dumps({'archive': path.name, 'sha256': hashlib.sha256(payload).hexdigest(), 'parts': parts}))
                helper = root / 'Download-SmartAI-Linux.sh'
                helper.write_text('#!/bin/sh\necho verified-download\n')
                files += parts + [file_record(manifest), file_record(helper)]
                path.unlink()
            else:
                files.append(file_record(path))
        (root / f'SmartAI-{target}.receipt.json').write_text(json.dumps({
            'schema': 1, 'target': target, 'commit': SHA, 'gui_smoke': 'passed',
            'setup_smoke': 'passed', 'bonsai_native_smoke': 'passed',
            'bundled_native_verified': 'not_applicable' if target.startswith('Windows') else 'passed',
            'files': files,
        }))
    mac = file_record(root / 'SmartAI-macOS-arm64-portable.zip')
    write_installer_zip(
        f'#!/bin/bash\nCOMMIT="{SHA}"\nARCHIVE_NAME="{mac["name"]}"\nARCHIVE_SHA256="{mac["sha256"]}"\n', root)
    return verify(root, SHA, include_installer=True)


def run_release_gate(tmp_path, *, tag='latest', fault=None):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is required to execute the actual publication gate')
    expected = package_fixture(tmp_path / 'dist')
    manifest = {'schema': 1, 'commit': SHA, 'assets': expected}
    assets = [{**row, 'digest': 'sha256:' + row['sha256'], 'state': 'uploaded'} for row in expected]
    assets += [{'name': name, 'size': 123, 'state': 'uploaded'} for name in sorted(OBSOLETE)]
    assets += [{'name': 'SmartAI-macOS-arm64.zip.backup', 'size': 123, 'state': 'uploaded'},
               {'name': 'manual-example.txt', 'size': 2, 'state': 'uploaded'}]
    for index, asset in enumerate(assets):
        asset['id'] = index + 1
    if fault in {'missing_part', 'missing_package', 'missing_installer'}:
        missing = {'missing_part': 'SmartAI-Linux-x86_64-portable.zip.part001',
                   'missing_package': 'SmartAI-Windows-x86_64-portable.zip',
                   'missing_installer': INSTALLER_ZIP}[fault]
        assets = [asset for asset in assets if asset['name'] != missing]
    elif fault == 'wrong_digest':
        assets[0]['digest'] = 'sha256:' + '0' * 64
    elif fault == 'missing_digest':
        assets[0].pop('digest')
    elif fault == 'wrong_size':
        assets[0]['size'] += 1
    elif fault == 'upload_in_progress':
        assets[0]['state'] = 'starter'
    elif fault == 'duplicate_asset':
        assets.append(dict(assets[0]))
    elif fault == 'wrong_manifest_commit':
        manifest['commit'] = 'b' * 40
    elif fault == 'duplicate_expected':
        manifest['assets'] = expected + [dict(expected[0])]
    elif fault == 'excess_assets':
        manifest['assets'] = expected + [{**expected[0], 'name': f'extra-{i}.zip'} for i in range(20)]
    (tmp_path / 'release-assets.json').write_text(json.dumps(manifest))
    (tmp_path / 'smartai-release-notes.md').write_text('Verified release notes\n')
    harness = r'''
const p = require('./fixture.json');
const context = {repo: {owner: 'test-owner', repo: 'test-repo'}, sha: p.sha,
  ref: p.tag === 'latest' ? 'refs/heads/master' : `refs/tags/${p.tag}`};
const calls = [];
let moved = false;
const github = {
  paginate: async (method, args) => { calls.push(['assets', args]); return p.assets; },
  rest: {
    repos: {
      getReleaseByTag: async args => { calls.push(['release', args]); return {data: {id: 42}}; },
      listReleaseAssets: function () {},
      deleteReleaseAsset: async args => {
        calls.push(['delete', args]);
        if (p.fault === 'delete_failed') throw Error('Cannot delete obsolete asset');
      },
      compareCommits: async args => {
        calls.push(['compare', args]);
        return {data: {status: p.fault === 'newer_release' ? 'behind' : p.fault === 'unrelated_release' ? 'diverged' : 'ahead'}};
      },
      updateRelease: async args => { calls.push(['publish', args]); },
    },
    git: {
      updateRef: async args => { moved = true; calls.push(['update', args]); },
      getRef: async args => {
        calls.push(['getRef', args]);
        const branch = args.ref.startsWith('heads/');
        const sha = branch ? (p.fault === 'superseded_head' ? 'b'.repeat(40) : context.sha)
          : (moved && p.fault !== 'tag_mismatch' ? context.sha : 'c'.repeat(40));
        return {data: {object: {type: 'commit', sha}}};
      },
    },
  },
};
const core = {notice: message => calls.push(['notice', message])};
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
new AsyncFunction('github', 'context', 'core', 'require', p.script)(github, context, core, require)
  .then(() => process.stdout.write(JSON.stringify({calls, error: null})))
  .catch(error => process.stdout.write(JSON.stringify({calls, error: error.message})));
'''
    (tmp_path / 'fixture.json').write_text(json.dumps({'script': final_script(), 'assets': assets, 'sha': SHA, 'fault': fault, 'tag': tag}))
    (tmp_path / 'gate.js').write_text(harness)
    result = subprocess.run([node, 'gate.js'], cwd=tmp_path, env={**os.environ, 'RELEASE_TAG': tag, 'RELEASE_TITLE': 'Verified release'},
                            capture_output=True, text=True, timeout=30, check=True)
    parsed = json.loads(result.stdout)
    parsed['assets'] = assets
    return parsed


def test_successful_release_removes_all_superseded_formats_after_verification(tmp_path):
    result = run_release_gate(tmp_path)
    assert result['error'] is None
    calls = result['calls']
    deleted = {args['asset_id'] for method, args in calls if method == 'delete'}
    assert {asset['name'] for asset in result['assets'] if asset['id'] in deleted} == OBSOLETE
    assert [args for method, args in calls if method == 'update'] == [{
        'owner': 'test-owner', 'repo': 'test-repo', 'ref': 'tags/latest', 'sha': SHA, 'force': True,
    }]
    order = [method for method, _ in calls]
    assert order.index('assets') < order.index('delete') < order.index('update') < order.index('publish')
    assert next(args for method, args in calls if method == 'publish')['body'] == 'Verified release notes\n'


@pytest.mark.parametrize('fault', ['missing_part', 'missing_package', 'missing_installer', 'wrong_digest', 'missing_digest',
                                  'wrong_size', 'upload_in_progress', 'duplicate_asset', 'wrong_manifest_commit',
                                  'duplicate_expected', 'excess_assets', 'superseded_head', 'newer_release', 'unrelated_release'])
def test_bad_or_superseded_replacement_never_prunes_or_publishes(tmp_path, fault):
    result = run_release_gate(tmp_path, fault=fault)
    assert result['error']
    assert not any(method in {'delete', 'update', 'publish'} for method, _ in result['calls'])


def test_failed_cleanup_does_not_move_tag_or_publish(tmp_path):
    result = run_release_gate(tmp_path, fault='delete_failed')
    assert 'Cannot delete' in result['error']
    assert not any(method in {'update', 'publish'} for method, _ in result['calls'])


def test_immutable_version_tag_is_never_rewritten(tmp_path):
    result = run_release_gate(tmp_path, tag='v1.2.3')
    assert result['error'] is None
    assert not any(method in {'update', 'getRef'} for method, _ in result['calls'])


def test_tag_readback_must_confirm_packaged_sha_before_publication(tmp_path):
    result = run_release_gate(tmp_path, fault='tag_mismatch')
    assert 'Rolling tag did not move' in result['error']
    assert not any(method in {'publish', 'notice'} for method, _ in result['calls'])


@pytest.mark.parametrize('name', ['SmartAI-macOS-arm64.dmg', 'SmartAI-native-cpu-Linux-X64.zip',
                                 'Reassemble-SmartAI-Linux-x86_64-portable.zip.ps1', 'SmartAI-macOS-arm64.bundle.json',
                                 'Install-SmartAI-macOS-Full.command'])
def test_build_rejects_reintroduced_public_clutter(tmp_path, name):
    package_fixture(tmp_path / 'dist')
    (tmp_path / 'dist' / name).write_bytes(b'obsolete distribution')
    with pytest.raises(RuntimeError, match='Unexpected files'):
        verify(tmp_path / 'dist', SHA, include_installer=True)


def test_release_rejects_installer_zip_without_execute_permission(tmp_path):
    root = tmp_path / 'dist'
    package_fixture(root)
    path = root / INSTALLER_ZIP
    with zipfile.ZipFile(path) as packaged:
        script = packaged.read(INSTALLER)
    info = zipfile.ZipInfo(INSTALLER)
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    with zipfile.ZipFile(path, 'w') as packaged:
        packaged.writestr(info, script)
    with pytest.raises(RuntimeError, match='executable command'):
        verify(root, SHA, include_installer=True)


def test_pipeline_bundles_native_artifacts_and_uploads_only_verified_allowlist():
    source = WORKFLOW.read_text(encoding='utf-8')
    package = source.split('\n  package:\n', 1)[1].split('\n  build-and-release:\n', 1)[0]
    assert 'needs: [preflight, native-build, reuse-native]' in package
    assert 'SMARTAI_NATIVE_RUNTIME_DIR:' in package
    assert 'run-id: ${{ needs.preflight.outputs.verified_run_id || github.run_id }}' in package
    assert source.index('- name: Build one Mac installer') < source.index('- name: Upload only the verified download set') < source.index('- name: ' + STEP_NAME)
    assert 'build_bonsai2_bundled_zip.py' not in source and 'build_macos_noadmin_portable.py' not in source
    assert "for item in rows:" in source and "str(Path('dist') / item['name'])" in source
    assert 'cancel-in-progress: false' in source
