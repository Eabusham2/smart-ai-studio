"""Exercise the shipped shell downloader with tiny local, model-free archives."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import zipfile

import pytest

from tools.build_release_package import split_large
from tools.portable_download import write_linux_downloader


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture
def release(tmp_path):
    source = tmp_path / 'release files'; source.mkdir()
    name = 'SmartAI-Linux-x86_64-portable.zip'
    archive = source / name
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('SmartAI-Linux/SmartAI.AppDir/AppRun', '#!/bin/sh\nprintf portable-ok\n')
        bundle.writestr('SmartAI-Linux/SmartAI.AppDir/usr/bin/fixture.txt', 'portable payload')
    expected = archive.read_bytes()
    split_large(archive, limit=256)
    manifest = source / (name + '.parts.json')
    return source, manifest, name, expected


@pytest.fixture
def local_release(release):
    source, manifest, name, expected = release
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(_QuietHandler, directory=str(source)))
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        yield source, manifest, name, expected, f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)


def _script(release, output):
    _, manifest, _, _, url = release
    output.mkdir(parents=True)
    return write_linux_downloader(manifest, output, repository='Eabusham2/smart-ai-studio',
                                 tag='latest', download_base=url)


def _run(script, *args, env_overrides=None):
    env = dict(os.environ, NO_PROXY='127.0.0.1', no_proxy='127.0.0.1')
    env.update(env_overrides or {})
    return subprocess.run(['sh', str(script), *map(str, args)], capture_output=True,
                          text=True, timeout=30, env=env)


requires_shell = pytest.mark.skipif(
    not sys.platform.startswith('linux') or any(shutil.which(tool) is None for tool in ('sh', 'curl', 'unzip')),
    reason='The Linux download helper needs its native shell/download/extraction tools')


@requires_shell
def test_download_joins_and_extracts_without_system_python(local_release, tmp_path):
    script = _script(local_release, tmp_path / 'installer with spaces')
    output = tmp_path / 'portable apps with spaces'
    # Only shell/download/ZIP tools are available: no Python executable in PATH.
    shell_tools = tmp_path / 'shell tools'; shell_tools.mkdir()
    for command in ('sh', 'curl', 'unzip', 'sha256sum', 'dirname', 'mkdir',
                    'mktemp', 'rm', 'wc', 'tr', 'cat', 'mv', 'chmod'):
        (shell_tools / command).symlink_to(shutil.which(command))
    result = _run(script, '--output-dir', output, env_overrides={'PATH': str(shell_tools)})
    assert result.returncode == 0, result.stderr
    _, _, name, expected, _ = local_release
    assert (output / name).read_bytes() == expected
    app = output / 'SmartAI-Linux/SmartAI.AppDir/AppRun'
    assert os.access(app, os.X_OK)
    assert subprocess.check_output([str(app)], text=True) == 'portable-ok'
    assert 'Launch with:' in result.stdout
    assert not list(output.glob('.smartai-download.*'))


@requires_shell
def test_download_only_reuses_verified_zip_and_preserves_existing_app(local_release, tmp_path):
    script = _script(local_release, tmp_path / 'helper')
    output = tmp_path / 'downloads'; (output / 'SmartAI-Linux').mkdir(parents=True)
    marker = output / 'SmartAI-Linux/user-data'; marker.write_text('keep')
    result = _run(script, '--output-dir', output, '--download-only')
    assert result.returncode == 0, result.stderr
    # Remove parts at the server: the second invocation must reuse its verified ZIP.
    for part in local_release[0].glob('*.part[0-9]*'):
        part.unlink()
    again = _run(script, '--output-dir', output, '--download-only')
    assert again.returncode == 0, again.stderr
    assert 'Reusing the verified' in again.stdout
    blocked = _run(script, '--output-dir', output)
    assert blocked.returncode != 0 and 'already exists' in blocked.stderr
    assert marker.read_text() == 'keep'


@requires_shell
@pytest.mark.parametrize('corruption', ['part', 'full_hash'])
def test_corruption_never_installs_or_publishes_a_zip(local_release, tmp_path, corruption):
    source, manifest, name, _, _ = local_release
    if corruption == 'part':
        part = next(source.glob('*.part000'))
        part.write_bytes(b'x' * part.stat().st_size)
    else:
        record = json.loads(manifest.read_text()); record['sha256'] = '0' * 64
        manifest.write_text(json.dumps(record))
    script = _script(local_release, tmp_path / 'helper')
    output = tmp_path / 'downloads'
    result = _run(script, '--output-dir', output)
    assert result.returncode != 0 and 'checksum mismatch' in result.stderr.lower()
    assert not (output / name).exists()
    assert not (output / 'SmartAI-Linux').exists()
    assert not list(output.glob('.smartai-download.*'))


@requires_shell
def test_existing_different_zip_is_preserved(local_release, tmp_path):
    script = _script(local_release, tmp_path / 'helper')
    output = tmp_path / 'downloads'; output.mkdir()
    archive = output / local_release[2]; archive.write_bytes(b'keep-existing-file')
    result = _run(script, '--output-dir', output, '--download-only')
    assert result.returncode != 0 and 'different checksum' in result.stderr
    assert archive.read_bytes() == b'keep-existing-file'
    assert not (output / 'SmartAI-Linux').exists()


@pytest.mark.parametrize('field,value', [
    ('archive', '../escape.zip'), ('name', '../escape'),
    ('name', 'SmartAI-Linux-x86_64-portable.zip.part001'), ('size', -1),
    ('sha256', '$(touch injected)'),
])
def test_generator_rejects_untrusted_names_order_sizes_and_hashes(release, tmp_path, field, value):
    _, manifest, _, _ = release
    record = json.loads(manifest.read_text())
    if field == 'archive':
        record[field] = value
    else:
        record['parts'][0][field] = value
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        write_linux_downloader(manifest, tmp_path, repository='Eabusham2/smart-ai-studio', tag='latest')
    assert not (tmp_path / 'Download-SmartAI-Linux.sh').exists()


def test_production_helper_requires_https_for_download_and_redirect(release, tmp_path):
    script = write_linux_downloader(release[1], tmp_path,
                                   repository='Eabusham2/smart-ai-studio', tag='latest')
    body = script.read_text()
    assert "--proto '=https' --proto-redir '=https'" in body
    assert '--https-only' in body
    with pytest.raises(ValueError, match='HTTPS'):
        write_linux_downloader(release[1], tmp_path, repository='Eabusham2/smart-ai-studio',
                               tag='latest', download_base='http://example.com/releases')
