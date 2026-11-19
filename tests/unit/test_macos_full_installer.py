"""Pinned installer integrity plus real macOS offline assembly/upgrade/rollback."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys

import pytest

from tools.build_macos_full_installer import INSTALLER, build
from tools.build_macos_noadmin_portable import (
    ASSEMBLER, KIT, MANIFEST, REPOS, build as build_kit,
)
from tests.unit.test_macos_noadmin_portable import _sample_bundle


COMMIT = '2' * 40


def _installer(tmp_path):
    data = _sample_bundle(tmp_path, full_app=True)
    from build_bonsai2_bundled_zip import _write_reassembler
    _write_reassembler(tmp_path, data['archive_name'], MANIFEST)
    kit = build_kit(tmp_path / MANIFEST, tmp_path / ASSEMBLER, tmp_path,
                    repo='Eabusham2/smart-ai-studio', tag='latest', commit=COMMIT)
    script = build(kit, tmp_path, repo='Eabusham2/smart-ai-studio',
                   tag='latest', commit=COMMIT)
    return script, data


def test_full_installer_is_executable_pinned_and_has_valid_shell_syntax(tmp_path):
    script, _ = _installer(tmp_path)
    content = script.read_text()
    assert COMMIT in content
    assert hashlib.sha256((tmp_path / KIT).read_bytes()).hexdigest() in content
    assert 'TARGET="${SMARTAI_FULL_TARGET:-/Applications/SmartAI.app}"' in content
    assert 'SMARTAI_PORTABLE_INSTALL_ONLY=1' in content
    assert '/usr/bin/open "$TARGET"' in content
    assert 'rm -rf "$CACHE_ROOT"' not in content
    assert 'xattr' not in content and 'spctl' not in content
    assert (tmp_path / (INSTALLER + '.sha256')).read_text() == (
        f'{hashlib.sha256(script.read_bytes()).hexdigest()}  {INSTALLER}\n')
    if os.name == 'posix':
        assert script.stat().st_mode & stat.S_IXUSR
        subprocess.run(['/bin/bash', '-n', str(script)], check=True)


def test_full_installer_rejects_mismatched_release_and_shell_injection(tmp_path):
    _installer(tmp_path)
    for overrides in ({'repo': 'owner/repo;echo bad'}, {'tag': 'latest$(id)'}, {'commit': 'not-a-sha'}):
        values = dict(repo='Eabusham2/smart-ai-studio', tag='latest', commit=COMMIT)
        values.update(overrides)
        with pytest.raises(ValueError):
            build(tmp_path / KIT, tmp_path, **values)
    with pytest.raises(ValueError, match='version isolation'):
        build(tmp_path / KIT, tmp_path, repo='Eabusham2/smart-ai-studio',
              tag='latest', commit='a' * 40)


NATIVE_MAC = sys.platform == 'darwin' and platform.machine().lower() == 'arm64'


def _offline_env(tmp_path, data):
    cache = tmp_path / 'cache with spaces'
    downloads = cache / ('model-downloads-' + COMMIT[:12])
    downloads.mkdir(parents=True)
    (downloads / '.full-installer-owner').write_text(COMMIT + '\n')
    shutil.copy2(tmp_path / KIT, downloads / KIT)
    for part in data['parts']:
        shutil.copy2(tmp_path / part['name'], downloads / part['name'])
    (cache / 'unrelated-user-file').write_text('keep me')
    apps = tmp_path / 'Applications with spaces'
    target = apps / 'SmartAI.app'
    target.mkdir(parents=True)
    (target / 'previous-version.txt').write_text('previous app')
    user_data = apps / 'SmartAI_Data'
    user_data.mkdir()
    (user_data / 'memory.db').write_bytes(b'preserved chat and learning data')
    return {**os.environ, 'SMARTAI_FULL_CACHE': str(cache),
            'SMARTAI_FULL_TARGET': str(target), 'SMARTAI_FULL_INSTALL_ONLY': '1',
            'SMARTAI_FULL_OFFLINE': '1', 'PATH': '/usr/bin:/bin'}, target, downloads


@pytest.mark.skipif(not NATIVE_MAC, reason='Runs actual macOS JXA, ditto, shasum and the full bundled installer')
def test_native_full_install_assembles_both_models_cleans_owned_downloads_and_reopens_offline(tmp_path):
    script, data = _installer(tmp_path)
    env, target, downloads = _offline_env(tmp_path, data)
    result = subprocess.run(['/bin/bash', str(script)], env=env, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, (result.stdout, result.stderr)
    code = target / 'Contents/Resources/app'
    assert (code / '.smartai-full-install-commit').read_text().strip() == COMMIT
    for repo in REPOS:
        model = code / 'preloaded_models' / repo.replace('/', '--')
        assert (model / 'config.json').is_file() and (model / 'model.safetensors').stat().st_size > 0
    assert not downloads.exists()
    assert (Path(env['SMARTAI_FULL_CACHE']) / 'unrelated-user-file').read_text() == 'keep me'
    assert not list(Path(env['SMARTAI_FULL_CACHE']).glob('work.*'))
    backups = list(target.parent.glob('SmartAI.backup-*.app'))
    assert len(backups) == 1 and (backups[0] / 'previous-version.txt').read_text() == 'previous app'
    assert (target.parent / 'SmartAI_Data/memory.db').read_bytes() == b'preserved chat and learning data'
    again = subprocess.run(['/bin/bash', str(script)], env=env, capture_output=True,
                           text=True, timeout=30)
    assert again.returncode == 0, again.stderr
    assert 'already installed' in again.stdout
    assert not list(target.parent.glob('.SmartAI-*'))
    # Reopening an applet can restart its Python session. An idempotent rerun
    # must leave an already-running model/training session alone.
    fake_ps = tmp_path / 'running-process.sh'
    fake_ps.write_text('#!/bin/bash\nprintf "%s\\n" "python $SMARTAI_FULL_TARGET/Contents/Resources/app/app_gui.py"\n')
    fake_ps.chmod(0o755)
    fail_open = tmp_path / 'must-not-open.sh'
    fail_open.write_text('#!/bin/bash\nexit 97\n')
    fail_open.chmod(0o755)
    script.write_text(script.read_text().replace('/bin/ps ax -o command=', '"' + str(fake_ps) + '"')
                      .replace('/usr/bin/open', '"' + str(fail_open) + '"'))
    running = subprocess.run(['/bin/bash', str(script)], env={**env, 'SMARTAI_FULL_INSTALL_ONLY': '0'},
                             capture_output=True, text=True, timeout=30)
    assert running.returncode == 0 and 'already running' in running.stdout


@pytest.mark.skipif(not NATIVE_MAC, reason='Exercises actual macOS assembly before a controlled final-rename failure')
def test_native_full_install_restores_old_app_when_publication_move_fails(tmp_path):
    script, data = _installer(tmp_path)
    env, target, downloads = _offline_env(tmp_path, data)
    move = tmp_path / 'fail-final-move.sh'
    move.write_text('#!/bin/bash\ncase "$1" in */.SmartAI-incoming-*) exit 42;; esac\nexec /bin/mv "$@"\n')
    move.chmod(0o755)
    # Only replace the publication move, not the real portable assembly path.
    script.write_text(script.read_text().replace('install_command /bin/mv "$INCOMING" "$TARGET"',
                                                'install_command "' + str(move) + '" "$INCOMING" "$TARGET"'))
    result = subprocess.run(['/bin/bash', str(script)], env=env, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode != 0
    assert (target / 'previous-version.txt').read_text() == 'previous app'
    assert downloads.is_dir()  # Reuse verified downloads on retry.
    assert (target.parent / 'SmartAI_Data/memory.db').read_bytes() == b'preserved chat and learning data'
    assert not list(target.parent.glob('.SmartAI-*'))


@pytest.mark.skipif(not NATIVE_MAC, reason='Checks native installer fail-closed behavior before extraction')
def test_native_full_installer_bad_kit_preserves_existing_install_and_unrelated_cache(tmp_path):
    script, data = _installer(tmp_path)
    env, target, downloads = _offline_env(tmp_path, data)
    (downloads / KIT).write_bytes(b'corrupted-download')
    result = subprocess.run(['/bin/bash', str(script)], env=env, capture_output=True,
                            text=True, timeout=30)
    assert result.returncode != 0 and 'missing or corrupt' in result.stderr
    assert (target / 'previous-version.txt').read_text() == 'previous app'
    assert (Path(env['SMARTAI_FULL_CACHE']) / 'unrelated-user-file').is_file()
    assert not list(Path(env['SMARTAI_FULL_CACHE']).glob('work.*'))
