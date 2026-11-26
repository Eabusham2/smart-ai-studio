"""Pinned portable-archive installation plus macOS offline upgrade/rollback."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import zipfile

import pytest

from tools.build_macos_full_installer import ARCHIVE, INSTALLER, RECEIPT, build


COMMIT = '2' * 40


def _installer(tmp_path):
    app = tmp_path / 'source/SmartAI.app'
    files = {
        'Contents/Info.plist': '<plist version="1.0"><dict/></plist>',
        'Contents/MacOS/applet': '#!/bin/sh\nexit 0\n',
        'Contents/Resources/app/app_gui.py': '# synthetic app\n',
        'Contents/Resources/app/launch_studio.sh': '#!/bin/sh\nexit 0\n',
        'Contents/Resources/app/runtime/python/bin/python3': '#!/bin/sh\nexit 0\n',
        'Contents/Resources/app/runtime/python/.smartai-deps-ready': 'fixture\n',
    }
    archive = tmp_path / ARCHIVE
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as packaged:
        for name, content in files.items():
            path = app / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            path.chmod(0o755 if content.startswith('#!/bin/sh') else 0o644)
            packaged.write(path, 'SmartAI.app/' + name)
    data = {'schema': 1, 'target': 'macOS-arm64', 'commit': COMMIT,
            'gui_smoke': 'passed', 'setup_smoke': 'passed', 'files': [
                {'name': ARCHIVE, 'size': archive.stat().st_size,
                 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}]}
    (tmp_path / RECEIPT).write_text(json.dumps(data))
    script = build(tmp_path / RECEIPT, tmp_path, repo='Eabusham2/smart-ai-studio',
                   tag='latest', commit=COMMIT)
    return script, data


def test_full_installer_is_executable_pinned_and_has_valid_shell_syntax(tmp_path):
    script, _ = _installer(tmp_path)
    content = script.read_text()
    assert COMMIT in content
    assert hashlib.sha256((tmp_path / ARCHIVE).read_bytes()).hexdigest() in content
    assert 'TARGET="${SMARTAI_FULL_TARGET-}"' in content
    assert '0) TARGET="$HOME/Applications/SmartAI.app"' in content
    assert 'KIT_' not in content and 'NoAdmin-Portable-Kit' not in content
    assert '/usr/bin/open "$TARGET"' in content
    assert 'rm -rf "$CACHE_ROOT"' not in content
    assert 'xattr' not in content and 'spctl' not in content
    assert not (tmp_path / (INSTALLER + '.sha256')).exists()
    if os.name == 'posix':
        assert script.stat().st_mode & stat.S_IXUSR
        subprocess.run(['/bin/bash', '-n', str(script)], check=True)


def test_full_installer_rejects_mismatched_release_and_shell_injection(tmp_path):
    _installer(tmp_path)
    for overrides in ({'repo': 'owner/repo;echo bad'}, {'tag': 'latest$(id)'}, {'commit': 'not-a-sha'}):
        values = dict(repo='Eabusham2/smart-ai-studio', tag='latest', commit=COMMIT)
        values.update(overrides)
        with pytest.raises(ValueError):
            build(tmp_path / RECEIPT, tmp_path, **values)
    with pytest.raises(ValueError, match='verified release'):
        build(tmp_path / RECEIPT, tmp_path, repo='Eabusham2/smart-ai-studio',
              tag='latest', commit='a' * 40)


def test_installer_is_not_emitted_for_corrupt_or_unverified_app_archive(tmp_path):
    script, data = _installer(tmp_path)
    script.unlink()
    (tmp_path / ARCHIVE).write_bytes(b'corrupted archive')
    with pytest.raises(ValueError, match='verified receipt'):
        build(tmp_path / RECEIPT, tmp_path, repo='Eabusham2/smart-ai-studio',
              tag='latest', commit=COMMIT)
    assert not script.exists()
    data['gui_smoke'] = 'failed'
    (tmp_path / RECEIPT).write_text(json.dumps(data))
    with pytest.raises(ValueError, match='verified release'):
        build(tmp_path / RECEIPT, tmp_path, repo='Eabusham2/smart-ai-studio',
              tag='latest', commit=COMMIT)


NATIVE_MAC = sys.platform == 'darwin' and platform.machine().lower() == 'arm64'


def _offline_env(tmp_path, data):
    cache = tmp_path / 'cache with spaces'
    downloads = cache / ('app-downloads-' + COMMIT[:12])
    downloads.mkdir(parents=True)
    (downloads / '.full-installer-owner').write_text(COMMIT + '\n')
    shutil.copy2(tmp_path / ARCHIVE, downloads / ARCHIVE)
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


def _system_candidate(script, path):
    """Keep native tests independent of an actual /Applications installation."""
    script.write_text(script.read_text().replace('SYSTEM_APP="/Applications/SmartAI.app"',
                                                'SYSTEM_APP=' + shlex.quote(str(path))))


@pytest.mark.skipif(not NATIVE_MAC, reason='Runs actual macOS JXA, ditto, shasum and the full bundled installer')
def test_native_full_install_cleans_owned_downloads_preserves_data_and_reopens_offline(tmp_path):
    script, data = _installer(tmp_path)
    env, target, downloads = _offline_env(tmp_path, data)
    result = subprocess.run(['/bin/bash', str(script)], env=env, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, (result.stdout, result.stderr)
    code = target / 'Contents/Resources/app'
    assert (code / '.smartai-full-install-commit').read_text().strip() == COMMIT
    assert (code / 'runtime/python/bin/python3').is_file()
    assert not (code / 'preloaded_models').exists()
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


@pytest.mark.skipif(not NATIVE_MAC, reason='Verifies the actual default destination under an isolated Mac home')
def test_native_installer_keeps_existing_user_applications_without_admin_access(tmp_path):
    script, data = _installer(tmp_path)
    env, target, _ = _offline_env(tmp_path, data)
    home = tmp_path / 'user home with spaces'
    home.mkdir()
    shutil.move(str(target.parent), home / 'Applications')
    env.pop('SMARTAI_FULL_TARGET')
    env['HOME'] = str(home)
    _system_candidate(script, tmp_path / 'no system app/SmartAI.app')
    result = subprocess.run(['/bin/bash', str(script)], env=env, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert (home / 'Applications/SmartAI.app/Contents/Resources/app/app_gui.py').is_file()
    assert 'administrator password' not in result.stdout
    assert (home / 'Applications/SmartAI_Data/memory.db').read_bytes() == b'preserved chat and learning data'


@pytest.mark.skipif(not NATIVE_MAC, reason='Checks a fresh Mac installation with the actual shell and ditto')
def test_native_fresh_installer_uses_user_applications_without_admin_access(tmp_path):
    script, data = _installer(tmp_path)
    env, _, _ = _offline_env(tmp_path, data)
    home = tmp_path / 'fresh user home'; home.mkdir()
    env.pop('SMARTAI_FULL_TARGET'); env['HOME'] = str(home)
    _system_candidate(script, tmp_path / 'no system app/SmartAI.app')
    result = subprocess.run(['/bin/bash', str(script)], env=env, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert (home / 'Applications/SmartAI.app/Contents/Resources/app/app_gui.py').is_file()
    assert not list((home / 'Applications').glob('SmartAI.backup-*'))
    assert 'administrator password' not in result.stdout


@pytest.mark.skipif(not NATIVE_MAC, reason='Checks preservation of recognized pre-cleanup Mac install locations')
@pytest.mark.parametrize('legacy', [False, True])
def test_native_installer_reuses_system_or_legacy_path_and_saved_data(tmp_path, legacy):
    script, data = _installer(tmp_path)
    env, target, _ = _offline_env(tmp_path, data)
    home = tmp_path / 'isolated user home'; home.mkdir()
    if legacy:
        destination = home / 'Applications/SmartAI-Portable/oldrelease12'
        destination.parent.mkdir(parents=True)
        shutil.move(str(target.parent), destination)
        target = destination / 'SmartAI.app'
        _system_candidate(script, tmp_path / 'no system app/SmartAI.app')
    else:
        _system_candidate(script, target)
    env.pop('SMARTAI_FULL_TARGET'); env['HOME'] = str(home)
    result = subprocess.run(['/bin/bash', str(script)], env=env, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert (target / 'Contents/Resources/app/app_gui.py').is_file()
    assert (target.parent / 'SmartAI_Data/memory.db').read_bytes() == b'preserved chat and learning data'
    assert len(list(target.parent.glob('SmartAI.backup-*.app'))) == 1
    assert not (home / 'Applications/SmartAI.app').exists()


@pytest.mark.skipif(not NATIVE_MAC, reason='Checks chooser results before a native Mac installer changes anything')
@pytest.mark.parametrize('selection', ['system', 'cancel', 'unrecognized'])
def test_native_multiple_copies_require_a_valid_explicit_selection(tmp_path, selection):
    script, data = _installer(tmp_path)
    env, system_app, downloads = _offline_env(tmp_path, data)
    home = tmp_path / 'user home'; home.mkdir()
    user_app = home / 'Applications/SmartAI.app'; user_app.mkdir(parents=True)
    (user_app / 'user-copy.txt').write_text('leave this copy')
    chooser = tmp_path / 'fixture chooser.sh'
    chooser.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$SMARTAI_TEST_CHOICES"\n'
                       'printf "%s\\n" "$SMARTAI_TEST_CHOICE"\n')
    chooser.chmod(0o755)
    script.write_text(script.read_text().replace('/usr/bin/osascript', shlex.quote(str(chooser))))
    _system_candidate(script, system_app)
    env.pop('SMARTAI_FULL_TARGET'); env['HOME'] = str(home)
    env['SMARTAI_TEST_CHOICES'] = str(tmp_path / 'choices.txt')
    env['SMARTAI_TEST_CHOICE'] = (str(system_app) if selection == 'system' else
                                 '' if selection == 'cancel' else str(tmp_path / 'other/SmartAI.app'))
    result = subprocess.run(['/bin/bash', str(script)], env=env, capture_output=True,
                            text=True, timeout=120)
    assert Path(env['SMARTAI_TEST_CHOICES']).read_text().splitlines() == ['-', str(user_app), str(system_app)]
    assert (user_app / 'user-copy.txt').read_text() == 'leave this copy'
    assert (system_app.parent / 'SmartAI_Data/memory.db').read_bytes() == b'preserved chat and learning data'
    if selection == 'system':
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert (system_app / 'Contents/Resources/app/app_gui.py').is_file()
    else:
        assert result.returncode != 0 and 'No recognized installation was selected' in result.stderr
        assert 'SMARTAI_FULL_TARGET=' in result.stderr
        assert (system_app / 'previous-version.txt').read_text() == 'previous app'
        assert downloads.is_dir()
        assert not list(Path(env['SMARTAI_FULL_CACHE']).glob('work.*'))


@pytest.mark.skipif(not NATIVE_MAC, reason='Checks native installer fail-closed behavior before extraction')
def test_native_full_installer_bad_archive_preserves_existing_install_and_unrelated_cache(tmp_path):
    script, data = _installer(tmp_path)
    env, target, downloads = _offline_env(tmp_path, data)
    (downloads / ARCHIVE).write_bytes(b'corrupted-download')
    result = subprocess.run(['/bin/bash', str(script)], env=env, capture_output=True,
                            text=True, timeout=30)
    assert result.returncode != 0 and 'missing or corrupt' in result.stderr
    assert (target / 'previous-version.txt').read_text() == 'previous app'
    assert (Path(env['SMARTAI_FULL_CACHE']) / 'unrelated-user-file').is_file()
    assert not list(Path(env['SMARTAI_FULL_CACHE']).glob('work.*'))
